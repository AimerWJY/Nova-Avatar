"""Reference-preserving warp/residual candidate. Android export is not implemented."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import SelfAttention, SIDE, WIDTH, HISTORY, MEL

ARCHITECTURE = 'reference_warp_residual_v1'


class ReferenceCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.image1 = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
        self.image2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.identity = nn.Linear(32, WIDTH, bias=False)
        self.audio = nn.Linear(MEL, WIDTH, bias=False)
        self.attention = SelfAttention()
        self.audio_context = nn.Linear(HISTORY*WIDTH, WIDTH, bias=False)
        self.motion = nn.Linear(WIDTH, WIDTH, bias=False)
        # Spatial audio conditioning instead of a constant channel-wise offset.
        self.modulate = nn.Linear(WIDTH, 32*8*8, bias=False)
        self.decoder1 = nn.Conv2d(64, 32, 3, padding=1, bias=False)
        self.decoder2 = nn.Conv2d(32, 16, 3, padding=1, bias=False)
        self.output = nn.Conv2d(16, 5, 3, padding=1)
        coordinates = 2*(torch.arange(SIDE, dtype=torch.float32)+.5)/SIDE-1
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing='ij')
        self.register_buffer('identity_grid', torch.stack((xx, yy), dim=-1)[None], persistent=False)
        self.reset_output_head()

    def reset_output_head(self):
        # The untrained model copies the reference exactly; it does not redraw it.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_identity(self, image):
        spatial = F.silu(self.image2(F.silu(self.image1(image))))
        return (spatial, image), self.identity(spatial.mean(dim=(-2, -1)))

    def forward_frame(self, audio, state, cache, identity):
        spatial, reference = cache
        tokens = self.attention(F.silu(self.audio(audio)))
        context = self.audio_context(tokens.flatten(1))
        next_state = self.motion(context+state+identity).tanh()
        modulation = self.modulate(next_state).reshape(-1, 32, 8, 8)
        modulation = F.interpolate(modulation, size=spatial.shape[-2:], mode='bilinear', align_corners=False)
        decoded = F.silu(self.decoder1(torch.cat((spatial, modulation), dim=1)))
        decoded = F.interpolate(decoded, size=(SIDE//2, SIDE//2), mode='bilinear', align_corners=False)
        decoded = F.silu(self.decoder2(decoded))
        decoded = F.interpolate(decoded, size=(SIDE, SIDE), mode='bilinear', align_corners=False)
        output = self.output(decoded)
        flow = output[:, :2].tanh().permute(0, 2, 3, 1)*(2*24/SIDE)
        warped = F.grid_sample(reference, self.identity_grid+flow, mode='bilinear',
                               padding_mode='border', align_corners=False)
        rgb = (warped+output[:, 2:].tanh()).clamp(0, 1)
        return rgb, next_state

    def forward(self, image, audio, state):
        return self.forward_frame(audio, state, *self.encode_identity(image))
