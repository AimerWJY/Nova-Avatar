"""Local learned mouth correction; fixed crop for the reviewed diagnostic portrait."""
import torch
from torch import nn
from torch.nn import functional as F

from .reference_model import ReferenceCandidate

ARCHITECTURE = 'mouth_refinement_v2'
PATCH_BOX = (103, 101, 153, 133)
CORE_BOX = (110, 108, 148, 128)


class MouthRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(8*40+128, 128), nn.SiLU(), nn.Linear(128, 32*8*8), nn.SiLU())
        self.encoder = nn.Sequential(nn.Conv2d(6, 16, 3, stride=2, padding=1), nn.SiLU(),
                                     nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.SiLU())
        self.refine = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.SiLU())
        self.output = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        # Blend only the patch boundary; leave the lips fully supervised.
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(50), indexing='ij')
        distance = torch.minimum(torch.minimum(xx, 49-xx), torch.minimum(yy, 31-yy))
        self.register_buffer('feather', (distance.float()/4).clamp(0, 1)[None, None], persistent=False)

    def forward(self, base, reference, audio, state):
        x0, y0, x1, y1 = PATCH_BOX
        crops = torch.cat((base[:, :, y0:y1, x0:x1], reference[:, :, y0:y1, x0:x1]), dim=1)
        crops = F.interpolate(crops, size=(64, 64), mode='bilinear', align_corners=False)
        encoded = self.encoder(crops)
        condition = self.condition(torch.cat((audio.flatten(1), state), dim=1)).reshape(-1, 32, 8, 8)
        condition = F.interpolate(condition, size=(16, 16), mode='bilinear', align_corners=False)
        output = self.fuse(torch.cat((encoded, condition), dim=1))
        output = self.refine(F.interpolate(output, size=(32, 32), mode='bilinear', align_corners=False))
        output = self.output(F.interpolate(output, size=(64, 64), mode='bilinear', align_corners=False)).tanh()
        output = F.interpolate(output, size=(32, 50), mode='bilinear', align_corners=False)*self.feather
        correction = F.pad(output, (x0, base.shape[-1]-x1, y0, base.shape[-2]-y1))
        return (base+correction).clamp(0, 1)


class MouthCandidate(ReferenceCandidate):
    def __init__(self):
        super().__init__()
        self.mouth = MouthRefinement()

    def forward_base_frame(self, audio, state, cache, identity):
        return super().forward_frame(audio, state, cache, identity)

    def forward_frame(self, audio, state, cache, identity):
        base, next_state = self.forward_base_frame(audio, state, cache, identity)
        return self.mouth(base, cache[1], audio, next_state), next_state


def mouth_luma(rgb):
    x0, y0, x1, y1 = CORE_BOX
    coefficients = rgb.new_tensor([.299, .587, .114])[None, :, None, None]
    return (rgb[:, :, y0:y1, x0:x1]*coefficients).sum(dim=1)


def dark_area_proxy(rgb):
    # Diagnostic proxy for this image, not landmark-based lip aperture or a sync score.
    return ((.22-mouth_luma(rgb))/.04).sigmoid().mean(dim=(-2, -1))
