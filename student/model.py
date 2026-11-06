"""PyTorch counterpart of mobile/native/candidate_probe.cpp, architecture v0."""
import math

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'recurrent_conditioned_rgb_v0'
SIDE, WIDTH, HISTORY, MEL, HEADS = 256, 128, 8, 40, 4


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(WIDTH, WIDTH)
        self.k = nn.Linear(WIDTH, WIDTH)
        self.v = nn.Linear(WIDTH, WIDTH)
        self.out = nn.Linear(WIDTH, WIDTH)

    def forward(self, tokens):
        batch, length, _ = tokens.shape
        q, k, v = [layer(tokens).reshape(batch, length, HEADS, WIDTH // HEADS).transpose(1, 2)
                   for layer in (self.q, self.k, self.v)]
        weights = ((q / math.sqrt(WIDTH // HEADS)) @ k.transpose(-1, -2)).softmax(dim=-1)
        return self.out((weights @ v).transpose(1, 2).reshape(batch, length, WIDTH))


class Candidate(nn.Module):
    """One RGB frame per forward; image and audio feature preprocessing are external."""
    def __init__(self):
        super().__init__()
        self.image1 = nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False)
        self.image2 = nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False)
        self.identity = nn.Linear(32, WIDTH, bias=False)
        self.audio = nn.Linear(MEL, WIDTH, bias=False)
        self.attention = SelfAttention()
        self.audio_context = nn.Linear(HISTORY * WIDTH, WIDTH, bias=False)
        self.motion = nn.Linear(WIDTH, WIDTH, bias=False)
        self.modulate = nn.Linear(WIDTH, 32, bias=False)
        self.decoder1 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.decoder2 = nn.Conv2d(32, 16, 3, padding=1, bias=False)
        self.rgb = nn.Conv2d(16, 3, 3, padding=1, bias=False)

    def encode_identity(self, image):
        spatial = F.silu(self.image2(F.silu(self.image1(image))))
        return spatial, self.identity(spatial.mean(dim=(-2, -1)))

    def forward_frame(self, audio, state, spatial, identity):
        tokens = self.attention(F.silu(self.audio(audio)))
        context = self.audio_context(tokens.flatten(1))
        next_state = F.silu(self.motion(context + state + identity))
        fused = spatial + self.modulate(next_state)[:, :, None, None]
        decoded = F.silu(self.decoder1(fused))
        decoded = F.interpolate(decoded, size=(SIDE // 2, SIDE // 2), mode='bilinear', align_corners=False)
        decoded = F.silu(self.decoder2(decoded))
        decoded = F.interpolate(decoded, size=(SIDE, SIDE), mode='bilinear', align_corners=False)
        return self.rgb(decoded).sigmoid(), next_state

    def forward(self, image, audio, state):
        return self.forward_frame(audio, state, *self.encode_identity(image))

    def ncnn_weights(self):
        """Node indices and FP32 row-major matrices, matching the native graph exactly."""
        projection = {0: self.image1, 2: self.image2, 5: self.identity, 6: self.audio,
                      10: self.audio_context, 13: self.motion, 15: self.modulate,
                      18: self.decoder1, 21: self.decoder2, 24: self.rgb}
        result = {node: [layer.weight] for node, layer in projection.items()}
        result[8] = [value for layer in (self.attention.q, self.attention.k, self.attention.v, self.attention.out)
                     for value in (layer.weight, layer.bias)]
        return [result.get(node, []) for node in range(26)]
