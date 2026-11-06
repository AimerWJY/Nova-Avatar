"""Fixed-shape batch-one encoder/frame boundaries for the v2 model."""
import torch
from torch import nn


def fixed_letterbox_mask(record, data):
    """Fixed-geometry export policy, permitted only for verified zero-padding in source AND targets."""
    import numpy as np
    from pathlib import Path
    from PIL import Image, ImageOps
    from .data import sha256
    source = Path(record['source_reference'])
    if sha256(source) != record['source_reference_sha256']:
        raise ValueError('original reference required to verify letterbox geometry')
    with Image.open(source) as original:
        resized = ImageOps.contain(ImageOps.exif_transpose(original), (256, 256), Image.Resampling.LANCZOS)
        width, height = resized.size
    x, y = (256-width)//2, (256-height)//2
    mask = np.zeros((256, 256), dtype=np.float32); mask[y:y+height, x:x+width] = 1
    outside = mask == 0
    if (data['reference'][outside] != 0).any() or (data['frames'][:, outside] != 0).any():
        raise ValueError('requested mask would alter non-padding reference/teacher pixels')
    return torch.from_numpy(mask)[None, None], {
        'version': 'fixed_letterbox_mask_v1', 'content_box': [x, y, x+width, y+height],
        'source_reference_sha256': record['source_reference_sha256'],
        'fixed_geometry_only': True, 'zero_padding_in_reference_and_teacher_verified': True}


class IdentityEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        cache, identity = self.model.encode_identity(image)
        return cache[0], identity


class FrameDecoder(nn.Module):
    def __init__(self, model, content_mask=None):
        super().__init__()
        self.model = model
        self.register_buffer('content_mask', content_mask)

    def forward(self, image, spatial, identity, audio, state):
        rgb, next_state = self.model.forward_frame(audio, state, (spatial, image), identity)
        if self.content_mask is not None:
            rgb = rgb*self.content_mask
        return rgb, next_state
