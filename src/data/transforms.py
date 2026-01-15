import torch
import torch.nn.functional as F
import random

class VideoTransform:
    def __init__(self, spatial_size: int = 224, training: bool = True):
        self.spatial_size = spatial_size
        self.training = training
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (C, T, H, W)
        frames = self._resize(frames, self.spatial_size)

        if self.training:
            frames = self._random_crop(frames)
            frames = self._horizontal_flip(frames)
            frames = self._color_jitter(frames)
            frames = self._speed_perturb(frames)
        else:
            frames = self._center_crop(frames)

        frames = (frames - self.mean) / self.std
        return frames

    def _resize(self, frames: torch.Tensor, size: int) -> torch.Tensor:
        c, t, h, w = frames.shape
        scale = size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frames = frames.view(c * t, 1, h, w)
        frames = F.interpolate(frames, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return frames.view(c, t, new_h, new_w)

    def _random_crop(self, frames: torch.Tensor) -> torch.Tensor:
        c, t, h, w = frames.shape
        if h > self.spatial_size and w > self.spatial_size:
            top = random.randint(0, h - self.spatial_size)
            left = random.randint(0, w - self.spatial_size)
            return frames[:, :, top:top+self.spatial_size, left:left+self.spatial_size]
        return self._center_crop(frames)

    def _center_crop(self, frames: torch.Tensor) -> torch.Tensor:
        c, t, h, w = frames.shape
        top = (h - self.spatial_size) // 2
        left = (w - self.spatial_size) // 2
        top = max(0, top)
        left = max(0, left)
        cropped = frames[:, :, top:top+self.spatial_size, left:left+self.spatial_size]
        if cropped.shape[2] < self.spatial_size or cropped.shape[3] < self.spatial_size:
            cropped = F.interpolate(
                cropped.view(c * t, 1, cropped.shape[2], cropped.shape[3]),
                size=(self.spatial_size, self.spatial_size),
                mode='bilinear', align_corners=False
            ).view(c, t, self.spatial_size, self.spatial_size)
        return cropped

    def _horizontal_flip(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            return frames.flip(dims=[3])
        return frames

    def _color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)

        frames = frames * brightness
        mean = frames.mean()
        frames = (frames - mean) * contrast + mean
        return frames.clamp(0, 1)

    def _speed_perturb(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            return frames

        c, t, h, w = frames.shape
        speed = random.uniform(0.8, 1.2)
        new_t = int(t * speed)

        if new_t == t:
            return frames

        frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W)
        frames = F.interpolate(frames, size=(h, w), mode='bilinear', align_corners=False)

        indices = torch.linspace(0, t - 1, new_t)
        indices_floor = indices.long().clamp(0, t - 1)

        resampled = frames[indices_floor]

        if new_t > t:
            step = new_t // t
            resampled = resampled[::step][:t]
        elif new_t < t:
            pad = t - new_t
            resampled = torch.cat([resampled, resampled[-1:].expand(pad, -1, -1, -1)])

        return resampled[:t].permute(1, 0, 2, 3)  # (C, T, H, W)
