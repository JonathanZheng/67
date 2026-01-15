import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path
import time

class GestureDetector:
    def __init__(self, model_path: str, device: str = None,
                 window_sec: float = 2.0, stride_sec: float = 0.5,
                 cooldown_sec: float = 1.5, threshold: float = 0.5,
                 num_frames: int = 48, spatial_size: int = 224):

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.fps = 30
        self.window_frames = int(window_sec * self.fps)
        self.stride_frames = int(stride_sec * self.fps)
        self.cooldown_sec = cooldown_sec
        self.threshold = threshold
        self.num_frames = num_frames
        self.spatial_size = spatial_size

        self.frame_buffer = deque(maxlen=self.window_frames)
        self.last_detection_time = 0
        self.score = 0
        self.frame_count = 0
        self.last_inference_frame = -self.stride_frames

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(self.device)

    def reset(self):
        self.frame_buffer.clear()
        self.last_detection_time = 0
        self.score = 0
        self.frame_count = 0
        self.last_inference_frame = -self.stride_frames

    def process_frame(self, frame: np.ndarray) -> dict:
        current_time = time.time()
        self.frame_buffer.append(frame)
        self.frame_count += 1

        result = {
            'score': self.score,
            'detected': False,
            'confidence': 0.0,
            'ready': len(self.frame_buffer) >= self.window_frames
        }

        if len(self.frame_buffer) < self.window_frames:
            return result

        if self.frame_count - self.last_inference_frame < self.stride_frames:
            return result

        if current_time - self.last_detection_time < self.cooldown_sec:
            return result

        self.last_inference_frame = self.frame_count

        with torch.no_grad():
            tensor = self._prepare_input()
            confidence = self.model(tensor).item()

        result['confidence'] = confidence

        if confidence > self.threshold:
            self.score += 1
            self.last_detection_time = current_time
            result['detected'] = True
            result['score'] = self.score

        return result

    def _prepare_input(self) -> torch.Tensor:
        frames = list(self.frame_buffer)
        n = len(frames)

        if n <= self.num_frames:
            indices = np.arange(n)
        else:
            indices = np.linspace(0, n - 1, self.num_frames, dtype=int)

        selected = [frames[i] for i in indices]
        while len(selected) < self.num_frames:
            selected.append(selected[-1])

        arr = np.stack(selected)  # (T, H, W, C)
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)

        c, t, h, w = tensor.shape
        scale = self.spatial_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        tensor = tensor.view(c * t, 1, h, w)
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
        tensor = tensor.view(c, t, new_h, new_w)

        top = (new_h - self.spatial_size) // 2
        left = (new_w - self.spatial_size) // 2
        tensor = tensor[:, :, top:top+self.spatial_size, left:left+self.spatial_size]

        if tensor.shape[2] != self.spatial_size or tensor.shape[3] != self.spatial_size:
            tensor = F.interpolate(
                tensor.view(c * t, 1, tensor.shape[2], tensor.shape[3]),
                size=(self.spatial_size, self.spatial_size),
                mode='bilinear', align_corners=False
            ).view(c, t, self.spatial_size, self.spatial_size)

        tensor = tensor.unsqueeze(0).to(self.device)  # (1, C, T, H, W)
        tensor = (tensor - self.mean) / self.std

        return tensor
