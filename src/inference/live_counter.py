import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
from pathlib import Path
import time

class LiveGestureCounter:
    def __init__(self, model_path: str, device: str = None,
                 window_frames: int = 48, stride_frames: int = 24,
                 spatial_size: int = 224, output_path: str = None):

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.spatial_size = spatial_size

        self.frame_buffer = deque(maxlen=window_frames)
        self.live_count = 0
        self.frame_idx = 0

        self.video_writer = None
        self.output_path = output_path
        self.fps = 30

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(self.device)

    def start_recording(self, output_path: str, frame_size: tuple):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, frame_size)

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def reset(self):
        self.frame_buffer.clear()
        self.live_count = 0
        self.frame_idx = 0

    def process_frame(self, frame: np.ndarray) -> dict:
        if self.video_writer:
            self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        self.frame_buffer.append(frame)
        self.frame_idx += 1

        result = {
            'live_count': self.live_count,
            'detected': False,
            'window_count': 0,
            'confidence': 0.0,
            'ready': len(self.frame_buffer) >= self.window_frames
        }

        if len(self.frame_buffer) < self.window_frames:
            return result

        if self.frame_idx % self.stride_frames != 0:
            return result

        with torch.no_grad():
            tensor = self._prepare_input()
            detection, count_logits = self.model(tensor)

            detection_prob = detection.item()
            result['confidence'] = detection_prob

            if detection_prob > 0.5:
                count = count_logits.argmax(dim=-1).item()
                self.live_count += count
                result['detected'] = True
                result['window_count'] = count
                result['live_count'] = self.live_count

        return result

    def _prepare_input(self) -> torch.Tensor:
        frames = list(self.frame_buffer)
        arr = np.stack(frames)
        tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0

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

        tensor = tensor.unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std

        return tensor
