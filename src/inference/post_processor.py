import torch
import torch.nn.functional as F
import numpy as np
import av
from pathlib import Path

class VideoPostProcessor:
    def __init__(self, model_path: str, device: str = None,
                 window_frames: int = 48, spatial_size: int = 224):

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.window_frames = window_frames
        self.spatial_size = spatial_size

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(self.device)

    def count_gestures(self, video_path: str) -> dict:
        frames = self._load_video(video_path)

        if len(frames) < self.window_frames:
            return {'total_count': 0, 'windows': [], 'error': 'Video too short'}

        total_count = 0
        window_results = []

        for i in range(0, len(frames) - self.window_frames + 1, self.window_frames):
            window = frames[i:i + self.window_frames]
            tensor = self._preprocess(window)

            with torch.no_grad():
                detection, count_logits = self.model(tensor)

            det_prob = detection.item()
            if det_prob > 0.5:
                count = count_logits.argmax(dim=-1).item()
                total_count += count
                window_results.append({
                    'start_frame': i,
                    'end_frame': i + self.window_frames,
                    'detected': True,
                    'count': count,
                    'confidence': det_prob
                })
            else:
                window_results.append({
                    'start_frame': i,
                    'end_frame': i + self.window_frames,
                    'detected': False,
                    'count': 0,
                    'confidence': det_prob
                })

        return {
            'total_count': total_count,
            'windows': window_results,
            'num_windows': len(window_results)
        }

    def _load_video(self, path: str) -> list:
        container = av.open(path)
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))
        container.close()
        return frames

    def _preprocess(self, frames: list) -> torch.Tensor:
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


def process_match_videos(model_path: str, video_paths: list) -> dict:
    processor = VideoPostProcessor(model_path)
    results = {}
    for path in video_paths:
        name = Path(path).stem
        results[name] = processor.count_gestures(path)
    return results
