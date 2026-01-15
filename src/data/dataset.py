import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import av

class GestureDataset(Dataset):
    def __init__(self, annotations_csv: str, root_dir: str, num_frames: int = 48,
                 transform=None, split: str = None):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform

        df = pd.read_csv(annotations_csv)

        if split:
            np.random.seed(42)
            indices = np.random.permutation(len(df))
            n = len(df)
            if split == 'train':
                df = df.iloc[indices[:int(0.7 * n)]]
            elif split == 'val':
                df = df.iloc[indices[int(0.7 * n):int(0.85 * n)]]
            elif split == 'test':
                df = df.iloc[indices[int(0.85 * n):]]

        self.annotations = df.reset_index(drop=True)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        video_path = self.root_dir / row['video_path']

        is_gesture = torch.tensor(row['is_gesture'], dtype=torch.float32)
        cycle_count = torch.tensor(min(row['cycle_count'], 11), dtype=torch.long)

        frames = self._load_video(video_path)

        if self.transform:
            frames = self.transform(frames)

        return frames, is_gesture, cycle_count

    def _load_video(self, path: Path) -> torch.Tensor:
        container = av.open(str(path))
        stream = container.streams.video[0]

        total_frames = stream.frames
        if total_frames == 0:
            frames_list = list(container.decode(video=0))
            total_frames = len(frames_list)
        else:
            frames_list = None

        if total_frames <= self.num_frames:
            indices = np.arange(total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        if frames_list is None:
            frames_list = list(container.decode(video=0))

        selected = []
        for i in indices:
            if i < len(frames_list):
                frame = frames_list[i].to_ndarray(format='rgb24')
                selected.append(frame)

        while len(selected) < self.num_frames:
            selected.append(selected[-1] if selected else np.zeros((224, 224, 3), dtype=np.uint8))

        container.close()

        frames = np.stack(selected)
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
        frames = frames / 255.0

        return frames
