import torch
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import GestureDataset
from src.data.transforms import VideoTransform
from src.models.movinet import create_model

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            outputs = model(frames)

            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            labels = labels.numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)

def main():
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / 'configs' / 'train_config.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = VideoTransform(spatial_size=config['data']['spatial_size'], training=False)

    test_dataset = GestureDataset(
        annotations_csv=str(project_root / 'data' / 'annotations.csv'),
        root_dir=str(project_root),
        num_frames=config['data']['frames_per_clip'],
        transform=transform,
        split='test'
    )

    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                             shuffle=False, num_workers=2)

    model = create_model(
        model_type='slow_r50',
        pretrained=False,
        dropout=config['model']['dropout']
    ).to(device)

    checkpoint_path = project_root / 'checkpoints' / 'best_model.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    preds, labels, probs = evaluate(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    neg_mask = labels == 0
    fpr = preds[neg_mask].sum() / neg_mask.sum() if neg_mask.sum() > 0 else 0

    print('Test Results:')
    print(f'  Accuracy:  {acc:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall:    {recall:.4f}')
    print(f'  F1 Score:  {f1:.4f}')
    print(f'  FPR:       {fpr:.4f}')
    print()
    print('Confusion Matrix:')
    print(confusion_matrix(labels, preds))

if __name__ == '__main__':
    main()
