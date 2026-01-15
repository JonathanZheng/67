import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import GestureDataset
from src.data.transforms import VideoTransform
from src.models.movinet import create_model

def train_epoch(model, loader, bce_loss, ce_loss, optimizer, device):
    model.train()
    total_loss = 0
    det_correct = 0
    count_correct = 0
    count_total = 0
    total = 0

    for frames, is_gesture, cycle_count in tqdm(loader, desc='Training'):
        frames = frames.to(device)
        is_gesture = is_gesture.to(device)
        cycle_count = cycle_count.to(device)

        optimizer.zero_grad()
        detection, count_logits = model(frames)

        detection_loss = bce_loss(detection, is_gesture)

        positive_mask = is_gesture == 1
        if positive_mask.any():
            counting_loss = ce_loss(count_logits[positive_mask], cycle_count[positive_mask])
        else:
            counting_loss = torch.tensor(0.0, device=device)

        loss = detection_loss + counting_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        det_preds = (detection > 0.5).float()
        det_correct += (det_preds == is_gesture).sum().item()
        total += is_gesture.size(0)

        if positive_mask.any():
            count_preds = count_logits[positive_mask].argmax(dim=-1)
            count_correct += (torch.abs(count_preds - cycle_count[positive_mask]) <= 1).sum().item()
            count_total += positive_mask.sum().item()

    det_acc = det_correct / total
    count_acc = count_correct / count_total if count_total > 0 else 0
    return total_loss / len(loader), det_acc, count_acc

def validate(model, loader, bce_loss, ce_loss, device):
    model.eval()
    total_loss = 0
    det_correct = 0
    count_correct = 0
    count_total = 0
    total = 0

    with torch.no_grad():
        for frames, is_gesture, cycle_count in tqdm(loader, desc='Validating'):
            frames = frames.to(device)
            is_gesture = is_gesture.to(device)
            cycle_count = cycle_count.to(device)

            detection, count_logits = model(frames)

            detection_loss = bce_loss(detection, is_gesture)

            positive_mask = is_gesture == 1
            if positive_mask.any():
                counting_loss = ce_loss(count_logits[positive_mask], cycle_count[positive_mask])
            else:
                counting_loss = torch.tensor(0.0, device=device)

            loss = detection_loss + counting_loss

            total_loss += loss.item()
            det_preds = (detection > 0.5).float()
            det_correct += (det_preds == is_gesture).sum().item()
            total += is_gesture.size(0)

            if positive_mask.any():
                count_preds = count_logits[positive_mask].argmax(dim=-1)
                count_correct += (torch.abs(count_preds - cycle_count[positive_mask]) <= 1).sum().item()
                count_total += positive_mask.sum().item()

    det_acc = det_correct / total
    count_acc = count_correct / count_total if count_total > 0 else 0
    return total_loss / len(loader), det_acc, count_acc

def main():
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / 'configs' / 'train_config.yaml'

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_transform = VideoTransform(spatial_size=config['data']['spatial_size'], training=True)
    val_transform = VideoTransform(spatial_size=config['data']['spatial_size'], training=False)

    train_dataset = GestureDataset(
        annotations_csv=str(project_root / 'data' / 'annotations.csv'),
        root_dir=str(project_root),
        num_frames=config['data']['frames_per_clip'],
        transform=train_transform,
        split='train'
    )

    val_dataset = GestureDataset(
        annotations_csv=str(project_root / 'data' / 'annotations.csv'),
        root_dir=str(project_root),
        num_frames=config['data']['frames_per_clip'],
        transform=val_transform,
        split='val'
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=2)

    model = create_model(
        pretrained=config['model']['pretrained'],
        dropout=config['model']['dropout']
    ).to(device)

    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )

    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = project_root / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)

    for epoch in range(config['training']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["training"]["epochs"]}')

        train_loss, train_det_acc, train_count_acc = train_epoch(
            model, train_loader, bce_loss, ce_loss, optimizer, device
        )
        val_loss, val_det_acc, val_count_acc = validate(
            model, val_loader, bce_loss, ce_loss, device
        )
        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}, Det Acc: {train_det_acc:.4f}, Count ±1 Acc: {train_count_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Det Acc: {val_det_acc:.4f}, Count ±1 Acc: {val_count_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            print('Saved best model')
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print('Early stopping triggered')
                break

    model.load_state_dict(torch.load(checkpoint_dir / 'best_model.pth'))
    model.eval()
    example = torch.randn(1, 3, config['data']['frames_per_clip'], config['data']['spatial_size'], config['data']['spatial_size']).to(device)
    traced = torch.jit.trace(model, example)
    traced.save(str(checkpoint_dir / 'model_traced.pt'))
    print(f'Saved traced model to {checkpoint_dir / "model_traced.pt"}')

if __name__ == '__main__':
    main()
