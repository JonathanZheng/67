# Six Seven Gesture Game

Competitive multiplayer game where players perform the "six seven" weighing scale hand gesture.

## The Gesture
- Both hands held out, palms up
- Hands alternate up/down like a balance scale
- Performed rapidly at 3-4 cycles per second

## How Scoring Works
- **During match**: Live estimated count shown (approximate)
- **After match**: Video is re-processed for accurate final count
- Winner determined by post-processed scores

## How to Play
1. Start the server: `python server/main.py`
2. Open http://localhost:8000 in two browser tabs
3. Create a room and have opponent join
4. Both players click "Ready"
5. Perform the gesture as fast as you can!
6. After the match, final scores are calculated from recorded video

## Setup

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Collect Training Data
1. Record videos of the gesture: place in `data/raw/positive/`
2. Record non-gesture videos: place in `data/raw/negative/`
3. Annotate: `python scripts/annotate.py`

### Train Model (on Kaggle)
1. Upload `data/` as a Kaggle dataset
2. Upload `notebooks/train_kaggle.ipynb`
3. Run notebook, download `model_traced.pt`
4. Place model in `checkpoints/`

### Run Demo
```bash
# Single player webcam test
python src/inference/webcam.py --model checkpoints/model_traced.pt

# Multiplayer server
python server/main.py
```

## Project Structure
```
67/
├── data/raw/           # Training videos (positive/negative)
├── src/
│   ├── data/           # Dataset and transforms
│   ├── models/         # Two-head model architecture
│   ├── training/       # Training and evaluation
│   └── inference/      # Live counter and post-processor
├── server/             # Multiplayer game server
├── client/             # Web UI
└── notebooks/          # Kaggle training notebook
```

## Model Architecture
Two-head design for better accuracy:
- **Detection head**: Binary classification (gesture or not) - trained on all data
- **Counting head**: Multi-class (0-11+ cycles) - trained on positives only
