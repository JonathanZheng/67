#!/usr/bin/env python3
import cv2
import pandas as pd
from pathlib import Path
import argparse
import sys

def load_annotations(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=['video_path', 'is_gesture', 'cycle_count', 'source'])

def save_annotations(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False)

def get_video_files(data_dir: Path) -> list:
    extensions = ['*.mp4', '*.mov', '*.avi', '*.webm', '*.mkv']
    files = []
    for ext in extensions:
        files.extend(data_dir.rglob(ext))
    return sorted(files)

def play_video(video_path: Path, slow_factor: float = 0.5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path}")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / (fps * slow_factor))

    window_name = f"Annotate: {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    print(f"\nPlaying at {slow_factor}x speed: {video_path.name}")
    print("[0-9] = cycle count | [+] = 10+ | [n] = no gesture | [s] = skip | [r] = replay | [q] = quit")
    print("[,] = slower | [.] = faster")

    current_slow = slow_factor

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            instructions = f"0-9=count | +=10+ | n=none | r=replay | Speed: {current_slow:.1f}x"
            cv2.putText(frame, instructions, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, frame)

            current_delay = int(1000 / (fps * current_slow))
            key = cv2.waitKey(current_delay) & 0xFF

            if ord('0') <= key <= ord('9'):
                cap.release()
                cv2.destroyWindow(window_name)
                return True, key - ord('0')
            elif key == ord('+') or key == ord('='):
                cap.release()
                cv2.destroyWindow(window_name)
                return True, 11
            elif key == ord('n'):
                cap.release()
                cv2.destroyWindow(window_name)
                return False, 0
            elif key == ord('s'):
                cap.release()
                cv2.destroyWindow(window_name)
                return None, None
            elif key == ord('q'):
                cap.release()
                cv2.destroyWindow(window_name)
                return 'quit', 'quit'
            elif key == ord('r'):
                break
            elif key == ord(','):
                current_slow = max(0.1, current_slow - 0.1)
            elif key == ord('.'):
                current_slow = min(2.0, current_slow + 0.1)

        print("Video ended. Press key to annotate or [r] to replay")
        key = cv2.waitKey(0) & 0xFF

        if ord('0') <= key <= ord('9'):
            cap.release()
            cv2.destroyWindow(window_name)
            return True, key - ord('0')
        elif key == ord('+') or key == ord('='):
            cap.release()
            cv2.destroyWindow(window_name)
            return True, 11
        elif key == ord('n'):
            cap.release()
            cv2.destroyWindow(window_name)
            return False, 0
        elif key == ord('s'):
            cap.release()
            cv2.destroyWindow(window_name)
            return None, None
        elif key == ord('q'):
            cap.release()
            cv2.destroyWindow(window_name)
            return 'quit', 'quit'

def main():
    parser = argparse.ArgumentParser(description='Annotate videos for gesture recognition')
    parser.add_argument('--data-dir', type=str, default='data/raw', help='Directory containing videos')
    parser.add_argument('--output', type=str, default='data/annotations.csv', help='Output CSV file')
    parser.add_argument('--slow', type=float, default=0.5, help='Playback speed (0.5 = half speed)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    csv_path = project_root / args.output

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    annotations = load_annotations(csv_path)
    annotated = set(annotations['video_path'].tolist())

    video_files = get_video_files(data_dir)
    unannotated = [f for f in video_files if str(f.relative_to(project_root)) not in annotated]

    print(f"Found {len(video_files)} videos, {len(unannotated)} unannotated")

    for i, video_path in enumerate(unannotated):
        print(f"\n[{i+1}/{len(unannotated)}]")

        is_gesture, cycle_count = play_video(video_path, args.slow)

        if is_gesture == 'quit':
            break
        elif is_gesture is None:
            continue
        else:
            rel_path = str(video_path.relative_to(project_root))
            source = 'positive' if 'positive' in rel_path else 'negative'

            new_row = pd.DataFrame([{
                'video_path': rel_path,
                'is_gesture': 1 if is_gesture else 0,
                'cycle_count': cycle_count,
                'source': source
            }])
            annotations = pd.concat([annotations, new_row], ignore_index=True)
            save_annotations(annotations, csv_path)
            print(f"Saved: gesture={is_gesture}, count={cycle_count}")

    cv2.destroyAllWindows()
    print(f"\nAnnotation complete. Total: {len(annotations)} videos")

if __name__ == '__main__':
    main()
