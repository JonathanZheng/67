import cv2
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.live_counter import LiveGestureCounter

def main():
    parser = argparse.ArgumentParser(description='Webcam gesture counting demo')
    parser.add_argument('--model', type=str, required=True, help='Path to traced model')
    parser.add_argument('--record', type=str, default=None, help='Path to save recorded video')
    args = parser.parse_args()

    counter = LiveGestureCounter(model_path=args.model)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if args.record:
        counter.start_recording(args.record, (640, 480))

    print('Press Q to quit, R to reset count')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = counter.process_frame(frame_rgb)

        color = (0, 255, 0) if result['detected'] else (255, 255, 255)
        cv2.putText(frame, f"Count: ~{result['live_count']}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        if result['ready']:
            cv2.putText(frame, f"Conf: {result['confidence']:.2f}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        if result['detected']:
            cv2.putText(frame, f"+{result['window_count']}", (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)

        cv2.imshow('Gesture Counter', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.reset()
            print('Count reset')

    counter.stop_recording()
    cap.release()
    cv2.destroyAllWindows()
    print(f'Final count: ~{counter.live_count}')

if __name__ == '__main__':
    main()
