import argparse, os, csv, time
import cv2
from app.utils import extract_landmarks_from_bgr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True, help='Class label to capture (e.g., A, B, HELLO)')
    ap.add_argument('--samples', type=int, default=200, help='Number of samples to capture')
    ap.add_argument('--out', default='data/landmarks.csv', help='CSV path to append to')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Could not open webcam')
        return

    count = 0
    print(f'Capturing {args.samples} samples for label "{args.label}"...')
    with open(args.out, 'a', newline='') as f:
        writer = csv.writer(f)
        while count < args.samples:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks, annotated = extract_landmarks_from_bgr(frame)
            cv2.putText(annotated, f'Label: {args.label}  Count: {count}/{args.samples}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow('Collecting', annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to stop early
                break
            if landmarks is not None:
                row = list(landmarks.flatten()) + [args.label]
                writer.writerow(row)
                count += 1
                time.sleep(0.02)
        cap.release()
        cv2.destroyAllWindows()
    print('Done. Saved to', args.out)

if __name__ == '__main__':
    main()
