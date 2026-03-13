import cv2
import numpy as np

def detect_cuts(video_path, threshold=0.2, min_gap_seconds=0.05):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    min_gap_frames = int(max(1, min_gap_seconds * fps))

    prev_hist = None
    cuts = []
    last_cut_frame = -10**9

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff >= threshold and (frame_idx - last_cut_frame) >= min_gap_frames:
                cuts.append(frame_idx / fps)
                last_cut_frame = frame_idx

        prev_hist = hist
        frame_idx += 1

    cap.release()
    return cuts

if __name__ == "__main__":
    video = "/Users/hamzahussein/Desktop/Data/Carton13.mp4"
    cut_times = detect_cuts(video_path=video, threshold=0.30, min_gap_seconds=0.1)

    print("Detected cuts at (seconds):")
    for t in cut_times:
        print(f"{t:.0f}")