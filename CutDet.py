#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class VideoItem:
    video_id: str
    category: str
    path: str

@dataclass
class Boundary:
    t: float
    typ: str

@dataclass
class EvalResult:
    video_id: str
    category: str
    method: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def norm_col(name: str) -> str:
    name = (name or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")

def detect_delim(sample: str) -> str:
    if ";" in sample and "," not in sample:
        return ";"
    if "," in sample and ";" not in sample:
        return ","
    return ";" if ";" in sample else ","

def read_csv(path: str) -> Tuple[List[Dict], List[str], str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        delim = detect_delim(f.readline())
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        return list(reader), reader.fieldnames or [], delim

def find_col(norm_map: dict, *candidates) -> Optional[str]:
    for c in candidates:
        if c in norm_map:
            return norm_map[c]
    return None


def read_videos_csv(path: str) -> List[VideoItem]:
    rows, fieldnames, _ = read_csv(path)
    if not fieldnames:
        raise ValueError(f"Empty videos CSV: {path}")
    nm = {norm_col(h): h for h in fieldnames if h}
    vid_col = find_col(nm, "video_id", "videoid")
    cat_col = find_col(nm, "category", "cat")
    path_col = find_col(nm, "path", "file", "filepath", "file_path")
    if not all([vid_col, cat_col, path_col]):
        raise ValueError(f"Videos CSV headers not recognized. Found: {fieldnames}")
    items = []
    for row in rows:
        vid = row.get(vid_col, "").strip().rstrip(";,")
        cat = row.get(cat_col, "").strip().rstrip(";,")
        vpath = row.get(path_col, "").strip().rstrip(";,")
        if vid and vpath:
            items.append(VideoItem(vid, cat, vpath))
    if not items:
        raise ValueError("No valid rows in videos CSV.")
    return items


def read_ground_truth_csv(path: str) -> Dict[str, List[Boundary]]:
    rows, fieldnames, _ = read_csv(path)
    if not fieldnames:
        raise ValueError(f"Empty ground truth CSV: {path}")
    nm = {norm_col(h): h for h in fieldnames if h and h.strip()}
    vid_col = find_col(nm, "video_id", "videoid")
    time_col = find_col(nm, "time_sec", "timesec", "time")
    type_col = find_col(nm, "transition_type", "transitiontype", "type")
    if not vid_col or not time_col:
        raise ValueError(f"Ground truth CSV headers not recognized. Found: {fieldnames}")
    gt: Dict[str, List[Boundary]] = {}
    for row in rows:
        vid = row.get(vid_col, "").strip().rstrip(";,")
        t_raw = row.get(time_col, "").strip().rstrip(";,")
        typ = (row.get(type_col, "OTHER") or "OTHER").strip().rstrip(";,").upper() or "OTHER"
        if not vid or not t_raw:
            continue
        try:
            gt.setdefault(vid, []).append(Boundary(float(t_raw), typ))
        except ValueError:
            continue
    if not gt:
        raise ValueError("No valid ground truth rows parsed.")
    for vid in gt:
        gt[vid].sort(key=lambda b: b.t)
    return gt


def greedy_match(gt_times: List[float], pred_times: List[float], tol: float) -> Tuple[int, int, int]:
    gt_used = [False] * len(gt_times)
    pred_used = [False] * len(pred_times)
    tp = 0
    for pi, pt in enumerate(pred_times):
        best_gi, best_dist = None, None
        for gi, gt in enumerate(gt_times):
            if gt_used[gi]:
                continue
            d = abs(gt - pt)
            if d <= tol and (best_dist is None or d < best_dist):
                best_gi, best_dist = gi, d
        if best_gi is not None:
            gt_used[best_gi] = True
            pred_used[pi] = True
            tp += 1
    return tp, sum(not u for u in pred_used), sum(not u for u in gt_used)

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def run_pyscenedetect(video_path: str, out_dir: str, threshold: float = 30.0) -> List[float]:
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception as e:
        raise RuntimeError("PySceneDetect not installed.") from e
    os.makedirs(out_dir, exist_ok=True)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video=open_video(video_path))
    boundaries = [float(end.get_seconds()) for _, end in scene_manager.get_scene_list()]
    return sorted(t for t in set(boundaries) if t > 0)


def load_transnetv2():
    if not hasattr(load_transnetv2, "_model"):
        try:
            from transnetv2 import TransNetV2
        except ImportError as e:
            raise RuntimeError("TransNetV2 not installed.") from e
        print("[TransNetV2] Loading model...")
        load_transnetv2._model = TransNetV2()
        print("[TransNetV2] Model loaded.")
    return load_transnetv2._model

def run_transnetv2(video_path: str) -> List[float]:
    import cv2
    model = load_transnetv2()
    _, single_frame_preds, _ = model.predict_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_preds)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    boundaries = [float(end_frame) / fps for _, end_frame in scenes]
    return sorted(b for b in set(boundaries) if b > 0)


def summarize(results: List[EvalResult]) -> Dict[Tuple[str, str], List[float]]:
    d: Dict[Tuple[str, str], List[float]] = {}
    for r in results:
        d.setdefault((r.category, r.method), []).append(r.f1)
    return d

def write_results_csv(path: str, results: List[EvalResult]) -> None:
    fields = ["video_id", "category", "method", "tp", "fp", "fn", "precision", "recall", "f1"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({"video_id": r.video_id, "category": r.category, "method": r.method,
                        "tp": r.tp, "fp": r.fp, "fn": r.fn,
                        "precision": f"{r.precision:.6f}", "recall": f"{r.recall:.6f}", "f1": f"{r.f1:.6f}"})

def write_summary_csv(path: str, summary: Dict[Tuple[str, str], List[float]]) -> None:
    fields = ["category", "method", "n_videos", "mean_f1", "median_f1", "std_f1"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (cat, method), vals in summary.items():
            n = len(vals)
            mean = sum(vals) / n if n else 0.0
            median = sorted(vals)[n // 2] if n else 0.0
            std = (sum((x - mean) ** 2 for x in vals) / (n - 1)) ** 0.5 if n >= 2 else 0.0
            w.writerow({"category": cat, "method": method, "n_videos": n,
                        "mean_f1": f"{mean:.6f}", "median_f1": f"{median:.6f}", "std_f1": f"{std:.6f}"})

def save_barplot(summary: Dict[Tuple[str, str], List[float]], out_path: str) -> None:
    categories = sorted(set(c for c, _ in summary))
    methods = sorted(set(m for _, m in summary))
    labels = [f"{c}\n{m}" for c in categories for m in methods]
    values = [(sum(vals) / len(vals)) if (vals := summary.get((c, m), [])) else 0.0 for c in categories for m in methods]
    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Mean F1")
    plt.title("Shot Boundary Detection: Mean F1 by Category & Method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def save_boxplot(summary: Dict[Tuple[str, str], List[float]], out_path: str) -> None:
    keys = sorted(summary)
    plt.figure()
    plt.boxplot([summary[k] for k in keys], tick_labels=[f"{k[0]}\n{k[1]}" for k in keys], showmeans=True)
    plt.ylim(0, 1)
    plt.ylabel("F1 per Video")
    plt.title("Shot Boundary Detection: F1 Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_csv", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--pyscene_threshold", type=float, default=27.0)
    ap.add_argument("--use_transnet", action="store_true", help="Enable TransNetV2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = read_videos_csv(args.videos_csv)
    gt = read_ground_truth_csv(args.gt_csv)
    print(f"Videos: {len(videos)} | GT entries: {len(gt)}")

    if args.use_transnet:
        load_transnetv2()

    results: List[EvalResult] = []

    for v in videos:
        if not os.path.exists(v.path):
            raise FileNotFoundError(f"Video not found: {v.path}")
        if v.video_id not in gt:
            raise ValueError(f"No ground truth for: {v.video_id}")
        gt_times = [b.t for b in gt[v.video_id]]

        ps_times = run_pyscenedetect(v.path, str(out_dir / "preds/pyscenedetect" / v.video_id), args.pyscene_threshold)
        tp, fp, fn = greedy_match(gt_times, ps_times, args.tol)
        results.append(EvalResult(v.video_id, v.category, "PySceneDetect", tp, fp, fn, *prf(tp, fp, fn)))

        if args.use_transnet:
            tn_times = run_transnetv2(v.path)
            tp, fp, fn = greedy_match(gt_times, tn_times, args.tol)
            results.append(EvalResult(v.video_id, v.category, "TransNetV2", tp, fp, fn, *prf(tp, fp, fn)))

        print(f"[OK] {v.video_id} ({v.category})")

    write_results_csv(str(out_dir / "results_per_video.csv"), results)
    summ = summarize(results)
    write_summary_csv(str(out_dir / "summary.csv"), summ)
    save_barplot(summ, str(out_dir / "plot_mean_f1_bar.png"))
    save_boxplot(summ, str(out_dir / "plot_f1_box.png"))
    print(f"\nDONE ✅  →  {out_dir}")


if __name__ == "__main__":
    main()