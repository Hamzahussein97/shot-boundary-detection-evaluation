#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import subprocess  # only used for optional TransNetV2
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class VideoItem:
    video_id: str
    category: str
    path: str


@dataclass
class Boundary:
    t: float      # time in seconds
    typ: str      # CUT / FADE / DISSOLVE / OTHER


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


# -----------------------------
# CSV robustness helpers
# -----------------------------
def _norm_col(name: str) -> str:
    """Normalize column names: lower + replace non-alnum with underscore."""
    name = (name or "").strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _detect_delimiter(sample: str) -> str:
    """Detect delimiter between ; and , (your files often use ;)"""
    if ";" in sample and "," not in sample:
        return ";"
    if "," in sample and ";" not in sample:
        return ","
    # if both appear, prefer ;
    if ";" in sample:
        return ";"
    return ","


def _read_csv_dicts(path: str) -> Tuple[List[Dict[str, str]], List[str], str]:
    """
    Read CSV with delimiter auto-detection and BOM handling.
    Returns (rows, fieldnames, delimiter).
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        delim = _detect_delimiter(first)
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delim)
        rows = [row for row in reader]
        return rows, (reader.fieldnames or []), delim


# -----------------------------
# Read input files (YOUR formats)
# -----------------------------
def read_videos_csv(path: str) -> List[VideoItem]:
    """
    Accepts headers like:
      video_id,category,path
    OR
      Video_Id;category;path
    Delimiter can be ; or ,
    """
    rows, fieldnames, _ = _read_csv_dicts(path)
    if not fieldnames:
        raise ValueError(f"Empty or invalid videos CSV: {path}")

    norm_map = {_norm_col(h): h for h in fieldnames if h is not None}

    # Video_Id / video_id / videoid
    vid_col = norm_map.get("video_id") or norm_map.get("videoid")
    if vid_col is None:
        for nk, orig in norm_map.items():
            if "video" in nk and "id" in nk:
                vid_col = orig
                break

    cat_col = norm_map.get("category") or norm_map.get("cat")
    path_col = norm_map.get("path") or norm_map.get("file") or norm_map.get("filepath") or norm_map.get("file_path")

    if vid_col is None or cat_col is None or path_col is None:
        raise ValueError(
            f"Videos CSV headers not recognized.\n"
            f"Found: {fieldnames}\n"
            f"Expected something like: Video_Id;category;path"
        )

    items: List[VideoItem] = []
    for row in rows:
        video_id = (row.get(vid_col, "") or "").strip().rstrip(";").rstrip(",")
        category = (row.get(cat_col, "") or "").strip().rstrip(";").rstrip(",")
        vpath = (row.get(path_col, "") or "").strip().rstrip(";").rstrip(",")

        if not video_id or not vpath:
            continue

        items.append(VideoItem(video_id=video_id, category=category, path=vpath))

    if not items:
        raise ValueError("No valid rows parsed from videos CSV (check paths and headers).")

    return items


def read_ground_truth_csv(path: str) -> Dict[str, List[Boundary]]:
    """
    Accepts headers like:
      video_id;boundary_id;time_sec;transition_type;
    (including possible trailing delimiter creating an empty column)
    Delimiter can be ; or ,
    """
    rows, fieldnames, _ = _read_csv_dicts(path)
    if not fieldnames:
        raise ValueError(f"Empty or invalid ground truth CSV: {path}")

    cleaned_fieldnames = [h for h in fieldnames if h and h.strip() != ""]
    norm_map = {_norm_col(h): h for h in cleaned_fieldnames}

    vid_col = norm_map.get("video_id") or norm_map.get("videoid")
    time_col = norm_map.get("time_sec") or norm_map.get("timesec") or norm_map.get("time")
    type_col = norm_map.get("transition_type") or norm_map.get("transitiontype") or norm_map.get("type")

    if vid_col is None or time_col is None:
        raise ValueError(
            f"Ground Truth CSV headers not recognized.\n"
            f"Found: {fieldnames}\n"
            f"Expected something like: video_id;boundary_id;time_sec;transition_type;"
        )

    gt: Dict[str, List[Boundary]] = {}
    for row in rows:
        vid = (row.get(vid_col, "") or "").strip().rstrip(";").rstrip(",")
        t_raw = (row.get(time_col, "") or "").strip().rstrip(";").rstrip(",")

        typ = (row.get(type_col, "OTHER") or "OTHER").strip().rstrip(";").rstrip(",").upper()
        if not typ:
            typ = "OTHER"

        if not vid or not t_raw:
            continue

        try:
            t = float(t_raw)
        except ValueError:
            continue

        gt.setdefault(vid, []).append(Boundary(t=t, typ=typ))

    for vid in gt:
        gt[vid].sort(key=lambda b: b.t)

    if not gt:
        raise ValueError("No valid ground truth rows parsed (check time_sec values).")

    return gt


# -----------------------------
# Matching + metrics
# -----------------------------
def greedy_match(gt_times: List[float], pred_times: List[float], tol: float) -> Tuple[int, int, int]:
    """
    Greedy 1-1 matching within tolerance window.
    Returns (TP, FP, FN).
    """
    gt_used = [False] * len(gt_times)
    pred_used = [False] * len(pred_times)

    tp = 0
    for pi, pt in enumerate(pred_times):
        best_gi = None
        best_dist = None
        for gi, gt in enumerate(gt_times):
            if gt_used[gi]:
                continue
            d = abs(gt - pt)
            if d <= tol and (best_dist is None or d < best_dist):
                best_gi = gi
                best_dist = d
        if best_gi is not None:
            gt_used[best_gi] = True
            pred_used[pi] = True
            tp += 1

    fp = sum(1 for used in pred_used if not used)
    fn = sum(1 for used in gt_used if not used)
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# -----------------------------
# PySceneDetect runner (Python API, NO CLI)
# -----------------------------
def run_pyscenedetect(video_path: str, out_dir: str, threshold: float = 30.0) -> List[float]:
    """
    Runs PySceneDetect via Python API (NO scenedetect command needed).
    Requires (inside your venv):
      python -m pip install scenedetect[opencv] opencv-python
    Returns boundary times in seconds (end of each detected scene).
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
    except Exception as e:
        raise RuntimeError(
            "PySceneDetect is not installed in this environment. "
            "Install with: python -m pip install scenedetect[opencv] opencv-python"
        ) from e

    os.makedirs(out_dir, exist_ok=True)

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    boundaries: List[float] = []
    for (_start, end) in scene_list:
        t = float(end.get_seconds())
        if t > 0:
            boundaries.append(t)

    return sorted(set(boundaries))


# -----------------------------
# TransNetV2 runner (optional)
# -----------------------------
def run_transnetv2_cli(
    video_path: str,
    out_dir: str,
    transnet_py: str,
    model_path: Optional[str] = None
) -> List[float]:
    """
    Calls your TransNetV2 script. Your script MUST output JSON like:
      {"transitions":[{"time_sec": 3.4}, {"time_sec": 8.1}]}
    """
    os.makedirs(out_dir, exist_ok=True)
    out_json = str(Path(out_dir) / "transnetv2_out.json")

    cmd = ["python", transnet_py, "--input", video_path, "--output", out_json]
    if model_path:
        cmd += ["--model", model_path]

    subprocess.run(cmd, check=True)

    if not os.path.exists(out_json):
        raise RuntimeError(f"TransNetV2 output JSON not found: {out_json}")

    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    transitions = data.get("transitions", [])
    times: List[float] = []
    for x in transitions:
        if isinstance(x, dict) and "time_sec" in x:
            times.append(float(x["time_sec"]))
        elif isinstance(x, (int, float)):
            times.append(float(x))

    return sorted(set(times))


# -----------------------------
# Aggregation + plots
# -----------------------------
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
            w.writerow({
                "video_id": r.video_id,
                "category": r.category,
                "method": r.method,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "precision": f"{r.precision:.6f}",
                "recall": f"{r.recall:.6f}",
                "f1": f"{r.f1:.6f}",
            })


def write_summary_csv(path: str, summary: Dict[Tuple[str, str], List[float]]) -> None:
    fields = ["category", "method", "n_videos", "mean_f1", "median_f1", "std_f1"]
    rows = []
    for (cat, method), vals in summary.items():
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n if n else 0.0
        median = vals_sorted[n // 2] if n else 0.0
        if n >= 2:
            var = sum((x - mean) ** 2 for x in vals_sorted) / (n - 1)
            std = var ** 0.5
        else:
            std = 0.0
        rows.append((cat, method, n, mean, median, std))

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cat, method, n, mean, median, std in rows:
            w.writerow({
                "category": cat,
                "method": method,
                "n_videos": n,
                "mean_f1": f"{mean:.6f}",
                "median_f1": f"{median:.6f}",
                "std_f1": f"{std:.6f}",
            })


def save_barplot(summary: Dict[Tuple[str, str], List[float]], out_path: str) -> None:
    categories = sorted(set(cat for (cat, _m) in summary.keys()))
    methods = sorted(set(m for (_cat, m) in summary.keys()))

    means = {}
    for cat in categories:
        for m in methods:
            vals = summary.get((cat, m), [])
            means[(cat, m)] = (sum(vals) / len(vals)) if vals else 0.0

    labels = []
    values = []
    for cat in categories:
        for m in methods:
            labels.append(f"{cat}\n{m}")
            values.append(means[(cat, m)])

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
    keys = sorted(summary.keys())  # (category, method)
    data = [summary[k] for k in keys]
    labels = [f"{k[0]}\n{k[1]}" for k in keys]

    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylim(0, 1)
    plt.ylabel("F1 per Video")
    plt.title("Shot Boundary Detection: F1 Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_csv", required=True, help="Path to Videos.csv")
    ap.add_argument("--gt_csv", required=True, help="Path to Ground Truth CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--tol", type=float, default=0.5, help="Tolerance window in seconds (default: 0.5)")
    ap.add_argument("--pyscene_threshold", type=float, default=27.0, help="PySceneDetect content threshold")

    ap.add_argument("--transnet_script", default="", help="Path to your TransNetV2 predict script (optional)")
    ap.add_argument("--transnet_model", default="", help="Path to TransNetV2 model (optional)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = read_videos_csv(args.videos_csv)
    gt = read_ground_truth_csv(args.gt_csv)

    print(f"Loaded videos: {len(videos)} from {args.videos_csv}")
    print(f"Loaded GT entries for: {len(gt)} videos from {args.gt_csv}")

    results: List[EvalResult] = []

    for v in videos:
        if not os.path.exists(v.path):
            raise FileNotFoundError(f"Video not found for {v.video_id}: {v.path}")
        if v.video_id not in gt:
            raise ValueError(f"No Ground Truth for video_id={v.video_id}. Check IDs match between files.")

        gt_times = [b.t for b in gt[v.video_id]]

        # ----- PySceneDetect (API)
        ps_dir = out_dir / "preds" / "pyscenedetect" / v.video_id
        ps_times = run_pyscenedetect(v.path, str(ps_dir), threshold=args.pyscene_threshold)
        tp, fp, fn = greedy_match(gt_times, ps_times, args.tol)
        p, r, f1 = prf(tp, fp, fn)
        results.append(EvalResult(v.video_id, v.category, "PySceneDetect", tp, fp, fn, p, r, f1))

        # ----- TransNetV2 (optional)
        if args.transnet_script.strip():
            tn_dir = out_dir / "preds" / "transnetv2" / v.video_id
            tn_times = run_transnetv2_cli(
                v.path, str(tn_dir),
                transnet_py=args.transnet_script,
                model_path=(args.transnet_model.strip() or None)
            )
            tp, fp, fn = greedy_match(gt_times, tn_times, args.tol)
            p, r, f1 = prf(tp, fp, fn)
            results.append(EvalResult(v.video_id, v.category, "TransNetV2", tp, fp, fn, p, r, f1))

        print(f"[OK] {v.video_id} ({v.category}) done.")

    results_csv = out_dir / "results_per_video.csv"
    write_results_csv(str(results_csv), results)

    summ = summarize(results)
    summary_csv = out_dir / "summary.csv"
    write_summary_csv(str(summary_csv), summ)

    plot_bar = out_dir / "plot_mean_f1_bar.png"
    plot_box = out_dir / "plot_f1_box.png"
    save_barplot(summ, str(plot_bar))
    save_boxplot(summ, str(plot_box))

    print("\nDONE ✅")
    print(f"- Results per video: {results_csv}")
    print(f"- Summary:          {summary_csv}")
    print(f"- Plot (bar):       {plot_bar}")
    print(f"- Plot (box):       {plot_box}")


if __name__ == "__main__":
    main()