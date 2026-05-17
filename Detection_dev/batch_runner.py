import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple


def discover_batch_recordings(
    dataset_root: str,
    csv_suffix: str = "--4.csv",
    video_ext: str = ".mp4",
) -> List[Tuple[str, str, str]]:
    recordings: List[Tuple[str, str, str]] = []
    csv_suffix_lower = csv_suffix.lower()

    for folder_label in ("alarm", "noalarm"):
        folder_path = os.path.join(dataset_root, folder_label)
        if not os.path.isdir(folder_path):
            print(f"[INFO] Missing folder: {folder_path}")
            continue

        for current_root, _, files in os.walk(folder_path):
            mp4_files = sorted([f for f in files if f.lower().endswith(video_ext)])
            for mp4_name in mp4_files:
                csv_candidates = sorted(
                    [f for f in files if f.lower().endswith(csv_suffix_lower)]
                )
                if not csv_candidates:
                    print(f"[WARNING] Missing CSV for: {mp4_name} in {current_root}")
                    continue

                target_csv = f"{Path(mp4_name).stem}{csv_suffix}"
                if target_csv in csv_candidates:
                    csv_name = target_csv
                else:
                    csv_name = csv_candidates[0]
                    if len(csv_candidates) > 1:
                        print(
                            f"[WARNING] Multiple CSV candidates for {mp4_name} in {current_root}. "
                            f"Using: {csv_name}"
                        )

                video_path = os.path.join(current_root, mp4_name)
                csv_path = os.path.join(current_root, csv_name)
                recordings.append((folder_label, video_path, csv_path))

    return recordings


def format_alarm_reasons(reason_counts: Dict[str, int]) -> str:
    if not reason_counts:
        return "none"

    parts: List[str] = []
    for reason in sorted(reason_counts.keys()):
        parts.append(f"{reason}({reason_counts[reason]})")
    return "; ".join(parts)


def save_batch_report_markdown(
    report_rows: List[Dict[str, str]],
    output_path: str,
    correct_count: int,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total = len(report_rows)
    accuracy = (correct_count / total) * 100.0 if total else 0.0

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for row in report_rows:
        expected_alarm = row["expected_alarm"] == "True"
        detected_alarm = row["alarm_detected"] == "True"

        if expected_alarm and detected_alarm:
            tp += 1
        elif not expected_alarm and not detected_alarm:
            tn += 1
        elif not expected_alarm and detected_alarm:
            fp += 1
        else:
            fn += 1

    false_alarm_rate = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    true_alarm_rate = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0

    def esc(value: str) -> str:
        return value.replace("|", "\\|")

    md_lines: List[str] = [
        "# Batch Report",
        "",
        f"- Quantity of recordings: **{total}**",
        f"- Correct classifications: **{correct_count}**",
        f"- Accuracy: **{accuracy:.2f}%**",
        f"- False alarm rate (FAR): **{false_alarm_rate:.2f}%**",
        f"- True alarm rate (TAR): **{true_alarm_rate:.2f}%**",
        "",
        "| # | expected_folder | expected_codes | detected_codes | alarm_detected | alarm_reasons | expected_alarm | match | video_path | csv_path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for idx, row in enumerate(report_rows, start=1):
        md_lines.append(
            f"| {idx} | {esc(row['expected_folder'])} | {esc(row.get('expected_codes', 'none'))} | "
            f"{esc(row.get('detected_codes', 'none'))} | {esc(row['alarm_detected'])} | "
            f"{esc(row['alarm_reasons'])} | {esc(row['expected_alarm'])} | {esc(row['match'])} | "
            f"{esc(row['video_path'])} | {esc(row['csv_path'])} |"
        )

    with open(output_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(md_lines) + "\n")


def build_batch_report_path(report_dir: str, report_date: date) -> str:
    os.makedirs(report_dir, exist_ok=True)
    date_stamp = report_date.strftime("%d.%m.%Y")
    file_name = f"batch_alarm_report_{date_stamp}.md"
    return os.path.join(report_dir, file_name)
