import argparse
import csv
import shutil
from pathlib import Path


def bucket_for_video(video_name: str) -> str | None:
    if "140000" in video_name or "150000" in video_name:
        return "train"
    if "160000" in video_name:
        return "val"
    if "170000" in video_name:
        return "test_candidates"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare seed round folders from preview manifest.")
    parser.add_argument("--manifest", required=True, help="Preview manifest.csv path")
    parser.add_argument("--output-root", required=True, help="Output root for seed round")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)

    images_root = output_root / "images"
    labels_root = output_root / "labels"
    reports_root = output_root / "reports"

    for split in ("train", "val", "test_candidates"):
        (images_root / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (labels_root / split).mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    copied_rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            split = bucket_for_video(row["source_video"])
            if split is None:
                continue

            src = Path(row["image_path"])
            dst = images_root / split / src.name
            if dst.exists() and not args.overwrite:
                pass
            else:
                shutil.copy2(src, dst)

            copied_rows.append(
                {
                    "split": split,
                    "image_name": src.name,
                    "image_path": str(dst.as_posix()),
                    "source_video": row["source_video"],
                    "video_offset_sec": row["video_offset_sec"],
                    "wall_clock": row["wall_clock"],
                }
            )

    split_manifest = reports_root / "seed_round1_manifest.csv"
    with split_manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["split", "image_name", "image_path", "source_video", "video_offset_sec", "wall_clock"],
        )
        writer.writeheader()
        writer.writerows(copied_rows)

    counts = {}
    for row in copied_rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1

    print("Seed round prepared:")
    for split in ("train", "val", "test_candidates"):
        print(f"  {split}: {counts.get(split, 0)} images")
    print(f"Manifest: {split_manifest}")


if __name__ == "__main__":
    main()
