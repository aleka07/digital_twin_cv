import argparse
import shutil
from pathlib import Path


def copy_split(src_root: Path, dst_root: Path, split: str) -> tuple[int, int]:
    src_images = src_root / "images" / split
    src_labels = src_root / "labels" / split
    dst_images = dst_root / "images" / split
    dst_labels = dst_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0

    for img_path in sorted(src_images.glob("*.jpg")):
        dst_path = dst_images / img_path.name
        shutil.copy2(img_path, dst_path)
        image_count += 1

    for lbl_path in sorted(src_labels.glob("*.txt")):
        dst_path = dst_labels / lbl_path.name
        shutil.copy2(lbl_path, dst_path)
        label_count += 1

    return image_count, label_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged article_pack dataset from multiple labeled rounds.")
    parser.add_argument("--output-root", required=True, help="Output merged dataset root")
    parser.add_argument("--sources", nargs="+", required=True, help="Source round roots")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists():
        shutil.rmtree(output_root)

    summary_lines = []
    total_train_images = 0
    total_val_images = 0

    for source_str in args.sources:
        source = Path(source_str)
        train_images, train_labels = copy_split(source, output_root, "train")
        val_images, val_labels = copy_split(source, output_root, "val")
        total_train_images += train_images
        total_val_images += val_images
        summary_lines.append(
            f"{source.name}: train={train_images}/{train_labels} val={val_images}/{val_labels}"
        )

    data_yaml = output_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "",
                "names:",
                "  0: person",
                "  1: vat",
            ]
        ),
        encoding="utf-8",
    )

    summary_path = output_root / "merge_summary.txt"
    summary_path.write_text(
        "\n".join(
            summary_lines
            + [
                f"total_train_images={total_train_images}",
                f"total_val_images={total_val_images}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Merged dataset created at {output_root}")
    print(f"train images: {total_train_images}")
    print(f"val images: {total_val_images}")
    print(f"data yaml: {data_yaml}")


if __name__ == "__main__":
    main()
