import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train first detector on article_pack seed_round1.")
    parser.add_argument("--model", default="yolo11x.pt", help="Base model path")
    parser.add_argument(
        "--data",
        default="article_pack/data/cam10/seed_round1/data.yaml",
        help="Dataset yaml path",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--freeze", type=int, default=22, help="Freeze backbone layers")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers; keep 0 on Windows sandbox")
    parser.add_argument(
        "--name",
        default="cam10_seed_round1_yolo11x",
        help="Experiment name under article_pack/experiments/detector/",
    )
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    project_dir = Path("article_pack/experiments/detector")
    project_dir.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Seed Detector Training")
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Epochs:  {args.epochs}")
    print(f"Batch:   {args.batch}")
    print(f"ImgSz:   {args.imgsz}")
    print(f"Freeze:  {args.freeze}")
    print(f"Workers: {args.workers}")
    print(f"Device:  {device}")
    print("=" * 60)

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        freeze=args.freeze,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        mosaic=0.3,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        degrees=8.0,
        translate=0.08,
        scale=0.25,
        fliplr=0.5,
        workers=args.workers,
        plots=True,
        verbose=True,
    )

    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else project_dir / args.name
    best_path = save_dir / "weights" / "best.pt"

    print("\nTraining complete.")
    print(f"Run dir: {save_dir}")
    if best_path.exists():
        print(f"Best weights: {best_path}")


if __name__ == "__main__":
    main()
