import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector on a fixed article_pack split.")
    parser.add_argument("--model", required=True, help="Path to trained detector weights")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--name", default="eval_run", help="Run name")
    args = parser.parse_args()

    from ultralytics import YOLO
    import torch

    project_dir = Path("article_pack/experiments/detector_eval")
    project_dir.mkdir(parents=True, exist_ok=True)
    device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Detector Evaluation")
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Split:   {args.split}")
    print(f"ImgSz:   {args.imgsz}")
    print(f"Batch:   {args.batch}")
    print(f"Workers: {args.workers}")
    print(f"Device:  {device}")
    print("=" * 60)

    model = YOLO(args.model)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else project_dir / args.name
    print("\nEvaluation complete.")
    print(f"Run dir: {save_dir}")
    print(f"P:        {metrics.box.mp:.3f}")
    print(f"R:        {metrics.box.mr:.3f}")
    print(f"mAP50:    {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    if len(metrics.box.p) > 0:
        print(f"Person:   P={metrics.box.p[0]:.3f} R={metrics.box.r[0]:.3f}")
    if len(metrics.box.p) > 1:
        print(f"Vat:      P={metrics.box.p[1]:.3f} R={metrics.box.r[1]:.3f}")


if __name__ == "__main__":
    main()
