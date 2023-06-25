from ultralytics import YOLO
import clearml


def main():
    clearml.browser_login()
    model = YOLO("yolov8n-seg.pt")
    model.train(
        # Project
        project="HuBMAP",
        name="first-run-yolov8n-seg",

        # Random Seed parameters
        deterministic=True,
        seed=42,

        # Data & model parameters
        data="./data/dataset/coco.yaml",
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=512,

        # Training parameters
        epochs=150,
        batch=5,
        workers=8,
        val=True,
        device=0,

        # Optimization parameters
        lr0=0.015,
        patience=8,
        optimizer="auto",
        momentum=0.937,
        weight_decay=0.0015
    )


if __name__ == '__main__':
    main()
