# Competition Notebooks

The Kaggle notebooks written for *HuBMAP — Hacking the Human Vasculature* (2023), curated and
renamed into pipeline order. Together they cover the full arc of the competition: exploratory
analysis, data engineering, two model families (Mask R‑CNN and YOLOv8), and the offline‑safe
submission path. Every number quoted below is taken directly from the notebooks' own executed
outputs or from the saved training artifacts — nothing is reconstructed.

> **On the two model families.** The competition metric is single‑class Average Precision at a
> single IoU threshold of **0.6** (`blood_vessel`). Both model families were trained on all three
> annotated types (`blood_vessel`, `glomerulus`, `unsure`) and filtered down to `blood_vessel` at
> submission time. The **best selected public score was AP @ 0.6 = 0.3317** (see the root
> [`README.md`](../README.md#-results--experiments) for the full submission timeline).

| # | Notebook | Role | Headline result (traceable) |
|---|----------|------|-----------------------------|
| 01 | [`01-eda-and-submission-pipeline.ipynb`](01-eda-and-submission-pipeline.ipynb) | EDA + submission scaffold | 7,033 tiles · 1,633 annotated · COCO‑RLE→zlib→base64 encoder validated end‑to‑end |
| 02 | [`02-dataset-to-yolo-seg-format.ipynb`](02-dataset-to-yolo-seg-format.ipynb) | Data engineering | polygons.jsonl → YOLO‑seg labels; 1,306 train / 327 val, 3 classes |
| 03 | [`03-offline-tools-setup.ipynb`](03-offline-tools-setup.ipynb) | Infra | offline install of `ultralytics` + `pycocotools 2.0.6` (internet‑banned kernels) |
| 04 | [`04-mask-rcnn-torchvision.ipynb`](04-mask-rcnn-torchvision.ipynb) | Model A | Mask R‑CNN ResNet50‑FPN, 10 epochs, val loss 1.185 → 0.946 |
| 05 | [`05-mask-rcnn-from-scratch-pipeline.ipynb`](05-mask-rcnn-from-scratch-pipeline.ipynb) | Model A (hand‑built loop) | plateaued (val `loss_mask` ≈ 0.53), early‑stopped at epoch 8 |
| 06 | [`06-yolov8-instance-segmentation.ipynb`](06-yolov8-instance-segmentation.ipynb) | Model B | YOLOv8x‑seg, 20 epochs, **Mask mAP@50 = 0.597**; `blood_vessel` mAP@50 = 0.690 |
| 07 | [`07-yolo-inference-and-submission.ipynb`](07-yolo-inference-and-submission.ipynb) | Submission | offline inference from `best.pt` → competition `submission.csv` |

---

## 01 — EDA & submission pipeline

Exploratory analysis plus a reusable **submission scaffold**. Establishes the dataset facts used
throughout: **7,033** tiles (`tile_meta.csv`), all **512×512**; **1,633** annotated images
(`polygons.jsonl`, Dataset 1 + Dataset 2); five annotated WSIs plus nine unannotated
(Dataset 3); donor demographics for four WSIs (age 53–73, 3 F / 1 M). Notes the analytical
catch that only **13 of the expected 14** WSI sources appear — WSI #5 is the likely held‑out
private‑test slide. Implements the exact competition encoding
(**COCO‑RLE → `zlib` → `base64`**) and exercises it with a random‑mask stub so the encoder is
verified before any real model exists. No metric is computed here by design.

## 02 — Dataset → YOLO‑seg format

A one‑call converter from the organizers' `polygons.jsonl` to the layout Ultralytics expects:
per‑image `.txt` files of normalized polygon vertices (`class x1 y1 x2 y2 …`) plus a `coco.yaml`.
Prints **1,306 train / 327 val** from 1,633 tiles at an 80/20 split, 3 classes. Background‑only
tiles are dropped. *Naming caveat carried into the case study:* despite the "COCO" title the
output is **YOLO‑seg**, not COCO JSON — the file name here is corrected to reflect the code. The
split is a deterministic index slice — **not shuffled or grouped by WSI** — which matters for the
generalization gap discussed in the root README.

## 03 — Offline tools setup

Solves the competition's hard constraint: scoring kernels have **no internet**, so third‑party
libraries must be shipped as a Kaggle dataset and installed from local files. Demonstrates
side‑loading `ultralytics` via `sys.path` and building `pycocotools 2.0.6` from source with
`pip install . --no-index`. Pure infrastructure — the RLE/eval APIs are made importable but not
exercised here.

## 04 — Mask R‑CNN (torchvision)

`maskrcnn_resnet50_fpn`, COCO‑pretrained, box/mask heads swapped for 4 classes
(3 types + background), `box_detections_per_img=500` to allow many small instances per tile.
80/20 split **stratified by category**, val = 452. 10 epochs, SGD lr 0.001. Train loss
**1.319 → 0.924**, val loss **1.185 → 0.946**, best at epoch 10 (~90 min total). The notebook's
*internal* mAP scorer is unreliable — its IoU threshold sweep is `np.arange(0.6, 6.5, 0.05)`
(an upper bound of 6.5 instead of ~0.95), which averages in dozens of structurally impossible
thresholds and collapses the reported number toward zero. Loss convergence is the trustworthy
signal from this notebook, not that internal score.

## 05 — Mask R‑CNN from scratch (hand‑built pipeline)

"From scratch" here means a **hand‑written training loop, dataset builder, and submission encoder**
in pure PyTorch (the model itself is still torchvision Mask R‑CNN, COCO‑pretrained). Train 1,388 /
val 245 at a 0.85 split. The run **plateaued** — val `loss_mask` frozen at ≈0.53 for all epochs,
early‑stopped at epoch 8. Traceable root cause: `StepLR.step()` is called **per batch** instead of
per epoch (`step_size=3`), collapsing the learning rate to near‑zero within the first epoch. The
author's own closing note flags fatigue and lists NMS / low‑score mask filtering as deliberately
left undone. A useful negative result on training‑loop discipline.

## 06 — YOLOv8 instance segmentation

The strongest model line. **YOLOv8x‑seg** (71.7 M params), COCO‑pretrained, `imgsz=512`, 20 epochs,
SGD `lr0=0.018` `momentum=0.947`, batch 4, seed 43. Per‑epoch metrics climb steadily to
**Mask mAP@50 = 0.597 / mAP@50‑95 = 0.412** overall at epoch 20. Per‑class validation:

| class | instances | Mask mAP@50 | Mask mAP@50‑95 |
|-------|-----------|-------------|----------------|
| **blood_vessel** (scored) | 2,955 | **0.690** | 0.338 |
| glomerulus | 104 | 0.921 | 0.794 |
| unsure | 177 | 0.180 | 0.103 |

`unsure` is near‑noise and dilutes multi‑class training; `patience=3` never fired, so the model was
still improving at epoch 20 (headroom left). See the saved sweep in the root README — SGD beat Adam
(0.60 vs 0.45 Mask mAP@50) and `close_mosaic` helped.

## 07 — YOLO inference & submission

The offline **submission‑only** notebook: load a trained `best.pt`, run inference over test tiles,
extract masks, binarize at 0.5, encode as COCO‑RLE → `zlib` → `base64`, write `submission.csv`
(`id, height, width, prediction_string`; per instance `0 <conf> <b64mask>`). Two implementation
notes surfaced during review and worth citing: the `conf=0.05` filter is applied *after* prediction
rather than passed to the predictor (so the predictor runs at the Ultralytics default 0.25), and a
torchvision preprocessing transform is defined but unused (YOLO loads the raw path). Neither breaks
the run; both are the kind of config/impl drift a rigorous pipeline would catch.
