# CNN + CAM / Grad-CAM on Canvas FashionMNIST (Localization with IoU)

This repository contains a PyTorch notebook that trains a small CNN on **FashionMNIST**, where each 28×28 image is randomly **pasted onto a larger 128×128 canvas**.  
After training, the notebook generates **Class Activation Maps (CAM)** and **Grad-CAM** overlays and evaluates rough object localization using **IoU** between:

- **Ground-truth bounding box** (known paste location), and
- **Predicted CAM bounding box** (thresholded heatmap region)

> Notebook: `CNN.ipynb`

---

## What’s inside

### Main steps
1. **Dataset construction**: `CanvasFashionMNIST`
   - Takes FashionMNIST samples (28×28)
   - Randomly places them on a 128×128 blank canvas
   - Returns `(canvas_image, class_label, bbox_true)`
2. **Model**: `CNN_CAM`
   - Convolutional feature extractor
   - Global Average Pooling (GAP)
   - Linear classifier (enables CAM from classifier weights)
3. **Training & validation**
   - CrossEntropyLoss + Adam
   - Tracks loss/accuracy
4. **Evaluation**
   - **Test ROC-AUC (OvR)** (multi-class)
   - Micro-average ROC curve
5. **Explainability / Localization**
   - CAM overlays + bounding box extraction via threshold
   - IoU computed per sample + mean IoU
   - Optional **Grad-CAM** overlays

---

## Outputs (saved automatically)

The notebook creates an `./out/` folder and saves:

- `loss_curve.png`
- `acc_curve.png`
- `roc_micro.png`
- `cam_overlay_0.png ... cam_overlay_{N-1}.png`
- `gradcam_overlay_0.png ... gradcam_overlay_{N-1}.png` (optional section)
- `metrics_summary.txt`

---

## Requirements

Recommended Python: **3.9+**

Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn matplotlib
```
Optional (for running notebooks):

```bash
pip install notebook
```
## How to run
## Option A — Run the notebook (recommended)
jupyter notebook

Open:

CNN.ipynb

The dataset will be downloaded automatically into ./data/ on first run.

## Option B — Export to a Python script (optional)

jupyter nbconvert --to script "CNN.ipynb"
python "CNN.py"

## Key configuration (in the notebook)

You can modify these constants near the top:

- CANVAS = 128 (canvas size)

- PASTE = 28 (patch size)

- EPOCHS = 15

- LR = 1e-3

- BATCH_TRAIN = 64

- BATCH_EVAL = 256

- CAM_THR = 0.4 (threshold for CAM→bbox)

- SEED = 42 (reproducibility)

## Method summary
## CAM (Class Activation Map)

Because the model uses GAP + Linear classifier, CAM is computed by combining:

- last conv feature maps

- classifier weights for the predicted class

Then the heatmap is:

- ReLU’d

- normalized to [0, 1]

- upsampled to canvas size (128×128)

## Bounding box from CAM

A predicted bbox is extracted by thresholding:

- cam >= CAM_THR
and taking the min/max x,y of active pixels.

Localization metric

IoU is computed between:

- true bbox (paste coordinates)

- pred bbox (from CAM)

## Suggested repo structure

.
├── CNN.ipynb
├── README.md
├── requirements.txt            # optional but recommended
├── data/                       # auto-created (FashionMNIST download)
└── out/                        # auto-created (plots + overlays)

## Author
### Dorsa Abbasi

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
