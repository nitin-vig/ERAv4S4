# MNIST — Train a Classifier in One Epoch

This project demonstrates how to train a strong MNIST digit classifier **in a single epoch** using PyTorch. The workflow lives in the notebook **`MNIST_Training_in_one_epoch.ipynb`**.

---

## ✨ Highlights

- **One-epoch training**: reach high accuracy with the right optimizer, LR schedule, and regularization.
- **Simple, readable PyTorch**: minimal boilerplate, easy to tweak.
- **Reproducible**: fixed seeds and deterministic settings included.

---

# 📐 Model Architecture

`Network` (PyTorch)

- Input: **[N, 1, 28, 28]** (MNIST grayscale)
- All convolutions: `kernel_size=3`, `stride=1`, `padding=0`
- Pooling: `MaxPool2d(kernel_size=2, stride=2)`
- Activations: `ReLU`
- Classifier: `Linear → ReLU → Linear` (10 logits)

### Layer-by-layer (with shapes)

| Stage | Layer | In → Out shape |
|---|---|---|
| Input | — | `[N, 1, 28, 28]` |
| Block 1 | `Conv2d(1→4, 3×3)` + ReLU | `[N, 1, 28, 28] → [N, 4, 26, 26]` |
|  | `Conv2d(4→8, 3×3)` + ReLU | `[N, 4, 26, 26] → [N, 8, 24, 24]` |
|  | `MaxPool2d(2,2)` | `[N, 8, 24, 24] → [N, 8, 12, 12]` |
| Block 2 | `Conv2d(8→16, 3×3)` + ReLU | `[N, 8, 12, 12] → [N, 16, 10, 10]` |
|  | `Conv2d(16→32, 3×3)` + ReLU | `[N, 16, 10, 10] → [N, 32, 8, 8]` |
|  | `MaxPool2d(2,2)` | `[N, 32, 8, 8] → [N, 32, 4, 4]` |
| Flatten | `reshape(-1, 32*4*4)` | `[N, 32, 4, 4] → [N, 512]` |
| Head | `Linear(512→30)` + ReLU | `[N, 512] → [N, 30]` |
|  | `Linear(30→10)` | `[N, 30] → **[N, 10]** (logits)` |

> Note: No softmax in the model — use `nn.CrossEntropyLoss` (it applies `log_softmax` internally). Apply `softmax` only at inference if you need probabilities.

---

# 🔢 Parameter Count (exact)

- `conv1 (1→4, 3×3)`: 4·1·3·3 + 4 = **40**
- `conv2 (4→8, 3×3)`: 8·4·3·3 + 8 = **296**
- `conv3 (8→16, 3×3)`: 16·8·3·3 + 16 = **1,168**
- `conv4 (16→32, 3×3)`: 32·16·3·3 + 32 = **4,640**
- `fc1 (512→30)`: 512·30 + 30 = **15,390**
- `out (30→10)`: 30·10 + 10 = **310**

**Total parameters: 21,844**

### (Optional) quick check in code
```python
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(total, trainable)  # both should be 21844
```

---

# 📊 Metrics

(Add these after your run. Template below.)

- **Train accuracy (1 epoch):** `XX.X%`
- **Validation accuracy (1 epoch):** `XX.X%`
- **Test accuracy (final):** `XX.X%`
- **Loss (train/val):** `X.XXX / X.XXX`
- **Confusion matrix:** (insert image or table)

### Snippets to log metrics

```python
# Accuracy helper
def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()

# After each epoch or at end:
print(f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")

# Confusion matrix (PyTorch)
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

model.eval()
all_preds, all_tgts = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        all_preds.append(logits.argmax(1).cpu())
        all_tgts.append(yb.cpu())
cm = confusion_matrix(torch.cat(all_tgts), torch.cat(all_preds))
print(cm)
```

---

# ⚙️ Recommended one-epoch settings (for good results fast)

- **Optimizer:** `AdamW(lr=1e-3, weight_decay=0.01)`
- **Scheduler:** `OneCycleLR(max_lr=1e-3, epochs=1, steps_per_epoch=len(train_loader), pct_start=0.3)`
- **Batch size:** 128–256
- **Loss:** `CrossEntropyLoss(label_smoothing=0.1)`
- **Light aug (optional):** small rotations/affine transforms

