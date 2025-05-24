#%%

import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader

from libs.ml.dataset import GroundStateDataset
from libs.ml.modeling import ClassificationCNN


Q = 2
L = 10
data_repeat = 200
epochs = 100000
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-2

train_set = GroundStateDataset(Q, L, L, data_repeat)
test_set = GroundStateDataset(Q, L, L, 1)

train_loader = DataLoader(train_set, batch_size, shuffle=True)
test_loader = DataLoader(test_set, Q, shuffle=False)
test_x, test_y_gt = next(iter(test_loader))


model = ClassificationCNN(Q, L, L)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss_sum = 0.0
    for x, y_gt in train_loader:
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y_gt)
        loss.backward()
        train_loss_sum += loss.item() * x.shape[0]
        optimizer.step()
    
    train_loss_avg = train_loss_sum / len(train_set)
    
    with torch.inference_mode():
        test_y_pred = model(test_x)
        test_loss_avg = criterion(test_y_pred, test_y_gt)
        test_acc = (test_y_pred.argmax(dim=1) == test_y_gt).float().mean()
    print(f"[Epoch {epoch:03d}] train_loss {train_loss_avg:.4f} test_loss {test_loss_avg:.4f} test_acc {test_acc:.4f}")

    if test_loss_avg < 0.01:
        break

#%%

from pathlib import Path
import numpy as np
import torch.nn.functional as F

data_root = Path("dataset/debug_dataset_v1")
data_dir = data_root / f"delta__swendsen_wang__q={Q}__L={L}"

with torch.inference_mode():
    pred_by_T = {}
    for path in sorted(data_dir.glob("*.npz")):
        T = float(path.name.replace("t=", "").replace(".npz", ""))
        samples = torch.from_numpy(np.load(path)["samples"])
        samples_one_hot = F.one_hot(samples.long(), Q).float().permute(0, 2, 1).reshape(-1, Q, L, L)
        pred_by_T[T] = torch.softmax(model(samples_one_hot), dim=-1).numpy()


norm_by_T = {T: np.linalg.norm(p, axis=1, ord=2) for T, p in pred_by_T.items()} 

#%%

import matplotlib.pyplot as plt
for t in sorted(norm_by_T.keys()):
    plt.hist(norm_by_T[t])
    plt.show()

#%%

# is it better than just taking mean of one-hot spin vectors?

with torch.inference_mode():
    mean_by_T = {}
    for path in sorted(data_dir.glob("*.npz")):
        T = float(path.name.replace("t=", "").replace(".npz", ""))
        samples = torch.from_numpy(np.load(path)["samples"])
        samples_one_hot = F.one_hot(samples.long(), Q).float().permute(0, 2, 1).reshape(-1, Q, L, L)
        print(samples)
        v = samples_one_hot.mean(dim=3).mean(dim=2)
        mean_by_T[T] = v / v.sum(dim=1, keepdims=True)


for t in sorted(mean_by_T.keys()):
    v = mean_by_T[t]
    plt.hist(torch.linalg.norm(v, ord=2, dim=1))
    plt.show()
