#%%

import torch
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader

from libs.ml.dataset import GroundStateDataset
from libs.ml.modeling import ClassificationCNN



def train_model_on_ordered_set(Q, L):
    torch.manual_seed(2048)

    data_repeat = 200
    epochs = 100000
    batch_size = 16
    learning_rate = 0.001
    # learning_rate_decayed = 1e-3
    weight_decay = 2.0

    train_set = GroundStateDataset(Q, L, L, data_repeat)
    test_set = GroundStateDataset(Q, L, L, 1)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, Q, shuffle=False)
    test_x, test_y_gt = next(iter(test_loader))


    model = ClassificationCNN(Q, L, L)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []

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
            train_losses.append(loss.item())

        train_loss_avg = train_loss_sum / len(train_set)
        
        with torch.inference_mode():
            test_y_pred = model(test_x)
            test_loss_avg = criterion(test_y_pred, test_y_gt)
            test_acc = (test_y_pred.argmax(dim=1) == test_y_gt).float().mean()
        print(f"[Epoch {epoch:03d}] train_loss {train_loss_avg:.4f} test_loss {test_loss_avg:.4f} test_acc {test_acc:.4f}")

        # if test_loss_avg < 1.0:
        #     optimizer.param_groups[0]["lr"] = learning_rate_decayed


        if test_loss_avg < 0.01:  # early stopping; otherwise the model becomes "overconfident", producing useless results
            break

    return model, train_losses

#%%

from pathlib import Path
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

data_root = Path("dataset/debug_dataset_v1")
output_root = Path("logs/supervised_learning/images")
output_root.mkdir(parents=True, exist_ok=True)


all_train_losses = {}
all_norm_means = {}

for Q in [2, 3, 4]:
    for L in [10, 20, 40]:
        model, train_losses = train_model_on_ordered_set(Q, L)
        all_train_losses[Q, L] = train_losses

        conf_name = f"delta__swendsen_wang__q={Q}__L={L}"

        data_dir = data_root / conf_name
        output_dir = output_root / conf_name
        output_dir.mkdir(exist_ok=True)

        with torch.inference_mode():
            pred_by_T = {}
            mean_by_T = {}
            mag_by_T = {}
            for path in sorted(data_dir.glob("*.npz")):
                T = float(path.name.replace("t=", "").replace(".npz", ""))
                samples = torch.from_numpy(np.load(path)["samples"])
                samples_one_hot = F.one_hot(samples.long(), Q).float().permute(0, 2, 1).reshape(-1, Q, L, L)
                pred_by_T[T] = torch.softmax(model(samples_one_hot), dim=-1).numpy()
                v = samples_one_hot.mean(dim=3).mean(dim=2)
                mean_by_T[T] = v / v.sum(dim=1, keepdims=True)
                samples_vec = torch.stack((torch.cos(samples), torch.sin(samples)), dim=2)
                mag_vec = samples_vec.mean(dim=1)
                mag = torch.linalg.norm(mag_vec, dim=1)
                mag_by_T[T] = mag.numpy()

        norm_by_T = {T: np.linalg.norm(p, axis=1, ord=2) for T, p in pred_by_T.items()} 
        dist_output_dir = output_dir / "dist"
        dist_output_dir.mkdir(exist_ok=True)

        rs, ms = [], []
        ts = sorted(norm_by_T.keys())
        for t in ts:
            try:
                plt.figure(figsize=(5, 4))
                plt.hist(norm_by_T[t], bins=20)
                plt.xlabel("R")
                plt.ylabel("Count")
                plt.title(f"Norm distribution, q={Q}, L={L}, T={t}")
                plt.savefig(dist_output_dir / f"norm_dist__t={t}.png", dpi=200)
            except:
                pass
            rs.append(norm_by_T[t].mean().item())

            try:
                plt.figure(figsize=(5, 4))
                plt.hist(mag_by_T[t], bins=20)
                plt.xlabel(r"$\langle m \rangle$")
                plt.ylabel("Count")
                plt.title(f"Magnetization distribution, q={Q}, L={L}, T={t}")
                plt.savefig(dist_output_dir / f"mag_dist__t={t}.png", dpi=200)
            except:
                pass
            ms.append(mag_by_T[t].mean().item())

            plt.close("all")

        plt.figure(figsize=(5, 4))
        plt.plot(ts, rs)
        plt.xlabel("T")
        plt.ylabel("R")
        plt.savefig(output_dir / f"norm_mean.png", dpi=200)

        all_norm_means[Q, L] = (ts, rs)

        plt.figure(figsize=(5, 4))
        plt.plot(ts, ms)
        plt.xlabel("")
        plt.savefig(output_dir / f"mag_mean.png", dpi=200)

        plt.close("all")

for q in [2, 3, 4]:
    plt.figure(figsize=(5, 4))
    for (Q, L), train_losses in all_train_losses.items():
        if Q != q:
            continue
        plt.plot(np.arange(1, len(train_losses) + 1), train_losses, label=f"q={Q}, L={L}")
    plt.xlabel("Iter")
    plt.ylabel("Train Loss")
    plt.title(f"Training Curve, q={q}")
    plt.legend()
    plt.savefig(output_root / f"training_curve__q={q}.png", dpi=200)

Tc_values = {2: 1.134, 3: 0.995, 4: 0.910}

for q in [2, 3, 4]:
    plt.figure(figsize=(5, 4))
    for (Q, L), (T, R) in all_norm_means.items():
        if Q != q:
            continue
        plt.plot(T, R, label=f"q={Q}, L={L}")
    plt.xlabel("T")
    plt.ylabel("R")
    plt.axvline(Tc_values[q], color='black', linestyle=':', label=f"Tc={Tc_values[q]:.3f}")
    plt.title(f"R, q={q}")
    plt.legend()
    plt.savefig(output_root / f"norm_mean__q={q}.png", dpi=200)
