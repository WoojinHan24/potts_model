"""
Unsupervised analysis of 2-D Potts model configurations using PCA + k-means
Reproducing the result from https://link.springer.com/article/10.1140/epjb/s10051-022-00453-3
"""

import pandas as pd
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit


def extract_Tc_half_m(q: int, L_values: list[int], out_root: Path = Path("logs/unsupervised_learning")):
    L_inv = []
    Tc_L = []
    for L in L_values:
        conf_name = f"delta__swendsen_wang__q={q}__L={L}"
        out_dir = out_root / conf_name
        df = pd.read_csv(out_dir / "order_parameter.csv").sort_values("T")
        ms = df["mean_centroid_norm"] / L
        ts = df["T"].values

        idx = np.where(ms > 0.5)[0]
        if len(idx) == 0 or idx[-1] + 1 >= len(ms):
            continue
        i = idx[-1]
        # Linear interpolate between ms[i] and ms[i+1]
        m0, m1 = ms[i], ms[i + 1]
        t0, t1 = ts[i], ts[i + 1]
        Tc_interp = t0 + (0.5 - m0) * (t1 - t0) / (m1 - m0)
        L_inv.append(1 / L)
        Tc_L.append(Tc_interp)

    plt.figure(figsize=(5, 4))
    plt.scatter(L_inv, Tc_L, label="Extracted $T_c$", color="black")

    def power_law(x, a, b, c):
        return a * x ** b + c

    popt, _ = curve_fit(power_law, L_inv, Tc_L, p0=(1.0, 1.0, 1.0))
    x_fit = np.linspace(0, max(L_inv), 100)
    y_fit = power_law(x_fit, *popt)
    plt.plot(x_fit, y_fit, "r--", label=f"Power-law fit: {popt[0]:.3f}x^{popt[1]:.3f} + {popt[2]:.3f}")

    plt.xlabel("$1/L$")
    plt.ylabel("$T_c$")
    plt.title(f"Scaling of predicted $T_c$ — q={q}")
    plt.legend()
    plt.savefig(out_root / f"Tc_scaling_q{q}.png", dpi=200)
    plt.close()


def _load_all_samples(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows, temps = [], []
    for npz in sorted(data_dir.glob("t=*.npz"), key=lambda p: float(p.stem.split("=")[1])):
        T = float(npz.stem.split("=")[1])
        arr = np.load(npz)["samples"]
        rows.append(arr)
        temps.extend([T] * arr.shape[0])
    return np.concatenate(rows, axis=0), np.asarray(temps, dtype=np.float32)


def _encode_trig(samples: np.ndarray, q: int) -> np.ndarray:
    """
    Encode spins like clock
    See footnote 1 in the paper
    """
    theta = 2 * math.pi * samples / q
    cos, sin = np.cos(theta), np.sin(theta)
    return np.stack((cos, sin), axis=-1).reshape(samples.shape[0], -1).astype(np.float32)


def analyze_configuration_set(
    q: int,
    L: int,
    data_root: Path = Path("dataset/debug_dataset_v1"),
    out_root: Path = Path("logs/unsupervised_learning"),
    random_seed: int = 42,
) -> None:
    """
    Perform PCA+ K-means analysis for a single (q, L) dataset
    """
    conf_name = f"delta__swendsen_wang__q={q}__L={L}"
    data_dir = data_root / conf_name
    out_dir = out_root / conf_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    X_int, temps = _load_all_samples(data_dir)
    X = _encode_trig(X_int, q)

    # PCA
    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X)

    # Scatter plot
    scatter_png = out_dir / "pca_scatter.png"
    fig = plt.figure(figsize=(7, 5))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=temps, cmap="viridis", s=4, alpha=0.6)
    plt.xlabel("$p_1$")
    plt.ylabel("$p_2$")
    plt.title(f"PCA Scatter Plot, q={q}, L={L}")
    plt.colorbar(sc, label="T")
    plt.tight_layout()
    plt.savefig(scatter_png)
    plt.close(fig)

    # K-means
    unique_T = np.unique(temps)
    assert len(unique_T) > 0
    order_param = []
    for T in sorted(unique_T):
        mask = temps == T
        km = KMeans(n_clusters=q, n_init="auto", random_state=random_seed)
        km.fit(X_pca[mask])
        m_mean = np.linalg.norm(km.cluster_centers_, axis=1).mean()
        order_param.append((T, m_mean))

    Ts, ms = zip(*order_param)

    # Output
    df = pd.DataFrame(order_param, columns=["T", "mean_centroid_norm"])
    df.to_csv(out_dir / "order_parameter.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(Ts, ms, marker="o")
    plt.xlabel("T")
    plt.ylabel("$\\langle m \\rangle$")
    plt.title(f"PCA + k-means order parameter — q={q}, L={L}")
    plt.tight_layout()
    plt.savefig(out_dir / "order_parameter.png", dpi=200)
    plt.close(fig)


Tc_values = {2: 1.134, 3: 0.995, 4: 0.910, 5: 0.851}


def plot_finite_size_effects(q: int, L_values: list[int], out_root: Path = Path("logs/unsupervised_learning")):
    plt.figure(figsize=(5, 4))
    for L in L_values:
        conf_name = f"delta__swendsen_wang__q={q}__L={L}"
        out_dir = out_root / conf_name
        df = pd.read_csv(out_dir / "order_parameter.csv")
        plt.plot(df["T"], df["mean_centroid_norm"] / L, marker="o", label=f"L={L}")
        
    if q in Tc_values:
        plt.axvline(Tc_values[q], color='black', linestyle=':', label=f"Tc={Tc_values[q]:.3f}")
    plt.xlabel("T")
    plt.ylabel("$\\langle m \\rangle / L$")
    plt.title(f"PCA + k-means order parameter — q={q}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_root / f"finite_size_comparison_q{q}.png", dpi=200)
    plt.close()

Q_range = [2, 3, 4, 5]
L_range = [10, 20, 30, 40, 50]

def main() -> None:
    for q in Q_range:
        for L in L_range:
            analyze_configuration_set(q, L)

    for q in Q_range:
        plot_finite_size_effects(q, L_range)

    for q in Q_range:
        extract_Tc_half_m(q, L_range)


if __name__ == "__main__":
    main()
