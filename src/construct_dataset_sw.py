"""
Usage: for q in 2 3 4 5 10; do; python src/construct_dataset_sw.py $q &; done
"""


from argparse import ArgumentParser
from functools import partial
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from libs.algorithms.swendsen_wang import run_swendsen_wang
from libs.models.potts_model import PottsModel



def idx_to_pos(Lx, i):
    return (i // Lx, i % Lx)


def periodic_neighbors(Lx, Ly):
    neighbors = []
    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y
            right = x * Ly + (y + 1) % Ly
            down = ((x + 1) % Lx) * Ly + y
            neighbors.append((i, right))
            neighbors.append((i, down))
    return neighbors


def ferro_J(s1, s2, J0=1.0):
    return np.where(s1 == s2, -J0, 0.0)


def main():
    parser = ArgumentParser()
    parser.add_argument("q", type=int)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    output_root = Path("dataset")
    output_root.mkdir(exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    thermalization_iters, num_samples, iter_per_sample, energy_log_period = 100000, 50000, 2000, 100

    L_range = [10, 20, 40, 80, 120]
    # q_range = [2, 3, 4, 5, 10]
    q_range = [args.q]
    T_range = [0.4, 0.5] + [i / 100 for i in range(60, 120)] + [1.2, 1.4]
    # T_range = [0.4, 0.8, 1.0, 1.5]
    for L in L_range:
        I = list(range(L * L))
        neighbors = periodic_neighbors(L, L)

        for q in q_range:
            plt.figure()
            output_dir = output_root / f"delta__swendsen_wang__q={q}__L={L}"
            output_dir.mkdir(exist_ok=True)
            mean_energy, std_energy = [], []
            for T in T_range:
                model = PottsModel(
                    q=q,
                    index_set=I,
                    index_to_position_map=partial(idx_to_pos, L),
                    neighbors=neighbors,
                    interaction=lambda s1, s2: ferro_J(s1, s2, q),
                )
                samples, energy_log = run_swendsen_wang(model, T, thermalization_iters, num_samples, iter_per_sample, energy_log_period)
                
                output_path = output_dir / f"t={T}.npz"
                np.savez_compressed(output_path, allow_pickle=False, samples=np.stack(samples, axis=0), energy_log=energy_log)
                mean_energy.append(np.mean(energy_log))
                std_energy.append(np.std(energy_log))
            
            plt.xlabel("T")
            plt.ylabel("Mean Energy")
            plt.errorbar(T_range, mean_energy, std_energy)
            plt.savefig(log_dir / f"energy_plot_q={q}__L={L}.png")


if __name__ == "__main__":
    main()
