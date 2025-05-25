"""
Usage: for q in 2 3 4 5 10; do; python src/construct_dataset_sw.py $q &; done
"""


from argparse import ArgumentParser
from functools import partial
from joblib import Parallel, delayed
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from libs.simulation.swendsen_wang import run_swendsen_wang
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


def run(seed, output_root, q, L, T, thermalization_iters, num_samples, iter_per_sample, energy_log_period):
    logging.info(f"Start generating dataset for q={q}, L={L}, T={T}")
    logging.info(f"Total iters: {thermalization_iters + num_samples * iter_per_sample}")
    I = list(range(L * L))
    neighbors = periodic_neighbors(L, L)

    np.random.seed(seed)

    output_dir = output_root / f"delta__swendsen_wang__q={q}__L={L}"
    output_dir.mkdir(exist_ok=True)
    model = PottsModel(
        q=q,
        index_set=I,
        index_to_position_map=partial(idx_to_pos, L),
        neighbors=neighbors,
        interaction=ferro_J,
    )
    samples, energy_log = run_swendsen_wang(model, T, thermalization_iters, num_samples, iter_per_sample, energy_log_period)
    
    output_path = output_dir / f"t={T}.npz"
    np.savez_compressed(output_path, allow_pickle=False, samples=np.stack(samples, axis=0), energy_log=energy_log)

    logging.info(f"Done! Generated dataset to {str(output_path)}")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    output_root = Path("dataset/debug_dataset_v1")
    output_root.mkdir(exist_ok=True)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    thermalization_iters, num_samples, iter_per_sample, energy_log_period = 100000, 1000, 2000, 100

    q_range = [2, 3, 4]#, 5, 10]
    L_range = [10, 20, 40]#, 80, 120]
    T_range_past = set([0.4, 0.5] + [i / 100 for i in range(60, 120, 5)] + [1.2, 1.4])
    T_range = sorted(set([i / 100 for i in range(60, 120, 1)]).difference(T_range_past))
    # T_range = [0.4, 0.8, 1.0, 1.5]
    args = [(q, L, T) for q in q_range for L in L_range for T in T_range]
    tasks = [delayed(run)(42, output_root, q, L, T, thermalization_iters, num_samples, iter_per_sample, energy_log_period) for q, L, T in args]
    Parallel(n_jobs=10)(tasks)

if __name__ == "__main__":
    main()
