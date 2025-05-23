import logging
from pathlib import Path
import numpy as np
from libs.algorithms.swendsen_wang import run_swendsen_wang
from libs.models.potts_model import PottsModel
import os
import time


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    Lx, Ly = 40, 40

    def idx_to_pos(i):
        return (i // Lx, i % Lx)

    I = list(range(Lx * Ly))

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

    neighbors = periodic_neighbors(Lx, Ly)

    def ferro_J(s1, s2, J0=1.0):
        return np.where(s1 == s2, -J0, 0.0)

    q = 3
    model = PottsModel(
        q=q,
        index_set=I,
        index_to_position_map=idx_to_pos,
        neighbors=neighbors,
        interaction=lambda s1, s2: ferro_J(s1, s2, q),
    )

    output_dir = Path("data")
    
    for q in [2, 3, 4, 5, 10]:
        for T in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
            ret = run_swendsen_wang(model, T, 100000, 2000)
            output_path = output_dir / f"delta__q={q}__t={T}__swendsen_wang.npz"
            np.savez_compressed(output_path, np.stack(ret, axis=0))


if __name__ == "__main__":
    main()
