import logging
import numpy as np
from libs.models.potts_model import PottsModel
import os


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
        return -J0 if s1 == s2 else 0.0

    def clock_J(s1, s2, q, J0=1.0):
        return -J0 * np.cos(2 * np.pi * (s1 - s2) / q)

    model = PottsModel(
        q=3,
        index_set=I,
        f_index_to_position=idx_to_pos,
        neighbors=neighbors,
        interaction=lambda s1, s2: ferro_J(s1, s2),
    )


if __name__ == "__main__":
    main()
