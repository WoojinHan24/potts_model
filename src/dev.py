import logging
import numpy as np
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

    def clock_J(s1, s2, q, J0=1.0):
        return -J0 * np.cos(2 * np.pi * (s1 - s2) / q)

    q = 3
    model = PottsModel(
        q=q,
        index_set=I,
        index_to_position_map=idx_to_pos,
        neighbors=neighbors,
        interaction=lambda s1, s2: clock_J(s1, s2, q),
    )
    
    # test code for verifying optimization for 2d lattice case
    # remove me later
    # check if the results match
    assert np.allclose(model.Hamiltonian(), model.Hamiltonian_2d_lattice_pbc())
    assert np.allclose(model.Hamiltonian(), model.Hamiltonian_respecting_interactions())
    
    n_warmup = 3
    n_time = 100
    
    for i in range(n_warmup):
        model.Hamiltonian_2d_lattice_pbc()
    tick = time.perf_counter()
    for i in range(n_time):
        model.Hamiltonian_2d_lattice_pbc()
    tock = time.perf_counter()
    print(f"np impl (assuming 2d lattice pbc) avg {(tock - tick) / n_time}s")
    
    for i in range(n_warmup):
        model.Hamiltonian_respecting_interactions()
    tick = time.perf_counter()
    for i in range(n_time):
        model.Hamiltonian_respecting_interactions()
    tock = time.perf_counter()
    print(f"np impl (using given neighbors) avg {(tock - tick) / n_time}s")
    
    for i in range(n_warmup):
        model.Hamiltonian()
    tick = time.perf_counter()
    for i in range(n_time):
        model.Hamiltonian()
    tock = time.perf_counter()
    print(f"for loop impl avg {(tock - tick) / n_time}s")


if __name__ == "__main__":
    main()
