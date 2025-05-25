import logging
import numpy as np
from libs.models.potts import PottsModel
from logging import info


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    Lx, Ly = 64, 64  # code below assumes Lx=Ly

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

    M = 100
    info(f"random ensemble of {M} creation")
    rng = np.random.default_rng(seed=42)
    PottsEnsemble = PottsModel.Generate_Monte_Carlo(
        M=M,
        rng=rng,
        q=q,
        index_set=I,
        index_to_position_map=idx_to_pos,
        neighbors=neighbors,
        interaction=ferro_J,
    )

    def get_energy_dist(list_of_potts_model):
        return [model.Hamiltonian() for model in list_of_potts_model]

    # here, should be potts model, with "above configuration"
    # TODO This is the idea to  implement RG in class, but finally, this must run wrapped.
    def get_group_spin(list_of_spins):
        return round(sum(list_of_spins) / len(list_of_spins))

    def get_2x2_blocks(L: int):
        k = int(L // 2)
        query = []
        for i in range(k):
            for j in range(k):
                top_left = (2 * i) * L + (2 * j)
                indices = (
                    top_left,
                    top_left + 1,
                    top_left + L,
                    top_left + L + 1,
                )
                decimated_index = i * k + j
                query.append((indices, decimated_index))
        return query

    L_step = Lx  # assume Lx=Ly

    def half_decimation(model, L):
        block_querry = get_2x2_blocks(L)
        decimated_spin = []
        k = int(L // 2)
        model.I = list(range(k * k))
        model.linked_info = periodic_neighbors(k, k)

        for indicies, _ in block_querry:
            decimated_spin.append(get_group_spin([model.S[i] for i in indicies]))

        model.S = decimated_spin
        # TODO better set function
        # updates model.

    E_dists = [(L_step, get_energy_dist(PottsEnsemble))]
    while L_step >= 2:
        info(f"grid size {L_step} initiated")

        for model in PottsEnsemble:
            half_decimation(model, L_step)

        info(f"energy dist updating")
        E_dists.append((L_step, get_energy_dist(PottsEnsemble)))
        L_step = int(L_step // 2)

    print(E_dists)


if __name__ == "__main__":
    main()
