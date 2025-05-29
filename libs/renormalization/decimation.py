import numpy as np
from logging import info
from libs.models.potts import PottsModel, periodic_neighbors, idx_to_pos, ferro_J


# can not afford a "general structure decimation" so SquareGrid with specific logic is used
class SquareDecimation:
    def __init__(
        self,
        k,  # grid size of 2^k x 2^k
        M,  # number of ensembles
        q,  # potts q
        seed,  # rng seed
        gathering_sequence=[  # list of functions returns "key", f:list_of_models -> data
            lambda list_of_models: ("Energy", get_energy_dist(list_of_models))
        ],
        gathering_keys=["Energy"],
    ):
        self.L = 2**k
        self.q = q
        self.I = list(range(self.L * self.L))
        self.rng = np.random.default_rng(seed)
        neighbors = periodic_neighbors(self.L, self.L)
        self.PottsEnsemble = PottsModel.Generate_Monte_Carlo(
            M=M,
            rng=self.rng,
            q=q,
            index_set=self.I,
            index_to_position_map=idx_to_pos,
            neighbors=neighbors,
            interaction=ferro_J,
        )

        self.gathering_sequence = gathering_sequence
        self.keys = gathering_keys
        self.gathered_data = dict()
        for key in gathering_keys:
            self.gathered_data[key] = []
            self.gathered_data["size"] = []

    def reciprocal_decimation(self):
        L_step = self.L
        while L_step >= 2:
            info(f"grid size {L_step} initiated")

            for model in self.PottsEnsemble:
                half_decimation(model, L_step)

            for fn in self.gathering_sequence:
                key, res = fn(self.PottsEnsemble)
                self.gathered_data[key].append(res)
                self.gathered_data["size"] = L_step

            L_step = int(L_step // 2)


def get_energy_dist(list_of_potts_model):
    return [model.Hamiltonian() for model in list_of_potts_model]


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
