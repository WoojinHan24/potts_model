import numpy as np


class PottsModel:
    def __init__(
        self,
        q,
        index_set,
        index_to_position_map,
        neighbors,
        interaction,
        field=lambda s: 0.0,
    ):
        """
        0~q spin possible
        index_set : I - total index map into int
        index_to_position_map : f(i) --> index to position function
        neighbors : [list of (i,j)] inform i,j are linked
        interaction : J(s1,s2) = s1 s2 interaction, function
        field : h(s1) = h, s1 magnetic field, function setting
        """
        self.q = q
        self.I = index_set
        self.f = index_to_position_map
        self.linked_info = neighbors
        self.J = interaction
        self.h = field
        self.S = np.random.randint(
            0, q, size=len(self.I)
        )  # keep this out smdy? monte carlo seperation

    def Hamiltonian(self) -> float:
        energy = 0.0
        for i, j in self.linked_info:
            si, sj = self.S[i], self.S[j]
            energy += self.J(si, sj)
        for i in self.I:
            energy += self.h(self.S[i])
        return energy

    def set_spin(self, i: int, value: int):
        assert 0 <= value < self.q
        self.S[i] = value

    def get_spin(self, i: int) -> int:
        return self.S[i]

    def spin_config(self) -> np.ndarray:
        return self.S.copy()

    def position(self, i: int):
        return self.f(i)
