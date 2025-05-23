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
            0, q, size=len(self.I), dtype=np.uint8
        )  # keep this out smdy? monte carlo seperation

    def Hamiltonian(self) -> float:
        energy = 0.0
        for i, j in self.linked_info:
            si, sj = self.S[i], self.S[j]
            energy += self.J(si, sj)
        for i in self.I:
            energy += self.h(self.S[i])
        return energy

    def Hamiltonian_2d_lattice_pbc(self) -> float:
        """
        assumes this is 2d square lattice with periodic boundary condition
        TODO: make this separate class, possibly inheriting PottsModel
        """
        L = int(np.sqrt(len(self.I)))
        S_2d = self.S.reshape(L, L)
        J_term_vertical = np.sum(self.J(S_2d, np.roll(S_2d, 1, axis=0)))
        J_term_horizontal = np.sum(self.J(S_2d, np.roll(S_2d, 1, axis=1)))
        h_term = np.sum(self.h(self.S))
        return J_term_vertical + J_term_horizontal + h_term

    def Hamiltonian_respecting_interactions(self) -> float:
        """
        TODO: maybe make this the Hamiltonian method?
        """
        S_pairs = self.S[self.linked_info]  # shape: [E, 2]
        J_term = np.sum(self.J(S_pairs[:, 0], S_pairs[:, 1]))
        h_term = np.sum(self.h(self.S))
        return J_term + h_term

    def set_spin(self, i: int, value: int):
        assert 0 <= value < self.q
        self.S[i] = value

    def get_spin(self, i: int) -> int:
        return self.S[i]

    def spin_config(self) -> np.ndarray:
        return self.S.copy()

    def position(self, i: int):
        return self.f(i)
