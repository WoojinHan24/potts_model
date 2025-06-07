import numpy as np
from logging import debug


class PottsModel:
    def __init__(
        self,
        q,
        index_set,
        index_to_position_map,
        neighbors,
        interaction,
        S,
        field=lambda s: 0.0,
    ):
        """
        0~q spin possible
        index_set : I - total index map into int
        index_to_position_map : f(i) --> index to position function
        neighbors : [list of (i,j)] inform i,j are linked
        interaction : J(s1,s2) = s1 s2 interaction, function
        S : Spin configuration of the index_set.
        field : h(s1) = h, s1 magnetic field, function setting
        """
        self.q = q
        self.I = index_set
        self.f = index_to_position_map
        self.linked_info = neighbors
        self.J = interaction
        self.h = field
        self.S = S

    def Hamiltonian(self) -> float:  # for general Hamiltonian calculations
        energy = 0.0
        debug(f"indicies set {len(self.I)}, with len {len(self.S)}")
        debug(
            f"while max idx in linked_info = {max(max(a, b) for a, b in self.linked_info)}"
        )
        for i, j in self.linked_info:
            si, sj = self.S[i], self.S[j]
            energy += self.J(si, sj)
        for i in self.I:
            energy += self.h(self.S[i])
        return energy

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

    @classmethod
    def Generate_Monte_Carlo(
        cls,
        M,
        rng,
        q,
        index_set,
        index_to_position_map,
        neighbors,
        interaction,
        field=lambda s: 0.0,
    ):
        """
        Generate M random PottsModel configurations using same structure.
        This inputs rng for reproductible results.
        """
        models = []
        for _ in range(M):
            spin_config = rng.integers(0, q, size=len(index_set))
            model = cls(
                q,
                index_set,
                index_to_position_map,
                neighbors,
                interaction,
                field=field,
                S=spin_config,
            )
            models.append(model)
        return models


def idx_to_pos(i, Lx, Ly):
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


def ferro_J(s1, s2, q, J0=1.0):
    return np.where(s1 == s2, -J0, 0.0)


def clock_J(s1, s2, q, J0=1.0):
    return -J0 * np.cos(2 * np.pi * (s1 - s2) / q)
