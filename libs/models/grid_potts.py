from .potts import PottsModel
import numpy as np


class GridPotts(PottsModel):
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

        # TODO Maybe make this part contains, Nx Ny, ...
        super.__init__(
            self,
            q,
            index_set,
            index_to_position_map,
            neighbors,
            interaction,
            S,
            field=lambda s: 0.0,
        )

    def Hamiltonian(self):
        return self.Hamiltonian_2d_lattice_pbc()

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
