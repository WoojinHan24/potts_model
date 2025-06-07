import numpy as np
import itertools
from logging import info, debug


def run_TRG_square_lattice(q, J_fn, J0, beta, N_keep, iteration):
    print("hello world")

    A = get_initial_A(q, J0, J_fn, beta)

    run_results = []
    for i in range(iteration):
        info(f"iteration {i+1} started for TRG")
        result_1 = get_T(A, N_keep)
        result_2 = get_T(A.transpose(1, 0, 2, 3), N_keep)
        debug(f"T granted, top 5 S_spectrum --- \n {result_1.S_spectrum[:5]}")
        debug(f"T tilde granted, top 5 S_spectrum --- \n {result_2.S_spectrum[:5]}")
        T1 = result_1.T
        T3 = result_1.Td
        T2 = result_2.T
        T4 = result_2.Td
        A = get_A(T1, T2, T3, T4)
        # TT = A check
        # TT = np.einsum(f"ijk,lmk->ijlm", T, T)
        # print(f"TT-A = {print_mat(TT-A)}")
        # print(f"A = {print_mat(A)}")

        run_results.append([result_1, result_2])

    # This method will lose final updated A (which is not vital since spectrum only matters)
    return run_results  # list of TRG_Results


class TRG_Results:
    def __init__(
        self,
        T,  # T matrix of the A
        Td,
        A,  # A this initial iter
        S_spectrum,  # S full spectrum of the A
    ):
        self.T = T
        self.Td = Td
        self.A = A
        self.S_spectrum = S_spectrum

    def get_neumann_entropy(self):
        s_values = np.array(self.S_spectrum, dtype=np.float64)

        probabilities = s_values**2

        sum_prob = np.sum(probabilities)
        if sum_prob > 0:
            probabilities /= sum_prob
        else:
            return 0.0

        nonzero_probabilities = probabilities[probabilities > 0]

        if nonzero_probabilities.size == 0:
            return 0.0

        entropy = -np.sum(nonzero_probabilities * np.log(nonzero_probabilities))

        return entropy


# deprecated
def symmetrize(A):
    # Hardcoded for rank 4
    assert len(A.shape) == 4, "This method only supports rank-4 tensor"

    # Exploit repeated permutations for optimization
    A = (A + A.transpose(0, 1, 3, 2) + A.transpose(0, 2, 1, 3)) / 3
    A = (A + A.transpose(0, 3, 2, 1)) / 2
    # So far, the above covers all permutations that keep rank 0 intact

    # Now enforce shifting permutations:
    A = (A + A.transpose(1, 2, 3, 0)) / 2
    A = (A + A.transpose(2, 3, 0, 1)) / 2

    return A


def print_mat(A):
    return np.min(A), np.max(A)


def get_initial_A(q, J0, J_fn, beta):
    # [s_up, s_right, s_down, s_left]
    A = np.zeros((q, q, q, q))

    for s1 in range(q):
        for s2 in range(q):
            for s3 in range(q):
                for s4 in range(q):
                    A[s1, s2, s3, s4] = 0
                    for s0 in range(q):
                        E = (
                            J_fn(s1, s0, q, J0)
                            + J_fn(s2, s0, q, J0)
                            + J_fn(s3, s0, q, J0)
                            + J_fn(s4, s0, q, J0)
                        )
                        A[s1, s2, s3, s4] += np.exp(-beta * E)

                    # print(f"A[{s1}, {s2}, {s3}, {s4}] = {A[s1, s2, s3, s4]}")

    # A is symmetric, positive elements
    # print(f"A-A^T = {np.min(A-A.T)}, {np.max(A-A.T)}")
    # print(f"A = {np.min(A)}, {np.max(A)}")
    return A


def get_T(A, N_keep):
    A = A / A.max()
    Ashape = A.shape
    q = Ashape[0]
    if Ashape[1] != q or Ashape[2] != q or Ashape[3] != q:
        raise ValueError(f"A shape wrong {A.shape}")

    N = min(N_keep, q * q)
    # [(q^2,q^2), q^2, (q^2,q^2)]
    U_full, S_full, Vh_full = np.linalg.svd(
        A.reshape((q * q, q * q)), full_matrices=True
    )
    S = S_full[:N]  # [q]
    # S normalization for each iteration
    # Note this step comes up in literatures, which is confining the system sizes
    # Physical meaning hide inside S[1]/S[0] it says.
    # without this steps, whole code diverges practically.
    S_sqrt = np.sqrt(S)

    U = U_full[:, :N] @ np.diag(S_sqrt)  # [q^2,q]
    Vh = np.diag(S_sqrt) @ Vh_full[:N, :]  # [q,q^2]

    # well estimated
    # print(f"A-U Vh = {A.reshape(q*q,q*q)-U@Vh}")

    # U == Vh^dagger
    # print(f"U-Vh^dagger = {U - Vh.conj().T}")

    # T T^dagger = A, T [q,q,q]
    T = U.reshape(q, q, N)
    Td = Vh.conj().T.reshape(q, q, N)

    # check whether TTd == A
    # TTd = np.einsum("ijk,klm->ijlm", T, T.conj().T)
    # print(f"TTd - A = {np.min(TTd-A)}, {np.max(TTd-A)}")

    result = TRG_Results(T, Td, A, S_full)
    return result


def get_A(T1, T2, T3, T4):
    # What we want to do:
    # # A = np.einsum("ija,kjb,klc,lid->abcd", T, T, T, T)
    # A1 = np.einsum("ija,kjb->ikab", T, T)
    # A2 = np.einsum("klc,lid->kicd", T, T)
    # A = np.einsum("ikab,kicd->abcd", A1, A2)

    # A1 = np.einsum("mpi,mnj->ijnp", T1, T2)
    # A2 = np.einsum("pol,nok->pnlk", T2, T1)
    # A = np.einsum("ijnp,pnlk->ijkl", A1, A2)

    A = np.einsum("abs,bct,cdu,dav->stuv", T1, T2, T3, T4, optimize="optimal")

    # A = symmetrize(A)
    return A
