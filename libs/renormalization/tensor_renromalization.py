import numpy as np
import itertools
from logging import info


def run_TRG_square_lattice(q, J_fn, J0, beta, N_keep, iteration):
    print("hello world")

    A = get_initial_A(q, J0, J_fn, beta)

    run_results = []
    for i in range(iteration):
        info(f"iteration {i+1} started for TRG")
        result = get_T(A, N_keep)
        info(f"T granted, top 5 S_spectrum --- \n {result.S_spectrum[:5]}")
        T = result.T
        A = get_A(T)
        # TT = A check
        # TT = np.einsum(f"ijk,lmk->ijlm", T, T)
        # print(f"TT-A = {print_mat(TT-A)}")
        # print(f"A = {print_mat(A)}")

        run_results.append(result)

    # This method will lose final updated A (which is not vital since spectrum only matters)
    return run_results  # list of TRG_Results


class TRG_Results:
    def __init__(
        self,
        T,  # T matrix of the A
        A,  # A this initial iter
        S_spectrum,  # S full spectrum of the A
    ):
        self.T = T
        self.A = A
        self.S_spectrum = S_spectrum

    def get_neumann_entropy(self):
        arr = np.array(self.S_spectrum, dtype=np.float64)

        arr /= arr.sum()
        nonzero = arr[arr > 0]

        entropy = -np.sum(nonzero * np.log(nonzero))

        return entropy


def symmetrize(A):
    symmetrized_A_sum = np.zeros_like(A, dtype=np.float64)

    num_axes = A.ndim
    axes_indices = list(range(num_axes))  # [0, 1, 2, 3]

    num_permutations = 0
    for p in itertools.permutations(axes_indices):
        symmetrized_A_sum += A.transpose(p)
        num_permutations += 1

    if num_permutations == 0:
        return symmetrized_A_sum

    return symmetrized_A_sum / num_permutations


def print_mat(A):
    return np.min(A), np.max(A)


def get_initial_A(q, J0, J_fn, beta):
    # [s_up, s_down, s_right, s_left]
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
    s_max = S[0]
    S_sqrt = np.sqrt(S / s_max)

    U = U_full[:, :N] @ np.diag(S_sqrt)  # [q^2,q]
    Vh = np.diag(S_sqrt) @ Vh_full[:N, :]  # [q,q^2]

    # well estimated
    # print(f"A-U Vh = {A.reshape(q*q,q*q)-U@Vh}")

    # U == Vh^dagger
    # print(f"U-Vh^dagger = {U - Vh.conj().T}")

    # T T^dagger = A, T [q,q,q]
    T = (0.5 * (U + Vh.conj().T)).reshape(q, q, N)

    # check whether TTd == A
    # TTd = np.einsum("ijk,klm->ijlm", T, T.conj().T)
    # print(f"TTd - A = {np.min(TTd-A)}, {np.max(TTd-A)}")

    result = TRG_Results(T, A, S_full)
    return result


def get_A(T):
    A = np.einsum("ija,kjb,klc,lid->abcd", T, T, T, T)
    A = symmetrize(A)
    return A
