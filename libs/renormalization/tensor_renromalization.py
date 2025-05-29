import numpy as np
from libs.models.potts import clock_J
import itertools


def dev():
    print("hello world")
    Lx, Ly = 64, 64
    q = 3
    J_fn = clock_J
    J0 = 1.0
    A = get_initial_A(q, J0, J_fn)

    N_keep = 16
    for _ in range(3):
        # These names follow https://tensornetwork.org/trg/
        F1, F3 = get_T(A, N_keep)
        F2, F4 = get_T(A.transpose(0, 3, 2, 1), N_keep)
        # TT = A check
        # TT = np.einsum(f"ijk,lmk->ijlm", T, T)
        # print(f"TT-A = {compare_mat(TT, A)}")
        
        # M = np.einsum("ija,jkb->ikab", T, T2)
        # A = np.einsum("ikab,kicd->abcd", M, M)
        # A = symmetrize(A)

        A = np.einsum("ija,kjb,klc,ild->abcd", F1, F4, F3, F2)

        # T symmetry check
        # print(f"T_ijk - T_jik", compare_mat(T, T.transpose(1, 0, 2)))
        # print(f"T2_ijk - T2_jik", compare_mat(T2, T2.transpose(1, 0, 2)))

        # Do we need symmetry?
        # reflection symmetry ok
        print(f"A_ijkl - A_jilk = {print_mat(A-A.transpose(1,0,3,2))}")
        # rotation symmetry ok
        print(f"A_ijkl - A_jkli = {print_mat(A-A.transpose(1,2,3,0))}")
        # TODO but no exchanges
        print(f"A_ijkl - A_jikl = {print_mat(A-A.transpose(1,0,2,3))}")


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


def compare_mat(A, B):
    absdiff = np.max(np.abs(A - B)).item()
    maxrel = absdiff / max(np.max(np.abs(A)).item(), np.max(np.abs(B)).item())
    meanrel = absdiff / ((np.mean(np.abs(A)).item() + np.mean(np.abs(B)).item()) / 2)
    return f"abs {absdiff} maxrel {maxrel} meanrel {meanrel}"


def get_initial_A(q, J0, J_fn):
    # [s_up, s_down, s_right, s_left]
    A = np.zeros((q, q, q, q))
    beta = 1

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
    S_sqrt = np.sqrt(S)

    U = U_full[:, :N] @ np.diag(S_sqrt)  # [q^2,q]
    Vh = np.diag(S_sqrt) @ Vh_full[:N, :]  # [q,q^2]

    # well estimated
    # print(f"A-U Vh = {A.reshape(q*q,q*q)-U@Vh}")

    # U == Vh^dagger
    # print(f"U-Vh^dagger = {U - Vh.conj().T}")

    # T T^dagger = A, T [q,q,q]
    # T = (0.5 * (U + Vh.conj().T)).reshape(q, q, N)

    # check whether TTd == A
    # TTd = np.einsum("ijk,klm->ijlm", T, T.conj().T)
    # print(f"TTd - A = {np.min(TTd-A)}, {np.max(TTd-A)}")
    UVh = (U @ Vh).reshape(q, q, q, q)
    print(f"UVh - A = {np.min(UVh-A)}, {np.max(UVh-A)}")

    return U.reshape(q, q, N), Vh.conj().T.reshape(q, q, N)
