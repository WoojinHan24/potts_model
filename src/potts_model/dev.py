import logging
from logging import debug
from libs.renormalization.decimation import SquareDecimation
from libs.renormalization.tensor_renromalization import run_TRG_square_lattice
from libs.models.potts import clock_J, ferro_J
import pickle
import numpy as np


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # SD = SquareDecimation(
    #     k=6,
    #     M=500,
    #     q=3,
    #     seed=42,
    # )
    # SD.reciprocal_decimation()

    # for key in SD.keys:
    #     print(
    #         f"{key} values at each {SD.gathered_data['size']} \n are {SD.gathered_data[key]}"
    #     )

    run_on = get_running_schematics()
    is_db = True
    if is_db:
        generate_db(run_on, "./dataset/potts_trg/2505311902/")

    analyze_db(run_on)


def get_running_schematics():
    beta_c = np.log(1 + np.sqrt(2)) / 2
    q = 3
    J_fn_str = "clock"
    J0 = -1.0
    beta = 0.5
    iteration = 5
    N_keep = 20

    running_condtn = []
    for k in [0.8, 0.9, 1.0, 1.1, 1.2]:
        running_condtn.append((q, J_fn_str, J0, beta_c * k, iteration, N_keep))


def generate_db(run_on, folder):
    for q, J_fn_str, J0, beta, iteration, N_keep in run_on:
        if J_fn_str == "clock":
            J_fn = clock_J
        else:
            J_fn = ferro_J
        run_results = run_TRG_square_lattice(
            q=q, J_fn=J_fn, J0=J0, beta=beta, N_keep=N_keep, iteration=iteration
        )

        with open(
            f"{folder}q({q})_J_fn({J_fn_str})_J0({J0:.4})_beta({beta:.4})_N_keep({N_keep})_iter({iteration}).pkl",
            "wb",
        ) as f:
            pickle.dump(run_results, f)


def analyze_db(run_on):
    for q, J_fn_str, J0, beta, iteration, N_keep in run_on:
        root = "./"
        with open(
            f"{root}dataset/potts_trg/q({q})_J_fn({J_fn_str})_J0({J0:.4})_beta({beta:.4})_N_keep({N_keep})_iter({iteration}).pkl",
            "rb",
        ) as f:
            run_results = pickle.load(f)
            debug(
                f"opened {root}dataset/potts_trg/q({q})_J_fn({J_fn_str})_J0({J0:.4})_beta({beta:.4})_N_keep({N_keep})_iter({iteration}).pkl"
            )


if __name__ == "__main__":
    main()
