import logging
from logging import debug
from libs.renormalization.decimation import SquareDecimation
from libs.renormalization.tensor_renromalization import run_TRG_square_lattice
from libs.models.potts import clock_J, ferro_J
import pickle
import numpy as np
import importlib.util
import os
import matplotlib.pyplot as plt


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

    folder = "./dataset/potts_trg/Ising_50iter/"
    get_running_schematics = load_running_schematics(f"{folder}running_schematics.py")
    run_on = get_running_schematics()
    is_db = True
    if is_db:
        generate_db(run_on, folder)

    analyze_db(run_on, folder).savefig(f"{folder}S-iter_plot.png")


def load_running_schematics(filepath):
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"No running schematics are ready")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.get_running_schematics


def generate_db(run_on, folder):
    for q, J_fn_str, J0, beta, iteration, N_keep in run_on:
        if J_fn_str == "clock":
            J_fn = clock_J
        elif J_fn_str == "ferro":
            J_fn = ferro_J
        else:
            raise ImportError(f"J_fn_str not ready")
        run_results = run_TRG_square_lattice(
            q=q, J_fn=J_fn, J0=J0, beta=beta, N_keep=N_keep, iteration=iteration
        )

        with open(
            f"{folder}q({q})_J_fn({J_fn_str})_J0({J0:.4})_beta({beta:.4})_N_keep({N_keep})_iter({iteration}).pkl",
            "wb",
        ) as f:
            pickle.dump(run_results, f)


def analyze_db(run_on, folder):
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_betas = sorted(list(set(item[3] for item in run_on)))

    cmap = plt.get_cmap("viridis", max(1, len(unique_betas)))
    beta_to_color = {beta: cmap(i) for i, beta in enumerate(unique_betas)}

    J_fn_title = ""
    plotted_labels = []
    for q, J_fn_str, J0, beta, total_iter_in_filename, N_keep in run_on:
        base_filename = (
            f"q({q})_J_fn({J_fn_str})_J0({J0:.4})_"
            f"beta({beta:.4})_N_keep({N_keep})_iter({total_iter_in_filename}).pkl"
        )
        full_filename = os.path.join(folder, base_filename)
        Tc = np.abs(J0 / np.log(1 + np.sqrt(2)) * 2)

        try:
            with open(full_filename, "rb") as f:
                run_results = pickle.load(f)

            entropies = [
                (result_1.get_neumann_entropy() + result_2.get_neumann_entropy()) / 2
                for result_1, result_2 in run_results
            ]
            iterations = list(range(len(entropies)))

            current_T_over_Tc = 1.0 / (beta * Tc)

            label_str = f"$T/T_c = {current_T_over_Tc:.2f}$ ($\\beta={beta:.3f}$)"

            ax.plot(
                iterations,
                entropies,
                marker="s",
                linestyle="-",
                linewidth=1,
                color=beta_to_color.get(beta),
                label=label_str,
            )
            plotted_labels.append(label_str)
            J_fn_title = f"q = {q}, {J_fn_str} model"
        except FileNotFoundError:
            print(f"Warning: File not found - {full_filename}. Skipping this entry.")

    ax.set_xlabel("Iter")
    ax.set_ylabel("Neumann Entropy $S$")
    ax.set_title(f"Neumann Entropy - Iter Step for {J_fn_title}")
    ax.legend()

    if plotted_labels:
        num_lines = len(plotted_labels)

        legend_ncol = 1
        if num_lines > 10:
            legend_ncol = 2

        lgd = ax.legend(
            loc="center left",
            bbox_to_anchor=(
                1.02,
                0.5,
            ),
            ncol=legend_ncol,
            fontsize="small",
            title="$T/T_c$ ($\\beta$)",
        )

        fig.subplots_adjust(right=0.70 if legend_ncol == 2 else 0.75)

    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    main()
