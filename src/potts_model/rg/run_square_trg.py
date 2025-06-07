import logging
from logging import debug, info
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

    folders = []
    folders.extend([f"./dataset/potts_trg/q({i})_potts/" for i in [2, 3, 4, 5, 6]])
    # folders.extend([f"./dataset/potts_trg/q({i})_potts_higher_N_keep/" for i in [3, 4]])
    # folders.extend(
    #     [f"./dataset/potts_trg/q({i})_potts_clock/" for i in [2, 3, 4, 5, 6]]
    # )
    for folder in folders:
        # folder = "./dataset/potts_trg/q(4)_potts/"
        get_running_schematics = load_running_schematics(
            f"{folder}running_schematics.py"
        )
        run_on = get_running_schematics()
        is_db = False
        is_analyze = False
        is_S_limit_plot = True
        near_Tc = True

        if is_db:
            generate_db(run_on, folder)

        if is_analyze:
            analyze_db(run_on, folder, near_Tc=near_Tc).savefig(
                f"{folder}S-iter{'near_Tc' if near_Tc is True else ''}_plot.png"
            )
        if is_S_limit_plot:
            top_singulars, S_lim_fig = plot_limit_behavior(
                run_on, folder, near_Tc=near_Tc
            )

            top_singulars.savefig(
                f"{folder}Top_10_singulars_{'near_Tc' if near_Tc is True else ''}_plot.png"
            )
            S_lim_fig.savefig(
                f"{folder}S_limit_{'near_Tc' if near_Tc is True else ''}_plot.png"
            )


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
        info(f"{folder}, beta = {beta} generation started")
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
        del run_results


def analyze_db(run_on, folder, near_Tc=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    plotting_run_on = sorted(run_on, key=lambda x: x[3])
    if near_Tc is True:
        q = plotting_run_on[0][0]
        J0 = plotting_run_on[0][2]
        beta_c = 1 / np.abs(J0 / np.log(1 + np.sqrt(q)))
        beta_min = beta_c * 0.89
        beta_max = beta_c * 1.11
        plotting_run_on = [
            item for item in plotting_run_on if beta_min <= item[3] <= beta_max
        ]

    unique_betas = sorted(list(set(item[3] for item in plotting_run_on)))

    cmap = plt.get_cmap("viridis", max(1, len(unique_betas)))
    beta_to_color = {beta: cmap(i) for i, beta in enumerate(unique_betas)}

    J_fn_title = ""
    plotted_labels = []

    print(f"loading {len(plotting_run_on)} numbers of file")
    for q, J_fn_str, J0, beta, total_iter_in_filename, N_keep in plotting_run_on:
        base_filename = (
            f"q({q})_J_fn({J_fn_str})_J0({J0:.4})_"
            f"beta({beta:.4})_N_keep({N_keep})_iter({total_iter_in_filename}).pkl"
        )
        full_filename = os.path.join(folder, base_filename)

        print(f"{base_filename} is loaded")

        Tc = np.abs(J0 / np.log(1 + np.sqrt(q)))

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


def plot_limit_behavior(run_on, folder, near_Tc=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    S_fig, S_ax = plt.subplots(figsize=(10, 6))

    plotting_run_on = sorted(run_on, key=lambda x: x[3])
    if near_Tc is True:
        q = plotting_run_on[0][0]
        J0 = plotting_run_on[0][2]
        beta_c = 1 / np.abs(J0 / np.log(1 + np.sqrt(q)))
        beta_min = beta_c * 0.89
        beta_max = beta_c * 1.11
        plotting_run_on = [
            item for item in plotting_run_on if beta_min <= item[3] <= beta_max
        ]

    unique_betas = sorted(list(set(item[3] for item in plotting_run_on)))

    cmap = plt.get_cmap("viridis", max(1, len(unique_betas)))
    beta_to_color = {beta: cmap(i) for i, beta in enumerate(unique_betas)}

    J_fn_title = ""
    plotted_labels = []

    print(f"loading {len(plotting_run_on)} numbers of file")
    T_lim = []
    S_lim = []
    for q, J_fn_str, J0, beta, total_iter_in_filename, N_keep in plotting_run_on:
        base_filename = (
            f"q({q})_J_fn({J_fn_str})_J0({J0:.4})_"
            f"beta({beta:.4})_N_keep({N_keep})_iter({total_iter_in_filename}).pkl"
        )
        full_filename = os.path.join(folder, base_filename)

        print(f"{base_filename} is loaded")

        Tc = np.abs(J0 / np.log(1 + np.sqrt(q)))

        try:
            with open(full_filename, "rb") as f:
                run_results = pickle.load(f)

            entropies = [
                (result_1.get_neumann_entropy() + result_2.get_neumann_entropy()) / 2
                for result_1, result_2 in run_results
            ]
            T_lim.append(1 / Tc / beta)
            S_lim.append(entropies[-1])

            current_T_over_Tc = 1.0 / (beta * Tc)

            label_str = f"$T/T_c = {current_T_over_Tc:.2f}$ ($\\beta={beta:.3f}$)"

            ax.plot(
                range(10),
                run_results[-1][0].S_spectrum[:10],
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

    ax.set_xlabel("Size orders")
    ax.set_ylabel("Singular values")
    ax.set_yscale("log")
    ax.set_title(f"Top 10 Singular values for {J_fn_title}")
    ax.legend()

    S_ax.plot(
        T_lim,
        S_lim,
        marker="s",
    )
    S_ax.set_xlabel("T/$T_c$")
    S_ax.set_ylabel("Neumann Entropy")
    S_ax.set_title(f"limiting behavior of S - T for {J_fn_title}")

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

    return fig, S_fig


if __name__ == "__main__":
    main()
