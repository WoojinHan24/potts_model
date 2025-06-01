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
        # --- 범례 처리 시작 ---
        num_lines = len(plotted_labels)

        # 옵션 1: 범례를 그래프 오른쪽에 배치 (여러 열 사용 가능)
        # ncol은 범례 항목 수에 따라 적절히 조절 (예: 1, 2, 또는 3)
        # bbox_to_anchor의 x값을 조절하여 그래프와의 간격 설정 (1.02는 그래프 바로 오른쪽)
        # fontsize를 조절하여 글자 크기 변경
        legend_ncol = 1  # 항목이 매우 많으면 2 이상으로 변경 가능
        if num_lines > 10:  # 예시: 10줄이 넘어가면 2열로
            legend_ncol = 2

        lgd = ax.legend(
            loc="center left",  # 범례의 왼쪽 중앙을 기준점으로
            bbox_to_anchor=(
                1.02,
                0.5,
            ),  # 기준점을 그래프 영역의 (1.02, 0.5) 위치로 이동
            ncol=legend_ncol,
            fontsize="small",  # 또는 'x-small'
            title="$T/T_c$ ($\\beta$)",  # 범례 제목 추가 (선택 사항)
        )

        # 범례를 위한 공간 확보
        # fig.tight_layout() 호출 전에 subplot 영역 조정
        # right 값을 줄여서 오른쪽 여백을 만듭니다. (예: 0.75는 오른쪽 25%를 여백으로)
        # 값은 범례의 너비에 따라 조절해야 합니다.
        fig.subplots_adjust(
            right=0.70 if legend_ncol == 2 else 0.75
        )  # ncol에 따라 조절

        # 옵션 2: 범례를 그래프 아래쪽에 배치 (여러 열 사용)
        # legend_ncol_bottom = min(num_lines, 4) # 아래쪽에 넓게 배치할 경우 열 개수 늘림
        # lgd = ax.legend(
        #     loc='upper center', # 범례의 위쪽 중앙을 기준점으로
        #     bbox_to_anchor=(0.5, -0.15), # 기준점을 그래프 아래쪽으로 이동 (-0.15는 간격)
        #     ncol=legend_ncol_bottom,
        #     fontsize='small',
        #     fancybox=True, # 테두리 스타일 (선택 사항)
        #     shadow=True    # 그림자 스타일 (선택 사항)
        # )
        # fig.subplots_adjust(bottom=0.25) # 아래쪽 여백 확보 (ncol 수와 글자 크기에 따라 조절)

        # --- 범례 처리 끝 ---

    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()  # Adjust layout to make sure everything fits

    return fig


if __name__ == "__main__":
    main()
