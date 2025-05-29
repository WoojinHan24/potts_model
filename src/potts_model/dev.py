import logging
from libs.renormalization.decimation import SquareDecimation
from libs.renormalization.tensor_renromalization import dev


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

    dev()


if __name__ == "__main__":
    main()
