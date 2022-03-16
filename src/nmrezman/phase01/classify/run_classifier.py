# %%

from .classifier import classifier
import argparse

# %%


def get_args_parser():
    parser = argparse.ArgumentParser(description="NM Results Management Phase 01", add_help=False)

    # Paths
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to location of the the model checkpoints (e.g., \"/path/to/checkpoints/phase01/\")."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data file to run inference on (e.g., \"/path/to/data/report.txt\")."
    )

    return parser



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser("NM Results Management Phase 01", parents=[get_args_parser()])
    args = parser.parse_args()

    # Get the report text from the specified file
    with open(args.data_path) as fh:
        args.data = fh.read()
   
    # Classify the report using defined model checkpoints
    output = classifier(
        data=args.data,
        model_path=args.model_path,
    )

    # Print the result
    print('====================================')
    print("Output:")
    [print(f"  {key}:", value) for key, value in output.items()]


# %%