# %%

# nmrezman
from .general import train_comment_model

# Misc
import argparse


# %%

desc_str = "Train Phase 02 QA Comment Extraction Model"

def get_args_parser():
    parser = argparse.ArgumentParser(description=desc_str, add_help=False)

    # Paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataframe file (e.g., \"/path/to/data/reports_df.gz\")."
    )

    # Output file names
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save model checkpoints (e.g., \"/path/to/results/phase02/comment/\")."
    )
    parser.add_argument(
        "--result_fname_prefix",
        type=str,
        required=False,
        default="results",
        help="Result file name prefix to save *.csv in output_dir."
    )

    return parser



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(desc_str, parents=[get_args_parser()])
    args = parser.parse_args()

    # Train the QA comment extraction model
    train_comment_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        result_fname_prefix=args.result_fname_prefix,
    )

