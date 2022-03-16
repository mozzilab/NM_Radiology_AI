# %%

# nmrezman
from .general import train_comment_model

# Misc
import argparse


# %%

desc_str = "Train Phase 01 Comment Extraction Model"

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
        "--model_checkpoint_name",
        type=str,
        required=True,
        help="Path / filename to save model checkpoints (e.g., \"/path/to/results/phase01/comment/comment_best_model.sav\")."
    )
    parser.add_argument(
        "--result_fname",
        type=str,
        required=True,
        help="Path / filename to save model evaluation metrics (e.g., \"/path/to/results/phase01/comment/comment_best_result.log\")."
    )

    return parser



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(desc_str, parents=[get_args_parser()])
    args = parser.parse_args()

    # Train the comment extraction model
    train_comment_model(
        data_path=args.data_path,
        model_checkpoint_name=args.model_checkpoint_name,
        result_fname=args.result_fname,
    )

