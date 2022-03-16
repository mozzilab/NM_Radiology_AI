# %%

# nmrezman
from .general import train_findings_model

# Misc
import argparse


# %%

desc_str = "Train Phase 01 Findings vs No Findings Model"

def get_args_parser():
    parser = argparse.ArgumentParser(description=desc_str, add_help=False)

    # Paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataframe file (e.g., \"/path/to/data/reports_df.gz\")."
    )
    parser.add_argument(
        "--glove_embedding_path",
        type=str,
        required=True,
        help="Path to GloVe word vector glove.6B.300d file (e.g., \"/path/to/data/glove.6B.300d.txt\")."
    )

    # Output file names
    parser.add_argument(
        "--model_checkpoint_name",
        type=str,
        required=True,
        help="Path / filename to save model checkpoints (e.g., \"/path/to/results/phase01/findings/findings_best_model.h5\")."
    )
    parser.add_argument(
        "--result_fname",
        type=str,
        required=True,
        help="Path / filename to save model evaluation metrics (e.g., \"/path/to/results/phase01/findings/findings_best_result.log\")."
    )
    parser.add_argument(
        "--tokenizer_fname",
        type=str,
        required=True,
        help="Path / filename to save tokenizer (e.g., \"/path/to/results/phase01/findings/tokenizer.gz\")."
    )

    return parser



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(desc_str, parents=[get_args_parser()])
    args = parser.parse_args()

    # Train the findings vs no findings model
    train_findings_model(
        data_path=args.data_path,
        glove_embedding_path=args.glove_embedding_path,
        model_checkpoint_name=args.model_checkpoint_name,
        result_fname=args.result_fname,
        tokenizer_fname=args.tokenizer_fname,
    )

