# %%

# nmrezman
from .general import pretrain_roberta_base

# Misc
import argparse


# %%

desc_str = "Pretrain RoBERTa base model"

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
        help="Path to save model checkpoints (e.g., \"/path/to/results/phase02/pretrain\")."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        required=True,
        help="Path to save logging data (e.g., \"/path/to/results/phase02/pretrain/logging\")."
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        required=False,
        help="Path to save wandb logs (e.g., \"/path/to/results/phase02/pretrain\")."
    )

    # Misc
    parser.add_argument(
        "--do_reporting",
        type=str,
        default="True",
        help="Whether 🤗 will report to logs to supported integrations (True or False)."
    )

    return parser



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(desc_str, parents=[get_args_parser()])
    args = parser.parse_args()

    # Pretrain the model weights based on radiology reports
    pretrain_roberta_base(
        data_path=args.data_path,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        wandb_dir=args.wandb_dir,
        do_reporting=args.do_reporting == "True",
    )

