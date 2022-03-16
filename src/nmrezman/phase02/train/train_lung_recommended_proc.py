# %%

# nmrezman
from .general import train_lung_recommended_proc_model

# Misc
import argparse


# %%

desc_str = "Train Phase 02 Lung Recommended Procedure Model"

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
        "--model_pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained model checkpoint (\"/path/to/results/phase02/pretrain/checkpoint-XXXXX\")."
    )

    # Output file names
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save model checkpoints (e.g., \"/path/to/results/phase02/lung_recommend/\")."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        required=True,
        help="Path to save logging data (e.g., \"/path/to/results/phase02/lung_recommend/logging\")."
    )
    parser.add_argument(
        "--result_fname",
        type=str,
        required=True,
        help="Path / filename to save model evaluation metrics (e.g., \"/path/to/results/phase02/lung_recommend/lung_recommend_best_result.log\")."
    )
    parser.add_argument(
        "--wandb_dir",
        type=str,
        required=False,
        default="/path/to/results/phase02/lung_recommend/",
        help="Path to save wandb logs (e.g., \"/path/to/results/phase02/lung_recommend/\")."
    )

    # Misc
    parser.add_argument(
        "--do_reporting",
        type=str,
        default="True",
        help="Whether 🤗 will report to logs to supported integrations (True or False)"
    )

    return parser



if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser(desc_str, parents=[get_args_parser()])
    args = parser.parse_args()

    # Train the lung recommended procedure model
    train_lung_recommended_proc_model(
        data_path=args.data_path,
        model_pretrained_path=args.model_pretrained_path,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        result_fname=args.result_fname,
        wandb_dir=args.wandb_dir,
        do_reporting=args.do_reporting == "True",
    )

