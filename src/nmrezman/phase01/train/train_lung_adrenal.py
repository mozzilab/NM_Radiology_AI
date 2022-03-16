# %%

# nmrezman
from .general import train_lung_adrenal_model

# Misc
import argparse


# %%

desc_str = "Train Phase 01 Lung vs Adrenal Findings Model"

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
        "--bioword_path",
        type=str,
        required=True,
        help="Path to BioWord word vector BioWordVec_PubMed_MIMICIII_d200.bin (e.g., \"/path/to/data/BioWordVec_PubMed_MIMICIII_d200.bin\")."
    )

    # Output file names
    parser.add_argument(
        "--model_checkpoint_name",
        type=str,
        required=True,
        help="Path / filename to save model checkpoints (e.g., \"/path/to/results/phase01/lung_adrenal/lung_adrenal_best_model.h5\")."
    )
    parser.add_argument(
        "--result_fname",
        type=str,
        required=True,
        help="Path / filename to save model evaluation metrics (e.g., \"/path/to/results/phase01/lung_adrenal/lung_adrenal_best_result.log\")."
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

    # Train the lung vs adrenal findings model
    train_lung_adrenal_model(
        data_path=args.data_path,
        bioword_path=args.bioword_path,
        model_checkpoint_name=args.model_checkpoint_name,
        result_fname=args.result_fname,
        tokenizer_fname=args.tokenizer_fname,
    )

