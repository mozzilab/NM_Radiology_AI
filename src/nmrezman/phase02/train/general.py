# %%

# Models and tokenizers
import joblib
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM
from simpletransformers.question_answering import QuestionAnsweringModel
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
from datasets import Dataset, DatasetDict

# nmrezman
from ...utils import generate_eval_report, get_impression, remove_drtag

# Misc
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import wandb


# %% [markdown]
# # Common classes and functions

# %%

class Reports_Dataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset which returns the tokenized report text and label.

    Args:
        encodings (`dict`):
        labels (`list`):
            List of integer (enumerated) labels
    """
    def __init__(self, encodings: dict, labels: list) -> None:
        self.encodings = encodings
        self.labels = labels
        return

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


def preprocess_note(report_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Preprocess the radiolgy report to get the impressions section

    Args:
        report_df (`pd.core.frame.DataFrame`):
            Dataframe with column ``note`` with the original radiology report

    Returns:
        Dataframe with a new column ``impression`` that contains the impression section, stripped of the doctor tag
    """

    # For each radiology report ("note"), get the impressions section and add it to the dataframe
    # This could be more NM-specific if radiologists annotate differently across hospital networks and may need to be modified to your needs
    report_df["impression"] = report_df["note"].apply(get_impression)

    # Remove the "dr" tag
    # This could be more NM-specific if radiologists annotate differently across hospital networks and may need to be modified to your needs
    report_df["impression"] = report_df["impression"].apply(remove_drtag)

    return report_df


# %% [markdown]
# # Pretrain RoBERTa base model

# %%

def group_texts(examples):
    # Sample chunked into size `block_size`
    block_size = 128

    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # We drop the small remainder. We could add padding if the model supported it rather than dropping it
    # This represents the maximum length based on the block size
    # You can customize this part to your needs
    max_length = (total_length // block_size) * block_size
    result = {k: [t[i : i + block_size] for i in range(0, max_length, block_size)] for k, t in concatenated_examples.items()}
    result["labels"] = result["input_ids"].copy()

    return result


def pretrain_roberta_base(
        data_path: str,
        output_dir: str,
        logging_dir: str,
        do_reporting: bool=True,
        wandb_dir: str=None,
    ) -> None:
    """
    Pretrain the model based on custom dataset

    Args:
        data_path (`str`):
            Path to the dataframe file with the reports and labels
        output_dir (`str`):
            Path to save model checkpoints
        logging_dir (`str`):
            Path to save 🤗 logging data
        do_reporting (`bool`):
            Boolean to determine whether 🤗 will report to logs to all (True) or no (False) supported integrations
        wandb_dir (`bool`):
            Path to save the wandb logging directory
    """

    # Read in the raw data
    modeling_df = joblib.load(data_path)

    # Do preprocessing to get reports ready for training
    # Extract the impression and clean up doctor tags, etc.
    modeling_df = preprocess_note(modeling_df)
    modeling_df = modeling_df[modeling_df["impression"].notnull()]
    modeling_df["impression"] = modeling_df["impression"].apply(lambda x: str(x.encode('utf-8')) +"\n"+"\n")

    # Split into train and test data
    train, test = train_test_split(modeling_df, test_size=0.2, random_state=7867)
    train = train.reset_index()
    test = test.reset_index()

    # Import the data into a dataset
    train_dataset = Dataset.from_pandas(train["impression"].to_frame())
    test_dataset = Dataset.from_pandas(test["impression"].to_frame())
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Tokenize the entire dataset
    # Disable the wanring for forking process as a result of using the tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained(
        "distilroberta-base",
        use_fast=True,
        padding_side="left",
    )
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["impression"], truncation=True, padding=True), 
        batched=True, 
        num_proc=1, 
        remove_columns=["impression"],
    )  

    # Group the text into chunks to get "sentence-like" data structure
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=1,
    )

    # Define a data collator to accomplish random masking
    # By doing this step in the `data_collator` (vs as a pre-processing step like we do for tokenization),
    # we ensure random masking is done in a new way each time we go over the data (i.e., per epoch)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # Define the model
    # You can try any type of Roberta models here: roberta-base, roberta-large
    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

    # Initialize wandb directory
    if do_reporting:
        os.makedirs(os.path.dirname(wandb_dir), exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project="findings",
        )
        report_to = "all"
    else:
        report_to = "none"

    # Define the training parameters and 🤗 Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_device_train_batch_size=32,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="epoch",
        seed=1,
        logging_dir=logging_dir,
        report_to=report_to,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    # Train!
    trainer.train()

    return


# %% [markdown]
# # Train Lung, Adrenal, or No Findings Model

# %%

def train_findings_model(
        data_path: str,
        model_pretrained_path: str,
        output_dir: str,
        logging_dir: str,
        result_fname: str,
        do_reporting: bool=True,
        wandb_dir: str=None,
    ) -> None:
    """
    Trains the Phase 02 Lung, Adrenal, or No Findings Model. 

    Args:
        data_path (`str`):
            Path to the dataframe file with the reports and labels
        model_pretrained_path (`str`):
            Path / filename to pretrained model checkpoint
        output_dir (`str`):
            Path to save model checkpoints
        logging_dir (`str`):
            Path to save 🤗 logging data
        result_fname (`str`):
            Path / filename to save model evaluation metrics
        do_reporting (`bool`):
            Boolean to determine whether 🤗 will report to logs to all (True) or no (False) supported integrations
        wandb_dir (`bool`):
            Path to save the wandb logging directory
    """   

    # Read in the raw data
    modeling_df = joblib.load(data_path)

    # Do preprocessing to get reports ready for training
    # Extract the impression and clean up doctor tags, etc.
    modeling_df = preprocess_note(modeling_df)

    # Encode the Lung, Adrenal, and No Finding into integer labels
    le = preprocessing.LabelEncoder()
    le.fit(modeling_df["selected_finding"])
    modeling_df["int_labels"] = le.transform(modeling_df["selected_finding"])

    # Split the data into train and test
    train_df, test_df = train_test_split(modeling_df, test_size=0.30, stratify=modeling_df["selected_finding"], random_state=133278)
    train_note = list(train_df["impression"])
    train_label = list(train_df["int_labels"])
    test_note = list(test_df["impression"])
    test_label = list(test_df["int_labels"])

    # Define the tokenizer (from a pre-trained checkpoint) and tokenize the notes
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, padding_side="left")
    train_encodings = tokenizer(train_note, truncation=True, padding=True)
    val_encodings = tokenizer(test_note, truncation=True, padding=True)   

    # Define the training dataset with tokenized notes and labels
    train_dataset = Reports_Dataset(train_encodings, train_label)
    test_dataset = Reports_Dataset(val_encodings, test_label)

    # Load the pretrained checkpoint that will be fine-tuned for the 3-label classification task
    model = AutoModelForSequenceClassification.from_pretrained(model_pretrained_path, num_labels=3)

    # Initialize wandb directory
    if do_reporting:
        os.makedirs(os.path.dirname(wandb_dir), exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project="findings",
        )
        report_to = "all"
    else:
        report_to = "none"

    # Define the training parameters and 🤗 Trainer
    training_args = TrainingArguments(
                        output_dir=output_dir,
                        num_train_epochs=3,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=8,
                        warmup_steps=100,
                        weight_decay=0.015,
                        logging_dir=logging_dir,
                        fp16=True,
                        logging_steps=100,
                        load_best_model_at_end=True,
                        evaluation_strategy="epoch",
                        do_predict=True,
                        save_strategy="epoch",
                        report_to=report_to,
    )
    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=test_dataset,
    )

    # Train!
    trainer.train()

    # Perform confusion matrix and save the results
    y_pred = trainer.predict(test_dataset)
    y_pred = np.argmax(y_pred.predictions, axis=1)
    generate_eval_report(test_label, y_pred, result_fname)

    return


# %% [markdown]
# # Train Comment Extraction Model

# %%

def find_all(input_str: str, search_str: str) -> list:
    """
    Find the index(s) of where `input_str` appears in `search_str`

    Args:
        input_str (`str`):
            Text to search for
        search_str (`str`):
            Text to search within

    Returns:
        List of start indexes where `input_str` starts in `search_str`
    """
    # Adpated from https://www.kaggle.com/cheongwoongkang/roberta-baseline-starter-simple-postprocessing
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = search_str.find(input_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def do_qa_format(data: pd.core.frame.DataFrame, data_format: str="train") -> list:
    """
    Finds all QAs for given context

    Args:
        data (`pd.core.frame.DataFrame`):
            Dataframe with columns "context", "question". "answer", and "report_id"
        data_format (`str`):
            String (options: "train", "eval", "predict") indicating how the data should be formatted

    Returns:
        List of dictionaries that define the context and question-answers
    """
    # Adpated from https://www.kaggle.com/cheongwoongkang/roberta-baseline-starter-simple-postprocessing

    if data_format not in ["train", "eval", "predict"]:
        raise ValueError("`data_format` is not 'train', 'eval', 'predict'")

    # Define column names here
    context_col_name = "context"
    question_col_name = "question"
    answer_col_name = "answer"
    qid_col_name = "index"

    output = []
    for idx, line in data.iterrows():
        context = line[context_col_name]
        qas = []
        question = line[question_col_name]
        qid = line[qid_col_name]
        if data_format != "predict":
            answers = []
            answer = line[answer_col_name] if data_format=="train" else "__None__"
            if type(answer) != str or type(context) != str or type(question) != str:
                print(context, type(context))
                print(answer, type(answer))
                print(question, type(question))
                continue
            if data_format == "train":
                # Find all indexes which answer appears in context
                # For each start index, generate a dictionary with start and end indexes and answer
                answer_starts = find_all(answer, context)
                for answer_start in answer_starts:
                    answers.append({
                        "answer_start": answer_start,               # Start index
                        "answer_end":answer_start+len(answer),      # End index
                        "text": answer.lower()},                    # Answer (lowercase)
                    )
                    break
            else:
                # "Answer" for evaluation data
                answers = [{"answer_start": 1000000, "text": "__None__"}]

            # For each context, define the QAs 
            qas.append({
                "question": question,                       # Question
                "id": qid,                                  # Row number
                "is_impossible": False,                     # Placeholder
                "answers": answers,                         # Answers (list of dicitonaries)
            })
        else:
            # For each context, define the question 
            qas.append({
                "question": question,                       # Question
                "id": qid,                                  # Row number
            })

        output.append({
            "context": context.lower(),                     # Context
            "qas": qas,                                     # All QAs for context (list of dictionaries)
        })
        
    return output


def jaccard_similarity(doc1: str, doc2: str) -> float:
    """
    Calculate the Jaccard Similarity Score for two texts

    Args:
        doc1 (`str`):
            The input text
        doc2 (`str`):
            The output text

    Returns:
        The Jaccard Similarity Score
    """

    # List the unique words in a document
    words_doc1 = set(str(doc1).lower().split())
    words_doc2 = set(str(doc2).lower().split())

    # Find the intersection of words list of doc1 & doc2
    intersection = words_doc1.intersection(words_doc2)

    # Find the union of words list of doc1 & doc2
    union = words_doc1.union(words_doc2)

    # Calculate Jaccard similarity score
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)


def train_comment_model(
        data_path: str,
        output_dir: str="comment_model",
        result_fname_prefix: str="results",
    ) -> None:
    """
    Trains the Comment Extraction Hhase 02 MLM model. 

    Args:
        data_path (`str`):
            Path to the dataframe file with the reports and labels
        output_dir (`str`):
            Path to save training and evaluation logging and results. Model checkpoints are saved in 
                <output_dir_str>/output_dir. Evaluation results are in `output_dir`
        result_fname_prefix (`str`):
            Result file name prefix to save *.csv and *.json in `output_dir`
    """

    # Read in the raw data
    modeling_df = joblib.load(data_path)

    # Can only train on data with lung or adrenal findings
    modeling_df = modeling_df[modeling_df["selected_label"]!="No label"]

    # Do preprocessing to get reports ready for training
    # Extract the impression and clean up doctor tags, etc.
    modeling_df = preprocess_note(modeling_df)

    # Get only the columns necessary from the dataframe and rename them
    # NOTE: column names:
    #   - selected_finding:     Lung or Adrenal Findings (No Findings not part of this dataset)     -> Question
    #   - impression:           Impression section of report                                        -> Context
    #   - selected_label:       "Comment" section                                                   -> Answer
    modeling_df = modeling_df[["selected_finding", "impression", "selected_label"]]
    modeling_df.columns = ["question", "context", "answer"]

    # Drop the samples that do not have a comment
    input_text = [j.context if j.answer in j.context else \
                 "" if j.answer == "No label" else \
                 "DROP" for _, j in tqdm(modeling_df.iterrows())]
    modeling_df["input_text"] = input_text
    modeling_df = modeling_df[modeling_df["input_text"] != "DROP"]
    modeling_df = modeling_df.reset_index()
    modeling_df = modeling_df[["index", "context", "answer", "question"]]

    # Split the data into train and test
    train_df, test_df = train_test_split(modeling_df, test_size=0.30, stratify=modeling_df["question"], random_state=133278)

    # Define training arguments for `simple_transformers` training
    args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "learning_rate": 5e-5,
        "num_train_epochs": 3,
        "max_seq_length": 192,
        "doc_stride": 64,
        "fp16": False,
        "n_best_size": 2,
        "output_dir": os.path.join(output_dir, "output_dir"),
        "cache_dir": os.path.join(output_dir, "cache_dir"),
        "tensorboard_dir": os.path.join(output_dir, "tensorboard_dir"),
    }

    # Define the `simple_transformers` model
    model = QuestionAnsweringModel(
                "roberta",
                "distilroberta-base",
                args=args,
                use_cuda=torch.cuda.is_available(),
    )

    # Prepare the data in Question-Answer compatible format
    qa_train = do_qa_format(train_df, data_format="train")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as outfile:
        json.dump(qa_train, outfile)
    qa_test = do_qa_format(test_df, data_format="predict")
    with open(os.path.join(output_dir, "test.json"), "w") as outfile:
        json.dump(qa_test, outfile)

    # Train!
    model.train_model(qa_train)

    # Compute predictions
    # Note that `.predict` will change `qa_test`
    # This line still works with `data_format`=`train`, but trying to be "correct" here...
    predictions = model.predict(qa_test)
    predictions_df = pd.DataFrame.from_dict(predictions)

    # Format the results output, compute jaccard score, and output to file
    test_df = test_df.reset_index(drop=True)
    test_df["predicted_answer"] = np.nan
    test_df["jaccard_score"] = np.nan
    for _, j in predictions_df.T.iterrows():
        idx = j[0]["id"]
        df_idx = test_df.index[test_df['index']==idx][0]
        predicted_answer = j[0]["answer"][0]
        true_answer = test_df.iloc[df_idx]["answer"]
        jaccard_score = jaccard_similarity(true_answer, predicted_answer)
        test_df.loc[test_df["index"]==idx, "predicted_answer"] = predicted_answer
        test_df.loc[test_df["index"]==idx, "jaccard_score"] = jaccard_score
    test_df.to_csv(os.path.join(output_dir, result_fname_prefix + ".csv"))

    return


# %% [markdown]
# # Train Lung Recommended Procedure Model

# %%

def train_lung_recommended_proc_model(
        data_path: str,
        model_pretrained_path: str,
        output_dir: str,
        logging_dir: str,
        result_fname: str,
        do_reporting: bool=True,
        wandb_dir: str=None,
    ) -> None: 
    """
    Trains the Lung Recommended Procedure Phase 02 MLM model. Recommends "Chest CT" or "Ambiguous" procedure for 
    "Lung Findings". 

    Args:
        data_path (`str`):
            Path to the dataframe file with the reports and labels
        model_pretrained_path (`str`):
            Path / filename to pretrained model checkpoint
        output_dir (`str`):
            Path to save model checkpoints
        logging_dir (`str`):
            Path to save 🤗 logging data
        result_fname (`str`):
            Path / filename to save model evaluation metrics
        do_reporting (`bool`):
            Boolean to determine whether 🤗 will report to logs to all (True) or no (False) supported integrations
        wandb_dir (`bool`):
            Path to save the wandb logging directory
    """   

    # Read in the raw data
    modeling_df = joblib.load(data_path)

    # Do preprocessing to get reports ready for training
    # Extract the impression and clean up doctor tags, etc.
    modeling_df = preprocess_note(modeling_df)

    # Get the portion of the dataset that includes lung findings
    modeling_df = modeling_df[modeling_df["selected_finding"]=="Lung Findings"]
    modeling_df["int_labels"] = [1 if "CT Chest" in x else 0 for x in modeling_df["selected_proc"]]

    # Split the data into train and test and get notes and labels
    train_df, test_df = train_test_split(modeling_df, test_size=0.30, stratify=modeling_df["int_labels"], random_state=133278)
    train_note = list(train_df["impression"])
    train_label = list(train_df["int_labels"])
    test_note = list(test_df["impression"])
    test_label = list(test_df["int_labels"])

    # Define the tokenizer (from a pre-trained checkpoint) and tokenize the notes
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True, padding_side="left")
    train_encodings = tokenizer(train_note, truncation=True, padding=True)
    val_encodings = tokenizer(test_note, truncation=True, padding=True)   

    # Define the training / test datasets with tokenized notes and labels
    train_dataset = Reports_Dataset(train_encodings, train_label)
    test_dataset = Reports_Dataset(val_encodings, test_label)

    # Fine-tune the model from the pre-trained checkpoint
    model = AutoModelForSequenceClassification.from_pretrained(model_pretrained_path, num_labels=2)

    # Initialize wandb directory
    if do_reporting:
        os.makedirs(os.path.dirname(wandb_dir), exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project="lung_recommended_proc",
        )
        report_to = "all"
    else:
        report_to = "none"

    # Define the training parameters and 🤗 Trainer
    training_args = TrainingArguments(
                        output_dir=output_dir,
                        num_train_epochs=3,
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=8,
                        warmup_steps=100,
                        weight_decay=0.015,
                        logging_dir=logging_dir,
                        fp16=True,
                        logging_steps=100,
                        load_best_model_at_end=True,
                        evaluation_strategy="epoch",
                        do_predict=True,
                        save_strategy="epoch",
                        report_to=report_to,
    )
    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=test_dataset,
    )

    # Train!
    trainer.train()

    # Perform confusion matrix and save the results
    y_pred = trainer.predict(test_dataset)
    y_pred = np.argmax(y_pred.predictions, axis=1)
    generate_eval_report(test_label, y_pred, result_fname)

    return


