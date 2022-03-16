# %%

# Models and tokenizers
import joblib
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import fasttext

import nltk
from xgboost import XGBClassifier
from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

# nmrezman
from ..models import get_bilstm_findings_classifier, get_bilstm_lung_adrenal_classifier, recommended_proc_model
from ...utils import generate_eval_report

# Misc
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

# Typing
from typing import Tuple, Union
import numpy.typing as npt
from keras.models import Sequential


# %%

def tokenize(
        x: pd.Series,
        max_num_words: int,
        max_sequence_length: int,
        tokenizer_fname: str=None,
        is_create: bool=True,
    ) -> Tuple[npt.NDArray, dict, int]:
    
    # Define the tokenizer
    # Lowercase the text; filter out special characters 
    if is_create:
        tokenizer = Tokenizer(num_words=max_num_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(x)
    else:
        tokenizer = joblib.load(tokenizer_fname)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)+1

    # Tokenize the notes
    # Prepend since radiology fidings are almost always located in the last section of the report
    x_tokenized = tokenizer.texts_to_sequences(x)
    x_tokenized = pad_sequences(x_tokenized, maxlen=max_sequence_length, padding="pre")

    # Save the tokenizer, which will be needed for classification and training other models
    if is_create:
        os.makedirs(os.path.dirname(tokenizer_fname), exist_ok=True)
        joblib.dump(tokenizer, tokenizer_fname)

    return x_tokenized, word_index, vocab_size


def train(
        x_tokenized: npt.NDArray,
        y: Union[list, npt.NDArray],
        model: Sequential,
        model_checkpoint_name: str="./output/best_model.h5",
        epochs: int=50,
        class_weight: dict=None,
        result_fname: str="./output/validation.log",
    ) -> None:
    """
    Train a Keras model
    """

    # Make dirs before wasting time running model
    os.makedirs(os.path.dirname(model_checkpoint_name), exist_ok=True)
    os.makedirs(os.path.dirname(result_fname), exist_ok=True)

    # Split the data into train and test
    train_x, test_x, train_y, test_y = train_test_split(x_tokenized, y, test_size=0.30, random_state=133278)

    # Clear the Keras backend, starting model training from scratch
    K.clear_session()

    # Train!
    batch_size = 100
    es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15,)
    mc = ModelCheckpoint(model_checkpoint_name, monitor="val_loss", mode="min", verbose=1, save_best_only=True,)
    model.fit(
        train_x,
        to_categorical(train_y),
        class_weight=class_weight,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, mc],
        verbose=1,
        validation_data=(test_x, to_categorical(test_y)),
    )

    # Load in the best model
    best_model = load_model(model_checkpoint_name)

    # Perform confusion matrix and save the results
    y_pred = best_model.predict(np.array(test_x))
    y_pred = np.argmax(y_pred, axis=1)
    generate_eval_report(test_y, y_pred, result_fname)

    return


# %% [markdown]
# # Train Findings vs No Findings Model

# %%

def train_findings_model(
        data_path: str,
        glove_embedding_path: str,
        model_checkpoint_name: str="findings_best_model.h5",
        result_fname: str="findings_best_result.log",
        tokenizer_fname: str="tokenizer.gz"
    ) -> None:
    """
    Trains the Findings vs No Findings Phase 01 BiLSTM model. 

    Args:
        data_path (`str`):
            Path to the dataframe file with the preprocessed impressions and labels in ``new_note`` and
            ``selected_finding`` columns, respectively
        glove_embedding_path (`str`):
            Path to the pre-downloaded GloVe Stanford pretrained word vectors ``glove.6B.300d`` as found at 
            https://nlp.stanford.edu/projects/glove/
        model_checkpoint_name (`str`):
            Path / filename to save model checkpoints
        result_fname (`str`):
            Path / filename to save model evaluation metrics
        tokenizer_fname (`str`):
            Path / filename to save tokenizer
    """

    # Import data
    # NOTE: this data has already been preprocessed, extracting the findings, removing Dr signature, etc.
    # See `from ..utils import preprocess_input`
    modeling_df = joblib.load(data_path)

    # Get preprocessed notes and labels as already indicated by the dataframe
    X = modeling_df["new_note"]
    labels = [0 if i == "No Findings" else 1 for i in modeling_df["selected_finding"]]

    # Define model constants
    max_sequence_length = 300       # Max length of report. Avg NM is ~250
    max_num_words = 15000           # Max number of words for init vocab; the actual vocab size used by the model will be different
    glove_embedding_dim = 300       # GloVe embedding dimension size

    # Tokenize the data
    X_tokenized, word_index, vocab_size = tokenize(
        x=X,
        max_num_words=max_num_words,
        max_sequence_length=max_sequence_length,
        tokenizer_fname=tokenizer_fname,
        is_create=True,
    )

    # Get GloVe embedding matrix
    # NOTE: Stanford pretrained word vectors glove.6B.300d were downloaded from https://nlp.stanford.edu/projects/glove/
    glove_embeddings_index = {}
    f = open(glove_embedding_path, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype="float32")
        except:
            pass
        glove_embeddings_index[word] = coefs
    f.close()

    glove_embedding_matrix = np.random.random((len(word_index) + 1, glove_embedding_dim))
    for word, i in word_index.items():
        glove_embedding_vector = glove_embeddings_index.get(word)
        if glove_embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                if len(glove_embedding_matrix[i]) != len(glove_embedding_vector):
                    print("could not broadcast input array from shape", str(len(glove_embedding_matrix[i])),
                        "into shape", str(len(glove_embedding_vector)), " Please make sure your"
                                                                    " EMBEDDING_DIM is equal to embedding_vector file ,GloVe,")
                    exit(1)
                glove_embedding_matrix[i] = glove_embedding_vector

    # Define the model
    model = get_bilstm_findings_classifier(
        max_sequence_length=max_sequence_length,
        max_num_words=vocab_size, 
        glove_embedding_dim=glove_embedding_dim,
        glove_embedding_matrix=glove_embedding_matrix,
    )

    # Train and evaluate
    train(
        x_tokenized=X_tokenized,
        y=labels,
        model=model,
        model_checkpoint_name=model_checkpoint_name,
        epochs=30, 
        result_fname=result_fname,
    )

    return


# %% [markdown]
# # Train Lung vs Adrenal Findings Model

# %%

def train_lung_adrenal_model(
        data_path: str,
        bioword_path: str,
        model_checkpoint_name: str="lung_adrenal_best_model.h5",
        result_fname: str="lung_adrenal_best_result.log",
        tokenizer_fname: str="tokenizer.gz"
    ) -> None:
    """
    Trains the Lung vs Adrenal Findings Phase 01 BiLSTM model. 

    Args:
        data_path (`str`):
            Path to the dataframe file with the preprocessed impressions and labels in ``new_note`` and
            ``selected_finding`` columns, respectively
        bioword_path (`str`):
            Path to the BioWordVec pretrained word vectors ``BioWordVec_PubMed_MIMICIII_d200.bin`` as from 
            https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
        model_checkpoint_name (`str`):
            Path / filename to save model checkpoints
        result_fname (`str`):
            Path / filename to save model evaluation metrics
        tokenizer_fname (`str`):
            Path / filename to save tokenizer
    """

    # Import data
    # NOTE: this data has already been preprocessed, extracting the findings, removing Dr signature, etc.
    # See `from ..utils import preprocess_input`
    modeling_df = joblib.load(data_path)

    # Get preprocessed notes and labels as already indicated by the dataframe
    X = modeling_df[modeling_df["selected_finding"]!="No Findings"]["new_note"]
    labels = modeling_df[modeling_df["selected_finding"]!="No Findings"]["selected_finding"]
    labels = [0 if i =="Lung Findings" else 1 for i in labels]

    # Define model constants 
    max_sequence_length = 300       # Max length of report. Avg NM is ~250
    max_num_words = 20000           # Max number of words for init vocab; the actual vocab size used by the model will be different
    bioword_embedding_dim = 200     # Bioword embedding dimension size

    # Tokenize the data
    X_tokenized, word_index, vocab_size = tokenize(
        x=X,
        max_num_words=max_num_words,
        max_sequence_length=max_sequence_length,
        tokenizer_fname=tokenizer_fname,
        is_create=False,
    )

    # Load the model for (bio) word embedding
    # NOTE: bioword word vector downloaded from: https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
    model = fasttext.load_model(bioword_path)

    # Prepare the word embedding matrix
    num_words = min(len(word_index) + 1, max_num_words)
    bioword_embedding_matrix = np.zeros((num_words, bioword_embedding_dim))
    for word, i in tqdm(word_index.items()):
        if i >= max_num_words:
            continue
        bioword_embedding_matrix[i] = model.get_word_vector(word)

    # Define the model
    model = get_bilstm_lung_adrenal_classifier(
        max_num_words=vocab_size, 
        bioword_embedding_dim=bioword_embedding_dim, 
        max_sequence_length=max_sequence_length, 
        bioword_embedding_matrix=bioword_embedding_matrix,
    )

    # Train and evaluate
    train(
        x_tokenized=X_tokenized,
        y=labels,
        model=model,
        model_checkpoint_name=model_checkpoint_name,
        epochs=50, 
        class_weight={0:1, 1:100},
        result_fname=result_fname,
    )

    return    


# %% [markdown]
# # Train Lung Recommended Procedure (Chest CT or Ambiguous) Model

# %%

def train_lung_recommended_proc_model(
        data_path: str,
        model_checkpoint_name: str="lung_recommend_best_model.h5",
        result_fname: str="lung_recommend_best_result.log",
        tokenizer_fname: str="tokenizer.gz",        
    ) -> None:
    """
    Trains the Lung Recommended Procedure Phase 01 BiLSTM model. Recommends "Chest CT" or "Ambiguous" procedure for 
    "Lung Findings". 

    Args:
        data_path (`str`):
            Path to the dataframe file with the preprocessed impressions and labels in ``new_note`` and
            ``selected_finding`` columns, respectively
        model_checkpoint_name (`str`):
            Path / filename to save model checkpoints
        result_fname (`str`):
            Path / filename to save model evaluation metrics
        tokenizer_fname (`str`):
            Path / filename to save tokenizer
    """

    # Import data
    # NOTE: this data has already been preprocessed, extracting the findings, removing Dr signature, etc.
    # See `from ..utils import preprocess_input`
    modeling_df = joblib.load(data_path)

    # Get the portion of the dataset that includes lung findings as already indicated by the dataframe
    X = modeling_df[modeling_df["selected_finding"]=="Lung Findings"]["new_note"]
    labels = modeling_df[modeling_df["selected_finding"]=="Lung Findings"]["selected_proc"]
    labels = [1 if i=="CT Chest" else 0 for i in labels]

    # Define model constants 
    max_sequence_length = 300       # Max length of report. Avg NM is ~250
    max_num_words = 20000           # Max number of words for init vocab; the actual vocab size used by the model will be different
    embedding_dim = 300             # embedding dimension size

    # Tokenize the data
    X_tokenized, word_index, vocab_size = tokenize(
        x=X,
        max_num_words=max_num_words,
        max_sequence_length=max_sequence_length,
        tokenizer_fname=tokenizer_fname,
        is_create=False,
    )

    # Define the model
    model = recommended_proc_model(
        max_num_words=max_num_words, 
        embedding_dim=embedding_dim, 
        input_length=np.array(X_tokenized).shape[1],
    )

    # Train and evaluate
    train(
        x_tokenized=np.array(X_tokenized),
        y=np.array(labels),
        model=model,
        model_checkpoint_name=model_checkpoint_name,
        epochs=50, 
        result_fname=result_fname,
    )

    return


# %% [markdown]
# # Train Comment Extraction Model

# %%

def train_comment_model(
        data_path: str,
        model_checkpoint_name: str="comment_best_model.sav",
        result_fname: str="comment_best_result.log",
    ) -> None:
    """
    Trains the Comment Extraction Phase 01 XGBoost model. 

    Args:
        data_path (`str`):
            Path to the dataframe file with the preprocessed impressions and labels in ``new_note`` and
            ``selected_finding`` columns, respectively
        model_checkpoint_name (`str`):
            Path / filename to save model checkpoints
        result_fname (`str`):
            Path / filename to save model evaluation metrics
    """

    # Import data
    # NOTE: this data has already been preprocessed, extracting the findings, removing Dr signature, etc.
    # See `from ..utils import preprocess_input`
    modeling_df = joblib.load(data_path)

    # Get the portion of the dataset that includes only findings
    only_findings_df = modeling_df[modeling_df["selected_finding"]!="No Findings"]

    # Split into train and test data
    train, hold_out = train_test_split(only_findings_df, test_size=0.2)

    # Tokenize into sentences. Model training is done on sentences vs the whole report
    nltk.download("punkt")
    main_row_sents = []
    sent_classifier=[]
    rpt_num=[]
    def get_sentence_classification_data(train):
        for idx, row in tqdm(train.iterrows()):
            row_sents = nltk.tokenize.sent_tokenize(row["note"])
            last_sentence_label = nltk.tokenize.sent_tokenize(row["selected_label"])[-1]
            for jdx, ele in enumerate(row_sents):
                if ele in last_sentence_label:
                    classifier = 1
                else:
                    classifier = 0
                sent_classifier.append(classifier)
                rpt_num.append(row["rpt_num"])
            main_row_sents.append(row_sents)
        return main_row_sents, sent_classifier, rpt_num

    main_row_sents, classifier, rpt_num = get_sentence_classification_data(train)

    # Create a dataframe sentence classifier
    flattened_sents = [i for sublist in main_row_sents for i in sublist]
    sent_class_df = pd.DataFrame()
    sent_class_df["sentence"] = flattened_sents
    sent_class_df["finding_sent"] = classifier
    sent_class_df["rpt_num"] = rpt_num

    # Get matrix of counts
    y = np.array(sent_class_df["finding_sent"])
    my_stop_words = ["the", "is", "are", "a" "there", "for", "in"]
    tvec = TfidfVectorizer(stop_words=my_stop_words, max_features=1000, ngram_range=(1,3))
    cvec = CountVectorizer(stop_words=my_stop_words, ngram_range=(1,4))

    # Define XGBoost classifier
    xgb = XGBClassifier(eval_metric="error", use_label_encoder=False)

    sent_class_df["X"] = sent_class_df["sentence"]

    # Get the XGBoost pipeline and train
    print("XgBoost results")
    print("+"*100)
    RUS_pipeline = make_pipeline(tvec, RandomUnderSampler(random_state=777), xgb)
    lr_cv(5, sent_class_df.X, sent_class_df.finding_sent, RUS_pipeline, "macro", model_checkpoint_name, result_fname)

    return


def lr_cv(
        splits: int, 
        X: pd.Series, 
        Y: pd.Series, 
        pipeline: Pipeline, 
        average_method: str, 
        model_checkpoint_name: str,
        result_fname: str,
    ) -> None:
    """
    Trains the comment extraction model

    Args:
        splits (`int`):
            Number of folds
        X (`pandas.Series`):
            Series of train and test X data
        Y (`pandas.Series`):
            Series of train and test Y data
        pipeline (`imblearn.pipeline.Pipeline`):
            Pipeline of transforms and resamples with a final estimator
        average_method (`str`):
            Type of averaging performed on the data
        model_checkpoint_name (`str`):
            Path / filename to save model checkpoints
        result_fname (`str`):
            Path / filename to save model evaluation metrics
    """
    
    # Train and evaluate!
    kfold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=777)
    accuracy = []
    precision = []
    recall = []
    finding_recall =[]
    no_finding_recall=[]
    finding_precision=[]
    no_finding_precision =[]
    f1 = []
    f1_all = []
    for train, test in kfold.split(X, Y):
        lr_fit = pipeline.fit(X[train], Y[train])
        prediction = lr_fit.predict(X[test])
        scores = lr_fit.score(X[test], Y[test])
        
        print("           no_finding    finding_present     ")
        accuracy.append(scores*100)
        
        p_score = precision_score(Y[test], prediction, average=None)
        print("precision:", p_score)
        precision.append(precision_score(Y[test], prediction, average=average_method)*100)
        finding_precision.append(p_score[1])
        no_finding_precision.append(p_score[0])
        
        r_score = recall_score(Y[test], prediction, average=None)
        print("recall:   ", r_score)
        recall.append(recall_score(Y[test], prediction, average=average_method)*100)
        finding_recall.append(r_score[1])
        no_finding_recall.append(r_score[0])

        f1.append(f1_score(Y[test], prediction, average=average_method)*100)
        f_score = f1_score(Y[test], prediction, average=None)
        f1_all.append(f_score)
        print("f1 score: ", f_score)
        print("-"*50)

    # Save the model
    os.makedirs(os.path.dirname(model_checkpoint_name), exist_ok=True)
    pickle.dump(pipeline, open(model_checkpoint_name, "wb"))
    # joblib.dump(pipeline, open(model_checkpoint_name, "wb"))

    # Print a summary of the results
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
    print("f1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
    print("Finding Recall:  %.2f%%" % (np.mean(finding_recall)*100))
    print("No Finding Recall: %.2f%%" % (np.mean(no_finding_recall)*100))
    print("Finding Precision: %.2f%%" % (np.mean(finding_precision)*100))
    print("No Finding Precision: %.2f%%" % (np.mean(no_finding_precision)*100))

    # Write results out to a file
    with open(result_fname, "w") as fh:
        fh.write("Classification Report:")
        fh.write("\n\taccuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracy), np.std(accuracy)))
        fh.write("\n\tprecision: %.2f%% (+/- %.2f%%)" % (np.mean(precision), np.std(precision)))
        fh.write("\n\trecall: %.2f%% (+/- %.2f%%)" % (np.mean(recall), np.std(recall)))
        fh.write("\n\tf1 score: %.2f%% (+/- %.2f%%)" % (np.mean(f1), np.std(f1)))
        fh.write("\n\tFinding Recall:  %.2f%%" % (np.mean(finding_recall)*100))
        fh.write("\n\tNo Finding Recall: %.2f%%" % (np.mean(no_finding_recall)*100))
        fh.write("\n\tFinding Precision: %.2f%%" % (np.mean(finding_precision)*100))
        fh.write("\n\tNo Finding Precision: %.2f%%" % (np.mean(no_finding_precision)*100))

        fh.write("\n\nAll Results:")
        for precision_no_finding, precision_finding, recall_no_finding, recall_finding, f1sc in zip(no_finding_precision, finding_precision, no_finding_recall, finding_recall, f1_all):
            fh.write("\n\t           no_finding    finding_present     ")
            fh.write(f"\n\tprecision: [{precision_no_finding:0.8f} {precision_finding:0.8f}]")
            fh.write(f"\n\trecall:    [{recall_no_finding:0.8f} {recall_finding:0.8f}]")
            fh.write(f"\n\tf1 score:  {f1sc}")
            fh.write("\n\t"+"-"*50)

    return


# %%
