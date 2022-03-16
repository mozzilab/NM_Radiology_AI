# %%

# Models
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam

# Typing
from typing import Any
import numpy.typing as npt


# %% 

def get_bilstm_findings_classifier(
        glove_embedding_matrix: npt.NDArray[Any], 
        max_sequence_length: int, 
        max_num_words: int=13281, 
        glove_embedding_dim: int=300,
    ) -> Sequential:
    """
    Phase 01 BiLSTM Finding vs No Findings model
    
    Args:
        glove_embedding_matrix ():
            GloVe embedding matrix
        max_sequence_length (`int`):
            Max length of a sequence
        max_num_words (`int`):
            Max number of words in vocab
        glove_embedding_dim (`int`):
            GloVe embedding dimension size

    Returns:
        Finding vs No Findings model
    """
    model = Sequential()
    model.add(Embedding(max_num_words,
                        glove_embedding_dim,
                        weights=[glove_embedding_matrix],
                        input_length=max_sequence_length,
                        trainable=False),
    )
    model.add(SpatialDropout1D(0.25))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Bidirectional(LSTM(200, return_sequences=True)))
    model.add(Bidirectional(LSTM(200)))
    model.add(Dropout(0.1))
    model.add(Dense(12))
    model.add(Dense(units=2, activation="softmax"))
    adam = Adam(learning_rate=0.0011)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    return model


def get_bilstm_lung_adrenal_classifier(
        bioword_embedding_matrix: npt.NDArray[Any],  
        max_sequence_length: int, 
        max_num_words: int=13281, 
        bioword_embedding_dim: int=200,
    ) -> Sequential:   
    """
    Phase 01 BiLSTM Lung vs Adrenal Findings model
    
    Args:
        bioword_embedding_matrix ():
            BioWord embedding matrix
        max_sequence_length (`int`):
            Max length of a sequence
        max_num_words (`int`):
            Max number of words in vocab
        bioword_embedding_dim (`int`):
            BioWord embedding dimension size

    Returns:
        Lung vs Adrenal Findings model
    """
    model = Sequential()
    model.add(Embedding(max_num_words,
                    bioword_embedding_dim,
                    weights=[bioword_embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
    model.add(Bidirectional(LSTM(50, return_sequences=True)))
    model.add(Bidirectional(LSTM(50)))  
    model.add(Dense(1000, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))
    adam = Adam(learning_rate=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["categorical_accuracy"])
    return model


def recommended_proc_model(
        max_num_words: int, 
        embedding_dim: int, 
        input_length: int,
    ) -> Sequential:
    """
    Phase 01 BiLSTM Lung Recommended Procedure model
    
    Args:
        max_num_words (`int`):
            Max number of words in vocab
        embedding_dim (`int`):
            Embedding dimension size
        input_length (`int`):
            Length of the input data

    Returns:
        Lung Recommended Procedure model
    """
    model = Sequential()
    model.add(Embedding(max_num_words, embedding_dim, input_length=input_length))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(100, return_sequences=True)))
    model.add(Bidirectional(LSTM(50)))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
    return model


# %%
