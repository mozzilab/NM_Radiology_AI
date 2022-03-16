# %%

# Models and tokenizers
import pickle 
import joblib
import nltk
import numpy as np	
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import imblearn

# nmrezman
from ...utils import preprocess_input

# Misc
import os

# Typing
from typing import Dict
from imblearn.pipeline import Pipeline


# %%

def extract_comments(clean_note: str, comments_model: Pipeline) -> str:
    """
    For reports with findings, extract the related comment that led to the finding.

    Args:
        clean_note (`str`):
            Preprocessed note
        comments_model (`imblearn.pipeline.Pipeline`):
            Pretrained comments model
    """

    # Define the tokenizer
    nltk.download('punkt')
    nltk_token = nltk.tokenize

    # Get a list of all the tokenized sentences in the the clean note
    comments_X = nltk_token.sent_tokenize(clean_note)

    # Get the highest probability predicted comment
    comment_pred_probs = [comments_model.predict_proba([i.encode("utf-8")])[:, 1][0] for i in comments_X]
    idx = comment_pred_probs.index(max(comment_pred_probs))
    predicted_comment = comments_X[idx]

    return comments_X[idx-1] + predicted_comment


def classifier(
        data: str,
        model_path: str,
) -> Dict[str, object]:
    """
    Results Management Classifier using biLSTMs according to Phase 01 of the project.

    Args:
        data (`str`):
            Radiologist report
        model_path (`str`):
            Path to the folder with model checkpoints and tokenizer

            .. note::

                The model weights and tokenizer should be located in the specified folder as:
                    - ``findings_best_model.h5``
                    - ``comment_best_model.sav``
                    - ``lung_adrenal_best_model.h5``
                    - ``lung_recommend_best_model.h5``
                    - ``tokenizer.gz``
                for the (i) Findings vs No Finding Model, (ii) Lung vs Adrenal Findings Model, (iii) Comment Extraction 
                Model, (iv) Lung Recommended Procedure model, and (v) tokenizer, respectively.

    Returns:
        A dictionary which includes the (1) recommended procedure, (2) nodule type (if found), (3) boolean indicating
        if a follow-up is required, and (4) the follow-up text (i.e., text of the report that indicates the finding) 
        as stored / referenced by the dictionary keys "procedure", "noduleType", "followUpFlag", "followUpText", 
        respectively

    Example::

        >>> report_txt = "a string with the radiology report text"
        >>> model_path = "/path/to/checkpoints/phase01/"
        >>> output = classifier(report_txt, model_path)
        >>> print("Output:")
        >>> [print(f"  {key}:", value) for key, value in output.items()]
        ... Output:
        ...   procedure: Chest CT
        ...   noduleType: Lung
        ...   followUpFlag: Findings Present
        ...   followUpText: several pulmonary micronodules. follow-up in one year recommended.
    """  

    # Define tokenizers
    findings_tokenizer = joblib.load(os.path.join(model_path, "tokenizer.gz"))

    # Load models
    #   - findings_model:               model detecting if there is a finding or no finding (`findings_model_dict`)
    #   - lung_adrenal_model:           model detecting if there is a lung or adrenal finding for reports with findings (`lung_adrenal_dict`)
    #   - comments_model:               extract the relevant portion of report that mentions the finding
    #   - lung_recommended_proc:        model to determine the recommended procedure for lung findings (`lung_recommended_proc_dict`)
    # NOTE: Depending on your application, the models can be loaded differently / elsewhere (e.g., as globals that are always loaded)
    findings_model = load_model(os.path.join(model_path, "findings_best_model.h5"), compile=True)
    comments_model = pickle.load(open(os.path.join(model_path, "comment_best_model.sav"), "rb"))
    lung_adrenal_model = load_model(os.path.join(model_path, "lung_adrenal_best_model.h5"), compile=True)
    lung_recommended_proc = load_model(os.path.join(model_path, "lung_recommend_best_model.h5"), compile=True)

    # Define label embeddings / enummerations for the three models
    # Findings vs no findings model output
    findings_model_dict = {
        0: "No Findings Present",
        1: "Findings Present",
    }
    # Lung vs adrenal findings model output
    lung_adrenal_dict = {
        0: "Lung",
        1: "Adrenal",
    }
    # Lung recommended procedure model output
    lung_recommended_proc_dict = {
        0: "Ambiguous",
        1: "Chest CT",
    }

    # Preprocess the note, getting the impression, removing doctor signatures, etc.
    clean_findings_imp = preprocess_input(data, is_phase_2=False)
    
    # Tokenize the impression
    X = findings_tokenizer.texts_to_sequences([clean_findings_imp])
    X = pad_sequences(X, maxlen=300, padding="pre")

    # Classify if there is a finding or not and get as layman text
    y_pred = findings_model.predict(X)
    y_pred_class = np.argmax(y_pred, axis=1)
    follow_up_flg = findings_model_dict[y_pred_class[0]]

    # Based on the finding, run through more model(s)
    if follow_up_flg == "Findings Present":
        # Finding detected; follow-up recommended
        
        # Classify if the finding the lung or adrenal
        lung_adrenal_pred = lung_adrenal_model.predict(X)
        lung_or_adrenal_finding = lung_adrenal_dict[np.argmax(lung_adrenal_pred)]
        
        # Get follow-up text
        final_comment = extract_comments(clean_findings_imp, comments_model)		

        # Based on the finding type, get the procedure recommendation
        if lung_or_adrenal_finding == "Lung":
            # Lung finding run through another model to determine recommended procedure
            lung_recommended_proc_pred = lung_recommended_proc.predict(X)
            lung_adrenal_recommended_proc_label = lung_recommended_proc_dict[np.argmax(lung_recommended_proc_pred)]
        else:
            # Only one recommendation for adrenal findings
            lung_adrenal_recommended_proc_label = "Endocrinology Referral"
            
    else:
        # No finding detected; no follow-up recommended, etc.
        final_comment = "NA"
        lung_or_adrenal_finding = "NA"
        lung_adrenal_recommended_proc_label = "NA"


    # Output includes classification and recommendations	
    output = {
        "procedure": lung_adrenal_recommended_proc_label,	# Recommended procedure. For "Lung" finding, "Ambiguous" or "Chest CT". For "Adrenal", "Endocrinology Referral"
        "noduleType": lung_or_adrenal_finding,				# Options defined in `finding_model_dict`
        "followUpFlag": follow_up_flg,						# "Boolean" for findings: "Findings Present" or "NA"
        "followUpText": final_comment,						# Follow-up text (i.e., text that indicates finding)
    }

    return output


