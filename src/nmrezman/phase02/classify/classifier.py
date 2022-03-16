# %%

# Models
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification
import numpy as np
import torch

# nmrezman
from ...utils import preprocess_input

# Misc
import os

# Type checking
from typing import Dict
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast


# %%

def run_comment_qa_model(
        question: str, 
        context_text: str, 
        model: RobertaPreTrainedModel, 
        tokenizer: RobertaTokenizerFast,
    ) -> str:
    """
    This function runs an extractive Question-Answering model that takes an input of question as predicted nodule finding 
    and context as the clean note. In short, for reports with a finding classified, finds the text in the report that 
    helped make that determination.

    Args:
        question (`str`): 
            Input "question". The model is only trained for two questions: "LUNG FINDINGS" and "ADRENAL FINDINGS"
        context_text (`str`): 
            The clean, preprocessed radiology note
        model (:py:class:`~transformers.models.roberta.modeling_roberta.RobertaPreTrainedModel`):
            The pretrained Question-Answering model
        tokenizer (:py:class:`~transformers.models.roberta.tokenization_roberta_fast.RobertaTokenizerFast`):
            The tokenizer used with the model

    Returns:
        Text span (extractive) from ``context_text`` (i.e., radiology note) that relates to lung findings (in other words,
        the report text related to the finding)

    Example::
        
        >>> from nmrezman.utils import preprocess_input
        >>> report = preprocess_input("Radiologist report text")
        >>> model = AutoModelForQuestionAnswering.from_pretrained("/path/to/checkpoints/phase02/comment_model")
        >>> tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True,)
        >>> comment = run_comment_qa_model("LUNG FINDINGS", report, model, tokenizer)
    """
    
    try:
        # Format inputs
        inputs = tokenizer.encode_plus(
            question, 
            context_text, 
            add_special_tokens=True, 
            return_tensors="pt", 
            truncation=True,
            max_length=512,
        )
        input_ids= inputs["input_ids"].tolist()[0]
        
        # Get the most likely beginning and end of answer with the argmax of the score
        answer_start_scores, answer_end_scores = model(**inputs)[:2]
        answer_start = int(torch.argmax(answer_start_scores))
        answer_end = int(torch.argmax(answer_end_scores)) + 1
    
        # Get string from tokenized representation for likely range
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        if len(answer) < 2:
            # The QA model was unable to predict the sentence
            # For robustness, return the input text
            return context_text
        else:
            return answer
    
    except:
        return context_text
    

def classifier(
        data: str,
        model_path: str,
) -> Dict[str, object]:
    """
    Results Management Classifier using masked language models (MLM) according to Phase 02 of the project.

    Args:
        data (`str`):
            Radiologist report text
        model_path (`str`):
            Path to the folder containing the model checkpoints' folders

            .. note::

                The model weights should be in folders named as:
                    - ``findings_model``
                    - ``comment_model``
                    - ``lung_recommended_proc_model``
                for the (i) Lung, Adrenal, or No Findings Model, (ii) Comment Extraction Model, and (iii) Lung
                Recommended Procedure model, respectively.

    Returns:
        A dictionary which includes the (1) recommended procedure, (2) nodule type (if found), (3) boolean indicating
        if a follow-up is required, and (4) the follow-up text (i.e., text of the report that indicates the finding) 
        as stored / referenced by the dictionary keys "procedure", "noduleType", "followUpFlag", "followUpText", 
        respectively

    Example::

        >>> report_txt = "a string with the radiology report text"
        >>> model_path = "/path/to/checkpoints/phase02/"
        >>> output = classifier(report_txt, model_path)
        >>> print("Output:")
        >>> [print(f"  {key}:", value) for key, value in output.items()]
        ... Output:
        ...   procedure: Chest CT
        ...   noduleType: Lung
        ...   followUpFlag: Findings Present
        ...   followUpText: several pulmonary micronodules. follow-up in one year recommended. 
        
    """

    # Define roberta tokenizers
    roberta_tok = AutoTokenizer.from_pretrained(
        "roberta-base", 
        use_fast=True, 
        padding_side="left", 
        truncation=True, 
        model_max_len=512,
    )
    distilroberta_tok = AutoTokenizer.from_pretrained(
        "distilroberta-base", 
        use_fast=True,
    )
    
    # Load models
    #   - findings_model:                   model detecting if there is a lung, adrenal, or no finding (`findings_model_dict`)
    #   - comment_model:                    extract the relevant portion of report that mentions the finding (extractive QA model, `comment_model_dict`) 
    #   - lung_recommended_proc_model:      model to determine the recommended procedure for lung findings (`lung_recommended_proc_dict`)
    # NOTE: Depending on your application, the models can be loaded differently / elsewhere (e.g., as globals that are always loaded)
    findings_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "findings_model"))
    comment_model = AutoModelForQuestionAnswering.from_pretrained(os.path.join(model_path, "comment_model"))
    lung_recommended_proc_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "lung_recommended_proc_model"))

    # Define label embeddings / enummerations for the three models
    # Language findings model output
    findings_model_dict = {
        0: "Adrenal", 
        1: "Lung", 
        2: "NA",
    }
    # Comment extraction (QA model) model input
    comment_model_dict = {
        0: "ADRENAL FINDINGS", 
        1: "LUNG FINDINGS",
    }
    # Lung recommended procedure model output
    lung_recommended_proc_dict = {
        0: "Ambiguous",
        1: "Chest CT", 
    }
    
    # Preprocess the note, getting the impression, removing doctor signatures, etc.
    clean_input_text = preprocess_input(data)

    # Tokenize the impression
    roberta_inputs = roberta_tok(
        clean_input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True, 
        padding=True,
    )

    # Classify the impression and get it as final output classification text
    roberta_output = findings_model(**roberta_inputs)[0]
    roberta_output = np.argmax(roberta_output.detach().numpy(), axis=-1)[0] # could be replaced with int(torch.argmax(roberta_output.detach(), axis=-1)[0]) but pytorch in past has funny behaviors (e.g., https://github.com/pytorch/pytorch/issues/22853)
    lung_or_adrenal_finding = findings_model_dict[roberta_output]

    # Based on the finding, run through more model(s)
    if lung_or_adrenal_finding != "NA":
        # Finding detected; follow-up recommended
        # Determine text in report that indicates finding
        follow_up_flg = "Findings Present"
        final_comment = run_comment_qa_model(comment_model_dict[roberta_output], clean_input_text, comment_model, distilroberta_tok)

        # Based on the finding type, get the procedure recommendation
        if lung_or_adrenal_finding == "Lung":
            # Lung finding run through another model to determine recommended procedure
            roberta_proc_output = lung_recommended_proc_model(**roberta_inputs)[0]
            lung_recommended_class = np.argmax(roberta_proc_output.detach().numpy(), axis=-1)[0]
            lung_adrenal_recommended_proc_label = lung_recommended_proc_dict[lung_recommended_class]
        elif lung_or_adrenal_finding == "Adrenal":
            # Only one recommendation for adrenal findings
            lung_adrenal_recommended_proc_label = "Endocrinology Referral"
        
    else:
        # No finding detected; no follow-up recommended, etc.
        follow_up_flg = "NA"
        final_comment = "NA"
        lung_adrenal_recommended_proc_label = "NA"


    # Output includes classification and recommendations
    output = {
        "procedure": lung_adrenal_recommended_proc_label,	# Recommended procedure. For "Lung" finding, "Ambiguous" or "Chest CT". For "Adrenal", "Endocrinology Referral"
        "noduleType": lung_or_adrenal_finding,				# Options defined in `findings_model_dict`
        "followUpFlag": follow_up_flg,						# "Boolean" for findings: "Findings Present" or "NA"
        "followUpText": final_comment,						# Follow-up text (i.e., text that indicates finding)
    }
    
    return output


# %%