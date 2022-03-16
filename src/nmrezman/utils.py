# %%

# Typing
from typing import Union, Tuple, List
import numpy.typing as npt

# Misc
import re
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# %%

def keyword_split(x: str, keywords: Union[List, Tuple], return_idx: int=2) -> str:
    """
    Extract portion of string given a list of possible delimiters (keywords) via partition method

    Args:
        x (`str`):
            Note to select text from
        keywords (`List` or `Tuple`):
            Possible keywords to partition `x` on, searching in order
        return_idx (`int`):
            Return index from :meth:`partition`'s 3-tuple

    Returns:
        [`str`] Partitioned note
    """

    for keyword in keywords:
        if x.partition(keyword)[2] !='':
            return x.partition(keyword)[return_idx]
    return x


def get_impression(note: str, is_phase_2: bool=True) -> str:
    """
    Retrieves the impression / conclusion section of the radiology report. If no impression / conclusion section is 
    found, the note is returned.

    Args:
        note (`str`): 
            Radiology report text
        is_phase_2 (`bool`):
            Boolean denoting whether note is to be Phase 01's (``False``) or Phase 02's (``True``) model(s)
    
    Returns:
        [`str`]: impression / conclusion section of the report
    """

    # In Phase 2, adding "impression" first improved the model performance
    base = [
            "impression:",
            "conclusion(s):",
            "conclusions:",
            "conclusion:",
    ]
    findings = [
            "finding:",
            "findings:",
    ]
    if not is_phase_2:
        # Phase 1
        impression_keywords = findings + base
    else:
        # Phase 2
        impression_keywords = base + findings
    return keyword_split(note, impression_keywords)


def remove_drtag(note: str) -> str:
    """
    Removes any signature by the the radiologist at the end of the note

    Args:
        note (`str`):
            Radiologist note
    
    Returns:
        [`str`] Clean note without radiologist signature
    """

    # These are probably specific to NM
    # You may need to change these depending on the format of your reports
    signature_keywords = [
        "&#x20",
        "final report attending radiologist:",
    ]
    return keyword_split(note, signature_keywords, return_idx=0)
    
    
def preprocess_input(note: str, is_phase_2: bool=True) -> str:
    """
    Complete the all preprocessing steps for getting clean impressions from the oirginal radiology report.

    Args:
        note (`str`):
            Entire original radiology report
        is_phase_2 (`bool`):
            Boolean indicating whether the note will be used for Phase 01 (``False``) or Phase 02 (``True``) models

    Returns:
        [`str`] The preprocessed impresion (i.e., clean note)
    """
    
    impressions = get_impression(str(note).lower(), is_phase_2)

    # Remove newlines breaks (resulting from cloud integration)
    # This may be specific to organization / computing platform
    impressions = re.sub(r'\\.br\\', '', impressions)
    impressions = re.sub(r'\n\n', ' ', impressions)
    impressions = re.sub(r'\n', ' ', impressions)

    # Remove Dr tag lines
    impressions = remove_drtag(impressions)

    return impressions


# %%

def generate_eval_report(y_valid: npt.NDArray, y_pred: npt.NDArray, result_fname: str) -> None:
    """
    Compute the classification report and confusion matrix, which are then saved to an output file.

    Args:
        y_valid (`npt.NDArray`):
            The true data
        y_valid (`npt.NDArray`):
            The data predicted by the model
        result_fname (`str`):
            Name (name and path) of where to save the results
    """
    
    # Perform confusion matrix and save the results
    report = classification_report(y_valid, y_pred)
    matrix = confusion_matrix(y_valid, y_pred)
    print(report)
    print(matrix)
    with open(result_fname, "w") as fh:
        fh.write("Classification Report:\n")
        fh.write(report)
        fh.write("\n\nConfusion Matrix:\n")
        fh.write(np.array2string(matrix, separator=", "))

    return

