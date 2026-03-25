"""
This file contains variables and parameters for the project,
such as file paths, model parameters, and other configuration settings.
"""

# The language model to use for translation
LANGUAGE_MODEL = "facebook/nllb-200-distilled-1.3B" # NLLB
# LANGUAGE_MODEL = "csebuetnlp/mT5_multilingual_XLSum" # mT5

# Source and target languages for translation
SOURCE_LANGUAGE = "eng_Latn" # FLORES-200 BCP-47 code for English if needed
TARGET_LANGUAGE = "ltz_Latn" # Luxembourgish
# TARGET_LANGUAGE = "lim_Latn" # Limburgish
# TARGET_LANGUAGE = "cat_Latn" # Catalan
# TARGET_LANGUAGE = "oci_Latn" # Occitan
# TARGET_LANGUAGE = "isl_Latn" # Icelandic
# TARGET_LANGUAGE = "fao_Latn" # Faroese
# TARGET_LANGUAGE = "ydd_Hebr" # Eastern Yiddish
# Add more languages as needed

# Datasets
FT_DATASET = "wikimedia/wikipedia" # Wikipedia dataset for fine-tuning (can be large, so we will limit the number of samples)x
EVAL_DATASET = "openlanguagedata/flores_plus" # FLORES-200

# Evaluation metric
EVAL_METRIC = "sacrebleu" # BLEU score for evaluation


# strictly use GPU
import torch
def which_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device
