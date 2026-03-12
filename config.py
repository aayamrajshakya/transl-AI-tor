
"""

This file contains variables and parameters for the project,
such as file paths, model parameters, and other configuration settings.

"""
HF_ACCESS_TOKEN = "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"

# The language model to use for translation
LANGUAGE_MODEL = "facebook/nllb-200-distilled-600M" # NLLB
# LANGUAGE_MODEL = "csebuetnlp/mT5_multilingual_XLSum" # mT5

# Source and target languages for translation
# SOURCE_LANGUAGE = "en"
SOURCE_LANGUAGE = "eng_Latn" # FLORES-200 BCP-47 code for English if needed
TARGET_LANGUAGE = "ltz_Latn" # Luxembourgish
# TARGET_LANGUAGE = "lim_Latn" # Limburgish
# TARGET_LANGUAGE = "cat_Latn" # Catalan
# TARGET_LANGUAGE = "oci_Latn" # Occitan
# TARGET_LANGUAGE = "isl_Latn" # Icelandic
# TARGET_LANGUAGE = "fao_Latn" # Faroese
# TARGET_LANGUAGE = "ydd_Hebr" # Eastern Yiddish
# Add more languages as needed

# Dataset for evaluation
EVAL_DATASET = "openlanguagedata/flores_plus" # FLORES-200

# Evaluation metric
EVAL_METRIC = "bleu" # BLEU score for evaluation