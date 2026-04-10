"""
This file contains variables and parameters for the project,
such as file paths, model parameters, and other configuration settings.
"""

# The language model to use for translation
LANGUAGE_MODEL = "facebook/nllb-200-distilled-1.3B" # https://huggingface.co/facebook/nllb-200-distilled-1.3B
# LANGUAGE_MODEL = "facebook/nllb-200-distilled-600M"   # https://huggingface.co/facebook/nllb-200-distilled-600M

# Source and target languages for translation
SOURCE_LANGUAGE = "eng_Latn" # FLORES-200 BCP-47 code for English if needed
TARGET_LANGUAGE = "ltz_Latn" # Luxembourgish

# Datasets
FT_DATASET = "fredxlpy/LuxAlign" # Dataset for fine-tuning; https://huggingface.co/datasets/fredxlpy/LuxAlign
EVAL_DATASET = "openlanguagedata/flores_plus" # Dataset for evaluation; https://huggingface.co/datasets/openlanguagedata/flores_plus

# Evaluation metric
EVAL_METRIC = "sacrebleu" # BLEU score for evaluation

# Training hyperparameters
EPOCHS = 5
TRAIN_BATCH_SIZE = 16
INFERENCE_BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_LENGTH = 128
