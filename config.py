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

# Datasets
FT_DATASET = "fredxlpy/LuxAlign" # Dataset for fine-tuning
EVAL_DATASET = "openlanguagedata/flores_plus" # Dataset for evaluation

# Evaluation metric
EVAL_METRIC = "sacrebleu" # BLEU score for evaluation

# Training hyperparameters
EPOCHS = 5                  # number of full passes over the training data
TRAIN_BATCH_SIZE = 128      # starting batch size for training; auto_find_batch_size will halve if OOM
INFERENCE_BATCH_SIZE = 32   # batch size for inference in eval_predict
