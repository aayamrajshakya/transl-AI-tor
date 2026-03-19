from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, get_dataset_split_names
import evaluate
import torch
from config import *
import time


# ANSI codes
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'

device = which_device() # use the best available device (gpu) or fallback to cpu


def initialize_translator(model_name: str):
    """
    This function initializes the tokenizer and model for translation based on the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, tie_word_embeddings=False,
                                                  torch_dtype="auto").to(device)
    model.eval()    # put model in evaluation mode
    model = torch.compile(model)    # this speeds up the runtime apparently
    return tokenizer, model


def load_eval_dataset(dataset_name: str, source_lang: str, target_lang: str):
    """
    This function loads the evaluation dataset using the Hugging Face Datasets library.
    """
    source_dataset = load_dataset(dataset_name, source_lang, split="devtest")
    source_texts = source_dataset["text"][:10]
    reference_dataset = load_dataset(dataset_name, target_lang, split="devtest")
    reference_texts = reference_dataset["text"][:10]
    return source_texts, reference_texts


def eval_predict(tokenizer, model, source_texts, target_lang):
    """
    This function generates translations for the source texts and returns the predictions.
    """
    if isinstance(source_texts, str):
        source_texts = [source_texts]
    inputs = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
    predictions = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    for original, translation in zip(source_texts, predictions):
        print(f"\n{RED}Original:{RESET} {original}")
        print(f"{GREEN}Translation:{RESET} {translation}")
    return predictions


def evaluate_translations(predictions, references, metric_name: str):
    """
    This function evaluates the translations using the specified evaluation metric.
    """
    metric = evaluate.load(metric_name)
    reference_list = [[reference] for reference in references]
    results = metric.compute(predictions=predictions, references=reference_list)
    return results


def main():
    start = time.time()
    tokenizer, model = initialize_translator(LANGUAGE_MODEL)
    print(f"Device used: {device}")
    print(f"Converting {RED}{SOURCE_LANGUAGE}{RESET} ==> {GREEN}{TARGET_LANGUAGE}{RESET}")
    print(get_dataset_split_names(EVAL_DATASET))
    source_texts, reference_texts = load_eval_dataset(EVAL_DATASET, SOURCE_LANGUAGE, TARGET_LANGUAGE)
    predictions = eval_predict(tokenizer, model, source_texts, TARGET_LANGUAGE)
    results = evaluate_translations(predictions, reference_texts, EVAL_METRIC)
    print(f"\n{RED}{EVAL_METRIC} results:{RESET}")
    print(", \n".join(f"{BLUE}{key}{RESET}: {val}" for key, val in results.items()))
    end = time.time()
    print(f"\nTotal time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
