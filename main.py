from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, get_dataset_split_names
import evaluate
import huggingface_hub
import config
import sys

def initialize_translator(model_name: str):
    """
    This function initializes the tokenizer and model for translation based on the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def translate_text(tokenizer, model, text: str, target_lang: str):
    """
    This function takes in text and uses the tokenizer and model to translate it from the source language to the target language.
    """
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    return translation
    
def load_eval_dataset(dataset_name: str, source_lang: str, target_lang: str):
    """
    This function loads the evaluation dataset using the Hugging Face Datasets library.
    """
    source_dataset = load_dataset(dataset_name, split="devtest", source_lang=source_lang)
    source_texts = source_dataset["text"]
    target_dataset = load_dataset(dataset_name, split="devtest", source_lang=target_lang)
    target_texts = target_dataset["text"]
    return source_texts, target_texts

def eval_predict(tokenizer, model, source_texts, target_lang):
    """
    This function generates translations for the source texts and returns the predictions.
    """
    predictions = []
    for text in source_texts:
        translation = translate_text(tokenizer, model, text, target_lang)
        predictions.append(translation)
    return predictions
    

def evaluate_translations(predictions, references, metric_name: str):
    """
    This function evaluates the translations using the specified evaluation metric.
    """
    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=predictions, references=references)
    return results
    

def main():
    huggingface_hub.login()
    print(get_dataset_split_names(config.EVAL_DATASET))
    tokenizer, model = initialize_translator(config.LANGUAGE_MODEL)
    source_texts, target_texts = load_eval_dataset(config.EVAL_DATASET, config.SOURCE_LANGUAGE, config.TARGET_LANGUAGE)
    predictions = eval_predict(tokenizer, model, source_texts, config.TARGET_LANGUAGE)
    results = evaluate_translations(predictions, target_texts, config.EVAL_METRIC)
    print(f"Evaluation results: {results}")
    sys.exit(0)

if __name__ == "__main__":
    main()