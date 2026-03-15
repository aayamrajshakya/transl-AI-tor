from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, get_dataset_split_names
import evaluate
import torch
import huggingface_hub
import config


# strictly use GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def initialize_translator(model_name: str):
    """
    This function initializes the tokenizer and model for translation based on the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, tie_word_embeddings=False).to(device)
    return tokenizer, model


def translate_text(tokenizer, model, text: str, target_lang: str):
    """
    This function takes in text and uses the tokenizer and model to translate it from the source language to the target language.
    """
    print(f"\n\033[31mOriginal text:\033[0m {text}")
    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    print(f"\033[32mTranslation:\033[0m {translation}")
    return translation


def load_eval_dataset(dataset_name: str, source_lang: str, target_lang: str):
    """
    This function loads the evaluation dataset using the Hugging Face Datasets library.
    """
    source_dataset = load_dataset(dataset_name, source_lang, split="devtest")
    source_texts = source_dataset["text"][:10]
    target_dataset = load_dataset(dataset_name, target_lang, split="devtest")
    target_texts = target_dataset["text"][:10]
    return source_texts, target_texts


def eval_predict(tokenizer, model, source_texts, target_lang):
    """
    This function generates translations for the source texts and returns the predictions.
    """
    model.eval()    # put model in evaluation mode
    predictions = []
    with torch.no_grad():
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
    # huggingface_hub.login()
    print(f"Device used: {device}")
    print(f"Converting \033[31m{config.SOURCE_LANGUAGE}\033[0m ==> \033[32m{config.TARGET_LANGUAGE}\033[0m")
    print(get_dataset_split_names(config.EVAL_DATASET))
    tokenizer, model = initialize_translator(config.LANGUAGE_MODEL)
    source_texts, target_texts = load_eval_dataset(config.EVAL_DATASET, config.SOURCE_LANGUAGE, config.TARGET_LANGUAGE)
    predictions = eval_predict(tokenizer, model, source_texts, config.TARGET_LANGUAGE)
    results = evaluate_translations(predictions, target_texts, config.EVAL_METRIC)
    print(f"\n\033[31m{config.EVAL_METRIC} results:\033[0m")
    print(", \n".join(f"\033[34m{key}\033[0m: {val}" for key, val in results.items()))


if __name__ == "__main__":
    main()
