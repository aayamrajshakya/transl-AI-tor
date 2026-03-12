from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, get_dataset_split_names
import evaluate
import huggingface_hub
import config

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
    validation_dataset = load_dataset(dataset_name, split="validation", )
    source_texts = validation_dataset["text"]
    target_texts = []
    return source_texts, target_texts
    

def evaluate_translations(predictions, references, metric_name: str):
    """
    This function evaluates the translations using the specified evaluation metric.
    """
    metric = evaluate.load(metric_name)
    results = metric.compute(predictions=predictions, references=references)
    return results
    

def main():
    huggingface_hub.login()  # Ensure you are logged in to Hugging Face Hub
    print(get_dataset_split_names(config.EVAL_DATASET))

if __name__ == "__main__":
    main()