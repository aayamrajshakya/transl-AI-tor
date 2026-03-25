from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
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

def load_translation_data(dataset_name: str, source_lang: str, target_lang: str):
    """
    This function loads Hugging Face datasets for a source and target language.
    """
    source_dataset = load_dataset(dataset_name, source_lang, split="train")
    target_dataset = load_dataset(dataset_name, target_lang, split="train")
    return source_dataset, target_dataset

def tokenize_ft_data(tokenizer, source_dataset, target_dataset):
    """
    This function tokenizes the fine-tuning data and prepares it for training.
    """
    inputs = tokenizer(source_dataset["text"], truncation=True).to(device)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_dataset["text"], truncation=True).to(device)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    return inputs, labels, data_collator

def compute_metrics(eval_pred):
    metric = evaluate.load(EVAL_METRIC)
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

def fine_tune_model(tokenizer, model, source_dataset, target_dataset, epochs=3, batch_size=16):
    """ This function fine-tunes the translation model on the provided source and target datasets."""
    # Tokenize the fine-tuning data
    inputs = tokenizer(source_dataset["text"], truncation=True).to(device)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(target_dataset["text"], truncation=True).to(device)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs["train"],
        eval_dataset=labels["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    

def eval_predict(tokenizer, model, source_dataset, target_lang):
    """
    This function generates translations for the source texts and returns the predictions.
    """

    source_texts = source_dataset["text"][:10] # limit to 10 samples for evaluation
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


def evaluate_translations(predictions, reference_dataset, metric_name: str):
    """
    This function evaluates the translations using the specified evaluation metric.
    """

    references = reference_dataset["text"][:10] # limit to 10 samples for evaluation

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
    source_dataset, reference_dataset = load_translation_data(EVAL_DATASET, SOURCE_LANGUAGE, TARGET_LANGUAGE)
    predictions = eval_predict(tokenizer, model, source_dataset, TARGET_LANGUAGE)
    results = evaluate_translations(predictions, reference_dataset, EVAL_METRIC)
    print(f"\n{RED}{EVAL_METRIC} results:{RESET}")
    print(", \n".join(f"{BLUE}{key}{RESET}: {val}" for key, val in results.items()))
    end = time.time()
    print(f"\nTotal time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
