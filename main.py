import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, get_dataset_split_names
import evaluate
import torch
from config import *
import time
import sys


# ANSI codes
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'


finetune_flag = 'finetune' in sys.argv
eval_flag = 'eval' in sys.argv

device = which_device() # use the best available device (gpu) or fallback to cpu


def cleanup(device):
    print("Clearing cache...")
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

def initialize_translator(model_name: str):
    """
    This function initializes the tokenizer and model for translation based on the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, tie_word_embeddings=False,
                                                  torch_dtype="auto").to(device)
    return tokenizer, model

def tokenize_function(corpus, tokenizer):
    tokenizer.src_lang = SOURCE_LANGUAGE
    tokenizer.tgt_lang = TARGET_LANGUAGE
    return tokenizer(corpus["en"], text_target=corpus["lb"], padding=True, return_tensors="pt", truncation=True)

def compute_metrics(eval_pred):
    metric = evaluate.load(EVAL_METRIC)
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

def fine_tune_model(tokenizer, model, dataset, epochs=5, batch_size=128):
    """ This function fine-tunes the translation model on the provided source and target datasets."""
    # Tokenize the fine-tuning data
    print(f"\n{BLUE}Fine-tuning the model...{RESET}")
    #tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    train_data = split_dataset["train"].map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    eval_data = split_dataset["test"].map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #logging_dir="./logs",
        #logging_steps=10,
        eval_strategy="epoch",
        #load_best_model_at_end=True,
        #remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print(f"Training on {len(split_dataset['train'])} samples...")
    trainer.train()
    

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
    cleanup(device) # free up GPU memory before starting
    tokenizer, model = initialize_translator(LANGUAGE_MODEL)
    print(f"Device used: {device}")

    if (finetune_flag):
        ft_data = load_dataset(FT_DATASET, name="lb-en")["train"]
        fine_tune_model(tokenizer, model, ft_data)
    
    if (eval_flag):
        model.eval()    # put model in evaluation mode
        model = torch.compile(model)    # this speeds up the runtime apparently
        print(f"Converting {RED}{SOURCE_LANGUAGE}{RESET} ==> {GREEN}{TARGET_LANGUAGE}{RESET}")
        print(get_dataset_split_names(EVAL_DATASET))
        source_eval = load_dataset(EVAL_DATASET, SOURCE_LANGUAGE, split="devtest")["text"]
        references_eval = load_dataset(EVAL_DATASET, SOURCE_LANGUAGE, split="devtest")["text"]
        predictions = eval_predict(tokenizer, model, source_eval, TARGET_LANGUAGE)
        results = evaluate_translations(predictions, references_eval, EVAL_METRIC)
        print(f"\n{RED}{EVAL_METRIC} results:{RESET}")
        print(", \n".join(f"{BLUE}{key}{RESET}: {val}" for key, val in results.items()))
    
    end = time.time()
    print(f"\nTotal time: {end - start:.2f}s")


if __name__ == "__main__":
    main()
