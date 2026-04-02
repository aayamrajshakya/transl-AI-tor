import sys
import evaluate
import gc
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, get_dataset_split_names
from config import *


# ANSI codes
RED = '\033[31m'
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'


finetune_flag = 'finetune' in sys.argv
eval_flag = 'eval' in sys.argv


def which_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = which_device() # use the best available device


def cleanup(device):
    print("Clearing cache...")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def initialize_translator(model_name):
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
    return tokenizer(corpus["en"], text_target=corpus["lb"], padding=False, truncation=True)  # padding=False, DataCollatorForSeq2Seq dynamically pads the inputs received


def fine_tune_model(tokenizer, model, dataset, epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE):
    """
    This function fine-tunes the translation model on the provided source and target datasets.
    """
    # Tokenize the fine-tuning data
    print(f"\n{BLUE}Fine-tuning the model...{RESET}")
    #tokenized_dataset = dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    split_dataset = dataset.train_test_split(test_size=0.3, seed=42)
    train_data = split_dataset["train"].map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})
    eval_data = split_dataset["test"].map(tokenize_function, batched=True, fn_kwargs={"tokenizer": tokenizer})

    metric = evaluate.load(EVAL_METRIC)  # loaded once here, not on every eval call

    # defined here as a closure so it has access to tokenizer and metric for decoding token IDs to strings
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = [[token if token != -100 else tokenizer.pad_token_id for token in label] for label in labels]  # replace -100 padding before decoding
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        #logging_dir="./logs",
        #logging_steps=10,
        eval_strategy="epoch",
        predict_with_generate=True,
        optim="adafactor",  # memory-efficient optimizer for large models, default is adamw_torch
        dataloader_num_workers=4,  # parallel data loading so GPU isn't waiting on CPU
        auto_find_batch_size=True,  # automatically tries to find the largest batch size that fits in memory, avoiding CUDA OOM errors
        #load_best_model_at_end=True,
        #remove_unused_columns=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    print(f"Training on {len(split_dataset['train'])} samples...")
    trainer.train()
    

def eval_predict(tokenizer, model, source_texts, target_lang, batch_size=INFERENCE_BATCH_SIZE):
    """
    This function generates translations for the source texts and returns the predictions.
    """
    if isinstance(source_texts, str):
        source_texts = [source_texts]
    predictions = []
    for i in range(0, len(source_texts), batch_size):  # batch to avoid out of memory on large datasets
        batch = source_texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang))
        batch_preds = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        for original, translation in zip(batch, batch_preds):
            print(f"\n{RED}Original:{RESET} {original}")
            print(f"{GREEN}Translation:{RESET} {translation}")
        predictions.extend(batch_preds)
    return predictions


def evaluate_translations(predictions, references, metric_name):
    """
    This function evaluates the translations using the specified evaluation metric.
    """
    metric = evaluate.load(metric_name)
    reference_list = [[reference] for reference in references]
    results = metric.compute(predictions=predictions, references=reference_list)
    return results


def main():
    cleanup(device) # free any leftover GPU memory from previous runs before starting
    tokenizer, model = initialize_translator(LANGUAGE_MODEL)
    print(f"Device used: {device}")

    if (finetune_flag):
        ft_data = load_dataset(FT_DATASET, name="lb-en")["train"]
        fine_tune_model(tokenizer, model, ft_data)
        cleanup(device) # free training-related tensors/gradients before eval

    if (eval_flag):
        model.eval()    # put model in evaluation mode
        model = torch.compile(model)    # this speeds up the runtime apparently
        print(f"Converting {RED}{SOURCE_LANGUAGE}{RESET} ==> {GREEN}{TARGET_LANGUAGE}{RESET}")
        print(get_dataset_split_names(EVAL_DATASET))
        source_eval = load_dataset(EVAL_DATASET, SOURCE_LANGUAGE, split="devtest")["text"]
        references_eval = load_dataset(EVAL_DATASET, TARGET_LANGUAGE, split="devtest")["text"]
        predictions = eval_predict(tokenizer, model, source_eval, TARGET_LANGUAGE)
        results = evaluate_translations(predictions, references_eval, EVAL_METRIC)
        print(f"\n{RED}{EVAL_METRIC} results:{RESET}")
        print(", \n".join(f"{BLUE}{key}{RESET}: {val}" for key, val in results.items()))


if __name__ == "__main__":
    main()
