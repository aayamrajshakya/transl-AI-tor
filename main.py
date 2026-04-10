import sys
import evaluate
import gc
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import numpy as np
from datasets import load_dataset
from config import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock between fast tokenizer threads and num_proc multiprocessing


# ANSI codes
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
RESET = "\033[0m"


finetune_flag = "finetune" in sys.argv
eval_flag = "eval" in sys.argv


def which_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


device = which_device()  # use the best available device


def cleanup(device):
    print("Clearing cache...")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # elif device.type == "mps":
    #     torch.mps.empty_cache()


def initialize_translator(model_name):
    """
    This function initializes the tokenizer and model for translation based on the given model name.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                #   tie_word_embeddings=False,
                                                  torch_dtype="auto").to(device)
    return tokenizer, model


def tokenize_function(corpus, tokenizer):
    return tokenizer(corpus["en"],
                     text_target=corpus["lb"],
                     padding=False, # padding=False, DataCollatorForSeq2Seq dynamically pads the inputs received
                     truncation=True,
                     max_length=MAX_LENGTH)


def fine_tune_model(tokenizer, model, dataset, epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE):
    """
    This function fine-tunes the translation model on the provided source and target datasets.
    """
    print(f"\n{BLUE}Fine-tuning the model...{RESET}")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)    # luxalign is already small, so setting higher test size doesn't benefit
    tokenizer.src_lang = SOURCE_LANGUAGE
    tokenizer.tgt_lang = TARGET_LANGUAGE
    num_proc = (4 if device.type == "cuda" else 1)  # num_proc for parallelizing tokenization
    train_data = split_dataset["train"].map(tokenize_function,
                                            batched=True,
                                            fn_kwargs={"tokenizer": tokenizer},
                                            num_proc=num_proc,
                                            )
    eval_data = split_dataset["test"].map(tokenize_function,
                                          batched=True,
                                          fn_kwargs={"tokenizer": tokenizer},
                                          num_proc=num_proc,
                                          )

    metric = evaluate.load(EVAL_METRIC) # loaded once here, not on every eval call

    # defined here as a closure so it has access to tokenizer and metric for decoding token IDs to strings
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)  # replace -100 padding before decoding
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           pad_to_multiple_of=8 if device.type == "cuda" else None)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",  # when to run evaluation; evaluate at the end of each epoch
        save_strategy="epoch",  # when to save checkpoints; save at the end of each epoch
        predict_with_generate=True, # whether to use generate to calc generative metrics (like BLEU)
        generation_num_beams=4, # improves generation quality by exploring multiple promising token paths instead of just the single best one
        generation_max_length=MAX_LENGTH,   # max length to use on each eval loop
        optim="adafactor",  # memory-efficient optimizer for large models, default is adamw_torch
        learning_rate=LEARNING_RATE,    # duhh
        warmup_steps=300,   # no. of training steps for which the LR is gradually increased from a small val to the initial LR
        weight_decay=0.01,  # adds penalty term to the loss func to prevent overfitting by keeping the wts small
        label_smoothing_factor=0.05,    # prevent overconfidence
        lr_scheduler_type="cosine", # smoother decay than linear; tapers off gently at the end
        logging_steps=50,   # log training loss every 50 steps
        report_to="none",   # disable auto-reporting of results and logs TensorBoard and others
        dataloader_num_workers=(4 if device.type == "cuda" else 0),  # parallel data loading only on cuda
        # auto_find_batch_size=True,  # automatically tries to find the largest batch size that fits in memory, avoiding CUDA OOM errors
        train_sampling_strategy="group_by_length",  # group samples of roughly the same length together to minimize padding and be more efficient
        gradient_accumulation_steps=2,  # no. of update steps to accumulate gradients bfr performing backward/update pass
        eval_accumulation_steps=16, # no. of prediction steps to accumulate the output tensors for, bfr moving results to CPU
        load_best_model_at_end=True,    # load the best checkpoint at the end of training
        metric_for_best_model="score",  # metric to use for comparing models when load_best_model_at_end=True
        greater_is_better=True, # whether higher metric values are better; in our case: higher BLEU is indeed better
        save_total_limit=2, # keeps the most recent checkpoint plus the best
        bf16=(device.type == "cuda"), # use bf16 if we're on cuda device
        dataloader_pin_memory=(device.type == "cuda"),  # speeds up CPU to GPU data transfer
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],   # stop if BLEU doesn't improve for 3 consecutive epochs
    )
    print(f"Training on {len(split_dataset['train'])} samples...")
    trainer.train()
    trainer.save_model()


def eval_predict(tokenizer, model, source_texts, target_lang, batch_size=INFERENCE_BATCH_SIZE):
    """
    This function generates translations for the source texts and returns the predictions.
    """
    if isinstance(source_texts, str):
        source_texts = [source_texts]
    predictions = []
    tokenizer.src_lang = SOURCE_LANGUAGE
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)  # compute once before the loop, not on every batch
    for i in tqdm(range(0, len(source_texts), batch_size), desc="Translating"):  # batch to avoid out of memory on large datasets
        batch = source_texts[i : i + batch_size]
        inputs = tokenizer(batch,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=MAX_LENGTH).to(device)
        with torch.inference_mode():    # disables "view tracking" and "version counter bumps," which are still active in no_grad
            translated_tokens = model.generate(**inputs,
                                               forced_bos_token_id=forced_bos_token_id,
                                               num_beams=4,
                                               no_repeat_ngram_size=3)  # prevents repeated phrases in output
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
    if len(sys.argv) < 2:
        print(f"{RED}PROVIDE REQD ARGUMENTS: finetune and/or eval{RESET}")
        sys.exit(1)

    print(f"Device used: {device}")
    
    cleanup(device)  # free any leftover GPU memory from previous runs before starting
    tokenizer, model = initialize_translator(model_name=LANGUAGE_MODEL)

    if finetune_flag:
        ft_data = load_dataset(path=FT_DATASET, name="lb-en")["train"]
        fine_tune_model(tokenizer=tokenizer,
                        model=model,
                        dataset=ft_data)
        cleanup(device)  # free training-related tensors/gradients before eval

    if eval_flag:
        model.eval()
        if device.type == "cuda":   # little effect on cpu device, apparently...
            model = torch.compile(model)
        print(f"Converting {RED}{SOURCE_LANGUAGE}{RESET} ==> {GREEN}{TARGET_LANGUAGE}{RESET}")
        # print(get_dataset_split_names(EVAL_DATASET))
        source_eval = load_dataset(path=EVAL_DATASET,
                                   name=SOURCE_LANGUAGE,
                                   split="devtest")["text"]
        references_eval = load_dataset(path=EVAL_DATASET,
                                       name=TARGET_LANGUAGE,
                                       split="devtest")["text"]
        predictions = eval_predict(tokenizer=tokenizer,
                                   model=model,
                                   source_texts=source_eval,
                                   target_lang=TARGET_LANGUAGE)
        results = evaluate_translations(predictions=predictions,
                                        references=references_eval,
                                        metric_name=EVAL_METRIC)
        
        print(f"\n{RED}{EVAL_METRIC} results:{RESET}")
        print(", \n".join(f"{BLUE}{key}{RESET}: {val}" for key, val in results.items()))


if __name__ == "__main__":
    main()
