import os
import sys
import logging

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset
import pyarrow as pa
from params import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def format_prompt(example):
    return f"""### INSTRUCTION: {example['instruction']}

    ### CONTEXT: {example['context']}
                            
    ### RESPONSE: {example['response']}
    """


def get_refactored_dolly15K_format(dataset_fraction=None):
    dataset_name = "databricks/databricks-dolly-15k"
    dataset = load_dataset(dataset_name, split="train")

    if dataset_fraction:
        assert dataset_fraction > 0 and dataset_fraction <= 1
        num_shards = int(1 / dataset_fraction)
        dataset = dataset.shard(num_shards, 0)

    samples = {"text": [], "category": []}
    for example in dataset:
        samples["text"].append(format_prompt(example))
        samples["category"].append(example["category"])

    dataset = Dataset(pa.Table.from_pydict(samples))
    return dataset


def main(dataset_fraction=None):
    dataset = get_refactored_dolly15K_format(dataset_fraction)

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    model = AutoModelForCausalLM.from_pretrained(
        src_model_name, quantization_config=bnb_config, device_map=device_map
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(src_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard",
    )

    # training_args.ddp_find_unused_parameters = False

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    trainer.train()

    #
    dst_model_name_model = dst_model_name + "/model"
    dst_model_name_token = dst_model_name + "/tokenizer"
    trainer.model.save_pretrained(dst_model_name_model)
    trainer.tokenizer.save_pretrained(dst_model_name_token)


if __name__ == "__main__":
    main()
