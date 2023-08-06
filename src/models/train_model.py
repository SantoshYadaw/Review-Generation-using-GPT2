import os
import logging
from tqdm.auto import tqdm

tqdm.pandas()

import pandas as pd

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    AutoConfig,
)
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"device: {device}")

# Constants
HOME_PATH = os.getcwd()
logger.info(f"HOME_PATH: {HOME_PATH}")

DATA_PATH = os.path.join(HOME_PATH, "data", "raw", "task2.csv")
logger.info(f"DATA_PATH: {DATA_PATH}")

SPLIT_DATA_PATH = os.path.join(HOME_PATH, "data", "processed", "split_data.csv")
logger.info(f"SPLIT_DATA_PATH: {SPLIT_DATA_PATH}")

# Set the path to save gpt2 model
MODEL_PATH = os.path.join(HOME_PATH, "models")
logger.info(f"model_path: {MODEL_PATH}")

SEED = 2023

# Specify special tokens for gpt2
bos = "<|endoftext|>"
eos = "<|EOS|>"
pad = "<|pad|>"
special_tokens_dict = {"eos_token": eos, "bos_token": bos, "pad_token": pad}

NUM_EPOCHS_TRAIN = 6  # total # of training epochs
BATCH_SIZE_TRAIN = 1  # batch size per device during training
BATCH_SIZE_EVAL = 1  # batch size for evaluation
WARMUP_STEPS = 200  # number of warmup steps for learning rate scheduler
WEIGHT_DECAY = 0.01  # strength of weight decay
LOGGING_DIR = MODEL_PATH  # directory for storing logs
PREDICTION_LOSS = True
SAVE_STEPS = 10000


def train_gpt2(data_path: str):
    """Train gpt2 model

    Args:
        data_path (str): Path to data
    """

    def tokenize_function(text: str):
        """
        Tokenize the given text
        """
        return gpt2_tokenizer(
            text["text"], padding=True, truncation=True, max_length=1024
        )

    # Load split data
    logging.info(f"Loading split data ...")
    data = pd.read_csv(data_path)
    data["text"] = data["text"].astype(str)
    data_train = data[data["split"] == "train"]
    data_val = data[data["split"] == "val"]

    # Add special tokens
    logging.info(f"Add special tokens ...")
    data["text"] = bos + " " + data["text"] + " " + eos

    # Initialise gpt2 tokenizer
    logging.info(f"Loading gpt2 tokenizer")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Checking what is the vocab size for gpt2
    logging.info(f"Words in vocabulary: {gpt2_tokenizer.vocab_size}")

    # the new token is added to the tokenizer
    num_added_toks = gpt2_tokenizer.add_special_tokens(special_tokens_dict)

    # Initialise the model configs and add special tokens
    logging.info(f"Initialise model configs and add special tokens ...")
    config = AutoConfig.from_pretrained(
        "gpt2",
        bos_token_id=gpt2_tokenizer.bos_token_id,
        eos_token_id=gpt2_tokenizer.eos_token_id,
        pad_token_id=gpt2_tokenizer.pad_token_id,
        output_hidden_states=False,
    )

    # Load GPT2 model with special tokens
    logging.info(f"Load gpt2 model ...")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    # Resize embedding to tokenizer dimensions
    logging.info(f"Resize gpt2 embedding to tokenizer dimensions ...")
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))

    # load the data using hugging face dataset
    logging.info(f"Create dataset ...")
    train_dataset = Dataset.from_pandas(data_train[["text"]])
    val_dataset = Dataset.from_pandas(data_val[["text"]])

    # Tokenize dataset
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=5,
        remove_columns=["text"],
    )
    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=5,
        remove_columns=["text"],
    )

    # Set the training arguments for gpt2. Using default
    logging.info(f"Set training arguments ...")
    training_args = TrainingArguments(
        output_dir=MODEL_PATH,  # output directory
        num_train_epochs=NUM_EPOCHS_TRAIN,  # total # of training epochs
        per_device_train_batch_size=BATCH_SIZE_TRAIN,  # batch size per device during training
        per_device_eval_batch_size=BATCH_SIZE_EVAL,  # batch size for evaluation
        warmup_steps=WARMUP_STEPS,  # number of warmup steps for learning rate scheduler
        weight_decay=WEIGHT_DECAY,  # strength of weight decay
        logging_dir=MODEL_PATH,  # directory for storing logs
        prediction_loss_only=PREDICTION_LOSS,
        save_steps=SAVE_STEPS,
    )

    # Initialise the data collator
    logging.info(f"Initialise data collator ...")
    data_collator = DataCollatorForLanguageModeling(tokenizer=gpt2_tokenizer, mlm=False)

    # Initialise huggingface trainer
    logging.info(f"Initialise huggingface trainer ...")
    trainer = Trainer(
        model=gpt2_model,  # instantiated gpt2 model to be trained
        args=training_args,  # training arguments, defined above
        data_collator=data_collator,
        train_dataset=tokenized_train_dataset,  # training dataset
        eval_dataset=tokenized_val_dataset,  # evaluation dataset
    )

    # Train model
    logging.info(f"Train gpt2 model ...")
    trainer.train()

    # Save trained model
    logging.info(f"Saving gpt2 model ...")
    trainer.save_model()
    gpt2_tokenizer.save_pretrained(MODEL_PATH)

    logging.info(f"Gpt2 model training complete. Artifacts stored: {MODEL_PATH}")


if __name__ == "__main__":

    # Train gpt2 model
    train_gpt2(SPLIT_DATA_PATH)
