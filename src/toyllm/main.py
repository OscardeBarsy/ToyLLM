from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names, Dataset
import numpy as np
import sys
import os

def get_tokenizer():
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False
    )

    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer

def get_trainer():
    trainer = trainers.BpeTrainer(
        vocab_size=8192,
        special_tokens=[
            "<|endoftext|>",
            "<|user|>",
            "<|assistant|>"
        ],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    return trainer


def get_dataset(dataset_name: str):
    ds_builder = load_dataset_builder(dataset_name)
    print(ds_builder.info.description)
    print(ds_builder.info.features)
    dataset = load_dataset(dataset_name)
    splits = get_dataset_split_names(dataset_name)



    return dataset, splits

def batch_iterator(dataset: Dataset, batch_size: int =1000):
    tok_dataset = dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]

def train_tokenizer(tokenizer: Tokenizer, trainer: trainers.BpeTrainer, dataset: Dataset, tokenizer_file: str = "tokenizer.json"):

    
    tokenizer.train_from_iterator(batch_iterator(dataset=dataset), trainer=trainer)
    tokenizer.save(tokenizer_file)
    return tokenizer

def encode(dataset: Dataset, tokenizer: Tokenizer):
    encoding = tokenizer.encode_batch(batch_iterator(dataset=dataset))
    return {"ids": [e.ids + tokenizer.token_to_id("<|endoftext|>") for e in encoding]}

def main():

    dataset_name = "roneneldan/TinyStories"
    tokenizer_file = "tokenizer.json"
    ids_file = "train.bin"

    dataset, splits = get_dataset(dataset_name=dataset_name)
    trainer = get_trainer()

    if os.path.exists(tokenizer_file):
        print("tokenizer already exists")
        tokenizer = Tokenizer.from_file(tokenizer_file)
    else:
        print("tokenizer file not found, training new tokenizer")
        tokenizer = get_tokenizer()
        if "train" in splits:
            tokenizer = train_tokenizer(tokenizer=tokenizer, trainer=trainer, dataset=dataset["train"], tokenizer_file=tokenizer_file)
        else:
            print("No train set in data set")

    if os.path.exists(ids_file):
        print("ids file already exists")
    else:
        if "train" in splits:
                    rows = dataset["train"].map(lambda batch: encode(b), batched=True)
                    ids = np.fromiter(
                        [id for row in rows["ids"] for id in row],
                        dtype=np.uint16
                    )
                    ids.tofile("train.bin")
        else:
            print("No train set in data set")
        


    print(dataset)

    return 0



if __name__ == '__main__':
    sys.exit(main())