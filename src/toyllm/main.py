from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, text_batches
from datasets import load_dataset, load_dataset_builder, get_dataset_split_names, Dataset


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


def train(tokeniser: Tokenizer, trainer: trainers.BpeTrainer):

    for i in range()
    tokeniser.train_from_iterator(text_batches("trai"))



