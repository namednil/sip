import json
import random
import sys
from typing import Tuple, Iterable, List, Union

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Sampler, RandomSampler, BatchSampler, DataLoader, SequentialSampler
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizerFast

import datasets

import tqdm

def load_tsv(fname, expect_first_line = None, lenient: bool = False):
    with open(fname) as f:
        it = iter(f)
        if expect_first_line is not None:
            first_line = next(it).strip()
            if expect_first_line != first_line:
                if lenient:
                    line = first_line.strip("\n").strip("\r")
                    if line:
                        yield line.split("\t")
                else:
                    raise ValueError(f"First line must be: '{expect_first_line}'")
        for line in it:
            line = line.strip("\n").strip("\r")
            if line:
                yield line.split("\t")

def prepare_task_dataset(path:str, tokenizer: AutoTokenizer, batch_size: int, random_order: bool = True, lenient: bool=False) -> DataLoader:
    def mapper(examples):
        d = tokenizer(examples["input"])
        if "output" in examples:
            d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]
        return d

    keys = ["input", "output"]
    d = {k: [] for k in keys}
    for row in load_tsv(path, "input\toutput", lenient=lenient):
        for x, k in zip(row, keys):
            d[k].append(x)
    dataset = Dataset.from_dict(d)

    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    ts = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])
    return DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)


def fst_to_vector(fst_tokenizer, num_states, fst: List[Tuple[int, str, str, int]]) -> np.array:
    assert len(fst[0]) == 4 or len(fst[0]) == 5

    fst_rep = np.zeros((len(fst), len(fst[0])), dtype=np.int64)
    for j, f in enumerate(fst):
        s, i, o, sp = f[:4]
        assert s < num_states-1 #last state is reserved for padding
        assert sp < num_states-1
        fst_rep[j, 0] = s

        i_encoded = fst_tokenizer(i)["input_ids"]
        assert len(i_encoded) == 1
        fst_rep[j, 1] = i_encoded[0]

        o_encoded = fst_tokenizer(o)["input_ids"]
        assert len(o_encoded) == 1
        fst_rep[j, 2] = o_encoded[0]

        fst_rep[j, 3] = sp

        if len(f) == 5:
            # for final state indicator
            fst_rep[j, 4] = f[4]
    return fst_rep

def batch_fsts(fst_reps: List[np.array], num_states, max_len=None) -> np.array:
    if max_len is None:
        max_len = max(len(x) for x in fst_reps)
    batched_fst_reps = np.zeros((len(fst_reps), max_len, len(fst_reps[0][0])), dtype=np.int64)
    # Set states to a padding index (last state)
    batched_fst_reps[:, :, 0] = num_states - 1
    batched_fst_reps[:, :, 3] = num_states - 1
    for i, x in enumerate(fst_reps):
        for j, f in enumerate(x):
            if max_len is not None and j >= max_len:
                continue
            batched_fst_reps[i, j] = f
    return batched_fst_reps


def load_fst_jsonl(path: str, tokenizer: AutoTokenizer, fst_tokenizer: Union[str, PreTrainedTokenizerFast], batch_size:int, num_states: int, random_order: bool = True,
                   max_len: int = None, max_n:int=None, map_f = None, filter_f = None):
    if isinstance(fst_tokenizer, str):
        fst_tokenizer = PreTrainedTokenizerFast.from_pretrained(fst_tokenizer)

    def mapper(examples):
        d = tokenizer(examples["input"])
        if "output" in examples:
            d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]
        return d

    if map_f is None:
        map_f = lambda x: x

    data = {"input": [], "output": [], "fst_rep": [], "task_ids": []}
    with open(path) as f:
        i = 0
        for line in f:
            d = json.loads(line)
            if filter_f is None or filter_f(d):
                data["input"].append(d["input"])
                data["output"].append(d["output"])

                if "task_id" in d:
                    data["task_ids"].append(d["task_id"])

                data["fst_rep"].append(fst_to_vector(fst_tokenizer, num_states, map_f(d["FST"])))

                i += 1
                if max_n is not None and i > max_n:
                    break

    if len(data["task_ids"]) == 0:
        del data["task_ids"]

    dataset = Dataset.from_dict(data)
    if random_order:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    ts = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataset = dataset.map(mapper, batched=False, remove_columns=["input", "output"])

    seq2seq_collator = DataCollatorForSeq2Seq(tokenizer)
    def collator_fn(features):
        fst_reps = []
        for x in features:
            fst_reps.append(x["fst_rep"])
            del x["fst_rep"]
        d = seq2seq_collator(features)
        d["fst_rep"] = torch.from_numpy(batch_fsts(fst_reps, num_states, max_len=max_len))

        if "task_id" in features[0]:
            d["task_ids"] = torch.from_numpy(np.array([x["task_id"] for x in features]))

        return d

    return DataLoader(dataset, collate_fn=collator_fn, batch_sampler=ts)



class RandomSplit:

    def __init__(self, path: str, tokenizer: AutoTokenizer, num_train:int, train_batch_size, test_batch_size = None, lenient=True):
        def mapper(examples):
            d = tokenizer(examples["input"])
            if "output" in examples:
                d["labels"] = tokenizer(text_target=examples["output"])["input_ids"]
            return d

        keys = ["input", "output"]
        data = []
        for row in load_tsv(path, "input\toutput", lenient=lenient):
            data.append(row)
        print("Random number to verify seed", random.randint(0, 100_000_000), file=sys.stderr)
        random.shuffle(data)
        train_data = data[:num_train]
        rest_data = data[num_train:]

        train_dataset = Dataset.from_list([ {k: v for k,v in zip(keys, row)} for row in train_data])
        rest_dataset = Dataset.from_list([ {k: v for k,v in zip(keys, row)} for row in rest_data])

        sampler = SequentialSampler(train_dataset)
        ts = BatchSampler(sampler, batch_size=train_batch_size, drop_last=False)
        dataset = train_dataset.map(mapper, batched=True, remove_columns=["input", "output"])
        self.train_loader = DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)


        sampler = SequentialSampler(rest_dataset)
        ts = BatchSampler(sampler, batch_size=train_batch_size if test_batch_size is None else test_batch_size, drop_last=False)
        dataset = rest_dataset.map(mapper, batched=True, remove_columns=["input", "output"])
        self.rest_loader = DataLoader(dataset, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_sampler=ts)

    def get_train_loader(self):
        return self.train_loader

    def get_rest_loader(self):
        return self.rest_loader
