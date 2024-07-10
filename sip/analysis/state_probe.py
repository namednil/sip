import json
from collections import namedtuple
from typing import Iterator, Tuple, Union

import numpy as np
import torch.nn
import transformers
from datasets import Dataset
from torch.nn import Parameter
from torch.utils.data import RandomSampler, SequentialSampler, BatchSampler, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForSeq2Seq

from sip.fst_pretrain import FSTPretrainingModel, SIPPreTrainingModel
from sip.data_loading import fst_to_vector, batch_fsts

import torch
import tqdm

def load_fst_jsonl_for_probing(path: str, tokenizer: AutoTokenizer, fst_tokenizer: Union[transformers.PreTrainedTokenizer, str], batch_size:int, num_states: int, random_order: bool = True,
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

    data = {"input": [], "output": [], "fst_rep": [], "task_ids": [], "state_seq": []}
    with open(path) as f:
        i = 0
        for line in tqdm.tqdm(f):
            d = json.loads(line)
            if filter_f is None or filter_f(d):
                data["input"].append(d["input"])
                data["output"].append(d["output"])

                data["state_seq"].append(d["state_seq"])

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
        state_seqs = []
        for x in features:
            fst_reps.append(x["fst_rep"])
            state_seqs.append(torch.from_numpy(np.array(x["state_seq"])))
            del x["fst_rep"]
            del x["state_seq"]
        d = seq2seq_collator(features)
        d["fst_rep"] = torch.from_numpy(batch_fsts(fst_reps, num_states, max_len=max_len))

        max_state_seq_length = max(len(x) for x in state_seqs)
        state_seq = torch.zeros((len(state_seqs), max_state_seq_length), dtype=torch.long)-100
        for i in range(len(state_seqs)):
            state_seq[i, :len(state_seqs[i])] = state_seqs[i]
        d["state_seq"] = state_seq

        if "task_id" in features[0]:
            d["task_ids"] = torch.from_numpy(np.array([x["task_id"] for x in features]))

        return d

    return DataLoader(dataset, collate_fn=collator_fn, batch_sampler=ts)



ProbeOutput = namedtuple('ProbeOutput', ['logits', 'loss'])

class StateProbe(torch.nn.Module):
    def __init__(self, max_num_states: int, model_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_layer = torch.nn.Linear(model_dim, max_num_states)
        self.loss = torch.nn.CrossEntropyLoss()

    def save_pretrained(self, path):
        torch.save(self.output_layer, path)

    @staticmethod
    def from_pretrained(path, **kwargs):
        output_layer = torch.load(path, **kwargs)
        probe = StateProbe(output_layer.out_features, output_layer.in_features)
        probe.output_layer = output_layer
        return probe

    def forward(self, data, model_output):
        state_seq = data.get("state_seq", None)
        input_ids = data["input_ids"] #shape (batch, seq_len)
        # grab the last hidden states corresponding to the input_ids but drop the last index
        # activations_on_input_ids = model_output.encoder_last_hidden_state[:, -input_ids.shape[1]:-1]  #remove EOS token at the end
        activations_on_input_ids = model_output.encoder_last_hidden_state[:, -input_ids.shape[1]:]  #don't remove the EOS token at the end.
        # assert activations_on_input_ids.shape[1] == input_ids.shape[1]-1
        activations = self.output_layer(activations_on_input_ids)

        loss = None
        if state_seq is not None:
            reshaped_activations = torch.einsum("bic -> bci", activations)
            # loss = self.loss(reshaped_activations, state_seq[:, 1:]) # cut off the initial state, which is always 0
            loss = self.loss(reshaped_activations, state_seq) # cut off the initial state, which is always 0

            # print("acc est", (torch.argmax(activations, dim=-1) == state_seq[:, 1:]).float().sum() / (state_seq[:, 1:] != -100).float().sum())
            # print("acc est", (torch.argmax(activations, dim=-1) == state_seq).float().sum() / (state_seq != -100).float().sum())

        return ProbeOutput(activations, loss)

STATE_PROBE_TRAIN_DATA = "data/pretrain/analysis/probing_train_pretrain_s4.jsonl"
STATE_PROBE_TEST_DATA = "data/pretrain/analysis/probing_test_pretrain_s4.jsonl"

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")

    fst_tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained("namednil/sip-fst-tokenizer")

    train_data_loader = load_fst_jsonl_for_probing(STATE_PROBE_TRAIN_DATA,
                                                   tokenizer, fst_tokenizer,
                                                  32, 5, random_order=True) # original experiments used batch size 64 but might not fit into memory

    test_data_loader = load_fst_jsonl_for_probing(STATE_PROBE_TEST_DATA,
                                                  tokenizer, fst_tokenizer,
                                                   32, 5, random_order=True)

    model = SIPPreTrainingModel.from_pretrained("namednil/sip-d4-pt")

    probe = StateProbe(5, model.model.get_output_embeddings().in_features)

    model.eval()
    
    model = model.to(0)
    probe = probe.to(0)
    
    optim = torch.optim.Adagrad(probe.parameters(), lr=0.1)

    # print(next(iter(data_loader)))
    for batch in train_data_loader:
        batch = {k: v.to(model.device) for k,v in batch.items()}
        batch_without_state_seq = dict(batch)
        del batch_without_state_seq["state_seq"]
        model_output = model(**batch_without_state_seq)
        r = probe(batch, model_output)
        print(r.loss)
        r.loss.backward()

        optim.step(closure=lambda: probe(batch, model_output).loss)
        optim.zero_grad()


    total = 0
    correct = 0
    all_correct = 0
    num_examples = 0

    with torch.no_grad():
        for batch in test_data_loader:
            batch =	{k: v.to(0) for k,v in batch.items()}
            batch_without_state_seq = dict(batch)
            del batch_without_state_seq["state_seq"]
            model_output = model(**batch_without_state_seq)
            r = probe(batch, model_output)
            preds = torch.argmax(r.logits, dim=-1)
            total += (batch["state_seq"] != -100).sum()
            correct += (preds == batch["state_seq"]).sum()

            all_correct += torch.all((preds == batch["state_seq"]) | (batch["state_seq"] == -100), dim=-1).sum()
            num_examples += batch["state_seq"].shape[0]

    print("State probe accuracy", correct.cpu().numpy()/total.cpu().numpy() * 100, "%")
    print("Sequence acc", all_correct.cpu().numpy() / num_examples * 100, "%")
    #Output:
    #State probe accuracy 99.28491090984431 %
    #Sequence acc 93.8928210313448 %
    
    probe.save_pretrained("models/probe_states_s4_32.pt")
    

