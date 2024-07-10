import transformers

import torch

from sip.embed_finetune import SIPFinetuningModel
from sip.data_loading import load_fst_jsonl, prepare_task_dataset
from sip.eval_tools import evaluate_on

import numpy as np
import random

def take_n(it, n):
    for i, x in enumerate(it):
        yield x
        if i == n:
            break

def rand_perm(length):
    l = list(range(length))
    random.shuffle(l)
    return torch.tensor(l, dtype=torch.int64)


if __name__ == "__main__":
    random.seed(42)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")
    
    for i in range(5):
        model = SIPFinetuningModel.from_pretrained(f"models/s4_{i}_length_full_ft/")
        
        model.eval()
        model = model.to(0)

        path = f"data/eval/task_s4_{i}_length_test.tsv"
        data = prepare_task_dataset(path, tokenizer, 48, random_order=False, lenient=True)

        n = np.inf
        # ~ n = 10
        print(path)
        orig_eval = evaluate_on(model, tokenizer, take_n(data, n))
        # ~ print("Evaluation with original prefix", )
        print(f"Eval\t{i}\torig\t"+"\t".join([str(x) for x in orig_eval]))

        # ~ print("Orig prefix", model.prefix_embedding)
        for _ in range(20):
            model.prefix_embedding = torch.nn.Parameter(model.prefix_embedding[:, rand_perm(model.prefix_length), :].detach())
            
            perm_eval = evaluate_on(model, tokenizer, take_n(data, n))
            
            print(f"Eval\t{i}\tperm\t"+"\t".join([str(x) for x in perm_eval]))

        # ~ print("Evaluating with permuted prefix", )


