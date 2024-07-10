import Levenshtein
import transformers

import random

from sip.fst_pretrain import FSTPretrainingModel, SIPPreTrainingModel
from sip.data_loading import load_fst_jsonl

import torch


def compute_agreement(model, tokenizer, data1, data2):
  correct, total, edit_dist, per = 0,0,0,0
  model.eval()

  def compute_batch_output(model, test_batch):
      test_batch = {k: v.to(model.device) for k, v in test_batch.items()}
      test_batch_inputs = dict(test_batch)
      del test_batch_inputs["labels"]
      r = tokenizer.batch_decode(model.generate(**test_batch_inputs, max_new_tokens=test_batch["labels"].shape[1] + 2,
                                                early_stopping="never", num_beams=1, no_repeat_ngram_size=0),
                                 skip_special_tokens=True)
      return r

  for test_batch1, test_batch2 in zip(data1, data2):

    r1 = compute_batch_output(model, test_batch1)
    r2 = compute_batch_output(model, test_batch2)

    assert len(r1) == len(r2)
    correct += sum( [x == y for x,y in zip(r1, r2)])
    total += len(r1)
    edit_dist += sum(Levenshtein.distance(x,y) for x,y in zip(r1, r2))
    per += sum(Levenshtein.distance(x,y)/max(1, len(y)) for x,y in zip(r1, r2))

    print(correct / total)

  return correct/total, edit_dist/total, per/total


def permute_states(transitions):
    states = set()
    for state, ilabel, olabel, nextstate, dest_is_final in transitions:
        states.add(state)
        states.add(nextstate)

    states = sorted(states)
    assert states[0] == 0
    states = states[1:]  # remove the first element (which is always 1)
    orig_order = list(states)
    for _ in range(10):
        random.shuffle(states)
        if states != orig_order:  # we got a different order
            break
    # We want to ensure that state 0 is again assigned the id 0, because that marks it as the start symbol

    old2new = {old: new for new, old in enumerate([0] + states)}
    assert old2new[0] == 0
    new_transitions = []
    for state, ilabel, olabel, nextstate, dest_is_final in transitions:
        new_transitions.append((old2new[state], ilabel, olabel, old2new[nextstate], dest_is_final))
    return new_transitions


def permute_transitions(transitions):
    l = list(transitions)
    random.shuffle(l)
    return l


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    random.seed(2347823)
    model = SIPPreTrainingModel.from_pretrained("namednil/sip-d4-pt")
    model.eval()
    model = model.to(0)
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")

    fst_tokenizer = transformers.AutoTokenizer.from_pretrained("namednil/sip-fst-tokenizer")

    path = "data/pretrain/dev_pretrain_s4.jsonl"
    batch_size = 32
    max_n = None
    data_1 = load_fst_jsonl(path, fst_tokenizer=fst_tokenizer,
                            tokenizer=tokenizer,
                            batch_size=batch_size,
                            num_states=15,
                            random_order=False,
                            max_n=max_n
                            )

    data_perm_states = load_fst_jsonl(path, fst_tokenizer=fst_tokenizer,
                                      tokenizer=tokenizer,
                                      batch_size=batch_size,
                                      num_states=15,
                                      random_order=False,
                                      map_f=permute_states,
                                      max_n=max_n
                                      )
                            
    data_perm_transitions = load_fst_jsonl(path, fst_tokenizer=fst_tokenizer,
                                           tokenizer=tokenizer,
                                           batch_size=batch_size,
                                           num_states=15,
                                           random_order=False,
                                           map_f=permute_transitions,
                                           max_n=max_n
                                           )
                            
    data_perm_both = load_fst_jsonl(path, fst_tokenizer=fst_tokenizer,
                                    tokenizer=tokenizer,
                                    batch_size=batch_size,
                                    num_states=15,
                                    random_order=False,
                                    map_f=lambda x: permute_transitions(permute_states(x)),
                                    max_n=max_n
                                    )

    print("Agreement perm states", compute_agreement(model, tokenizer, data_1, data_perm_states))
    print("Agreement perm transitions", compute_agreement(model, tokenizer, data_1, data_perm_transitions))
    print("Agreement perm both", compute_agreement(model, tokenizer, data_1, data_perm_both))


# Results with:
# ~ Running provided command: python -m sip.analysis.agreement
# ~ Agreement perm states (0.9704, 0.0588, 0.0026706743879092863)
# ~ Agreement perm transitions (0.9832, 0.0278, 0.001388993852496627)
# ~ Agreement perm both (0.9712, 0.0586, 0.002692758534095434)

