import dataclasses
import json
import math
import os
import sys
from typing import List, Tuple, Set, Iterable

import random

import numpy as np

import pynini, pywrapfst

import time
import tqdm

from sip.data_gen.utils import gen_pair, one_step, FSTCollection, random_subset, replace_arc, fst_to_json

# use some ASCII control codes to take special meaning.
SYMBOL_ID = 17
SYMBOL_TO_UPPER = 18
SYMBOL_TO_LOWER = 19


def postprocess_for_sampling(fst: pynini.Fst, alphabet, total_vocab):
    """
    Replaces special symbols (SYMBOL_ID, SYMBOL_TO_UPPER, SYMBOL_TO_LOWER)
    by transitions that implement this.
    :param fst:
    :param alphabet:
    :return:
    """
    id_mapping = [(ord(a), ord(a)) for a in alphabet]
    upper_mapping = [(ord(a), ord(a.upper())) for a in alphabet if a != a.upper() and a in total_vocab]
    lower_mapping = [(ord(a), ord(a.lower())) for a in alphabet if a != a.lower() and a in total_vocab]

    fst = fst.copy()
    for state in fst.states():
        replace_arc(fst, state, SYMBOL_ID, id_mapping)
        replace_arc(fst, state, SYMBOL_TO_UPPER, upper_mapping)
        replace_arc(fst, state, SYMBOL_TO_LOWER, lower_mapping)

    return fst

def my_gen_random_fst(num_states, num_final, alphabet, add_sink = False, id_prob:float=0.25,
                      p_drop_letter: float = 0.0,
                      p_special : float = 0.0, make_initial_final: bool = False):
    fst = pynini.Fst()
    fst.add_states(num_states + int(add_sink))
    fst.set_start(0)
    states = list(range(num_states + int(add_sink)))
    # Make all states accepting for now except the initial state
    for state in range(0 if make_initial_final else 1, num_states):
        fst.set_final(state, 0)

    output_alphabet = [0] + [ord(c) for c in alphabet]
    input_alphabet = alphabet
    for s in range(num_states): #don't go over sink
        if random.random() < p_special:
            chosen_func = random.choice([SYMBOL_ID, SYMBOL_TO_LOWER, SYMBOL_TO_UPPER])
            fst.add_arc(s, pywrapfst.Arc(chosen_func, chosen_func, 0, random.choice(states)))
            continue
        for c in input_alphabet:
            dest = random.choice(states)
            if random.random() < p_drop_letter:
                continue

            if random.random() < id_prob:
                fst.add_arc(s, pywrapfst.Arc(ord(c), ord(c), 0, dest))
            else:
                fst.add_arc(s, pywrapfst.Arc(ord(c), random.choice(output_alphabet), 0, dest))

    # Trim the FST to get a connected graph
    fst.connect()

    not_final_weight = pynini.Weight("tropical", "infinity")
    # Make a random subset of size max(0, len(states) - num_final) non-final
    remaining_states = list(fst.states())
    for state in random_subset(remaining_states, max(0, len(remaining_states) - num_final)):
        fst.set_final(state, not_final_weight)

    copied_fst = fst.copy()

    fst = fst.minimize(allow_nondet=False)
    # Check that buggy? FST minimization didn't introduce any transitions that have epsilon on the input tape
    minimized_ok = True
    for state in fst.states():
        for arc in fst.arcs(state):
            if arc.ilabel == 0:
                minimized_ok = False
                break
            elif arc.ilabel in [SYMBOL_ID, SYMBOL_TO_LOWER, SYMBOL_TO_UPPER] and arc.ilabel != arc.olabel:
                minimized_ok = False
                break
    if not minimized_ok or fst.num_states() > copied_fst.num_states():
        fst = copied_fst

    return fst

vocab = [chr(x) for x in range(32, 127)]
vocab = vocab + [chr(i) for i in range(592, 687+1)] # add unicode characters for IPA symbols.
vocab = sorted(set(vocab))
#these characters have special meaning in OpenFST, and cannot be written into FAR
vocab.remove("]")
vocab.remove("[")
vocab.remove(chr(92)) # backslash, this messes things up as well!

if __name__ == "__main__":
    os.makedirs("data/pretrain", exist_ok=True)
    t0 = time.time()
    random.seed(55)


    print(vocab)

    fst_collection = FSTCollection()
    num_data_points = 50_000
    # num_data_points = 10_000
    num_fsts = 2*num_data_points
    num_ex_per_task = 5
    seeds = [random.randint(0, 100000000000) for _ in range(num_fsts)]

    DESIRED_MAX_STATES = 4

    name = f"pretrain_s{DESIRED_MAX_STATES}"

    max_num_states = 0

    for seed in tqdm.tqdm(seeds):
        num_states = random.randint(2, DESIRED_MAX_STATES)
        vocab_size = random.randint(5, 25)
        num_final = random.randint(1, num_states)
        my_vocab = list(vocab)
        random.shuffle(my_vocab)
        chosen_vocab = "".join(my_vocab[:vocab_size])
        fst = my_gen_random_fst(num_states, num_final, chosen_vocab,
                                p_special=0.15,
                                p_drop_letter=0.4, id_prob=0.2)

        max_num_states = max(max_num_states, fst.num_states())

        fst_for_sampling = postprocess_for_sampling(fst, chosen_vocab, vocab)
        for state in fst_for_sampling.states():
            for arc in fst_for_sampling.arcs(state):
                if arc.ilabel in [SYMBOL_ID, SYMBOL_TO_LOWER, SYMBOL_TO_UPPER] or arc.olabel in [SYMBOL_ID, SYMBOL_TO_LOWER, SYMBOL_TO_UPPER]:
                    raise ValueError()

        if fst.num_states() > 0 and not bool(fst.properties(pynini.ACYCLIC, True)):
            # only collect FSTs with cycles, so language is infinite
            fst_collection.maybe_add(fst, chosen_vocab)

        if len(fst_collection) > num_data_points:
            break

    fst_collection = fst_collection.to_list()
    random.shuffle(fst_collection)

    if len(fst_collection) < num_data_points:
        print(len(fst_collection), num_data_points)
        raise ValueError("fst collection not large enough")

    # split into train/dev/test

    collection_ids = list(range(min(num_data_points, len(fst_collection))))
    random.shuffle(collection_ids)

    num_train_ex = int(0.8 * len(collection_ids))
    num_easy_dev_ex = min(1000, num_train_ex)
    num_dev_ex = min(1000, int(0.1 * len(collection_ids)))
    num_test_ex = min(1000, int(0.1 * len(collection_ids)))

    length_restriction = one_step(vocab).closure(1, 35)

    curr_train = 0
    curr_dev = 0
    curr_test = 0
    curr_easy_dev = 0

    max_length_json = 0
    task_id = 0

    max_digits = len(str(len(fst_collection)))
    with (open(f"data/pretrain/train_{name}.jsonl", "w") as f_train,
          pynini.Far(f"data/pretrain/train_{name}.far", mode="w") as far_train,
          pynini.Far(f"data/pretrain/dev_{name}.far", mode="w") as far_dev,
          pynini.Far(f"data/pretrain/test_{name}.far", mode="w") as far_test,
          open(f"data/pretrain/dev_{name}.jsonl", "w") as f_dev,
          open(f"data/pretrain/easy_dev_{name}.jsonl", "w") as easy_dev_f,
          open(f"data/pretrain/test_{name}.jsonl", "w") as f_test):

        for fst, chosen_vocab in tqdm.tqdm(fst_collection):

            fst_for_sampling = postprocess_for_sampling(fst, chosen_vocab, vocab)
            train_fst = pynini.compose(length_restriction, fst_for_sampling)

            if train_fst.num_states() == 0:
                # Occasionally, this might happen, e.g. if the have a LOWER operation but no characters can be converted to lowercase (vocab is all symbols)
                # and this transition is the only way to get to a final state.
                continue

            task_id += 1

            fst_as_json = fst_to_json(fst)
            max_length_json = max(max_length_json, len(fst_as_json))
            data_points = []
            for _ in range(num_ex_per_task):
                inp, o = gen_pair(train_fst, seed=random.randint(0, 100000000000))
                # assert pynini.compose(pynini.accep(inp, token_type="utf8"), train_fst).num_states() > 0
                # assert pynini.compose(train_fst, pynini.accep(o, token_type="utf8")).num_states() > 0
                assert pynini.compose(pynini.compose(pynini.accep(inp, token_type="utf8"), train_fst), pynini.accep(o, token_type="utf8")).num_states() > 0

                data_points.append({"FST": fst_as_json, "input": inp, "output": o, "task_id": task_id})

            task_id_s = str(task_id)
            task_id_s = "0" * (max_digits - len(task_id_s)) + task_id_s

            if curr_train < num_train_ex:
                f = f_train
                curr_train += 1
                far = far_train
            elif curr_dev < num_dev_ex:
                f = f_dev
                curr_dev += 1
                far = far_dev
            elif curr_test < num_test_ex:
                f = f_test
                curr_test += 1
                far = far_test
            else:
                break

            far.add(task_id_s, fst)

            for datapoint in data_points:
                f.write(json.dumps(datapoint))
                f.write("\n")

            #If we are still generating training data, generate some easy dev examples (= known tasks but unkown strings) as well
            if curr_train <= num_train_ex and curr_easy_dev < num_easy_dev_ex:
                curr_easy_dev += 1
                inputs = set(datapoint["input"] for datapoint in data_points)
                excluded_inputs = pynini.union(*[pynini.accep(input, token_type="utf8") for input in inputs])

                sigma_star = one_step(chosen_vocab).closure()

                allowed_inputs = pynini.difference(sigma_star, excluded_inputs)
                easy_dev_fst = pynini.compose(allowed_inputs, train_fst)

                for _ in range(num_ex_per_task):
                    inp, o = gen_pair(easy_dev_fst, seed=random.randint(0, 100000000000))

                    assert inp not in inputs
                    assert pynini.compose(pynini.accep(inp, token_type="utf8"), train_fst).num_states() > 0

                    easy_dev_f.write(json.dumps({"FST": fst_as_json, "input": inp, "output": o, "task_id": task_id}))
                    easy_dev_f.write("\n")

    print("Max num. states", max_num_states)
    print("Max num. transitions", max_length_json)