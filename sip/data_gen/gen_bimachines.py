import dataclasses
import json
import math
import sys
from typing import List, Tuple, Set, Iterable

import random

import numpy as np

import pynini, pywrapfst

import time
import tqdm

from sip.data_gen.bimachine import gen_random_bimachine_fst
from sip.data_gen.utils import gen_pair, one_step, FSTCollection, random_subset, replace_arc, fst_to_json

# use some ASCII control codes to take special meaning.


vocab = [chr(x) for x in range(32, 127)]
vocab = vocab + [chr(i) for i in range(592, 687+1)] # add unicode characters for IPA symbols.
vocab = sorted(set(vocab))
#these characters have special meaning in OpenFST, and cannot be written into FAR
vocab.remove("]")
vocab.remove("[")
vocab.remove(chr(92)) # backslash, this messes things up as well!


def remap_states_to_initial_0(fst: pynini.Fst):
    """
    Creates a copy of the FST where 0 is the initial state if necessary. Otherwise return original FST.
    :param fst:
    :return:
    """
    if fst.start() == 0:
        return fst

    old2new = dict()
    for i in fst.states():
        if i == fst.start():
            old2new[i] = 0
        elif i < fst.start():
            old2new[i] = i + 1
        else:
            old2new[i] = i

    f = pynini.Fst()
    f.add_states(fst.num_states())
    f.set_start(0)
    for s in fst.states():
        for arc in fst.arcs(s):
            f.add_arc(old2new[s], pywrapfst.Arc(arc.ilabel, arc.olabel, arc.weight, old2new[arc.nextstate]))
        f.set_final(old2new[s], fst.final(s))
    return f



if __name__ == "__main__":
    t0 = time.time()
    random.seed(55)

    print(vocab)

    fst_collection = FSTCollection()
    # num_data_points = 50_000
    num_data_points = 50_000
    num_fsts = 2*num_data_points
    num_ex_per_task = 5
    seeds = [random.randint(0, 100000000000) for _ in range(num_fsts)]

    DESIRED_MAX_STATES = 7
    max_states = DESIRED_MAX_STATES-1

    name = f"pretrain_bimachines_s{DESIRED_MAX_STATES}"

    max_num_states = 0

    for seed in tqdm.tqdm(seeds):
        num_l_states = random.randint(2, max_states // 2)
        num_r_states = random.randint(2, max_states // num_l_states)
        if random.random() < 0.5:
            num_l_states, num_r_states = num_r_states, num_l_states

        vocab_size = random.randint(5, 25)
        my_vocab = list(vocab)
        random.shuffle(my_vocab)
        chosen_vocab = "".join(my_vocab[:vocab_size])

        fst = gen_random_bimachine_fst(num_l_states, num_r_states, chosen_vocab,
                                       p_drop_letter=0.6, id_prob=0.2).to_fst()
        if fst.num_states() > 0 and not bool(fst.properties(pynini.ACYCLIC, True)):
            # only collect FSTs with cycles, so language is infinite
            fst = remap_states_to_initial_0(fst)   # doesn't to be actually used but we'll keep it for peace of mind.

            max_num_states = max(max_num_states, fst.num_states())

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
    prefix = "data/pretrain/bimachines"
    with (open(f"{prefix}/train_{name}.jsonl", "w") as f_train,
          pynini.Far(f"{prefix}/train_{name}.far", mode="w") as far_train,
          pynini.Far(f"{prefix}/dev_{name}.far", mode="w") as far_dev,
          pynini.Far(f"{prefix}/test_{name}.far", mode="w") as far_test,
          open(f"{prefix}/dev_{name}.jsonl", "w") as f_dev,
          open(f"{prefix}/easy_dev_{name}.jsonl", "w") as easy_dev_f,
          open(f"{prefix}/test_{name}.jsonl", "w") as f_test):

        for fst, chosen_vocab in tqdm.tqdm(fst_collection):

            train_fst = pynini.compose(length_restriction, fst)

            if train_fst.num_states() == 0:
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