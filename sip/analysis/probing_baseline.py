# Appendix D.2 of the paper.

from sip.analysis.state_probe import STATE_PROBE_TEST_DATA
from sip.data_gen.gen_probing_data import postprocess_for_sampling
from sip.data_gen.recursion_limit import FasterFSTAccess, identify_state_sequence
from sip.data_gen.utils import SYMBOL_EPSILON

from pynini import Fst, Arc
import json

import random

def str_to_arc(s: str):
    if s == chr(SYMBOL_EPSILON):
        return 0
    return ord(s)

def json_to_fst(j: list):
    fst = Fst()
    fst.add_state()
    fst.set_start(0)
    for state, ilabel, olabel, nextstate, dest_is_final in j:
        if state >= fst.num_states():
            fst.add_state()
    for state, ilabel, olabel, nextstate, dest_is_final in j:
        fst.add_arc(state, Arc(str_to_arc(ilabel), str_to_arc(olabel), 0, nextstate))
        if dest_is_final == 2:
            fst.set_final(nextstate, 0)
    return fst


def naive_guess_fst_state(fst: Fst, s: str):
    """
    Makes a naive guess for each position, what state the FST is in by randomly selecting a state that has an incoming arc
    that is compatible with the current character.
    :param fst:
    :param s:
    :return:
    """
    state = fst.start()
    states = [state]
    for c in s:
        options = set()
        for state in fst.states():
            for arc in fst.arcs(state):
                if arc.ilabel == ord(c):
                    options.add(arc.nextstate)
        states.append(random.choice(sorted(options)))
    return states


vocab = [chr(x) for x in range(32, 127)]
# Only use characters that take a single token when tokenized with ByT5, so don't include
# [chr(i) for i in range(592, 687+1)]
vocab = sorted(set(vocab))
#these characters have special meaning in OpenFST, and cannot be written into FAR
vocab.remove("]")
vocab.remove("[")
vocab.remove(chr(92)) # backslash, this messes things up as well!



if __name__ == "__main__":
    random.seed(3454)
    total = 0
    correct = 0
    total_sent = 0
    total_correct = 0

    with open(STATE_PROBE_TEST_DATA) as f:
        for line in f:
            j = json.loads(line)
        # line = next(iter(f))
        # j = json.loads(line)
            fst = json_to_fst(j["FST"])
            post_processed_fst = postprocess_for_sampling(fst, vocab, vocab)
            faster_access_fst = FasterFSTAccess(post_processed_fst)
            gold_state_sequence = identify_state_sequence(faster_access_fst, j["input"])
            naive_guess = naive_guess_fst_state(post_processed_fst, j["input"])
            assert len(naive_guess) == len(gold_state_sequence)
            total += len(gold_state_sequence)
            correct += sum(x == y for x, y in zip(gold_state_sequence, naive_guess))
            total_sent += 1
            total_correct += (gold_state_sequence == naive_guess)
    print("Token acc", correct/total)
    print("Seq acc", total_correct/total_sent)



        # fst.draw("output_fst.dot", portrait=True)