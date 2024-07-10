from typing import List, Tuple

import pynini
import pywrapfst
from pywrapfst import Arc
from pynini import Fst
import pynini
import random

SYMBOL_EPSILON = 20
def fst_to_json(fst: pynini.Fst):
    s = []
    def arc_to_str(label: int):
        if label == 0:
            return chr(SYMBOL_EPSILON)
        return chr(label)

    assert fst.start() == 0

    for state in fst.states():
        for arc in fst.arcs(state):
            dest_is_final = int(pynini.Weight("tropical", "infinity") != fst.final(arc.nextstate)) + 1
            # final state => 2, non-final state => 1, (can use 0 for padding)
            s.append((state, arc_to_str(arc.ilabel), arc_to_str(arc.olabel), arc.nextstate, dest_is_final))
    return s


def outgoing_arc_ilabels(fst: pynini.Fst, state: int):
    for arc in fst.arcs(state):
        yield arc.ilabel

def replace_arc(fst: pynini.Fst, state, old_arc_ilabel, new_maps: List[Tuple[int, int]]):
    """
    If an arc with ilabel old_arc_ilabel is present at state, remove it and replace it by
    new arcs that go to the same state as the removed arc and use the mapping provided in new_maps.

    Operates in-place.
    :param fst:
    :param state:
    :param old_arc:
    :param new_arcs:
    :return:
    """
    backup_arcs = list(fst.arcs(state))
    fst.delete_arcs(state)
    target_state = None
    for arc in backup_arcs:
        if arc.ilabel == old_arc_ilabel:
            target_state = arc.nextstate
        else:
            fst.add_arc(state, arc)
    if target_state is not None:
        for ilabel, olabel in new_maps:
            fst.add_arc(state, pywrapfst.Arc(ilabel, olabel, 0, target_state))

def random_subset(s, n):
    s = list(s)
    random.shuffle(s)
    return s[:n]

def one_step(alphabet):
    fst = Fst()
    fst.add_state()
    fst.add_state()
    fst.set_start(0)
    fst.set_final(1, 0)
    for c in alphabet:
        fst.add_arc(0, Arc(ord(c), ord(c), 0, 1))
    return fst
def gen_pair(fst, token_type="utf8", **kwargs):
    """
    Generate an input/output pair from a transducer.
    :param fst:
    :param kwargs:
    :return:
    """
    output_auto = pynini.randgen(fst, **kwargs)
    input_auto = pynini.randgen(output_auto.copy().invert(), **kwargs)
    output_str = output_auto.string(token_type=token_type)
    input_str = input_auto.string(token_type=token_type)
    return input_str, output_str

def graph_hash(fst):
    """
    Returns a hash of an fst that is invariant to the numbering of the states
    :param fst:
    :return:
    """
    s = 0
    for state in fst.states():
        for arc in fst.arcs(state):
            s += fst.num_arcs(state) * (hash((arc.ilabel, arc.olabel)))
    return s
class FSTCollection:
    def __init__(self):
        self.hash2fst = dict()
        self.size = 0

    def maybe_add(self, f: Fst, vocab = None):
        hash = graph_hash(f)
        if hash in self.hash2fst:
            if vocab is not None:
                if not any(pywrapfst.isomorphic(graph, f) for graph, _ in self.hash2fst[hash]):
                    self.hash2fst[hash].append((f, vocab))
            else:
                if not any(pywrapfst.isomorphic(graph, f) for graph in self.hash2fst[hash]):
                    self.hash2fst[hash].append(f)
                    self.size += 1
        else:
            if vocab is not None:
                self.hash2fst[hash] = [(f, vocab)]
            else:
                self.hash2fst[hash] = [f]
            self.size += 1

    def __contains__(self, element):
        hash = graph_hash(element)
        if hash not in self.hash2fst:
            return False
        for obj in self.hash2fst[hash]:
            if isinstance(obj, tuple):
                graph, _ = obj
                if pywrapfst.isomorphic(graph, element):
                    return True
            else:
                if pywrapfst.isomorphic(obj, element):
                    return True
        # hash known but object unseen
        return False

    def __len__(self):
        return self.size

    def to_list(self):
        s = []
        for graphs in self.hash2fst.values():
            s.extend(graphs)
        return s

def restrict_input(fst, a):
    """
    Restrict the input of fst to the automaton a
    """
    return a @ fst
def write_tsv(fname, data):
    with open(fname, "w") as f:
        f.write("input\toutput\n")
        for x,y in data:
            f.write(x)
            f.write("\t")
            f.write(y)
            f.write("\n")