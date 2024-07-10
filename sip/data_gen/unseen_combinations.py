import dataclasses
import itertools
import math
import sys
from typing import List, Tuple, Set, Iterable

import random

import pynini, pywrapfst

from sip.data_gen.eval_automaton_gen import sample_fst_for_eval
from sip.data_gen.utils import restrict_input


@dataclasses.dataclass(frozen=True)
class FSTEdge:
    source: int
    ilabel: int
    olabel: int
    target: int


def dfs(fst: pynini.Fst) -> Tuple[Set[FSTEdge], Set[FSTEdge]]:
    initial_edge = FSTEdge(None, None, None, fst.start())
    agenda = [initial_edge]
    seen = set()
    used = set()
    all_edges = set()
    while agenda:
        edge = agenda.pop()
        current = edge.target
        if current in seen:
            continue
        seen.add(current)
        used.add(edge)
        children = []
        for arc in fst.arcs(current):
            children.append(FSTEdge(current, arc.ilabel, arc.olabel, arc.nextstate))
        # put those edges first that don't do anything interesting so they are explored first by the DFS
        # (enables us to withhold interesting combinations more often)
        children.sort(key=lambda arc: arc.ilabel == arc.olabel)
        agenda.extend(children)
        all_edges.update(children)
    used.remove(initial_edge)

    return used, all_edges

def random_edge_pairs(crucial_edges: Set[FSTEdge], all_edges: Set[FSTEdge], exclude_self_loops: bool, n: int) -> Iterable[Tuple[FSTEdge, FSTEdge]]:
    """
    Randomly picks adjacent edges (e.g. q0 -> q1 -> q2) from among the non-crucial edges
    :param crucial_edges: Edges which may not be excluded and which will not appear in pairs
    :param all_edges: all edges of FST
    :param exclude_self_loops:
    :return:
    """
    edges_into = dict()
    edges_out_of = dict()
    nodes = set()
    for edge in all_edges - crucial_edges:
        for node, d in itertools.product([edge.target, edge.source], [edges_into, edges_out_of]):
            if node not in d:
                d[node] = []
        if exclude_self_loops and edge.source == edge.target:
            continue
        edges_into[edge.target].append(edge)
        edges_out_of[edge.source].append(edge)
        nodes.add(edge.source)
        nodes.add(edge.target)
    #We have to ensure that each edge that we exclude is only excluded on one side
    # otherwise it's excluded from both FSTs and doesn't contribute to the training data at all
    node_list = sorted(nodes)
    left_set = set()
    right_set = set()
    changed = True
    i = 0
    while changed and i < n:
        changed = False
        for node in node_list:
            if i >= n:
                break
            edges_into[node] = [n for n in edges_into[node] if n not in right_set]
            edges_out_of[node] = [n for n in edges_out_of[node] if n not in left_set]
            if len(edges_into.get(node, [])) > 0 and len(edges_out_of.get(node, [])) > 0:
                changed = True
                i += 1
                random.shuffle(edges_into[node])
                random.shuffle(edges_out_of[node])
                left_edge, right_edge = edges_into[node][0], edges_out_of[node][0]
                left_set.add(left_edge)
                right_set.add(right_edge)
                yield left_edge, right_edge



def split_by_arcs(fst: pynini.Fst, arcs1: List[FSTEdge], arcs2: List[FSTEdge]) -> Tuple[pynini.Fst, pynini.Fst]:
    """
    Creates copies for each element of arcs. In that copy the respective arc is removed.
    ASSUMES UNWEIGHTED AUTOMATON (because weights are not hashable)
    :param fst:
    :param arcs:
    :return:
    """
    copies = []
    for _ in range(2):
        copy = pynini.Fst()

        copy.add_states(fst.num_states())
        copy.set_start(fst.start())
        # Set final weights
        for s in fst.states():
            copy.set_final(s, fst.final(s))

        copies.append(copy)

    all_arcs = {FSTEdge(state, arc.ilabel, arc.olabel, arc.nextstate) for state in fst.states() for arc in fst.arcs(state)}

    for copy, excluded_arcs in zip(copies, [arcs1, arcs2]):
        for edge in all_arcs:
            if edge not in excluded_arcs:
                copy.add_arc(edge.source, pywrapfst.Arc(edge.ilabel, edge.olabel, 0, edge.target))

    return tuple(copies)


class NoUnseenCombinationsPossibleException(BaseException):
    pass
def gen_compgen_variants(fst, max_bigrams):

    # Identify crucial edges (not unique) that cannot be deleted
    crucial_edges, all_edges = dfs(fst)
    # print("crucial edges", crucial_edges)
    # Randomly select up to k pairs of edges to be removed
    edge_pairs = list(random_edge_pairs(crucial_edges, all_edges, True, max_bigrams))
    if len(edge_pairs) == 0:
        raise NoUnseenCombinationsPossibleException()
    l, r = tuple(zip(*edge_pairs))

    #Create two copies of the fst, one with the edges in l removed, the other with edges from r removed
    train_1, train_2 = split_by_arcs(fst, l, r)
    # train_1.connect()
    # train_2.connect()
    train_fst = pynini.union(train_1, train_2)

    # Create test fst that accepts strings that are in the original FST
    # but not in the train fst, i.e. it uses unseen combinations of transitions.
    test_lang = pynini.project(fst, "input")
    test_lang = pynini.difference(test_lang, pynini.project(train_fst, "input"))

    test_fst = restrict_input(fst, test_lang)

    return train_fst, test_fst, (l, r)


if __name__ == "__main__":
    random.seed(9367442)

    fst, chosen_vocab = sample_fst_for_eval(4)
    fst.draw("orig.dot", portrait=True)
    train_fst, test_fst, (l, r) = gen_compgen_variants(fst, 3)
    print("l", l)
    print("r", r)
    train_fst.draw("train.dot", portrait=True)
    test_fst.draw("test.dot", portrait=True)