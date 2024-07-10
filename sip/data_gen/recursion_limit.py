import dataclasses

import pynini
import pywrapfst

from sip.data_gen.utils import one_step, gen_pair


@dataclasses.dataclass(frozen=True)
class CountingState:
    main_state: int
    counts: tuple[int]
    own_state_id: int


def inc_index(tupl: tuple[int], i) -> tuple[int]:
    l = list(tupl)
    l[i] += 1
    return tuple(l)

def limit_recursion(fst: pynini.Fst, max_depth: int) -> pynini.Fst:
    """
    Takes an FST and limits its recursion to max_depth, i.e. it will only transduce (or accept) strings
    such that when run the original FST, no state will be encountered more than max_depth.

    NOTE: the output depends on the FST, not only on the language it represents,
    e.g. (ab)* and ((ab)*)* behave differently here but represent the same language
    :param fst:
    :param max_depth:
    :return:
    """
    new_fst = pynini.Fst()
    initial_state = new_fst.add_state()
    new_fst.set_start(initial_state)

    agenda = [CountingState(fst.start(), fst.num_states() * (0,) , initial_state)]
    while agenda:
        current = agenda.pop()

        if any(c >= max_depth for c in current.counts):
            # maximum depth reached, don't pursue this any further
            continue

        for arc in fst.arcs(current.main_state):
            target_state = CountingState(arc.nextstate, inc_index(current.counts, current.main_state), new_fst.add_state())
            new_fst.set_final(target_state.own_state_id, fst.final(arc.nextstate))
            new_fst.add_arc(current.own_state_id, pywrapfst.Arc(arc.ilabel, arc.olabel, arc.weight, target_state.own_state_id))
            agenda.append(target_state)
    new_fst.connect()

    # Sanity check: the domain of the new_fst must be a subset of the domain of the old fst
    should_be_empty = pynini.difference(pynini.project(new_fst, "input"), pynini.project(fst, "input"))
    assert should_be_empty.num_states() == 0

    return new_fst


class FasterFSTAccess:
    def __init__(self, fst: pynini.Fst):
        self.fst = fst
        self.fst_dict = [dict() for _ in range(fst.num_states())]
        for state in fst.states():
            for arc in fst.arcs(state):
                if arc.ilabel not in self.fst_dict[state]:
                    self.fst_dict[state][arc.ilabel] = set()
                self.fst_dict[state][arc.ilabel].add(arc.nextstate)

    def get_successors(self, state, c):
        try:
            return self.fst_dict[state][c]
        except KeyError:
            return set()

def count_state_visits(fst: FasterFSTAccess, s):
    """
    For each state, counts how many times it is visited by the input. Assumes a functional FST.
    :param fst:
    :param s:
    :return:
    """
    if isinstance(s, str):
        s = [ord(c) for c in s]
    # The FST might not be deterministic, so go forward, identify at which state we stopped and then track backwards
    curr_state_set = {fst.fst.start()}
    backpointers = dict()
    for i, c in enumerate(s):
        new_state_set = set()
        for state in curr_state_set:
            for new_state in fst.get_successors(state, c):
                new_state_set.add(new_state)
                backpointers[(i, new_state)] = state
        curr_state_set = new_state_set

    remaining_states = [state for state in curr_state_set if fst.fst.final(state) != pynini.Weight("tropical", "infinity")]
    assert len(remaining_states) > 0, "FST doesn't accept the string"
    assert len(remaining_states) == 1, "FST doesn't seem to be functional"

    state = remaining_states[0]
    counter = fst.fst.num_states() * [0]
    counter[fst.fst.start()] = 1
    for i in range(len(s)-1, -1, -1):
        counter[state] += 1
        state = backpointers[(i, state)]
    return counter

import random

def identify_state_sequence(fst: FasterFSTAccess, s):
    """
    Runs the string through the FST and returns the sequence of states that were visited.
    Assumes a functional FST.
    :param fst:
    :param s:
    :return:
    """
    if isinstance(s, str):
        s = [ord(c) for c in s]
    # The FST might not be deterministic, so go forward, identify at which state we stopped and then track backwards
    curr_state_set = {fst.fst.start()}
    backpointers = dict()
    for i, c in enumerate(s):
        new_state_set = set()
        for state in curr_state_set:
            for new_state in fst.get_successors(state, c):
                new_state_set.add(new_state)
                backpointers[(i, new_state)] = state
        curr_state_set = new_state_set

    remaining_states = [state for state in curr_state_set if fst.fst.final(state) != pynini.Weight("tropical", "infinity")]
    assert len(remaining_states) > 0, "FST doesn't accept the string"
    assert len(remaining_states) == 1, "FST doesn't seem to be functional"

    state = remaining_states[0]
    state_seq = [state]
    for i in range(len(s)-1, -1, -1):
        state = backpointers[(i, state)]
        state_seq.append(state)
    return list(reversed(state_seq))




if __name__ == "__main__":
    f = pynini.Fst()
    f.add_states(2)
    f.set_start(0)
    f.set_final(1)
    f.add_arc(0, pywrapfst.Arc(ord("a"), ord("a"), 0, 0))
    f.add_arc(0, pywrapfst.Arc(ord("a"), ord("b"), 0, 1))
    # assert count_state_visits(FasterFSTAccess(f), "aaa") == [3, 1]
    print(identify_state_sequence(FasterFSTAccess(f), "aaa"))

    # f = pynini.union("xyz", pynini.closure(pynini.union("ab", "dc"), 1))
    # f.rmepsilon()
    # f.minimize()
    #
    # l = limit_recursion(f, 2)
    #
    # assert ("abab" @ l).num_states() > 0
    # assert ("abdc" @ l).num_states() > 0
    # assert ("ababab" @ l).num_states() == 0
    # assert ("abdcab" @ l).num_states() == 0
    #
    # f = pynini.union("xyz", pynini.closure(pynini.union(pynini.cross("ab", "xy"), "dc"), 1)).closure(1)
    #
    # l = limit_recursion(f, 2)
    #
    # assert ("abab" @ l).num_states() > 0
    # assert ("abdc" @ l).num_states() > 0
    # assert ("abababab" @ l).num_states() > 0
    # assert ("ababababab" @ l).num_states() == 0
    #
    # f.draw("limit_rec_full.dot", portrait=True)
    # l.draw("limit_rec.dot", portrait=True)

