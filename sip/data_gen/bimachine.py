import dataclasses
from typing import List, Tuple, Dict

import pynini
from pynini import Fst

import pywrapfst

from sip.data_gen.gen_pretrain import  my_gen_random_fst


def repr2str(s):
    if s == 0:
        return "ℇ"
    return repr(chr(s))

@dataclasses.dataclass
class Edge:
    l: int
    r: int
    sigma: int
    m: int
    l_: int
    r_: int

    def __repr__(self):
        return f"Edge(l={self.l}, r={self.r}, sigma={repr2str(self.sigma)}, m={repr2str(self.m)}, l_={self.l_}, r_={self.r_})"

def get_char2states(auto: Fst) -> Dict[int, List[Tuple[int, int]]]:
    char2states = dict()
    for s in auto.states():
        for arc in auto.arcs(s):
            if arc.ilabel not in char2states:
                char2states[arc.ilabel] = []
            char2states[arc.ilabel].append((s, arc.nextstate))
    return char2states
class Bimachine:

    def __init__(self, left_auto: Fst, right_auto: Fst, output_f: Dict[Tuple[int, int, int], int]):
        """

        :param left_auto:
        :param right_auto:
        :param output_f: maps l, sigma, r to an output symbol
        """
        self.left_auto = left_auto
        self.right_auto = right_auto

        # Make sure the automata are trim
        left_auto.connect()
        right_auto.connect()

        assert left_auto.properties(pynini.ACCEPTOR, True) and left_auto.properties(pynini.I_DETERMINISTIC, True), "left_auto must be a deterministic automaton"
        assert right_auto.properties(pynini.ACCEPTOR, True) and right_auto.properties(pynini.I_DETERMINISTIC, True), "right_auto must be a deterministic automaton"

        #TODO: check that all states are final

        self.output_f = output_f

        self.left_char2states = get_char2states(left_auto)
        self.right_char2states = get_char2states(right_auto)

    def to_fst(self):
        """
        Follows the straightforward construction in Proposition 6.2.1 in "Finite State Techniques"
        by Mihov and Schulz (https://doi.org/10.1017/9781108756945), also uses their notation.
        :return:
        """
        edges = []
        states = set()
        for sigma in self.left_char2states:
            for (l, l_) in self.left_char2states[sigma]: # l -> l_ by reading sigma in left auto
                if sigma in self.right_char2states:
                    for (r_, r) in self.right_char2states[sigma]: # r_ -> r by reading sigma in right auto
                        states.add((l, r))
                        states.add((l_, r_))
                        edges.append(Edge(l, r, sigma, self.output_f[l, sigma, r_], l_, r_))
        state2i = {s:i+1 for i,s in enumerate(states)}
        f = pynini.Fst()
        f.add_states(len(state2i)+1)
        f.set_start(0)
        for edge in edges:
            f.add_arc(state2i[(edge.l, edge.r)], pynini.Arc(edge.sigma, edge.m, 0.0, state2i[(edge.l_, edge.r_)]))

        right_initial = self.right_auto.start()
        left_initial = self.right_auto.start()

        for s, id in state2i.items():
            if s[0] == left_initial:
                # Make states initial that use left initial state
                # simulate this using epsilon transitions
                f.add_arc(0, pynini.Arc(0, 0, 0.0, id))

            # Make states final that use right initial state
            if s[1] == right_initial:
                f.set_final(id)

        f.rmepsilon()  # remove the epsilon transitions that we just introduced
        f.connect()

        return f

import random

def gen_f_output(n1, n2, vocab, p_id:float):
    d = dict()
    int_vocab = [ord(c) for c in vocab]
    for l in range(n1):
        for r in range(n2):
            for sigma in int_vocab:
                if random.random() < p_id:
                    d[l, sigma, r] = sigma
                else:
                    d[l, sigma, r] = random.choice(int_vocab + [0])
    return d


def gen_random_bimachine_fst(l_states: int, r_states: int, vocab, id_prob: float = 0.2, p_drop_letter:float=0.0):
    assert l_states > 0
    assert r_states > 0

    # TODO: using the special chars (SYMBOL_ID etc.) doesn't work well when sampling FSTs like this:
    # in order for SYMBOL_ID to make it through, it must have been sampled in both automata on compatible paths, which is quite unlikely
    # instead, we will usually get it in one automaton and the other doesn't accept it and then the bimachine accepts nothing
    # this has to be solved in the generation of the output function, or after converting the bimachine to an FST
    # (although the latter might be tricky to ensure the result remains a function!)

    left = my_gen_random_fst(l_states, l_states, vocab, p_special=0, p_drop_letter=p_drop_letter, make_initial_final=True).project("input")
    right = my_gen_random_fst(r_states, r_states, vocab, p_special=0, p_drop_letter=p_drop_letter, make_initial_final=True).project("input")

    output_f = gen_f_output(left.num_states(), right.num_states(), vocab, p_id=id_prob)

    return Bimachine(left, right, output_f)

if __name__ == "__main__":
    import sys

    random.seed(55122)
    bm = gen_random_bimachine_fst(5, 6, "abcdefg")

    bm.left_auto.draw("left.dot", portrait=True)
    bm.right_auto.draw("right.dot", portrait=True)

    fst = bm.to_fst()

    fst.draw("bimachine.dot", portrait=True)

    sys.exit()
    compiler = pywrapfst.Compiler()
    s=f"""
    0 1 {ord('a')} {ord('a')}
    1 2 {ord('b')} {ord('b')}
    0 0.0
    1 0.0
    2 0.0
    """
    print(s, file=compiler)

    left_auto = pynini.Fst.from_pywrapfst(compiler.compile())


    compiler = pywrapfst.Compiler()
    s=f"""
    0 1 {ord('a')} {ord('a')}
    0 1 {ord('b')} {ord('b')}
    1 1 {ord('a')} {ord('a')}
    1 1 {ord('b')} {ord('b')}
    0 0.0
    1 0.0
    """
    print(s, file=compiler)

    right_auto = pynini.Fst.from_pywrapfst(compiler.compile())

    f_output = {(0, ord("a"), 1): 0,
                (0, ord("a"), 0): ord("c"),
                (1, ord("b"), 1): 0,
                (1, ord("b"), 0): ord("d")}

    bimachine = Bimachine(left_auto, right_auto, f_output)

    fst = bimachine.to_fst()
    fst.draw("bimachine_fst.dot", portrait=True)


    ###########################################################################################

    compiler = pywrapfst.Compiler()
    s=f"""
    0 0 {ord('a')} {ord('a')}
    0 1 {ord('b')} {ord('b')}
    0 0.0
    1 0.0
    """
    print(s, file=compiler)

    left_auto = pynini.Fst.from_pywrapfst(compiler.compile())


    compiler = pywrapfst.Compiler()
    s=f"""
    0 1 {ord('a')} {ord('a')}
    0 2 {ord('b')} {ord('b')}
    1 1 {ord('a')} {ord('a')}
    2 2 {ord('a')} {ord('a')}
    0 0.0
    1 0.0
    2 0.0
    """
    print(s, file=compiler)

    right_auto = pynini.Fst.from_pywrapfst(compiler.compile())

    f_output = {(0, ord("b"), 0): ord("b"),
                (0, ord("a"), 0): ord("a"),
                (0, ord("a"), 1): ord("a"),
                (0, ord("a"), 2): 0}

    bimachine = Bimachine(left_auto, right_auto, f_output)

    fst = bimachine.to_fst()
    fst.draw("bimachine2_fst.dot", portrait=True)

    from utils import gen_pair

    compiler = pywrapfst.Compiler()
    s=f"""
    0 1 {ord('a')} {ord('b')}
    0 1 {ord('a')} {ord('c')}
    1 0 {ord('c')} {ord('a')}
    2 0 {ord('d')} {ord('a')}
    0 0.0
    1 0.0
    """
    print(s, file=compiler)

    weird_auto = pynini.Fst.from_pywrapfst(compiler.compile())

    weird_auto = pynini.compose("a", weird_auto)
    from collections import Counter
    counter = Counter()
    for i in range(10000):
        x,y = gen_pair(weird_auto, seed=i)
        counter.update([y])
    print(counter)
