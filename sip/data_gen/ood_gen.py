
import pynini
from sip.data_gen.gen_pretrain import my_gen_random_fst, postprocess_for_sampling, vocab
import random

from sip.data_gen.recursion_limit import limit_recursion, count_state_visits, FasterFSTAccess
from sip.data_gen.unseen_combinations import gen_compgen_variants
from sip.data_gen.utils import one_step, gen_pair
from collections import Counter
from sip.data_gen.eval_automaton_gen import sample_fst_for_eval

class DataGen:
    def __init__(self, fst, chosen_vocab, vocab):
        self.fst = fst
        self.sampling_fst = postprocess_for_sampling(fst, chosen_vocab, vocab)
        self.train_length = one_step(vocab).closure(3, 15)
    def gen_train_ex(self) -> tuple[str, str]:
        raise NotImplementedError()

    def gen_test_ex(self) -> tuple[str, str]:
        raise NotImplementedError()

    @property
    def name(self):
        raise NotImplementedError()


class IidGen(DataGen):

    def __init__(self, fst, chosen_vocab, vocab):
        super().__init__(fst, chosen_vocab, vocab)
        self.train_length_restricted = pynini.compose(self.train_length, self.sampling_fst)

    def gen_train_ex(self):
        return gen_pair(self.train_length_restricted, seed=random.randint(0, 2**31-1))

    def gen_test_ex(self) -> tuple[str, str]:
        return self.gen_train_ex()

    @property
    def name(self):
        return "iid"


class LengthGen(DataGen):
    def __init__(self, fst, chosen_vocab, vocab):
        super().__init__(fst, chosen_vocab, vocab)
        k = 3 # deepest recursion allowed in training
        self.train_data, self.test_data = self.create_recursion_split(self.sampling_fst,
                                                                      vocab, range(2, 31), range(k+1), range(k+1, 31))
        random.shuffle(self.train_data)
        random.shuffle(self.test_data)
        self.train_index = 0
        self.test_index = 0

    @staticmethod
    def create_recursion_split(fst: pynini.Fst, vocab, length_range, train_range, test_range, samples=5_000):
        assert len(set(train_range) & set(test_range)) == 0

        train_data = []
        test_data = []
        fst_faster_access = FasterFSTAccess(fst)
        for l in length_range:
            length_l = pynini.compose(one_step(vocab).closure(l, l), fst)
            for _ in range(samples):
                i, o = gen_pair(length_l, seed=random.randint(0, 2 ** 31 - 1))
                max_recursion_count = max(count_state_visits(fst_faster_access, i))
                if max_recursion_count in train_range:
                    train_data.append((i, o))
                elif max_recursion_count in test_range:
                    test_data.append((i, o))
        return train_data, test_data

    def gen_train_ex(self) -> tuple[str, str]:
        i, o = self.train_data[self.train_index]
        self.train_index += 1
        return i, o

    def gen_test_ex(self) -> tuple[str, str]:
        i, o = self.test_data[self.test_index]
        self.test_index += 1
        return i, o

    @property
    def name(self):
        return "length"


class UnseenCombinationsGen(DataGen):
    def __init__(self, fst, chosen_vocab, vocab):
        super().__init__(fst, chosen_vocab, vocab)
        self.train_length = one_step(chosen_vocab).closure(5, 20)
        train_fst, test_fst, (self.l, self.r) = gen_compgen_variants(fst, 20)
        self.train_fst = pynini.compose(self.train_length, postprocess_for_sampling(train_fst, chosen_vocab, vocab))
        self.test_fst = pynini.compose(self.train_length, postprocess_for_sampling(test_fst, chosen_vocab, vocab))

    def gen_train_ex(self) -> tuple[str, str]:
        i, o = gen_pair(self.train_fst, seed=random.randint(0, 2**31-1))
        restr_i = pynini.compose(pynini.accep(i, token_type="utf8"), self.sampling_fst)
        test_fst_restr = pynini.compose(pynini.accep(i, token_type="utf8"), self.test_fst)
        assert restr_i.num_states() > 0
        assert test_fst_restr.num_states() == 0, "Training example should not be accepted by test FST"
        return i, o

    def gen_test_ex(self) -> tuple[str, str]:
        i, o = gen_pair(self.test_fst, seed=random.randint(0, 2**31-1))
        restr_i = pynini.compose(pynini.accep(i, token_type="utf8"), self.sampling_fst)
        train_fst_restr = pynini.compose(pynini.accep(i, token_type="utf8"), self.train_fst)
        assert restr_i.num_states() > 0
        assert train_fst_restr.num_states() == 0, "Test example should not be accepted by training FST"
        return i, o

    @property
    def name(self):
        return "uc"






