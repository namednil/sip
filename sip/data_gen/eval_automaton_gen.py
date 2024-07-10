import pynini
import random

from sip.data_gen.bimachine import gen_f_output, Bimachine
from sip.data_gen.gen_pretrain import my_gen_random_fst

# vocabulary: printable ascii characters
vocab = [chr(x) for x in range(32, 127)]
vocab = sorted(set(vocab))
# these characters have special meaning in OpenFST, and cannot be written into FAR
vocab.remove("]")
vocab.remove("[")
vocab.remove(chr(92))  # backslash, this messes things up as well!

def sample_fst_for_eval(num_states, attempts: int = 50) -> tuple[pynini.Fst, str]:
    # vocab_size = 4 # for debugging purposes
    vocab_size = 25 # fix vocab size, to reduce variance
    num_final = random.randint(1, num_states)
    my_vocab = list(vocab)
    random.shuffle(my_vocab)
    chosen_vocab = "".join(my_vocab[:vocab_size])
    for _ in range(attempts):
        fst = my_gen_random_fst(num_states, num_final, chosen_vocab,
                                p_special=0.15,
                                p_drop_letter=0.4, id_prob=0.2)
        if fst.num_states() == num_states:
            return fst, chosen_vocab
    raise ValueError(f"Didn't manage to draw FST with {num_states} states")

def sample_bimachine_fst_for_eval(num_l_states, num_r_states, attempts: int = 50):
    vocab_size = 25 # fix vocab size, to reduce variance
    my_vocab = list(vocab)
    random.shuffle(my_vocab)
    chosen_vocab = "".join(my_vocab[:vocab_size])

    def sample_auto(num_states):
        for _ in range(attempts):
            fst = my_gen_random_fst(num_states, num_states, chosen_vocab,
                                    p_special=0, p_drop_letter=0.4, make_initial_final=True)
            if fst.num_states() == num_states:
                return fst
        raise ValueError(f"Didn't manage to draw FST with {num_states} states")

    left_auto = sample_auto(num_l_states).project("input")
    right_auto = sample_auto(num_r_states).project("input")
    output_f = gen_f_output(num_l_states, num_r_states, chosen_vocab, p_id=0.2)
    return Bimachine(left_auto, right_auto, output_f).to_fst(), chosen_vocab
