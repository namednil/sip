import random

from sip.data_gen.eval_automaton_gen import sample_fst_for_eval, vocab, sample_bimachine_fst_for_eval
from sip.data_gen.ood_gen import LengthGen, UnseenCombinationsGen
from sip.data_gen.unseen_combinations import gen_compgen_variants, NoUnseenCombinationsPossibleException
from sip.data_gen.utils import write_tsv, FSTCollection
import pynini

import tqdm
import pickle

if __name__ == "__main__":
    random.seed(2347623)

    num_train_examples = 5000
    num_test_examples = 1000

    num_l_states = 5
    num_r_states = 4

    # num_tasks = 5
    num_tasks = 10

    sampled_fsts = []

    for i in tqdm.tqdm(range(num_tasks)):
        fst, chosen_vocab = sample_bimachine_fst_for_eval(num_l_states, num_r_states)

        datagen = LengthGen(fst, chosen_vocab, vocab)
        sampled_fsts.append(datagen)

        train_data = []
        test_data = []
        for _ in range(num_train_examples):
            train_data.append(datagen.gen_train_ex())
        for _ in range(num_test_examples):
            test_data.append(datagen.gen_test_ex())

        write_tsv(f"data/eval/task_bimachine_s{num_l_states}.{num_r_states}_{i}_{datagen.name}_train.tsv", train_data)
        write_tsv(f"data/eval/task_bimachine_s{num_l_states}.{num_r_states}_{i}_{datagen.name}_test.tsv", test_data)

    with open(f"data/FSTs/bimachines_s{num_l_states}.{num_r_states}_length.pickle", "wb") as f:
        pickle.dump(sampled_fsts, f)

    # Unseen combinations. Sample new automata

    random.seed(14159265)

    sampled_fsts = []

    for i in tqdm.tqdm(range(num_tasks)):
        fst, chosen_vocab = sample_bimachine_fst_for_eval(num_l_states, num_r_states)

        print(fst.num_states())

        datagen = UnseenCombinationsGen(fst, chosen_vocab, vocab)
        sampled_fsts.append(datagen)

        train_data = []
        test_data = []
        for _ in range(num_train_examples):
            train_data.append(datagen.gen_train_ex())
        for _ in range(num_test_examples):
            test_data.append(datagen.gen_test_ex())

        write_tsv(f"data/eval/task_bimachine_s{num_l_states}.{num_r_states}_{i}_{datagen.name}_train.tsv", train_data)
        write_tsv(f"data/eval/task_bimachine_s{num_l_states}.{num_r_states}_{i}_{datagen.name}_test.tsv", test_data)

    with open(f"data/FSTs/bimachines_s{num_l_states}.{num_r_states}_uc.pickle", "wb") as f:
        pickle.dump(sampled_fsts, f)


