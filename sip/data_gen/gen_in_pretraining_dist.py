import pickle
import random

from sip.data_gen.eval_automaton_gen import sample_fst_for_eval, vocab
from sip.data_gen.ood_gen import LengthGen, UnseenCombinationsGen
from sip.data_gen.unseen_combinations import gen_compgen_variants, NoUnseenCombinationsPossibleException
from sip.data_gen.utils import write_tsv, FSTCollection
import pynini

import tqdm

if __name__ == "__main__":
    num_train_examples = 5000
    num_test_examples = 1000

    for num_states in [4, 5, 7, 10]:

        random.seed(2347623)

        collection = FSTCollection()
        with pynini.Far("data/pretrain/train_pretrain_s4.far", mode="r") as f:
            for key, fst in f:
                collection.maybe_add(fst)

            f.reset()
            assert f.get_fst() in collection

        num_tasks = 5

        sampled_fsts = []

        for i in tqdm.tqdm(range(num_tasks)):
            fst, chosen_vocab = sample_fst_for_eval(num_states)

            if fst in collection:
                raise ValueError("FST seen in pre-training data. Please adjust random seed to avoid sampling a seen FST.")

            datagen = LengthGen(fst, chosen_vocab, vocab)
            sampled_fsts.append(datagen)

            train_data = []
            test_data = []
            for _ in range(num_train_examples):
                train_data.append(datagen.gen_train_ex())
            for _ in range(num_test_examples):
                test_data.append(datagen.gen_test_ex())

            write_tsv(f"data/eval/task_s{num_states}_{i}_{datagen.name}_train.tsv", train_data)
            write_tsv(f"data/eval/task_s{num_states}_{i}_{datagen.name}_test.tsv", test_data)

        with open(f"data/FSTs/fsts_s{num_states}_length.pickle", "wb") as f:
            pickle.dump(sampled_fsts, f)

        # Unseen combinations. Sample new automata

        random.seed(1432347623)

        sampled_fsts = []

        i = 0
        while i < num_tasks:
            fst, chosen_vocab = sample_fst_for_eval(num_states)

            if fst in collection:
                raise ValueError("FST seen in pre-training data. Please adjust random seed to avoid sampling a seen FST.")
            try:
                datagen = UnseenCombinationsGen(fst, chosen_vocab, vocab)
                sampled_fsts.append(datagen)
            except NoUnseenCombinationsPossibleException as ex:
                print(ex)
                print("Not to worry. We sample a new FST for which we can get such a split.")
                continue

            train_data = []
            test_data = []
            for _ in range(num_train_examples):
                train_data.append(datagen.gen_train_ex())
            for _ in range(num_test_examples):
                test_data.append(datagen.gen_test_ex())

            write_tsv(f"data/eval/task_s{num_states}_{i}_{datagen.name}_train.tsv", train_data)
            write_tsv(f"data/eval/task_s{num_states}_{i}_{datagen.name}_test.tsv", test_data)

            i += 1

        with open(f"data/FSTs/fsts_s{num_states}_uc.pickle", "wb") as f:
            pickle.dump(sampled_fsts, f)


