
import pickle

from scipy.optimize import linear_sum_assignment

import numpy as np
import pandas as pd
import seaborn as sns

import scipy
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

import Levenshtein
def reorganize_matrix(matrix):
    """
    Permutes the columns such that the predicted states (columns) align best with the gold states (rows).
    :param matrix:
    :return:
    """
    rows, cols = linear_sum_assignment(matrix.transpose(1, 0), maximize=True)
    m = np.zeros_like(matrix)
    for r, c in zip(rows, cols):
        m[:, c] = matrix[:, r]
    return m, cols
def hamming_dist(x, y):
    return sum([a != b for a,b in zip(x, y)])
def approx_translate_pred_to_gold(cols, state_seq):
    return [cols[i] for i in state_seq]

if __name__ == "__main__":
    state_seq_correct = []
    outputs_correct = []
    state_seq_hamming_dist = []
    output_levenshtein = []
    tasks = []
    for task in range(5):
        with open(f"analysis_full_ft/details_{task}_test.pkl", "rb") as f:
            confusion_matrix, data = pickle.load(f)
        matrix, translation = reorganize_matrix(confusion_matrix)
        for dp in data:
            translated_state_seq = approx_translate_pred_to_gold(translation, dp["pred_state_seq"])
            state_seq_correct.append(int(translated_state_seq == dp["gold_state_seq"]))
            tasks.append(task)
            outputs_correct.append(dp["output_correct"])
            state_seq_hamming_dist.append(hamming_dist(translated_state_seq, dp["gold_state_seq"]))
            output_levenshtein.append(Levenshtein.distance(dp["gold_output"], dp["pred_output"]))

    df = pd.DataFrame({"output_correct": outputs_correct, "state_seq_correct": state_seq_correct,
                       "task": tasks, "state_seq_hamming": state_seq_hamming_dist, "output_edit": output_levenshtein})

    # print(df.groupby(["task", "state_seq_correct"]).aggregate(["count", "mean"]))
    print(df.groupby("state_seq_correct").mean())
    # print(df.groupby("output_correct").mean())
    # print(df.mean())
    # sns.violinplot(df, y="output_correct", x="state_seq_correct")
    # plt.show()
    # print(df.groupby(["state_seq_correct"]).mean())
    # print("Pearson rho", scipy.stats.pearsonr(df["output_edit"], df["state_seq_hamming"]))

    def statistic(x, y):
        return np.mean(x) - np.mean(y)

    # print(scipy.stats.permutation_test([df.query("state_seq_correct == 1")["output_correct"], df.query("state_seq_correct == 0")["output_correct"]], statistic, vectorized=False,
    #                                    n_resamples=20_000, alternative="greater"))
