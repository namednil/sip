
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

def compute_average_perm_matrix(matrices):
    avg = sum(matrices)
    return avg / avg.sum(axis=1, keepdims=True) * 100
def make_plot(matrix, title):
    # print(matrix)
    # sns.heatmap(matrix, annot=True, fmt=".1f", square=True, cbar=False, cmap=sns.light_palette("seagreen"))
    # sns.heatmap(matrix, annot=True, fmt=".1f", square=True, cbar=False, cmap=sns.light_palette("seagreen"), vmin=0, vmax=100)
    # sns.heatmap(matrix, annot=True, fmt=".1f", square=True, cbar=False, cmap=sns.color_palette("flare_r"), vmin=0, vmax=100)
    sns.heatmap(matrix, annot=True, fmt=".1f", square=True, cbar=False, cmap=sns.cubehelix_palette(start=0.2, rot=-.5, as_cmap=True), vmin=0, vmax=100)
    plt.xlabel("Predicted state")
    plt.ylabel("Gold state")
    plt.title(title)

if __name__ == "__main__":
    import seaborn as sns
    from matplotlib import pyplot as plt

    import sys
    
    state_seq_correct = []
    outputs_correct = []
    state_seq_hamming_dist = []
    output_levenshtein = []
    tasks = []
    task = 0

    sns.set(style="ticks", font_scale=1.3)

    d = {"hamming_dist": [], "task_id": [], "data": []}

    fine_tuning_type = "full_ft"
    for split in ["test", "train"]:
        all_m = []
        all_matrix = []
        # for split in ["train"]:
        for task in range(5):
            with open(f"data/analysis/analysis_{fine_tuning_type}/details_{task}_{split}.pkl", "rb") as f:
                confusion_matrix, data = pickle.load(f)
                matrix, _ = reorganize_matrix(confusion_matrix.transpose(1, 0))
                matrix = matrix.transpose(1, 0)
                all_matrix.append(matrix)

                print(confusion_matrix)
                print(matrix)
                print("===")

                _, translation = reorganize_matrix(confusion_matrix)

                seq_correct = 0
                # print(len(data))
                for dp in data:
                    approx_translation = approx_translate_pred_to_gold(translation, dp["pred_state_seq"])
                    seq_correct += approx_translation == dp["gold_state_seq"]
                    d["hamming_dist"].append(hamming_dist(dp["gold_state_seq"], approx_translation))
                    d["task_id"].append(task)
                    d["data"].append(split)
                # print(confusion_matrix)
                # print(matrix)
                # m, _ = reorganize_matrix(confusion_matrix)
                # print("acc", (m * np.eye(m.shape[0])).sum() / m.sum())
                # print("whole seq acc", seq_correct / len(data))
                # print(m)
                # print("==")
                # all_m.append(m)

        make_plot(compute_average_perm_matrix(all_matrix), "Training data" if split=="train" else "Test data")
        plt.savefig(f"confusion_matrix_{fine_tuning_type}_{split}.pdf", bbox_inches='tight')
        plt.clf()
        plt.show()