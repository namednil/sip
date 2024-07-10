import numpy as np
import torch
import transformers

from sip.analysis.state_probe import StateProbe
from sip.data_gen.recursion_limit import identify_state_sequence, FasterFSTAccess
from sip.embed_finetune import SIPFinetuningModel

import pickle

from sip.data_loading import prepare_task_dataset

np.set_printoptions(suppress=True)

if __name__ == "__main__":
    probe = StateProbe.from_pretrained("models/probe_states_s4_32.pt", map_location="cpu")

    tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")

    with open(f"data/FSTs/fsts_s4_length.pickle", "rb") as f:
        fst_data = pickle.load(f)

    # print(fst)
    probe = probe.to(0)
    for i in range(5):
        model = SIPFinetuningModel.from_pretrained(f"models/s4_{i}_length_full_ft/")
        model = model.to(0)
        model.eval()
        fst = fst_data[i].sampling_fst
        faster_access_fst = FasterFSTAccess(fst)
                
        for split in ["train", "test"]:
            data = prepare_task_dataset(f"data/eval/task_s4_{i}_length_{split}.tsv", tokenizer, 16)
            confusion_matrix = np.zeros((4, 4)) # [gold_index, predicted state index]
            correct = 0
            total = 0
            output_data = []
            for batch in data:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                model_output = model(**batch)

                test_batch_inputs = dict(batch)
                del test_batch_inputs["labels"]
                r = tokenizer.batch_decode(
                    model.generate(**test_batch_inputs, max_new_tokens=batch["labels"].shape[1] + 2, num_beams=1,
                                   no_repeat_ngram_size=0),
                    skip_special_tokens=True)
                gold = tokenizer.batch_decode(100 * (batch["labels"] == -100) + batch["labels"],
                                              skip_special_tokens=True)  # replace -100 by 0
                # loss += model_output.loss.detach().cpu().numpy()
                # The following only works because the ByT5 tokenizer seems to actually implement a bijection!
                input_strings = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                state_sequences = [identify_state_sequence(faster_access_fst, o) for o in input_strings]
                preds = probe(batch, model_output)
                preds = torch.argmax(preds.logits, dim=-1).cpu().numpy() #shape (batch, input seq len)
                for state_seq, pred_state_seq, input_str, pred_output, gold_output in zip(state_sequences, preds,
                                                                                          input_strings, r, gold):
                    # print(state_seq, pred_state_seq[:len(state_seq)], pred_output == gold_output)
                    correct += pred_output == gold_output
                    total += 1
                    output_data.append({"input_str": input_str, "gold_output": gold_output,
                                        "pred_output": pred_output,
                                        "gold_state_seq": state_seq,
                                        "pred_state_seq": pred_state_seq[:len(state_seq)],
                                        "output_correct": pred_output == gold_output})
                    for g, p in zip(state_seq, pred_state_seq):
                        confusion_matrix[g, p] += 1

            print(f"Dataset {i} and split {split}.")
            print("Confusion matrix (rows: gold state, column: extracted state).")
            print(confusion_matrix)
            print("Acc", correct / total)
            with open(f"data/analysis/analysis_full_ft/details_{i}_{split}.pkl", "wb") as f:
                pickle.dump((confusion_matrix, output_data), f)
            # print("Average batch loss", loss / len(data))
