# Appendix D.2 of the paper.

import transformers

from sip.analysis.state_probe import load_fst_jsonl_for_probing, StateProbe, STATE_PROBE_TRAIN_DATA, \
    STATE_PROBE_TEST_DATA
from sip.fst_pretrain import FSTPretrainingModel, SIPPreTrainingModel

import torch

if __name__ == "__main__":
    torch.manual_seed(2345682)
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")

    fst_tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained("namednil/sip-fst-tokenizer")

    train_data_loader = load_fst_jsonl_for_probing(STATE_PROBE_TRAIN_DATA,
                                                   tokenizer, fst_tokenizer,
                                                  32, 5, random_order=True) # original experiments used batch size 64 but might not fit into memory

    test_data_loader = load_fst_jsonl_for_probing(STATE_PROBE_TEST_DATA,
                                                  tokenizer, fst_tokenizer,
                                                   32, 5, random_order=True)

    model = SIPPreTrainingModel.from_pretrained("namednil/sip-d4-pt")
    
    model.model = transformers.AutoModelForSeq2SeqLM.from_config(transformers.AutoConfig.from_pretrained("google/byt5-small"))

    probe = StateProbe(5, model.model.get_output_embeddings().in_features)

    model.eval()
    
    model = model.to(0)
    probe = probe.to(0)
    
    optim = torch.optim.Adagrad(probe.parameters(), lr=0.1)

    # print(next(iter(data_loader)))
    for batch in train_data_loader:
        batch = {k: v.to(0) for k,v in batch.items()}
        batch_without_state_seq = dict(batch)
        del batch_without_state_seq["state_seq"]
        with torch.no_grad():
            model_output = model(**batch_without_state_seq)
        r = probe(batch, model_output)
        print(r.loss)
        r.loss.backward()

        optim.step(closure=lambda: probe(batch, model_output).loss)
        optim.zero_grad()


    total = 0
    correct = 0
    all_correct = 0
    num_examples = 0

    with torch.no_grad():
        for batch in test_data_loader:
            batch =	{k: v.to(0) for k,v in batch.items()}
            batch_without_state_seq = dict(batch)
            del batch_without_state_seq["state_seq"]
            with torch.no_grad():
                model_output = model(**batch_without_state_seq)
            r = probe(batch, model_output)
            preds = torch.argmax(r.logits, dim=-1)
            total += (batch["state_seq"] != -100).sum()
            correct += (preds == batch["state_seq"]).sum()

            all_correct += torch.all((preds == batch["state_seq"]) | (batch["state_seq"] == -100), dim=-1).sum()
            num_examples += batch["state_seq"].shape[0]

    print("State probe accuracy", correct.cpu().numpy()/total.cpu().numpy() * 100, "%")
    print("Sequence acc", all_correct.cpu().numpy() / num_examples * 100, "%")
    #Output:
    #State probe accuracy 42.54188125258983 %
    #Sequence acc 7.077856420626897 %
    
    probe.save_pretrained("models/probe_states_on_byt5_random.pt")
    

