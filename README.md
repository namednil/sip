# SIP

This is the code for the ACL 2024 paper [SIP: Injecting a Structural Inductive Bias into a Seq2Seq Model by Simulation](https://arxiv.org/abs/2310.00796).

## Using SIP without full installation
If you just want to play with our model and apply it to a downstream task (rather than pre-training/reproducing the exact experiments), then you can use it as follows:

```python
import transformers, torch
tokenizer = transformers.AutoTokenizer.from_pretrained("google/byt5-small")
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("namednil/sip-d4", trust_remote_code=True)
# (always make sure to check the remote code on Huggingface!)

# Construct an optimizer that uses the SIP-finetuning procedure:
optimizer = model.get_optimizer(torch.optim.Adam, prefix_lr=1.0, lr=3e-4)
# ... fine-tune the model as usual

# The above code uses a random initialization of the tunable prefix of SIP. 
# If you don't want that and have more control over the length of the tunable prefix, run:

config = transformers.AutoConfig.from_pretrained("namednil/sip-d4", trust_remote_code=True)
config.random_selection = False
config.prefix_length = 50 
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("namednil/sip-d4", config=config, trust_remote_code=True)
```

## Setup

If you want to reproduce our experiments, this will require a full installation. First set up a new environment as follows:
```shell
conda create -n sip python=3.10
conda activate sip
conda install -c conda-forge pynini # FST library
# install pytorch (we used v 2.2.1 but newer versions such as 2.6.0 should work fine as well)
#e.g. via
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Clone git repository
git clone https://github.com/namednil/sip

# Install remaining pip requirements
# (potentially uncomment neptune.ai dependency if desired)
cd sip
pip install -r requirements.txt
```

## Reproduction of fine-tuning experiments

### Data
For your convenience, we've included the data for our experiments in `data.zip`. This file has been password protected to avoid unintentional data contamination (password: SIP).

If you want to re-generate the synthetic data, you need to run the files `sip/data_gen/gen_in_pretraining_dist.py` and `sip/data_gen/gen_bimachine_eval.py`.

### Fine-tuning

Each fine-tuning experiment is described by a configuration file in `configs/finetune`, grouped by the datasets. 

Each configuration file also uses a logger. By default, it will try to log experiments to `neptune.ai` to a specific project.
If you prefer to log to the standard output, use `"logger": {  f: "TqdmLogger.create"  },` instead. You can also define your own logger in `logger.py`.

The configuration files refer to environment variables for the random seed and data files. For example, to run `SIP-d4` on the first of the length generalization tasks with 4 FST states, you need to run:
```shell
export seed="762354"
export train="data/eval/task_s4_0_length_train.tsv"
export test="data/eval/task_s4_0_length_test.tsv"
python config_evaluator.py configs/finetune/synthetic/sip-d4.jsonnet
```

Or on a few-shot text editing task:
```shell
export seed=1234
export num_train=5
export path="data/pbe_strings_track_tsv/name-combine-4-long.sl.tsv" 
python config_evaluator.py configs/finetune/text_editing/sip-d4.jsonnet
```

## Reproduction of pre-training

To generate the pre-training data, go to the main directory and call
```
python -m sip.data_gen.gen_pretrain
```
this will create the files `data/pretrain/[train|dev|test|easy_dev]_s4.jsonl`. Each line is a json object that encodes
a pre-training instance: an input/output string pair and a list of FST transitions. Each transition has the format (p, symbol read, symbol written, q, is q final) where p is the state we're coming from and q is the state we're going to.
epsilon is encoded as the value 20 (see sip.data_gen.utils).

The easy_dev files contain instances with FSTs seen during training but unseen strings, whereas the dev and test files contain unseen FSTs.

To pre-train the model, run:
```
python config_evaluator.py configs/pretrain_non_hub/pretrain_SIP_d4.jsonnet
```

To fine-tune a model that was pre-trained from a configuration in `configs/pretrain_non_hub/`, you will need to use one of the configs
in `configs/finetune_non_hub/`.
Unfortunately, the models produced in this way are currently not easy to upload to the HuggingFace hub, and we're hoping to provide a conversion script soon.

# Citation
```
@inproceedings{lindemann-etal-2024-sip,
    title = "SIP: Injecting a Structural Inductive Bias into a Seq2Seq Model by Simulation",
    author = "Lindemann, Matthias  and
      Koller, Alexander  and
      Titov, Ivan",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2310.00796",
}
```