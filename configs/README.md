# Configuration Files

## Which configuration files ones do I need?

- If you want to fine-tune sip-d4 (as stored on the Hugging Face hub) 
or reproduce baseline results with ByT5 or Charsiu, you need the files in `finetune`.
- If you want to pre-train your own model (or baseline), you will need `pretrain_non_hub`. 
Running these configuration files will produce models that cannot easily be stored to the Hugging Face hub and you will need to store these files locally. We aim to provide conversion scripts later to make it easier to share these models. These models are also not compatible with the scripts in `finetune` (see below).
- If you want to fine-tune a model that was produced by **non-hub pre-training**, you will need to use `finetune_non_hub` scripts.


## Overview

```
.
├── finetune
│   ├── g2p
│   │   ├── byt5.jsonnet    <-- ByT5 baseline
│   │   ├── charsiu.jsonnet <-- Charsiu multilingual grapheme-to-phoneme model
│   │   └── sip-d4.jsonnet
│   ├── synthetic
│   │   ├── byt5.jsonnet    <-- ByT5 baseline
│   │   └── sip-d4.jsonnet
│   └── text_editing
│       ├── byt5.jsonnet    <-- ByT5 baseline
│       ├── charsiu.jsonnet <-- Charsiu multilingual grapheme-to-phoneme model
│       └── sip-d4.jsonnet
├── finetune_non_hub
│   ├── g2p
│   │   ├── set.jsonnet   <-- Set baseline from Wu et al. 2022
│   │   ├── sip-d4.jsonnet
│   │   └── te.jsonnet    <-- Task Embedding baseline (see Section 5.2)
│   ├── synthetic
│   │   ├── naive.jsonnet <-- Naive pretraining baseline (see Section 5.2)
│   │   ├── set.jsonnet   <-- Set baseline from Wu et al. 2022
│   │   ├── sip-d4.jsonnet
│   │   └── te.jsonnet    <-- Task Embedding baseline (see Section 5.2)
│   └── text_editing
│       ├── set.jsonnet   -- Set baseline from Wu et al. 2022
│       ├── sip-d4.jsonnet
│       └── te.jsonnet    -- Task Embedding baseline (see Section 5.2)
├── pretrain_non_hub
│   ├── pretrain_SIP_d4.jsonnet   -- SIP-d4 (Section 5.2)
│   ├── pretrain_SIP_d4+.jsonnet  -- SIP-d4+ (Section 5.5)
│   ├── pretrain_SIP_nd7.jsonnet  -- SIP-nd7 (Section 5.5)
│   ├── pretrain_T5_SIP.jsonnet   -- SIP but with T5-base initialization instead of ByT5 (Appendix B.3)
│   └── pretrain_TE.jsonnet       -- TE baseline (Section 5.2); uses a second GPU to store task embeddings.

```
## Understanding configuration files

We use `jsonnet` files, which are files that can be executed to produce a json file (or a python dictionary). 
These files can refer to environment variables (`std.extVar`) and make it easy to structure the configuration.

Each configuration file consists of imports statements, a logger and a list of steps. Each step is represented as dictionary with a `name`
and a function that should be called, as identified under the dictionary key `f`. 
The remaining dictionary keys refer to arguments of the function to be called. 
Arguments might themselves be the result of applying function - this is indicated by passing a dictionary that again has an `f` key.

Occasionally, the function receiving an argument would like a partial function - this is marked with `[lazy]` instead of `f`. 
For example, passing `"optimizer": {"[lazy]": "torch.optim.Adam", "lr": 3e-4}` to `sip.task_finetune.finetune_model` 
does *not* immediately create an `optim.Adam` object but allows `finetune_model` to pass the model parameters to the optimizer before instantiating it.

### FST tokenizer?
We use a separate 'tokenizer' for the FST transitions that maps characters to IDs. 
We don't reuse the ByT5 tokenizer because it would split non-ASCII symbols into multiple tokens (bytes) (see also Section 4.1 of the paper).
