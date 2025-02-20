local num_states = 15;

local fst_tokenizer_path = "unicode_char_tokenizer_ipa.json";

local train_data_path = "data/pretrain/train_pretrain_s4.jsonl";
local dev_data_path = "data/pretrain/dev_pretrain_s4.jsonl";
local easy_dev_data_path = "data/pretrain/easy_dev_pretrain_s4.jsonl";
local test_data_path = "data/pretrain/test_pretrain_s4.jsonl";


local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
                        pretrained_model_name_or_path: "google/byt5-small"
                                };
                                
local data_loader(fname, batch_size) = {
        "f": "load_fst_jsonl",
        "batch_size": batch_size,
        "path": fname,
        "tokenizer": tokenizer,
        "fst_tokenizer": fst_tokenizer_path,
        "num_states": num_states,

} ;


{
  "imports": ["import transformers",
   "from sip.data_loading import *", "from sip.pretraining import *", "from sip.embed_finetune import *",
    "from sip.fst_pretrain import *"],
  "logger": {
    f: "TqdmLogger.create",
  },
  "steps": [

   {
    "name": "pretrain",
    "f": "pretrain",
    "model": {
        "f": "create_fst_pretraining_model",

        "machine_embedder": {
                "[lazy]": "create_simple_fst_embedder",
                "num_states": num_states,
                "fst_tokenizer_path": fst_tokenizer_path,
                "state_embedding_dim": 64,
                "token_embedding_dim": 256,
                "final_state_embedding_dim": 16
        },

        "model": {
            f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            pretrained_model_name_or_path: "google/byt5-small"
            },
    },

    "tokenizer": tokenizer,

    "train_data_loader": data_loader(train_data_path, 10),
    "easy_validation_data_loader": data_loader(easy_dev_data_path, 32),
    "validation_data_loader": data_loader(dev_data_path, 32),

    "test_data_loader": data_loader(test_data_path, 32),

    "optimizer": {"[lazy]": "torch.optim.Adam", "lr": 5e-4},
    "num_epochs": 20,

    "logger": "[logger]",

    "num_accumulation_steps": 3,

    "save_dir": "models/YOUR_MODEL",

    "train_data_path": train_data_path


   }

   ]
}
