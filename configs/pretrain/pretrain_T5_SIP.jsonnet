local num_states = 15;

local fst_tokenizer_path = "namednil/sip-fst-tokenizer";

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
        "fst_tokenizer_path": fst_tokenizer_path,
        "num_states": num_states

} ;


{
  "imports": ["import transformers", "from meta_adapters.metalearner import *", "from meta_adapters.bart_style_byt5_tok import *",
   "from meta_adapters.data_loading import *", "from meta_adapters.pretraining import *", "from meta_adapters.embed_finetune import *",
    "from meta_adapters.fst_pretrain import *"],
  "logger": {
    f: "NeptuneLogger.create",
    "project": "<NAME>/<PROJECT>"
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
            f: "create_model_shrink_vocab",
            model_str: "t5-base",
            vocab_size: 385 # vocab size of ByT5
       },

    },

    "lr_scheduler": {
        "[lazy]": "transformers.get_constant_schedule_with_warmup",
        "num_warmup_steps": 100
    },
    
    "pass_num_training_steps_to_scheduler": false,

    "tokenizer": tokenizer,

    "train_data_loader": data_loader(train_data_path, 16),
    "easy_validation_data_loader": data_loader(easy_dev_data_path, 32),
    "validation_data_loader": data_loader(dev_data_path, 32),

    "test_data_loader": data_loader(test_data_path, 32),

    "optimizer": {"[lazy]": "torch.optim.Adam", "lr": 1e-4},
    "num_epochs": 20, #TODO

    "logger": "[logger]",

    "num_accumulation_steps": 2,

    "save_dir": "models/w_fsts_pretrain_s4_t5_prefine",

    "train_data_path": train_data_path


   }

   ]
}
