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
    f: "NeptuneLogger.create",
    "project": "<NAME>/<PROJECT>"
  },
  "steps": [

   {
    "name": "pretrain",
    "f": "pretrain",
    "model": {
        "f": "create_task_embedding_model",
        
        "num_tasks": 40001,
        "prefix_length": 50,
        "ensure_task_id": true,
        
        "internal_embedding_dim": 180,
        
        "parallel": true,

        "model": {
            f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            pretrained_model_name_or_path: "google/byt5-small"
            },
    },


    "tokenizer": tokenizer,

    "train_data_loader": data_loader(train_data_path, 10),
    "easy_validation_data_loader": data_loader(easy_dev_data_path, 24),
    "validation_data_loader": null,

    "test_data_loader": null,

    "optimizer_groups": [
            ["^embedding.*", {"lr": 1.0}], # give task embeddings a high learning rate to adapt quickly to a task
            [".*", {"lr": 5e-4}]
    ],

    "num_epochs": 20, #TODO

    "logger": "[logger]",

    "num_accumulation_steps": 3,

    "save_dir": "models/w_fsts_pretrain_s4_32_task_embedding_longer2",
    
    "device": null,
    

    "train_data_path": train_data_path


   }

   ]
}
