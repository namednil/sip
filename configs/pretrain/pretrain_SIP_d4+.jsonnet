local num_states = 15;

local fst_tokenizer_path = "namednil/sip-fst-tokenizer";

local train_data_path = "data/pretrain/s4_more/train_pretrain_s4_more.jsonl";
local dev_data_path = "data/pretrain/s4_more/dev_pretrain_s4_more.jsonl";
local easy_dev_data_path = "data/pretrain/s4_more/easy_dev_pretrain_s4_more.jsonl";
local test_data_path = "data/pretrain/s4_more/test_pretrain_s4_more.jsonl";


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
  "imports": ["import transformers", "from meta_adapters.metalearner import *",
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
        "f": "SIPPreTrainingModel.from_pretrained",
        "path": "namednil/sip-d4-pt"
    },

    "tokenizer": tokenizer,

    "train_data_loader": data_loader(train_data_path, 10),
    "easy_validation_data_loader": data_loader(easy_dev_data_path, 32),
    "validation_data_loader": data_loader(dev_data_path, 32),

    "test_data_loader": data_loader(test_data_path, 32),

    "optimizer": {"[lazy]": "torch.optim.Adam", "lr": 5e-4},
    "num_epochs": 10,

    "logger": "[logger]",

    "num_accumulation_steps": 3,

    "save_dir": "models/w_fsts_pretrain_s4_then_more",

    "train_data_path": train_data_path


   }

   ]
}
