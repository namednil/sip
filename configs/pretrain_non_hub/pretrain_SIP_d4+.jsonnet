local num_states = 15;

local fst_tokenizer_path = "unicode_char_tokenizer_ipa.json";

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
        "f": "FSTPretrainingModel.from_pretrained",
        "path": "models/YOUR_MODEL"
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

    "save_dir": "models/YOUR_MODEL_s4_more",

    "train_data_path": train_data_path


   }

   ]
}
