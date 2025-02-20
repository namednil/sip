local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
            pretrained_model_name_or_path: "google/byt5-small"
        };

local task_train_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 32,
        "path": std.extVar("train"),
        "tokenizer": tokenizer
};

local task_val_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 64,
        "path":	std.extVar("test"),
        "tokenizer": tokenizer
};


{
  "random_seed": std.parseJson(std.extVar("seed")),
  "numpy_seed": std.parseJson(std.extVar("seed")),
  "pytorch_seed": std.parseJson(std.extVar("seed")),
  
  "imports": ["import transformers", 
   "from sip.data_loading import *",
     "from sip.embed_finetune import *", "from sip.task_finetune import *"],
  "logger": {
    f: "NeptuneLogger.create",
    "project": "<NAME>/<PROJECT>"
  },
  "steps": [

   {
    "name": "finetune",
    "f": "finetune_model",

    "model": {
        f: "create_struct_prefix",
        "prefix_length": 50,
        "model_str":  "models/w_fsts_pretrain_s4_32_task_embedding_longer2",
    },

    "tokenizer": tokenizer,

    "train_data_loader": task_train_loader,
    "validation_data_loader": task_val_loader,
    "optimizer": {"[lazy]": "torch.optim.Adam", "lr": 5e-4 },
    "num_epochs": 50,

    
    "optimizer_groups": [
        [".*prefix_embedding.*", {"lr": 1.0}],
        [".*", {"lr": 5e-4}]
    ],

    "logger": "[logger]",
    
    "grad_scale": 1.0,
  
   }
   
   ]
}

