local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
            pretrained_model_name_or_path: "google/byt5-small"
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

    "train_data_loader": null,
    "validation_data_loader": null,
    
    "dataset_splitter": {
       "f": "RandomSplit",
       "path": std.extVar("path"),
       "tokenizer": tokenizer,
       "num_train": std.parseJson(std.extVar("num_train")),
       "train_batch_size": 16
    },
    
    "optimizer": {"[lazy]": "torch.optim.Adam", "lr": 3e-4},
    "num_epochs": 200,

    "optimizer_groups": [
        [".*prefix_embedding.*", {"lr": 1.0}],
        [".*", {"lr": 3e-4}]
    ],

    
    "logger": "[logger]",
    
    "grad_scale": 1.0,
    
    "num_accumulation_steps": 1,
    "eval_only_last_epochs": true
  
   }
   
   ]
}

