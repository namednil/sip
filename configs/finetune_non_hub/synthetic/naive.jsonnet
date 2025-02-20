local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
            pretrained_model_name_or_path: "google/byt5-small"
        };

local task_train_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 48,
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

      model: {f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
                pretrained_model_name_or_path: std.extVar("load_model")
             },

    "tokenizer": tokenizer,

    "train_data_loader": task_train_loader,
    "validation_data_loader": task_val_loader,
    "optimizer": {"[lazy]": "torch.optim.Adam", "lr": 5e-4},
    "num_epochs": 50,

    "logger": "[logger]",
    
    "grad_scale": 1.0,
  
   }
   
   ]
}

