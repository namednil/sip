import os

import torch
from transformers import PreTrainedModel

from config_evaluator import Lazy

def create_task_embedding_model(**kwargs):
    return TaskEmbeddingModel(**kwargs)

class TaskEmbeddingModel(torch.nn.Module):

    def __init__(self, model: PreTrainedModel,
                 num_tasks: int,
                 prefix_length: int,
                 ensure_task_id: bool,
                 internal_embedding_dim: int,
                 parallel: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.embedding_dim = self.model.get_input_embeddings().embedding_dim
        self.internal_embedding_dim = internal_embedding_dim
        self.num_tasks = num_tasks
        self.prefix_length = prefix_length
        self.embeddings = torch.nn.Embedding(self.num_tasks, prefix_length*self.internal_embedding_dim)
        self.up_proj = torch.nn.Linear(self.internal_embedding_dim, self.embedding_dim)
        self.ensure_task_id = ensure_task_id
        
        self.parallel = parallel
        
        if parallel:
          # put embeddings on GPU 1.          
          self.embeddings = self.embeddings.to(1)
          self.up_proj = self.up_proj.to(1)
          self.embedding_device = 1
          
          self.model = self.model.to(0) #put main model on GPU 0


    @property
    def device(self):
        return self.model.device

    def prepare_input(self, kwargs):
        if not self.parallel:
           self.embedding_device = kwargs["input_ids"].device
           
        embedded_inputs = self.model.get_input_embeddings()(kwargs["input_ids"])
        batch_size = embedded_inputs.shape[0]

        if "fst_rep" in kwargs:
            del kwargs["fst_rep"]

        if "task_ids" not in kwargs:
            if self.ensure_task_id:
                raise ValueError("Need task ids")
            else:
                # Go with embedding no 0 by default
                task_embeddings = self.embeddings(torch.zeros(batch_size, dtype=torch.long, device=self.embedding_device))
        else:
            # if we don't have a task embedding (because the task id is higher than what we prepared for, e.g. test data
            # with unseen tasks), simply choose task 0
            inside_training = kwargs["task_ids"] < self.num_tasks #shape (batch,)
            lookup = inside_training * kwargs["task_ids"]
            task_embeddings = self.embeddings(lookup.to(self.embedding_device)) #shape (batch, prefix_length * embedding dim)

            del kwargs["task_ids"]

        rep = torch.unflatten(task_embeddings, 1, sizes=(self.prefix_length, self.internal_embedding_dim)) #shape (batch, prefix, internal embedding dim)
        rep = self.up_proj(rep)
        
        rep = rep.to(embedded_inputs.device)

        embedded_inputs = torch.cat([rep, embedded_inputs], dim=1)

        del kwargs["input_ids"]
        kwargs["inputs_embeds"] = embedded_inputs

        if "attention_mask" in kwargs:
            ones = torch.ones((batch_size, rep.shape[1]), device=embedded_inputs.device, dtype=kwargs["attention_mask"].dtype)
            input_mask = torch.cat([ones, kwargs["attention_mask"]], dim=1)
            kwargs["attention_mask"] = input_mask

        return kwargs

    def forward(self, **kwargs):
        return self.model(**self.prepare_input(kwargs))

    def generate(self, **kwargs):
        return self.model.generate(**self.prepare_input(kwargs))

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        torch.save(self.embeddings, os.path.join(path, "task_embeddings.pt"))
