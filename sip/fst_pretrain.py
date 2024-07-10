import os
from abc import ABC
from typing import Dict, Any, Tuple, Optional

import torch.nn
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PretrainedConfig, T5Config, \
    T5ForConditionalGeneration

from config_evaluator import Lazy
from sip.data_loading import load_fst_jsonl


# These are the original functions/classes that were used in the experiments for the paper.
# However, they don't make it very easy to share the models. Hence, there is "hf-compatible" code below for easier model sharing.

def create_fst_pretraining_model(**kwargs):
    return FSTPretrainingModel(**kwargs)


class MachineEmbedder(ABC,  torch.nn.Module):

    def __init__(self, trafo_embedding_dim: int):
        super().__init__()
        self.trafo_embedding_dim = trafo_embedding_dim
    def prepare_input(self, kwargs: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        raise NotImplementedError()

def create_simple_fst_embedder(num_states: int,
                 fst_tokenizer_path: str,
                 state_embedding_dim: int,
                 token_embedding_dim: int,
                **kwargs):
    return SimpleFSTEmbedder(num_states, fst_tokenizer_path, state_embedding_dim, token_embedding_dim, **kwargs)

class SimpleFSTEmbedder(MachineEmbedder):
    def __init__(self,  num_states: int,
                 fst_tokenizer_path: str,
                 state_embedding_dim: int,
                 token_embedding_dim: int,
                 mlp_hidden_dim: Optional[int] = None,
                 final_state_embedding_dim: int = 0,
                 num_final_state_info: int = 3,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_embeddings = torch.nn.Embedding(num_states, state_embedding_dim)
        self.fst_tokenizer = PreTrainedTokenizerFast(tokenizer_file=fst_tokenizer_path)
        self.token_embeddings = torch.nn.Embedding(self.fst_tokenizer.vocab_size, token_embedding_dim)
        self.final_state_embedding = torch.nn.Embedding(num_final_state_info, final_state_embedding_dim)


        self.down_project = None
        if mlp_hidden_dim is not None:
            self.down_project = torch.nn.Linear(state_embedding_dim * 2 + token_embedding_dim * 2 + final_state_embedding_dim, mlp_hidden_dim)
            self.dropout = torch.nn.Dropout(0.1)
            self.output_layer = torch.nn.Linear(mlp_hidden_dim, self.trafo_embedding_dim)
        else:
            self.input_layer = torch.nn.Linear(state_embedding_dim * 2 + token_embedding_dim * 2 + final_state_embedding_dim, self.trafo_embedding_dim)

    @property
    def device(self):
        return self.state_embeddings.device

    def prepare_input(self, kwargs):
        fst_rep = kwargs["fst_rep"] #shape (batch, transition count, 4 or 5)
        del kwargs["fst_rep"]


        from_rep = self.state_embeddings(fst_rep[:, :, 0]) #shape (batch, transition count, embed dim)
        to_rep = self.state_embeddings(fst_rep[:, :, 3]) #shape (batch, transition count, embed dim)
        io_rep = self.token_embeddings(fst_rep[:, :, 1:3]) #shape (batch, transition count, 2, embed dim)
        io_rep = torch.flatten(io_rep, start_dim=2)

        if fst_rep.shape[-1] == 5:
            final_state_rep = self.final_state_embedding(fst_rep[:, :, 4])
            flat_total_fst_rep = torch.cat([from_rep, to_rep, io_rep, final_state_rep],
                                           dim=2)  # shape (batch, transition, 2* state embed dim + 2 * token embed dim)
        else:
            flat_total_fst_rep = torch.cat([from_rep, to_rep, io_rep], dim=2) #shape (batch, transition, 2* state embed dim + 2 * token embed dim)

        if self.down_project is not None:
            fst_embed = self.output_layer(self.dropout(torch.nn.functional.gelu(self.down_project(flat_total_fst_rep))))
        else:
            fst_embed = self.input_layer(flat_total_fst_rep) # shape (batch, transition, trafo embedding dim)

        return fst_embed, kwargs

class FSTPretrainingModel(torch.nn.Module):

    def __init__(self, model: PreTrainedModel,
                 machine_embedder: Lazy[MachineEmbedder],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.machine_embedder = machine_embedder.run(trafo_embedding_dim = self.model.get_input_embeddings().embedding_dim)

    @property
    def device(self):
        return self.model.device
    def prepare_input(self, kwargs):
        embedded_inputs = self.model.get_input_embeddings()(kwargs["input_ids"])
        batch_size = embedded_inputs.shape[0]

        machine_rep, kwargs = self.machine_embedder.prepare_input(kwargs) #shape(

        embedded_inputs = torch.cat([machine_rep, embedded_inputs], dim=1)

        del kwargs["input_ids"]
        kwargs["inputs_embeds"] = embedded_inputs

        if "task_ids" in kwargs:
            del kwargs["task_ids"]

        if "attention_mask" in kwargs:
            ones = torch.ones((batch_size, machine_rep.shape[1]), device=embedded_inputs.device, dtype=kwargs["attention_mask"].dtype)
            input_mask = torch.cat([ones, kwargs["attention_mask"]], dim=1)
            kwargs["attention_mask"] = input_mask

        return kwargs

    def forward(self, **kwargs):
        return self.model(**self.prepare_input(kwargs))

    def generate(self, **kwargs):
        return self.model.generate(**self.prepare_input(kwargs))

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
        torch.save(self.machine_embedder, os.path.join(path, "machine_embedder_params.pt"))

    @staticmethod
    def from_pretrained(path):
        class Dummy:
            def run(self, *args, **kwargs):
                return None

        revived_pretrained_model = FSTPretrainingModel(transformers.AutoModelForSeq2SeqLM.from_pretrained(path),
                                                       Dummy())
        revived_pretrained_model.machine_embedder = torch.load(os.path.join(path, "machine_embedder_params.pt"))

        return revived_pretrained_model



#### Hugging-face compatible version

class SIPPreTrainingConfig(T5Config):

    model_type = "sip_pretrain"

    def __init__(self,
                 num_states: int=16,
                 fst_tokenizer_vocab_size:int=308,
                 state_embedding_dim: int = 64,
                 token_embedding_dim: int = 256,
                 mlp_hidden_dim: Optional[int] = None,
                 final_state_embedding_dim: int = 0,
                 num_final_state_info: int = 3,
                 **kwargs):
        self.state_embedding_dim = state_embedding_dim
        self.token_embedding_dim = token_embedding_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.final_state_embedding_dim = final_state_embedding_dim
        self.num_final_state_info = num_final_state_info
        self.fst_tokenizer_vocab_size = fst_tokenizer_vocab_size
        self.num_states = num_states

        super().__init__(**kwargs)


class SIPPreTrainingModel(PreTrainedModel):
    config_class = SIPPreTrainingConfig

    def __init__(self, config: SIPPreTrainingConfig):
        super().__init__(config)

        self.model = T5ForConditionalGeneration(config)

        self.state_embeddings = torch.nn.Embedding(config.num_states, config.state_embedding_dim)
        self.token_embeddings = torch.nn.Embedding(config.fst_tokenizer_vocab_size, config.token_embedding_dim)
        self.final_state_embedding = torch.nn.Embedding(config.num_final_state_info, config.final_state_embedding_dim)

        self.down_project = None
        if config.mlp_hidden_dim is not None:
            self.down_project = torch.nn.Linear(config.state_embedding_dim * 2 + config.token_embedding_dim * 2 + config.final_state_embedding_dim, config.mlp_hidden_dim)
            self.dropout = torch.nn.Dropout(0.1)
            self.output_layer = torch.nn.Linear(config.mlp_hidden_dim, config.d_model)
        else:
            self.input_layer = torch.nn.Linear(config.state_embedding_dim * 2 + config.token_embedding_dim * 2 + config.final_state_embedding_dim, config.d_model)


    def prepare_input(self, kwargs):
        embedded_inputs = self.model.get_input_embeddings()(kwargs["input_ids"])
        batch_size = embedded_inputs.shape[0]


        ### embed FST
        fst_rep = kwargs.pop("fst_rep")  # shape (batch, transition count, 4 or 5)

        from_rep = self.state_embeddings(fst_rep[:, :, 0])  # shape (batch, transition count, embed dim)
        to_rep = self.state_embeddings(fst_rep[:, :, 3])  # shape (batch, transition count, embed dim)
        io_rep = self.token_embeddings(fst_rep[:, :, 1:3])  # shape (batch, transition count, 2, embed dim)
        io_rep = torch.flatten(io_rep, start_dim=2)

        if fst_rep.shape[-1] == 5:
            final_state_rep = self.final_state_embedding(fst_rep[:, :, 4])
            flat_total_fst_rep = torch.cat([from_rep, to_rep, io_rep, final_state_rep],
                                           dim=2)  # shape (batch, transition, 2* state embed dim + 2 * token embed dim)
        else:
            flat_total_fst_rep = torch.cat([from_rep, to_rep, io_rep],
                                           dim=2)  # shape (batch, transition, 2* state embed dim + 2 * token embed dim)

        if self.down_project is not None:
            machine_rep = self.output_layer(self.dropout(torch.nn.functional.gelu(self.down_project(flat_total_fst_rep))))
        else:
            machine_rep = self.input_layer(flat_total_fst_rep)  # shape (batch, transition, trafo embedding dim)
        # done embedding fst

        # Concatenate FST embedding to the front of the embedded inputs:
        embedded_inputs = torch.cat([machine_rep, embedded_inputs], dim=1)

        del kwargs["input_ids"]
        kwargs["inputs_embeds"] = embedded_inputs

        if "task_ids" in kwargs:
            del kwargs["task_ids"]

        if "attention_mask" in kwargs:
            ones = torch.ones((batch_size, machine_rep.shape[1]), device=embedded_inputs.device, dtype=kwargs["attention_mask"].dtype)
            input_mask = torch.cat([ones, kwargs["attention_mask"]], dim=1)
            kwargs["attention_mask"] = input_mask

        return kwargs

    def forward(self, **kwargs):
        return self.model(**self.prepare_input(kwargs))

    def generate(self, **kwargs):
        return self.model.generate(**self.prepare_input(kwargs))









