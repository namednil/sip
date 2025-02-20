import json
import os
import pickle
from typing import Optional, List, Callable, Mapping, Any, Union

import torch.nn
import transformers
from transformers import AutoTokenizer, PretrainedConfig, T5Config, PreTrainedModel, T5ForConditionalGeneration, \
    AutoModelForSeq2SeqLM



# These are the original functions/classes that were used in the experiments for the paper.
# However, they don't make it very easy to share the models. Hence, there is "hf-compatible" code below for easier model sharing.

def create_struct_prefix(*args, **kwargs):
    return StructuredPrefixEmbeddingModel(*args, **kwargs)

def load_struct_prefix_with_init(model_str: str,
                                  fst_tokenizer_path: str,
                                   tokenizer: AutoTokenizer,
                                   num_examples: int,
                                   prefix_length: int,
                                   random_selection: bool = False,
                                   fst_file_path:str = None,
                                   map_location = None,
                                   *args, **kwargs):
    import sip.fst_pretrain
    from sip.data_loading import load_fst_jsonl
    machine_embedder = torch.load(os.path.join(model_str, "machine_embedder_params.pt"), map_location=map_location,
                                  weights_only=False)

    num_states = machine_embedder.state_embeddings.num_embeddings

    if fst_file_path is None:
        # Open file from standard location.
        fst_file_path = os.path.join(model_str, "pretraining_sample.jsonl")

    data_loader = iter(load_fst_jsonl(fst_file_path, tokenizer, fst_tokenizer=fst_tokenizer_path, num_states=num_states,
                                      batch_size=num_examples, random_order=random_selection,
                                      max_len=prefix_length))
    batch = next(data_loader)

    activations, _ = machine_embedder.prepare_input(batch) #shape (batch, prefix length, embed dim)
    init = activations.mean(dim=0).unsqueeze(0)

    struct_prefix_model = StructuredPrefixEmbeddingModel(model_str, init.shape[1], *args, **kwargs)
    struct_prefix_model.prefix_embedding = torch.nn.Parameter(init.detach(), requires_grad=True)

    return struct_prefix_model


class StructuredPrefixEmbeddingModel(torch.nn.Module):

    def __init__(self, model_str: str,
                 prefix_length: int,
                 adapter_str: str = None,
                 init_strs: Optional[List[str]] = None,
                 tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
                 ignore_mismatched_sizes: bool = False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_str,
                                                                        ignore_mismatched_sizes=ignore_mismatched_sizes)
        self.model_str = model_str

        self.prefix_length = prefix_length

        self.adapter_str = adapter_str

        if adapter_str:
            self.model.add_adapter("task_adapter", adapter_str, set_active=True)

        if init_strs is not None:
            assert tokenizer is not None
            toks = tokenizer(init_strs, padding="max_length", max_length=prefix_length, return_tensors="pt", truncation=True)["input_ids"] # shape (batch, seq_len)
            embeded_toks = self.model.get_input_embeddings()(toks).mean(0, keepdim=True) #shape (1, seq_len, embedding dim)
            self.prefix_embedding = torch.nn.Parameter(embeded_toks.detach())
        else:
            self.prefix_embedding = torch.nn.Parameter(torch.empty(1, self.prefix_length, self.model.get_input_embeddings().embedding_dim))
            torch.nn.init.normal_(self.prefix_embedding)

    def save_pretrained(self, path):
        """
        Saves adapter (if any) and definitely saves prefix embeddings;
        :param path:
        :return:
        """
        self.model.save_all_adapters(path)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"model_str": self.model_str,
                       "prefix_length": self.prefix_length,
                       "adapters_to_load": list(self.model.active_adapters.flatten()) if self.model.active_adapters is not None else [],
                       }, f)
        # with open(os.path.join(path, "active_adapters.pickle"), "wb") as f:
        #     pickle.dump(self.model.active_adapters, f)
        torch.save(self.prefix_embedding, path+"/prefix_embedding.pt")


    @staticmethod
    def from_pretrained(path, add_adapter: Optional[str] = None):
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model = StructuredPrefixEmbeddingModel(model_str=config["model_str"], prefix_length=config["prefix_length"])
        assert len(config["adapters_to_load"]) <= 1

        for adapter_name in config["adapters_to_load"]:
            model.model.load_adapter(os.path.join(path, adapter_name), set_active=True)

        model.prefix_embedding = torch.load(os.path.join(path, "prefix_embedding.pt"),
                                                         map_location=torch.device('cpu') if not torch.cuda.is_available() else None) #(1, seq_length, embedding dim)
        assert model.prefix_length == model.prefix_embedding.shape[1]

        if add_adapter is not None:
            model.model.add_adapter("task_adapter", add_adapter, set_active=True)

        return model


    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def dump_reps(self, group_name, h5f, **kwargs):
        """
        Dumps representations for encoder/decoder states
        :return:
        """
        self(**kwargs, )
        raise NotImplementedError()

    @property
    def device(self):
        return self.model.device

    def prepare_input(self, kwargs):
        """
        Prepends the prefix to the given input.
        :param kwargs:
        :return:
        """
        input_ids = kwargs["input_ids"]

        embedded_inputs = self.model.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]

        prefix = torch.repeat_interleave(self.prefix_embedding, batch_size, 0) #shape (batch, prefix length, embed dim)

        kwargs = dict(kwargs)

        embedded_inputs = torch.cat([prefix, embedded_inputs], dim=1)  # shape (batch, prefix + seq length, embed dim)

        del kwargs["input_ids"]
        kwargs["inputs_embeds"] = embedded_inputs

        if "attention_mask" in kwargs:
            ones = torch.ones((batch_size, self.prefix_length), device=embedded_inputs.device, dtype=kwargs["attention_mask"].dtype)
            input_mask = torch.cat([ones, kwargs["attention_mask"]], dim=1)
            kwargs["attention_mask"] = input_mask

        return kwargs

    def forward(self, **kwargs):
        return self.model(**self.prepare_input(kwargs))

    def generate(self, **kwargs):
        return self.model.generate(**self.prepare_input(kwargs))



# Hf-compatible version

class SIPFinetuningModelConfig(T5Config):

    model_type = "sip_finetune"
    
    def __init__(self,
                 num_examples: int = 32,
                 prefix_length: int = 50,
                 random_selection: bool = True,
                 # don't change these unless you change what the prefix of the model is initialized with:
                 prefix_max_init_length: int = 70,
                 num_precomputed_examples: int = 400,
                 **kwargs):
        # These are all about the initialization of the prefix.
        self.num_examples = num_examples
        self.prefix_length = prefix_length
        self.random_selection = random_selection
        self.prefix_max_init_length = prefix_max_init_length
        self.num_precomputed_examples = num_precomputed_examples
        super().__init__(**kwargs)

class SIPFinetuningModel(PreTrainedModel):
    config_class = SIPFinetuningModelConfig

    def __init__(self, config: SIPFinetuningModelConfig):
        super().__init__(config)

        self.model = T5ForConditionalGeneration(config)

        # Initialize the prefix with NaNs.
        self.register_buffer("prefix_init_tensor", torch.zeros(config.num_precomputed_examples, config.prefix_max_init_length, config.d_model))

        # There are two cases: (1) we initialize the model after SIP-pretraining, i.e. the tunable prefix is not set
        # and (2) the model has been fine-tuned on downstream data, and hence there is meaningful data in the tunable prefix

        # Initialize the prefix with NaNs. If we initialize from SIP-pretraining, this will not be overwritten by a custom version of from_pretrained
        # if we initialize after fine-tuning, the NaNs will be overwritten anyway.

        self.prefix_embedding = torch.nn.Parameter(torch.nan + torch.zeros((1, self.config.prefix_length, self.config.d_model)))
        self.prefix_has_been_initialized = False

    def _initialize_prefix(self):
        prefix_init_tensor = self.prefix_init_tensor
        if self.config.random_selection:
            # randomize selection of FSTs to average for initialization the prefix.
            prefix_init_tensor = prefix_init_tensor[torch.randperm(prefix_init_tensor.shape[0]), :, :]

        prefix_init_tensor = prefix_init_tensor[:self.config.num_examples, :self.config.prefix_length,
                             :]  # shape (num ex, prefix length, d model)
        self.prefix_embedding.data.copy_(prefix_init_tensor.mean(dim=0, keepdims=True))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        model = super(SIPFinetuningModel, cls).from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if torch.all(model.prefix_embedding.isnan()):
            model._initialize_prefix()
        return model


    def prepare_input(self, kwargs):
        """
        Prepends the prefix to the given input.
        :param kwargs:
        :return:
        """
        input_ids = kwargs["input_ids"]

        embedded_inputs = self.model.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]

        prefix = torch.repeat_interleave(self.prefix_embedding, batch_size, 0) #shape (batch, prefix length, embed dim)

        kwargs = dict(kwargs)

        embedded_inputs = torch.cat([prefix, embedded_inputs], dim=1)  # shape (batch, prefix + seq length, embed dim)

        del kwargs["input_ids"]
        kwargs["inputs_embeds"] = embedded_inputs

        if "attention_mask" in kwargs:
            ones = torch.ones((batch_size, self.config.prefix_length), device=embedded_inputs.device, dtype=kwargs["attention_mask"].dtype)
            input_mask = torch.cat([ones, kwargs["attention_mask"]], dim=1)
            kwargs["attention_mask"] = input_mask

        return kwargs

    def forward(self, **kwargs):
        return self.model(**self.prepare_input(kwargs))

    def generate(self, **kwargs):
        return self.model.generate(**self.prepare_input(kwargs))


    def get_optimizer(self, optimizer: Callable[..., torch.optim.Optimizer], prefix_lr:float = 1.0, **kwargs) -> torch.optim.Optimizer:
        """
        Return an optimizer that uses a different learning rate (typically higher) for the prefix than for the rest of the model.
        """

        prefix_params = []
        other_params = []
        for name, param in self.named_parameters():
            if name == "prefix_embedding":
                prefix_params.append(param)
            else:
                other_params.append(param)
        return optimizer(params=[{"params": prefix_params, "lr": prefix_lr}, {"params": other_params}], **kwargs)


from transformers import AutoConfig, AutoModel

AutoConfig.register("sip_finetune", SIPFinetuningModelConfig)
AutoModel.register(SIPFinetuningModelConfig, SIPFinetuningModel)
AutoModelForSeq2SeqLM.register(SIPFinetuningModelConfig, SIPFinetuningModel)


