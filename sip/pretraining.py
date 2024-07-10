import os.path
import random
import re
from typing import Optional, List, Dict, Tuple

import torch.optim.optimizer
import transformers

import numpy as np

from config_evaluator import Lazy
from logger import Logger
from sip.training_utils import get_optimizer
from sip.gauss_prior import Regularizer
from sip.eval_tools import evaluate_on, hack_t5_parallelize

import tqdm

def create_random_task_adapter(model, adapter_str: str = "houlsby"):
    model.add_adapter("task_adapter", adapter_str)
    model.active_adapters = "task_adapter"
    return model

def save_all_adapters(model, where: str):
    model.save_all_adapters(where)

def save_pretrained(model, where: str):
    model.save_pretrained(where)

def load_adapters(model, adapters, set_active: bool = True):
    for adapter in adapters:
        model.load_adapter(adapter, set_active=set_active)
    return model

def scale_grad(model, scaling):
    if scaling is None:
        return
    length = 0.0
    for p in model.parameters():
        if p.grad is not None:
            length += torch.square(p.grad).sum()
    length = torch.sqrt(length)
    if length > scaling:
        length = length / scaling
        for p in model.parameters():
            if p.grad is not None:
                p.grad /= length


def _read_lines_and_sample(fname, num:int, outf):
    with open(fname) as f:
        lines = f.readlines()
        random.shuffle(lines)
        lines = lines[:num]
    with open(outf, "w") as f:
        f.writelines(lines)

def set_seeds(python_seed, pytorch_seed, numpy_seed):
    random.seed(python_seed)
    torch.manual_seed(pytorch_seed)
    np.random.seed(numpy_seed)

def create_random_model(model_str:str, **kwargs):
    return transformers.AutoModelForSeq2SeqLM.from_config(transformers.AutoConfig.from_pretrained(model_str, **kwargs))

def create_model_resize_vocab(model_str, vocab_size: int):
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_str)
    model.config.vocab_size = vocab_size
    d_model = model.get_output_embeddings().in_features
    model.set_input_embeddings(torch.nn.Embedding(vocab_size, d_model))
    model.set_output_embeddings(torch.nn.Linear(d_model, vocab_size))
    if hasattr(model, "final_logits_bias"):
        model.register_buffer("final_logits_bias", torch.zeros(1))
    return model

def create_model_shrink_vocab(model_str, vocab_size: int):
    """
    Loads a pretrained model but shrinks the vocabulary size. The output layer is randomly initialized.
    The input embeddings are initialized to the first vocab_size embeddings of the old model.
    :param model_str:
    :param vocab_size:
    :return:
    """
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_str)
    assert vocab_size < model.config.vocab_size
    model.config.vocab_size = vocab_size
    d_model = model.get_output_embeddings().in_features

    model.set_output_embeddings(torch.nn.Linear(d_model, vocab_size))

    embeddings = torch.nn.Embedding.from_pretrained(model.get_input_embeddings().weight[:vocab_size], freeze=False)
    model.set_input_embeddings(embeddings)
    if hasattr(model, "final_logits_bias"):
        model.register_buffer("final_logits_bias", torch.zeros(1))
    return model

def pretrain(model,
             tokenizer,
             train_data_loader,
             easy_validation_data_loader,
             validation_data_loader,
             save_dir: str,
             train_data_path: str = None, # for taking a sample to put into save_dir
             test_data_loader = None,
             num_epochs: int = 10,
             python_seed: int = 363166917,
             pytorch_seed: int = 682506085,
             numpy_seed: int = 161354504,
             device: str = "cuda:0",
             optimizer: Lazy[torch.optim.Optimizer] = None,
             lr_scheduler: Lazy[torch.optim.lr_scheduler.LRScheduler] = None,
             logger: Optional[Logger] = None,
             grad_scale: Optional[float] = None,
             optimizer_groups: Optional[List[Tuple[str, Dict]]] = None,
             hack_parallelize: bool = False,
             num_accumulation_steps: int = 1,
             sample_size_for_save_dir: int = 400,
             regularizer: Lazy[Regularizer] = None,
             pass_num_training_steps_to_scheduler: bool = True):

    set_seeds(python_seed, pytorch_seed, numpy_seed)

    optimizer = get_optimizer(model, optimizer, optimizer_groups)

    if train_data_path is not None:
        with open(train_data_path, "r"):
            pass


    if hack_parallelize:
        model = hack_t5_parallelize(model)
    elif device is None:
      device = model.device # get device from model
    else:
        if hasattr(model, "model_parallel"):
            model.deparallelize()

        model = model.to(device)

    if regularizer is not None:
        regularizer = regularizer.run(initial_point=model)

    if logger is None:
        logger = Logger()

    model.train()

    if lr_scheduler is not None:
        if pass_num_training_steps_to_scheduler:
            lr_scheduler = lr_scheduler.run(optimizer=optimizer, num_training_steps=num_epochs * len(
                train_data_loader) // num_accumulation_steps)
        else:
            lr_scheduler = lr_scheduler.run(optimizer=optimizer)


    loss = 0
    batch_count = 0
    for _ in range(num_epochs):
        for batch_id, batch in enumerate(logger.progress_bar(train_data_loader)):
            batch = {k: v.to(device) for k,v in batch.items()}
            r = model(**batch)
            if regularizer is not None:
                r.loss += regularizer.apply_reg(model)
            loss += r.loss.detach().cpu().numpy()
            r.loss.backward()
            batch_count += 1
            if batch_count % num_accumulation_steps == 0:
                scale_grad(model, grad_scale)
                optimizer.step()
                optimizer.zero_grad()
                logger.log_metrics("pretrain", {"loss": loss / num_accumulation_steps})
                loss = 0
                if lr_scheduler is not None:
                    lr_scheduler.step()

        # Easy Validation
        if easy_validation_data_loader is not None:
            acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(easy_validation_data_loader))
            logger.log_metrics("pretrain_easy_dev", {"acc": acc, "edit_dist": edit, "per": per})
            print("Easy validation", {"acc": acc, "edit_dist": edit, "per": per})

        #Normal validation
        if validation_data_loader is not None:
            acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(validation_data_loader))
            logger.log_metrics("pretrain_dev", {"acc": acc, "edit_dist": edit, "per": per})
            print("Validation", {"acc": acc, "edit_dist": edit, "per": per})

        model.train()

    if hack_parallelize:
        model.deparallelize()

    if test_data_loader is not None:
        acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(test_data_loader))
        logger.log_metrics("pretrain_test", {"acc": acc, "edit_dist": edit, "per": per})
        print("Validation", {"acc": acc, "edit_dist": edit, "per": per})

    model.save_pretrained(save_dir)

    _read_lines_and_sample(train_data_path, sample_size_for_save_dir, os.path.join(save_dir, "pretraining_sample.jsonl"))

    return model



