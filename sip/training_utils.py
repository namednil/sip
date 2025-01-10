import re
from config_evaluator import Lazy

import torch
import numpy as np


def get_optimizer(model, optimizer, optimizer_groups):
    if optimizer is None:
        optimizer = Lazy(dict(), torch.optim.Adam)

    if optimizer_groups:
        # Example config
        # "optimizer_groups": [
        #     [".*prefix_embedding.*", {"lr": 1.0}],
        #     [".*lm_head.*", {"lr": 1e-5}],
        #     [".*", {"lr": 0.0}]  # all other parameters are frozen
        # ]

        groups = []
        for regex, hyperparam in optimizer_groups:
            h = dict(hyperparam)
            h["params"] = []
            groups.append(h)

        for name, param in model.named_parameters():
            for (regex, _), group in zip(optimizer_groups, groups):
                if re.match(regex, name):
                    group["params"].append(param)
                    break
        # Exclude groups with learning rate 0
        new_groups = []
        for d in groups:
            if "lr" in d and d["lr"] == 0.0:
                for param in d["params"]:
                    param.requires_grad_(False)
            else:
                new_groups.append(d)
        optimizer = optimizer.run(params=new_groups)
    else:
        optimizer = optimizer.run(params=model.parameters())

    return optimizer





