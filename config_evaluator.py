import random
from typing import TypeVar, Callable, Generic

import numpy as np
import rjsonnet
import importlib
import json

T = TypeVar('T')

from logger import *


class Lazy(Generic[T]):
    def __init__(self, args, f: Callable[..., T]):
        self.args = args
        self.f = f

    def run(self, **kwargs) -> T:
        d = dict(self.args)
        d.update(kwargs)
        return self.f(**d)

def config_eval(fname, **kwargs):
    d = dict(json.loads(rjsonnet.evaluate_file(fname, ext_vars=dict(os.environ), **kwargs)))
    imports = d.get("imports", [])
    backup_globals = dict(globals())
    for x in imports:
        exec(x, globals())

    if "random_seed" in d:
        random.seed(d["random_seed"])

    if "numpy_seed" in d:
        np.random.seed(d["numpy_seed"])

    if "pytorch_seed" in d:
        import torch
        torch.manual_seed(d["pytorch_seed"])

    results = dict()

    def rec_eval(subdict, logger):
        if subdict == "[logger]":
            if logger is None:
                raise ValueError("[logger] is not defined")
            return logger

        if not isinstance(subdict, dict):
            return subdict

        elif "[lazy]" in subdict:
            fname = subdict["[lazy]"]
            args = {k: rec_eval(v, logger) for k,v in subdict.items() if k != "[lazy]"}

            return Lazy(args, eval(fname))

        elif "[ref]" in subdict:
            if subdict["[ref]"] not in results:
                raise ValueError(f"Wanted to get result from step '{subdict['[ref]']}'"
                                 f" but that could not be found (yet). Available are: {list(results.keys())}")
            return results[subdict["[ref]"]]
        elif "f" in subdict:
            fname = subdict["f"]
            r = {k: rec_eval(v, logger) for k,v in subdict.items() if k != "f"}
            return eval(fname)(**r)

    if not isinstance(d["steps"], list): # if there's a single step only, then we don't need to specify a list
        d["steps"] = [d["steps"]]

    logger = None
    if "logger" in d:
        logger : Logger = rec_eval(d["logger"], None)
        logger.log_config(d)
        with open(fname) as f:
            logger.log_jsonnet(f.read())

    #TODO: before running everything, do a dry-run to see if all function names are defined

    for step in d["steps"]:
        if "name" not in step and "f" not in step:
            raise ValueError("'name' and 'f' missing for step")
        elif "name" not in step:
            raise ValueError(f"Name missing for step. It's value for f is: {step['f']}")
        name = step["name"]
        copy = dict(step)
        del copy["name"]
        results[name] = rec_eval(copy, logger)

    # Restore globals() from backup
    for k,v in backup_globals.items():
        globals()[k] = v
    for key in list(globals()):
        if key not in backup_globals:
            del globals()[key]

    logger.stop_logging()

    return results

# Code for tests
def _add(x,y):
    return x+y

if __name__ == "__main__":
    import sys
    config_eval(sys.argv[1])
    # d = rjsonnet.evaluate_file("configs/test.jsonnet")
    # lazy_func = config_eval("configs/lazy_test.jsonnet")["s1"]
    # print(lazy_func(y=1))
