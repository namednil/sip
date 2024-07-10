import json
import os
import sys
from typing import Dict

import neptune
import tqdm


class Logger:

    def ___init__(self, **kwargs):
        pass

    @staticmethod
    def create(**kwargs):
        raise NotImplementedError()
    def progress_bar(self, data_loader):
        return data_loader
    def log_config(self, config: Dict):
        pass

    def log_jsonnet(self, text):
        pass
    def log_metrics(self, name: str, metrics: Dict[str, float]):
        pass

    def stop_logging(self):
        pass


class TqdmLogger(Logger):

    @staticmethod
    def create(**kwargs):
        return TqdmLogger()
    def progress_bar(self, data_loader):
        mininterval = 0.1
        if "TQDM_INTERVAL" in os.environ:
            mininterval = float(os.environ["TQDM_INTERVAL"])
        self.pb = tqdm.tqdm(data_loader, mininterval=mininterval)
        return self.pb
    def log_config(self, config: Dict):
        pass
        # print("CONFIG")
        # print(config)

    def log_metrics(self, name:str, metrics: Dict[str, float]):
        self.pb.set_description(name + ": " + ", ".join(f"{k}={metrics[k]}" for k in sorted(metrics)))



from neptune.utils import stringify_unsupported
class NeptuneLogger(Logger):
    def __init__(self, **kwargs):
        self.run = neptune.init_run(**kwargs)

    def log_config(self, config: Dict):
        # Replace the list in steps by a dictionary because neptune can't handle lists... :(
        d = {f"step_{i+1}": s for i,s in enumerate(config["steps"])}
        config = dict(config)
        config["steps"] = d
        self.run["config"] = stringify_unsupported(config)
        self.run["config_text"] = json.dumps(config, indent=4)
        self.run["config/argv"] = stringify_unsupported(sys.argv)

    def log_jsonnet(self, text):
        self.run["jsonnet"] = text

    def stop_logging(self):
        self.run.stop()
    def log_metrics(self, name: str, metrics: Dict[str, float]):
        for k,v in metrics.items():
            self.run[name + "_" + k].append(v)
    @staticmethod
    def create(**kwargs):
        return NeptuneLogger(**kwargs)


