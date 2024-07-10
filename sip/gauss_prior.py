import torch
import re


class Regularizer:
    def __init__(self, initial_point: torch.nn.Module):
        pass

    def apply_reg(self, model: torch.nn.Module) -> torch.Tensor:
        raise NotImplementedError()

class IsotropicGaussPrior(Regularizer):
    def __init__(self, initial_point: torch.nn.Module, coeff: float, regex: str = ".*", invert_regex: bool = False):
        """
        Regex determines which parameters the prior should be applied to
        :param initial_point:
        :param regex:
        :param invert_regex:
        """
        super().__init__(initial_point)
        self.coeff = coeff
        self.initial_params = {k: v.detach().clone() for k, v in initial_point.named_parameters()
                               if (not invert_regex and re.fullmatch(regex, k)) or (invert_regex and re.fullmatch(regex, k) is None)}

    def apply_reg(self, model: torch.nn.Module):
        reg = 0
        model_dict = dict(model.named_parameters())
        for k,v in self.initial_params.items():
            reg += torch.sum(torch.square(model_dict[k] - v))
        return self.coeff * reg


