from typing import Optional, List, Dict, Tuple

import torch.optim.optimizer

from config_evaluator import Lazy
from logger import Logger
from sip.training_utils import get_optimizer
from sip.gauss_prior import Regularizer
from sip.eval_tools import evaluate_on, MovingAvg

from sip.pretraining import scale_grad
from sip.data_loading import RandomSplit

def finetune_model(model,
                  tokenizer,
                  train_data_loader,
                  validation_data_loader,
                  dataset_splitter: Optional[RandomSplit] = None,
                  num_epochs: int = 110,
                  moving_avg_steps: int = 10,
                  device: str = "cuda:0",
                  optimizer: Lazy[torch.optim.Optimizer] = None,
                  logger: Optional[Logger] = None,
                  grad_scale: Optional[float] = None,
                  optimizer_groups: Optional[List[Tuple[str, Dict]]] = None,
                  num_accumulation_steps: int = 1,
                  eval_only_last_epochs: bool = False,
                  regularizer: Lazy[Regularizer] = None,
                  use_deterministic_algorithms: bool = False
                  ):
    optimizer = get_optimizer(model, optimizer, optimizer_groups)

    torch.use_deterministic_algorithms(use_deterministic_algorithms)

    model = model.to(device)

    if dataset_splitter is not None:
        if train_data_loader is not None or validation_data_loader is not None:
            raise ValueError("dataset_splitter given, so train_data_loader and validation_data_loader must be None")
        train_data_loader = dataset_splitter.get_train_loader()
        validation_data_loader = dataset_splitter.get_rest_loader()

    if regularizer is not None:
        regularizer = regularizer.run(initial_point=model)

    avg_acc = MovingAvg(moving_avg_steps)
    avg_edit = MovingAvg(moving_avg_steps)
    avg_per = MovingAvg(moving_avg_steps)

    if logger is None:
        logger = Logger()

    batch_count = 0
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for batch_id, batch in enumerate(logger.progress_bar(train_data_loader)):
            batch = {k: v.to(device) for k,v in batch.items()}
            r = model(**batch)
            if regularizer is not None:
                r.loss += regularizer.apply_reg(model)
            epoch_loss += r.loss.detach().cpu().numpy()
            r.loss.backward()
            batch_count += 1
            if batch_count % num_accumulation_steps == 0:
                scale_grad(model, grad_scale)
                optimizer.step()
                optimizer.zero_grad()
        logger.log_metrics("finetune_train", {"loss": epoch_loss})
        print("loss", epoch_loss)

        # Evaluate
        if (not eval_only_last_epochs) or epoch >= num_epochs - moving_avg_steps:
            acc, edit, per = evaluate_on(model, tokenizer, logger.progress_bar(validation_data_loader))
            logger.log_metrics("finetune_dev", {"acc": acc, "edit_dist": edit, "per": per,
                                                            f"acc_avg_{moving_avg_steps}": avg_acc.append(acc),
                                                            f"edit_dist_avg_{moving_avg_steps}": avg_edit.append(edit),
                                                            f"per_avg_{moving_avg_steps}": avg_per.append(per)})
            print({"acc": acc, "edit_dist": edit, "per": per})

    return model



