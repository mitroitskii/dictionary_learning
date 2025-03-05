"""
Training dictionaries
"""

import json
import os
from collections import defaultdict
import torch as t
from tqdm import tqdm
import wandb

from .trainers.crosscoder import CrossCoderTrainer
from .utils import ExitHandler


def get_stats(
    trainer,
    act: t.Tensor,
    deads_sum: bool = True,
):
    with t.no_grad():
        act, act_hat, f, losslog = trainer.loss(
            act, step=0, logging=True, return_deads=True
        )

    # L0
    l0 = (f != 0).float().detach().cpu().sum(dim=-1).mean().item()

    out = {
        "l0": l0,
        **{f"{k}": v for k, v in losslog.items() if k != "deads"},
    }
    if losslog["deads"] is not None:
        total_feats = losslog["deads"].shape[0]
        out["frac_deads"] = (
            losslog["deads"].sum().item() / total_feats
            if deads_sum
            else losslog["deads"]
        )

    # fraction of variance explained
    if act.dim() == 2:
        # act.shape: [batch, d_model]
        # fraction of variance explained
        total_variance = t.var(act, dim=0).sum()
        residual_variance = t.var(act - act_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
    else:
        # act.shape: [batch, layer, d_model]
        total_variance_per_layer = []
        residual_variance_per_layer = []

        for l in range(act_hat.shape[1]):
            total_variance_per_layer.append(
                t.var(act[:, l, :], dim=0).cpu().sum())
            residual_variance_per_layer.append(
                t.var(act[:, l, :] - act_hat[:, l, :], dim=0).cpu().sum()
            )
            out[f"cl{l}_frac_variance_explained"] = (
                1 - residual_variance_per_layer[l] /
                total_variance_per_layer[l]
            )
        total_variance = sum(total_variance_per_layer)
        residual_variance = sum(residual_variance_per_layer)
        frac_variance_explained = 1 - residual_variance / total_variance

    out["frac_variance_explained"] = frac_variance_explained.item()
    return out


def log_stats(
    trainer,
    step: int,
    act: t.Tensor,
    activations_split_by_head: bool,
    transcoder: bool,
    stage: str = "train",
):
    with t.no_grad():
        log = {}
        if activations_split_by_head:  # x.shape: [batch, pos, n_heads, d_head]
            act = act[..., 0, :]
        if not transcoder:
            stats = get_stats(trainer, act)
            log.update({f"{stage}/{k}": v for k, v in stats.items()})
        else:  # transcoder
            x, x_hat, f, losslog = trainer.loss(act, step=step, logging=True)
            # L0
            l0 = (f != 0).float().sum(dim=-1).mean().item()
            log[f"{stage}/l0"] = l0

        # log parameters from training
        log["step"] = step
        trainer_log = trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            log[f"{stage}/{name}"] = value

        wandb.log(log, step=step)


@t.no_grad()
def run_validation(
    trainer,
    validation_data,
    step: int = None,
):
    l0 = []
    frac_variance_explained = []
    frac_variance_explained_per_feature = []
    deads = []
    if isinstance(trainer, CrossCoderTrainer):
        frac_variance_explained_per_layer = defaultdict(list)
    for val_step, act in enumerate(tqdm(validation_data, total=len(validation_data), desc="Validating")):
        act = act.to(trainer.device)
        stats = get_stats(trainer, act, deads_sum=False)
        l0.append(stats["l0"])
        if "frac_deads" in stats:
            deads.append(stats["frac_deads"])
        if "frac_variance_explained" in stats:
            frac_variance_explained.append(stats["frac_variance_explained"])
        if "frac_variance_explained_per_feature" in stats:
            frac_variance_explained_per_feature.append(
                stats["frac_variance_explained_per_feature"]
            )

        if isinstance(trainer, CrossCoderTrainer):
            for l in range(act.shape[1]):
                if f"cl{l}_frac_variance_explained" in stats:
                    frac_variance_explained_per_layer[l].append(
                        stats[f"cl{l}_frac_variance_explained"]
                    )

    log = {}
    if len(deads) > 0:
        log["val/frac_deads"] = t.stack(deads).all(dim=0).float().mean().item()
    if len(l0) > 0:
        log["val/l0"] = t.tensor(l0).mean().item()
    if len(frac_variance_explained) > 0:
        log["val/frac_variance_explained"] = t.tensor(
            frac_variance_explained).mean()
    if len(frac_variance_explained_per_feature) > 0:
        frac_variance_explained_per_feature = t.stack(
            frac_variance_explained_per_feature
        ).cpu()  # [num_features]
        log["val/frac_variance_explained_per_feature"] = (
            frac_variance_explained_per_feature
        )
    if isinstance(trainer, CrossCoderTrainer):
        for l in frac_variance_explained_per_layer:
            log[f"val/cl{l}_frac_variance_explained"] = t.tensor(
                frac_variance_explained_per_layer[l]
            ).mean()
    if step is not None:
        log["step"] = step
    wandb.log(log, step=step)

    return log


def save_model(trainer, checkpoint_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    t.save(trainer.model.state_dict(), os.path.join(save_dir, checkpoint_name))


def trainSAE(
    data,
    trainer_config,
    use_wandb=False,
    wandb_entity="",
    wandb_project="",
    steps=None,
    save_steps=None,
    save_dir=None,
    log_steps=None,
    activations_split_by_head=False,
    validate_every_n_steps=None,
    validation_data=None,
    transcoder=False,
    run_cfg={},
    end_of_step_logging_fn=None,
    save_last_eval=True,
    start_of_training_eval=False,
):
    """
    Train SAE using the given trainer
    """
    assert not (
        validation_data is None and validate_every_n_steps is not None
    ), "Must provide validation data if validate_every_n_steps is not None"

    trainer_class = trainer_config["trainer"]
    del trainer_config["trainer"]
    trainer = trainer_class(**trainer_config)

    wandb_config = trainer.config | run_cfg

    wandb.init(
        entity=wandb_entity,
        project=wandb_project,
        config=wandb_config,
        name=wandb_config["wandb_name"],
        mode="disabled" if not use_wandb else "online",
    )

    # make save dir, export config
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        # save config
        config = {"trainer": trainer.config}
        try:
            config["buffer"] = data.config
        except:
            pass
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    # Setup save function for graceful exit
    def save_model_on_exit():
        if save_dir is not None:
            save_model(trainer, "model_interrupted.pt", save_dir)

    # Initialize signal handler
    exit_handler = ExitHandler(save_function=save_model_on_exit)

    for step, act in enumerate(tqdm(data, total=steps, desc="Training")):
        # Check if we need to exit due to interrupt
        if exit_handler.should_exit():
            exit_handler.save_and_exit()

        if steps is not None and step >= steps:
            break

        act = act.to(trainer.device)

        # logging
        if log_steps is not None and step % log_steps == 0 and step != 0:
            log_stats(trainer, step, act,
                      activations_split_by_head, transcoder)

        # saving
        if save_steps is not None and step > 0 and step % save_steps == 0:
            print(f"Saving at step {step}")
            print()
            if save_dir is not None:
                save_model(trainer, f"checkpoint_{step}.pt", save_dir)

        # training
        trainer.update(step, act)

        if (
            validate_every_n_steps is not None
            and step % validate_every_n_steps == 0
            and (start_of_training_eval or step > 0)
        ):
            print(f"Validating at step {step}")
            print()
            logs = run_validation(trainer, validation_data, step=step)
            try:
                os.makedirs(save_dir, exist_ok=True)
                t.save(logs, os.path.join(save_dir, f"eval_logs_{step}.pt"))
            except:
                pass

        if end_of_step_logging_fn is not None:
            end_of_step_logging_fn(trainer, step)
    
    if validation_data is not None:
        try:
            last_eval_logs = run_validation(trainer, validation_data, step=step)
            if save_last_eval:
                os.makedirs(save_dir, exist_ok=True)
            t.save(last_eval_logs, os.path.join(
                save_dir, "last_eval_logs.pt"))
        except Exception as e:
            print(f"Error during final validation: {str(e)}")
            print()

    # save final SAE
    if save_dir is not None:
        save_model(trainer, "ae.pt", save_dir)

    if use_wandb:
        wandb.finish()
