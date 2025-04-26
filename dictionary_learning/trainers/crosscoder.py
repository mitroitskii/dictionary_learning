"""
Implements the standard SAE training scheme.
"""

import torch as th
import torch.nn as nn
from ..trainers.trainer import SAETrainer
from ..config import DEBUG
from ..dictionary import CrossCoder, BatchTopKCrossCoder
from collections import namedtuple
from tqdm import tqdm
from typing import Optional
from ..trainers.trainer import (
    get_lr_schedule,
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
)
from ..utils import dtype_to_str

# TODO: add activation dtype and type conversion to other types of dicts
# - modify dictionary.py to support activation dtype
# - modify trainers

class CrossCoderTrainer(SAETrainer):
    """
    Standard SAE training scheme for cross-coding.
    """

    def __init__(
        self,
        dict_class=CrossCoder,
        activation_dtype = "float32",
        num_layers=2,
        activation_dim=512,
        dict_size=64 * 512,
        lr=1e-3,
        l1_penalty=1e-1,
        warmup_steps=1000,  # lr warmup period at start of training and after each resample
        resample_steps=None,  # how often to resample neurons
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="CrossCoderTrainer",
        submodule_name=None,
        compile=False,
        dict_class_kwargs={},
        pretrained_ae=None,
        use_mse_loss=False,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.compile = compile
        self.use_mse_loss = use_mse_loss
        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

        # convert activation dtype to th.dtype
        if activation_dtype.lower() == "float32":
            activation_dtype = th.float32
        elif activation_dtype.lower() == "float16":
            activation_dtype = th.float16
        elif activation_dtype.lower() == "bfloat16":
            activation_dtype = th.bfloat16
        else:
            raise ValueError(f"Unsupported activation dtype: {activation_dtype}")

        # initialize dictionary
        if pretrained_ae is None:
            self.ae = dict_class(
                activation_dtype, activation_dim, dict_size, num_layers=num_layers, **dict_class_kwargs
            )
        else:
            self.ae = pretrained_ae

        if compile:
            self.ae = th.compile(self.ae)

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.wandb_name = wandb_name

        if device is None:
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.ae.to(self.device)

        self.resample_steps = resample_steps

        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = th.zeros(self.ae.dict_size, dtype=int).to(
                self.device
            )
        else:
            self.steps_since_active = None

        self.optimizer = th.optim.Adam(self.ae.parameters(), lr=lr)
        if resample_steps is None:

            def warmup_fn(step):
                return min(step / warmup_steps, 1.0)

        else:

            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.0)

        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_fn
        )

    def resample_neurons(self, deads, activations):
        with th.no_grad():
            if deads.sum() == 0:
                return
            self.ae.resample_neurons(deads, activations)
            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()["state"]
            ## encoder weight
            state_dict[0]["exp_avg"][:, :, deads] = 0.0
            state_dict[0]["exp_avg_sq"][:, :, deads] = 0.0
            ## encoder bias
            state_dict[1]["exp_avg"][deads] = 0.0
            state_dict[1]["exp_avg_sq"][deads] = 0.0
            ## decoder weight
            state_dict[3]["exp_avg"][:, deads, :] = 0.0
            state_dict[3]["exp_avg_sq"][:, deads, :] = 0.0

    def loss(self, x, logging=False, return_deads=False, **kwargs):
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = th.linalg.norm(x - x_hat, dim=-1).mean()
        mse_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        if self.use_mse_loss:
            recon_loss = mse_loss
        else:
            recon_loss = l2_loss
        l1_loss = f.norm(p=1, dim=-1).mean()
        deads = (f <= 1e-4).all(dim=0)
        if self.steps_since_active is not None:
            # update steps_since_active
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0

        loss = recon_loss + self.l1_penalty * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "mse_loss": mse_loss.item(),
                    "sparsity_loss": l1_loss.item(),
                    "loss": loss.item(),
                    "deads": deads if return_deads else None,
                },
            )

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(
                self.steps_since_active > self.resample_steps / 2, activations
            )

    @property
    def config(self):
        return {
            "dict_class": (
                self.ae.__class__.__name__
                if not self.compile
                else self.ae._orig_mod.__class__.__name__
            ),
            "trainer_class": self.__class__.__name__,
            "activation_dtype": dtype_to_str(self.ae.activation_dtype),
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "lr": self.lr,
            "l1_penalty": self.l1_penalty,
            "warmup_steps": self.warmup_steps,
            "resample_steps": self.resample_steps,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "use_mse_loss": self.use_mse_loss,
            "sparsity_loss_type": str(self.ae.sparsity_loss_type),
            "sparsity_loss_alpha_sae": self.ae.sparsity_loss_alpha_sae,
            "sparsity_loss_alpha_cc": self.ae.sparsity_loss_alpha_cc,
        }


class BatchTopKCrossCoderTrainer(SAETrainer):
    def __init__(
        self,
        steps: int,  # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        k: int,
        layer: int,
        lm_name: str,
        num_layers: int = 2,
        activation_dtype = "float32",
        dict_class: type = BatchTopKCrossCoder,
        lr: Optional[float] = None,
        auxk_alpha: float = 1 / 32,
        warmup_steps: int = 1000,
        decay_start: Optional[int] = None,  # when does the lr decay start
        threshold_beta: float = 0.999,
        threshold_start_step: int = 1000,
        seed: Optional[int] = None,
        device: Optional[str] = None,
        wandb_name: str = "BatchTopKSAE",
        submodule_name: Optional[str] = None,
        pretrained_ae: Optional[BatchTopKCrossCoder] = None,
        dict_class_kwargs: dict = {},
    ):
        super().__init__(seed)
        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name
        self.steps = steps
        self.decay_start = decay_start
        self.warmup_steps = warmup_steps
        self.k = k
        self.threshold_beta = threshold_beta
        self.threshold_start_step = threshold_start_step

        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

        # convert activation dtype to th.dtype
        if activation_dtype.lower() == "float32":
            activation_dtype = th.float32
        elif activation_dtype.lower() == "float16":
            activation_dtype = th.float16
        elif activation_dtype.lower() == "bfloat16":
            activation_dtype = th.bfloat16
        else:
            raise ValueError(f"Unsupported activation dtype: {activation_dtype}")

        # initialize dictionary
        if pretrained_ae is None:
            self.ae = dict_class(
                activation_dtype, activation_dim, dict_size, num_layers, k, **dict_class_kwargs
            )
        else:
            self.ae = pretrained_ae

        if device is None:
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        if lr is not None:
            self.lr = lr
        else:
            # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
            scale = dict_size / (2**14)
            self.lr = 2e-4 / scale**0.5

        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        self.top_k_aux = activation_dim // 2  # Heuristic from B.1 of the paper
        self.num_tokens_since_fired = th.zeros(dict_size, dtype=th.long, device=device)
        self.logging_parameters = [
            "effective_l0",
            "running_deads",
            "pre_norm_auxk_loss",
        ]
        self.dict_class_kwargs = dict_class_kwargs
        self.effective_l0 = -1
        self.running_deads = -1
        self.pre_norm_auxk_loss = -1

        self.optimizer = th.optim.Adam(
            self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999)
        )

        lr_fn = get_lr_schedule(steps, warmup_steps, decay_start=decay_start)

        self.scheduler = th.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

    def get_auxiliary_loss(
        self,
        residual_BD: th.Tensor,
        post_relu_f: th.Tensor,
        post_relu_f_scaled: th.Tensor,
    ):
        batch_size, num_layers, model_dim = residual_BD.size()
        # reshape to (batch_size, num_layers*model_dim)
        residual_BD = residual_BD.reshape(batch_size, -1)
        dead_features = self.num_tokens_since_fired >= self.dead_feature_threshold
        self.running_deads = int(dead_features.sum())

        if dead_features.sum() > 0:
            k_aux = min(self.top_k_aux, dead_features.sum())

            auxk_latents_scaled = th.where(
                dead_features[None], post_relu_f_scaled, -th.inf
            ).detach()

            # Top-k dead latents
            auxk_acts_scaled, auxk_indices = auxk_latents_scaled.topk(
                k_aux, sorted=False
            )
            auxk_buffer_BF = th.zeros_like(post_relu_f)
            row_indices = (
                th.arange(post_relu_f.size(0), device=post_relu_f.device)
                .view(-1, 1)
                .expand(-1, auxk_indices.size(1))
            )
            auxk_acts_BF = auxk_buffer_BF.scatter_(
                dim=-1, index=auxk_indices, src=post_relu_f[row_indices, auxk_indices]
            )

            # Note: decoder(), not decode(), as we don't want to apply the bias
            x_reconstruct_aux = self.ae.decoder(auxk_acts_BF, add_bias=False)
            x_reconstruct_aux = x_reconstruct_aux.reshape(batch_size, -1)
            l2_loss_aux = (
                (residual_BD.float() - x_reconstruct_aux.float())
                .pow(2)
                .sum(dim=-1)
                .mean()
            )

            self.pre_norm_auxk_loss = l2_loss_aux

            # normalization from OpenAI implementation: https://github.com/openai/sparse_autoencoder/blob/main/sparse_autoencoder/kernels.py#L614
            residual_mu = residual_BD.mean(dim=0)[None, :].broadcast_to(
                residual_BD.shape
            )
            loss_denom = (
                (residual_BD.float() - residual_mu.float()).pow(2).sum(dim=-1).mean()
            )
            normalized_auxk_loss = l2_loss_aux / loss_denom

            return normalized_auxk_loss.nan_to_num(0.0)
        else:
            self.pre_norm_auxk_loss = -1
            return th.tensor(0, dtype=residual_BD.dtype, device=residual_BD.device)

    def update_threshold(self, f_scaled: th.Tensor):
        device_type = "cuda" if f_scaled.is_cuda else "cpu"
        active = f_scaled[f_scaled > 0]

        if active.size(0) == 0:
            min_activation = 0.0
        else:
            min_activation = active.min().detach().to(dtype=th.float32)

        if self.ae.threshold < 0:
            self.ae.threshold = min_activation
        else:
            self.ae.threshold = (self.threshold_beta * self.ae.threshold) + (
                (1 - self.threshold_beta) * min_activation
            )

    def loss(self, x, step=None, logging=False, use_threshold=False, **kwargs):
        f, f_scaled, active_indices_F, post_relu_f, post_relu_f_scaled = self.ae.encode(
            x, return_active=True, use_threshold=use_threshold
        )  # (batch_size, dict_size)
        # l0 = (f != 0).float().sum(dim=-1).mean().item()

        if step > self.threshold_start_step and not logging:
            self.update_threshold(f_scaled)

        x_hat = self.ae.decode(f)

        e = x - x_hat
        assert e.shape == x.shape

        self.effective_l0 = self.k

        num_tokens_in_step = x.size(0)
        did_fire = th.zeros_like(self.num_tokens_since_fired, dtype=th.bool)
        did_fire[active_indices_F] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0

        mse_loss = e.pow(2).sum(dim=-1).mean()
        l2_loss = th.linalg.norm(e, dim=-1).mean()
        auxk_loss = self.get_auxiliary_loss(e.detach(), post_relu_f, post_relu_f_scaled)
        loss = l2_loss + self.auxk_alpha * auxk_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "mse_loss": mse_loss.item(),
                    "l2_loss": l2_loss.item(),
                    "auxk_loss": auxk_loss.item(),
                    "loss": loss.item(),
                    "deads": ~did_fire,
                    "threshold": self.ae.threshold.item(),
                    "sparsity_weight": self.ae.get_sparsity_loss_weight().mean().item(),
                },
            )

    def update(self, step, x):
        x = x.to(self.device)
        if step == 0:
            median = self.geometric_median(x)
            median = median.to(self.device)
            self.ae.decoder.bias.data = median

        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        th.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()

        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "BatchTopKCrossCoderTrainer",
            "dict_class": "BatchTopKCrossCoder",
            "lr": self.lr,
            "steps": self.steps,
            "auxk_alpha": self.auxk_alpha,
            "warmup_steps": self.warmup_steps,
            "decay_start": self.decay_start,
            "threshold_beta": self.threshold_beta,
            "threshold_start_step": self.threshold_start_step,
            "top_k_aux": self.top_k_aux,
            "seed": self.seed,
            "activation_dtype": dtype_to_str(self.ae.activation_dtype),
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "k": self.ae.k.item(),
            "sparsity_loss_type": str(self.ae.sparsity_loss_type),
            "sparsity_loss_alpha_sae": self.ae.sparsity_loss_alpha_sae,
            "sparsity_loss_alpha_cc": self.ae.sparsity_loss_alpha_cc,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "dict_class_kwargs": self.dict_class_kwargs,
        }

    @staticmethod
    def geometric_median(points: th.Tensor, max_iter: int = 100, tol: float = 1e-5):
        # points.shape = (num_points, num_layers, model_dim)
        guess = points.mean(dim=0)
        prev = th.zeros_like(guess)
        weights = th.ones((len(points), points.shape[1]), device=points.device)

        for _ in range(max_iter):
            prev = guess
            weights = 1 / th.norm(points - guess, dim=-1)  # (num_points, num_layers)
            weights /= weights.sum(dim=0, keepdim=True)  # (num_points, num_layers)
            guess = (weights.unsqueeze(-1) * points).sum(
                dim=0
            )  # (num_layers, model_dim)
            if th.all(th.norm(guess - prev, dim=-1) < tol):
                break

        return guess
