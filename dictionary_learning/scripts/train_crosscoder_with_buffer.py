"""
Train a Crosscoder using live activations from models.

This script uses the ActivationBuffer to collect activations from two models in real-time
and trains a CrossCoder on these paired activations, with proper train/validation split
and configurable data types. There are two buffers, one for training and one for validation.
"""

import argparse
import random
import os
import torch as th
from datasets import load_dataset
from nnsight import LanguageModel

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import CrossCoder
from dictionary_learning.trainers import CrossCoderTrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator, get_submodule

# Set high precision for float32 matrix multiplications to improve performance and numerical stability for GPUs with f32 tensor cores (does nothing otherwise)
th.set_float32_matmul_precision('high') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base and Finetuned models arguments
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="Base model to use")
    parser.add_argument("--ft-model", type=str, default="agentica-org/DeepScaleR-1.5B-Preview",
                        help="Finetuned model to use")
    parser.add_argument("--layer", type=int, default=20,
                        help="Layer to extract activations from")
    parser.add_argument("--io", type=str, default="out", choices=["out", "in"],
                        help="Which side of the layer to get activations from (input or output)")
    parser.add_argument("--model-devices", type=str, nargs="+", default=["cuda:1", "cuda:1"],
                        help="Devices to load models on (auto, cpu, cuda) for base and finetuned models [BASE_MODEL, FINTUNED_MODEL]. Provide one device per model.")
    parser.add_argument("--models-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Data type for bases / ft models activations (bf16, fp16, or fp32)")

    # CrossCoder arguments
    parser.add_argument("--expansion-factor", type=int, default=32,
                        help="Dictionary size as multiple of activation dimension")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="How many token activations to load into crosscoder at once")
    parser.add_argument("--train-tokens", type=int, default=150_000_000,
                        help="Number of tokens to train the crosscoder on")
    parser.add_argument("--mu", type=float, default=3.6e-2,
                        help="L1 penalty")
    parser.add_argument("--same-init-for-all-layers", action="store_true",
                        help="Use same initialization for all layers")
    parser.add_argument("--norm-init-scale", type=float, default=0.005,
                        help="Scale for normal initialization")
    parser.add_argument("--init-with-transpose", action="store_true",
                        help="Initialize decoder with transpose of encoder")
    parser.add_argument("--encoder-layers", type=int, default=None, nargs="+",
                        help="Encoder layer sizes")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model")

    parser.add_argument("--out-dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="Data type for crosscoder parameters")
    parser.add_argument("--crosscoder-device", type=str, default="cuda:0", choices=["auto", "cpu", "cuda"],
                        help="Device to train CrossCoder on (auto, cpu, cuda)")

    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, default="mitroitskii/OpenR1-Math-220k-formatted",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Local path to dataset. If provided, loads from local path instead of HuggingFace")
    parser.add_argument("--dataset-field", type=str, default="message_qwen1.5b",
                        help="Field name in the dataset to use (e.g., 'text', 'message_qwen1.5b')")
    parser.add_argument("--context-length", type=int, default=1024,
                        help="Context length for tokenization")

    # Buffer arguments
    parser.add_argument("--buffer-ctx-batch-size", type=int, default=1,
                        help="How many input contexts (sequences of tokens) to run through the models (and buffer activations for) at once.")
    parser.add_argument("--buffer-size-multiplier", type=int, default=150,
                        help="How many times the buffer size should be larger than the crosscoder batch size")
    parser.add_argument("--buffer-device", type=str, default="cuda:0",
                        help="Device to store buffer activations on (auto, cpu, cuda).")

    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--resample-steps", type=int, default=None,
                        help="Resample dead neurons every N steps")
    
    # Misc arguments
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb-entity", type=str, default="",
                        help="Weights & Biases entity")
    parser.add_argument("--disable-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--save-every-n-steps", type=int, default=5000,
                        help="Save model every N steps")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable saving model checkpoints")

    args = parser.parse_args()

    print(f"Training args: {args}")
    print()

    # Set random seeds for reproducibility for all random number generators
    random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    # Map string dtype arguments to torch dtypes
    dtype_map = {
        "bf16": th.bfloat16,
        "fp16": th.float16,
        "fp32": th.float32
    }
    model_dtype = dtype_map[args.models_dtype]
    out_dtype = dtype_map[args.out_dtype]

    # Load models with specified dtype
    print(f"Loading models: {args.base_model} and {args.ft_model}")
    print()
    base_model = LanguageModel(
        args.base_model, device_map=args.model_devices[0], torch_dtype=model_dtype)
    ft_model = LanguageModel(
        args.ft_model, device_map=args.model_devices[1], torch_dtype=model_dtype)

    base_submodule = get_submodule(base_model, args.layer)
    ft_submodule = get_submodule(ft_model, args.layer)
    submodule_name = f"resid_post_layer_{args.layer}"

    # Load the dataset
    print(f"Loading dataset: {args.dataset_name}")
    print()
    if args.dataset_path is not None:
        print(
            f"Loading from local path: {args.dataset_path}")
        print()
        generator = hf_dataset_to_generator(args.dataset_path, split="train", field=args.dataset_field)
    else:
        # Load from Hugging Face
        generator = hf_dataset_to_generator(args.dataset_name, split="train", field=args.dataset_field)

    # Determine activation dimension
    if args.io == "out":
        if hasattr(base_submodule, 'out_features'):
            activation_dim = base_submodule.out_features
        else:
            activation_dim = base_model.config.hidden_size
    else:
        if hasattr(base_submodule, 'in_features'):
            activation_dim = base_submodule.in_features
        else:
            activation_dim = base_model.config.hidden_size

    # Dictionary size calculation
    dictionary_size = args.expansion_factor * activation_dim

    # Buffer size calculation
    # Check buffer size multiplier and ensure it's positive
    if args.buffer_size_multiplier <= 0:
        print(
            f"Warning: buffer_size_multiplier must be positive, got {args.buffer_size_multiplier}. Setting to 1.")
        print()
        args.buffer_size_multiplier = 1
    else:
        print(f"Using buffer_size_multiplier: {args.buffer_size_multiplier}")
        print()

    # Check that batch size is greater than or equal to context length
    if args.batch_size < args.context_length:
        raise ValueError(f"Batch size ({args.batch_size}) must be greater than or equal to context length ({args.context_length})")
    
    num_ctxs_per_crosscoder_batch = args.batch_size // args.context_length
    num_ctxs_in_buffer = num_ctxs_per_crosscoder_batch * args.buffer_size_multiplier

    # Total number of batches to train on
    max_steps = args.train_tokens // args.batch_size
    print(f"Total number of steps: {max_steps}")
    print()

    # Create buffer for training
    print("Creating activation buffer for training...")
    print()
    train_buffer = ActivationBuffer(
        data=generator,
        models=[base_model, ft_model],
        submodules=[base_submodule, ft_submodule],
        d_submodule=activation_dim,
        io=args.io,
        n_ctxs=num_ctxs_in_buffer,
        ctx_len=args.context_length,
        refresh_batch_size=args.buffer_ctx_batch_size,
        out_batch_size=args.batch_size,
        out_dtype=out_dtype,
        device=args.buffer_device,
    )

    # Set up trainer config
    trainer_cfg = {
        "lm_name": f"{args.ft_model}-{args.base_model}",
        "models_dtype": args.models_dtype,
        "dataset_name": args.dataset_name,
        "trainer": CrossCoderTrainer,
        "dict_class": CrossCoder,
        "layer": args.layer,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "batch_size": args.batch_size,
        "refresh_batch_size": args.buffer_ctx_batch_size,
        "ctx_len": args.context_length,
        "max_tokens": args.train_tokens,
        "l1_penalty": args.mu,
        "lr": args.lr,
        "resample_steps": args.resample_steps,
        "device": args.crosscoder_device,
        "warmup_steps": 1000,
        "submodule_name": submodule_name,
        "seed": args.seed,
        "compile": True,
        "dict_class_kwargs": {
            "same_init_for_all_layers": args.same_init_for_all_layers,
            "norm_init_scale": args.norm_init_scale,
            "init_with_transpose": args.init_with_transpose,
            "encoder_layers": args.encoder_layers,
        },
        "pretrained_ae": (
            CrossCoder.from_pretrained(args.pretrained)
            if args.pretrained is not None
            else None
        ),
        "wandb_name": f"L{args.layer}-mu{args.mu:.1e}-lr{args.lr:.0e}"
        + (f"-{args.run_name}" if args.run_name is not None else ""),
    }

    # Train the sparse autoencoder (SAE)
    trainSAE(
        data=train_buffer,
        trainer_config=trainer_cfg,
        use_wandb=not args.disable_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project="crosscoder",
        log_steps=50,
        steps=max_steps,
        save_steps=args.save_every_n_steps,
        save_dir=args.save_dir if not args.no_checkpoints else None,
    )
