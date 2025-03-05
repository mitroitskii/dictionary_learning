"""
Train a Crosscoder using pre-computed activations.

Activations are assumed to be stored in the directory specified by `--activation-store-dir`, organized by model and dataset:
    activations/<base-model>/<dataset>/<submodule-name>/
"""

import argparse
import os
import random
import datetime
from pathlib import Path
import torch as th

from dictionary_learning.cache import PairedActivationCache
from dictionary_learning import CrossCoder  # from dictionary.py
from dictionary_learning.trainers import CrossCoderTrainer  # from crosscoder.py
from dictionary_learning.training import trainSAE  # from training.py

# Set high precision for float32 matrix multiplications to improve performance and numerical stability for GPUs with f32 tensor cores (does nothing otherwise)
th.set_float32_matmul_precision('high') 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--base-model", type=str, default="gemma-2-2b",
                        help="Base model to use")
    parser.add_argument("--ft-model", type=str, default="gemma-2-2b-it",
                        help="Fine-tuned model to use")
    parser.add_argument("--layer", type=int, default=13,
                        help="Layer to extract activations from")
    
    # CrossCoder arguments
    parser.add_argument("--crosscoder-device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to train the crosscoder on (auto, cpu, cuda)") 
    # Model initialization arguments
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

    # Dataset arguments
    parser.add_argument("--activation-store-dir", type=str, default="activations",
                        help="Directory where activations are stored")
    parser.add_argument("--dataset", type=str, nargs="+", default=["fineweb", "lmsys_chat"],
                        help="Datasets to use for training")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of dataloader workers")

    # Training arguments
    parser.add_argument("--expansion-factor", type=int, default=32,
                        help="Dictionary size as multiple of activation dimension")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="How many token activations to load into crosscoder at once")
    parser.add_argument("--mu", type=float, default=1e-1,
                        help="L1 penalty")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum number of training steps")
    parser.add_argument("--resample-steps", type=int, default=None,
                        help="Resample dead neurons every N steps")

    # Validation arguments
    parser.add_argument("--validate-every-n-steps", type=int, default=10000,
                        help="Validate every N steps")
    parser.add_argument("--validation-batch-size", type=int, default=8192,
                        help="Validation batch size")
    parser.add_argument("--validation-size", type=int, default=10**6,
                        help="Number of token activations for validation")


    # Misc arguments
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name for logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--wandb-entity", type=str, default="",
                        help="Weights & Biases entity")
    parser.add_argument("--disable-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--no-checkpoints", action="store_true",
                        help="Disable saving model checkpoints")

    args = parser.parse_args()

    print(f"Training args: {args}")

    # Set random seeds for reproducibility for all random number generators
    random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)

    print(f"Training on device={args.device}.")

    # Set up activation cache paths
    print(f"Loading activations from {args.activation_store_dir}")
    activation_store_dir = Path(args.activation_store_dir)
    base_model_dir = activation_store_dir / args.base_model
    ft_model_dir = activation_store_dir / args.ft_model
    caches = []
    submodule_name = f"layer_{args.layer}_out"

    # Load all requested datasets
    print(f"Loading datasets: {args.dataset}")
    for dataset in args.dataset:
        base_model_dataset = base_model_dir / dataset
        ft_model_dataset = ft_model_dir / dataset
        print(f"Adding paired cache from {dataset}")
        caches.append(
            PairedActivationCache(
                base_model_dataset / submodule_name,
                ft_model_dataset / submodule_name,
            )
        )

    # Combine all datasets
    dataset = th.utils.data.ConcatDataset(caches)

    # Get activation dimension from the first example, compute dictionary size
    activation_dim = dataset[0].shape[1]
    dictionary_size = args.expansion_factor * activation_dim
    print(
        f"Activation dimension: {activation_dim}, Dictionary size: {dictionary_size}")

    # Create train/validation split
    print(
        f"Creating train/validation split with validation size {args.validation_size}")
    train_dataset, validation_dataset = th.utils.data.random_split(
        dataset, [len(dataset) - args.validation_size, args.validation_size]
    )
    print(
        f"Training on {len(train_dataset)} token activations, validating on {len(validation_dataset)}")

    # Create dataloaders
    print(
        f"Creating dataloaders with batch size {args.batch_size} and {args.workers} workers")
    dataloader = th.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    validation_dataloader = th.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.validation_batch_size,
        shuffle=False,  # No need to shuffle validation data
        num_workers=args.workers,
        pin_memory=True,
    )

    # Set up trainer config
    trainer_cfg = {
        "lm_name": f"{args.ft_model}-{args.base_model}",
        "trainer": CrossCoderTrainer,
        "dict_class": CrossCoder,
        "layer": args.layer,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "batch_size": args.batch_size,
        "ctx_len": args.context_length,
        "l1_penalty": args.mu,
        "lr": args.lr,
        "resample_steps": args.resample_steps,
        "device": args.device,
        "warmup_steps": 1000,
        "seed": args.seed,
        "compile": True,
        "wandb_name": f"L{args.layer}-mu{args.mu:.1e}-lr{args.lr:.0e}"
        + (f"-{args.run_name}" if args.run_name is not None else ""),
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
    }

    # Train the sparse autoencoder (SAE)
    print("Starting training...")
    trainSAE(
        data=dataloader,
        trainer_config=trainer_cfg,
        validate_every_n_steps=args.validate_every_n_steps,
        validation_data=validation_dataloader,
        use_wandb=not args.disable_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project="crosscoder",
        log_steps=50,
        steps=args.max_steps,
        save_steps=args.validate_every_n_steps,
        save_dir=os.path.join(args.save_dir, f"L{args.layer}_mu{args.mu:.1e}_lr{args.lr:.0e}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}") if not args.no_checkpoints else None,
    )
