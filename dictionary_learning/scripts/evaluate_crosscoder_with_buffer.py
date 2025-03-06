"""
Train a Crosscoder using live activations from models.

This script uses the ActivationBuffer to collect activations from two models in real-time
and trains a CrossCoder on these paired activations, with proper train/validation split
and configurable data types. There are two buffers, one for training and one for validation.
"""

import argparse
import random
import os
import datetime
import json
import torch as th
from nnsight import LanguageModel

from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator, get_submodule, load_dictionary
from dictionary_learning.evaluation import evaluate

# Set high precision for float32 matrix multiplications to improve performance and numerical stability for GPUs with f32 tensor cores (does nothing otherwise)
th.set_float32_matmul_precision('high')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base and Finetuned models arguments
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="Base model to use")
    parser.add_argument("--ft-model", type=str, default="agentica-org/DeepScaleR-1.5B-Preview",
                        help="Finetuned model to use")
    parser.add_argument("--io", type=str, default="out", choices=["out", "in"],
                        help="Which side of the layer to get activations from (input or output)")
    parser.add_argument("--model-devices", type=str, nargs="+", default=["cuda:1", "cuda:1"],
                        help="Devices to load models on (auto, cpu, cuda) for base and finetuned models [BASE_MODEL, FINTUNED_MODEL]. Provide one device per model.")
    parser.add_argument("--models-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="Data type for bases / ft models activations (bf16, fp16, or fp32)")

    # CrossCoder arguments
    parser.add_argument("--crosscoder-path", type=str, default=None,
                        help="Path to the trained CrossCoder dictionary")
    parser.add_argument("--out-dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"],
                        help="Data type for crosscoder parameters")
    parser.add_argument("--crosscoder-device", type=str, default="cuda:0", choices=["auto", "cpu", "cuda"],
                        help="Device to train CrossCoder on (auto, cpu, cuda)")

    # Dataset arguments
    parser.add_argument("--dataset-name", type=str, default="mitroitskii/OpenR1-Math-220k-formatted",
                        help="HuggingFace dataset name")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Local path to dataset. If provided, loads from local path instead of HuggingFace")
    parser.add_argument("--dataset-split", type=str, default="test",
                        help="Dataset split to use")
    parser.add_argument("--dataset-field", type=str, default="message_qwen1.5b",
                        help="Field name in the dataset to use (e.g., 'text', 'message_qwen1.5b')")
    parser.add_argument("--n-eval-ctx", type=int, default=10000,  # ~ 10 million tokens (if ctx_len is 1024)
                        help="Number of contexts to evaluate on")

    # Buffer arguments
    parser.add_argument("--buffer-ctx-batch-size", type=int, default=1,
                        help="How many input contexts (sequences of tokens) to run through the models (and buffer activations for) at once.")
    parser.add_argument("--buffer-size-multiplier", type=int, default=150,
                        help="How many times the buffer size should be larger than the crosscoder batch size")
    parser.add_argument("--buffer-device", type=str, default="cuda:0",
                        help="Device to store buffer activations on (auto, cpu, cuda).")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--evaluations-dir", type=str, default="evaluations",
                        help="Directory to save evaluation results")

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

    # Load the trained crosscoder
    print(f"Loading trained crosscoder from: {args.crosscoder_path}")
    print()
    crosscoder, config = load_dictionary(
        args.crosscoder_path, args.crosscoder_device)
    
    # Load models with specified dtype
    print(f"Loading models: {args.base_model} and {args.ft_model}")
    print()
    base_model = LanguageModel(
        args.base_model, device_map=args.model_devices[0], torch_dtype=model_dtype)
    ft_model = LanguageModel(
        args.ft_model, device_map=args.model_devices[1], torch_dtype=model_dtype)

    activation_dim = config["trainer"]["activation_dim"]
    batch_size = config["trainer"]["batch_size"]
    context_length = config["buffer"]["ctx_len"]
    layer = config["trainer"]["layer"]
    
    base_submodule = get_submodule(base_model, layer)
    ft_submodule = get_submodule(ft_model, layer)

    # Load the dataset
    print(f"Loading evaluation dataset: {args.dataset_name}")
    print()
    if args.dataset_path is not None:
        print(
            f"Loading from local path: {args.dataset_path}")
        print()
        generator = hf_dataset_to_generator(
            args.dataset_path, split=args.dataset_split, field=args.dataset_field)
    else:
        # Load from Hugging Face
        generator = hf_dataset_to_generator(
            args.dataset_name, split=args.dataset_split, field=args.dataset_field)

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


    num_ctxs_per_crosscoder_batch = batch_size // context_length
    num_ctxs_in_buffer = num_ctxs_per_crosscoder_batch * args.buffer_size_multiplier

    # Create buffer for evaluation
    print("Creating activation buffer for evaluation...")
    print()
    eval_buffer = ActivationBuffer(
        data=generator,
        models=[base_model, ft_model],
        submodules=[base_submodule, ft_submodule],
        d_submodule=activation_dim,
        io=args.io,
        n_ctxs=num_ctxs_in_buffer,
        ctx_len=context_length,
        refresh_batch_size=args.buffer_ctx_batch_size,
        out_batch_size=batch_size,
        out_dtype=out_dtype,
        device=args.buffer_device,
    )

    
    eval_results = evaluate(
        crosscoder,
        eval_buffer,
        device=args.crosscoder_device,
        n_batches=args.n_eval_ctx // num_ctxs_per_crosscoder_batch,
        return_loss_recovered=False, # NOTE: loss_recovered does not currently work with more than one model
    )

    mu = config["trainer"]["mu"]
    lr = config["trainer"]["lr"]

    if args.evaluations_dir is not None:
        os.makedirs(args.evaluations_dir, exist_ok=True)

        # Create results dictionary with evaluation results and training config
        results_with_config = {
            "eval_results": eval_results,
            "training_config": config
        }

        with open(os.path.join(args.evaluations_dir, f"eval_results_L{layer}_mu{mu:.1e}_lr{lr:.0e}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w") as f:
            json.dump(results_with_config, f)

    print(eval_results)
