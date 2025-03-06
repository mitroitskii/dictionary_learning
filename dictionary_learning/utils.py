from datasets import load_dataset
import zstandard as zstd
import io
import json
import signal
import time
import sys

from .trainers.top_k import AutoEncoderTopK
from .trainers.batch_top_k import BatchTopKSAE
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
    CrossCoder,
)

class ExitHandler:
    """
    Signal handler for exit with model saving.

    Handles SIGINT (Ctrl+C) signals to allow:
    - First Ctrl+C: Save model and exit
    - Second Ctrl+C within timeout: Exit immediately without saving
    """

    def __init__(self, save_function=None, timeout=5.0):
        """
        Initialize the signal handler.

        Args:
            save_function: Function to call to save model state before exiting.
                           Should take no arguments.
            timeout: Time window in seconds for detecting double Ctrl+C.
        """
        self.save_function = save_function
        self.timeout = timeout
        self.last_sigint_time = 0
        self.exit_now = False
        signal.signal(signal.SIGINT, self._handler)

    def _handler(self, sig, frame):
        """
        Handle SIGINT signal.
        """
        current_time = time.time()

        if current_time - self.last_sigint_time < self.timeout:
            # Second Ctrl+C within timeout period
            print("\nReceived second interrupt. Exiting immediately without saving...")
            print()
            sys.exit(1)

        # First Ctrl+C or beyond timeout window
        self.last_sigint_time = current_time
        
        print("\nInterrupt received. Will save model and exit...")
        print()
        print(
            f"Press Ctrl+C again within {self.timeout} seconds to exit without saving.")
        print()
        self.exit_now = True

    def should_exit(self):
        """
        Check if the script should exit.
        
        If we're in a training loop, this flag allows it to complete the current step
        before saving and exiting, which can help prevent corrupted checkpoints

        Returns:
            bool: True if exit signal has been received, False otherwise.
        """
        return self.exit_now

    def save_and_exit(self):
        """
        Call the save function and exit.
        """
        if self.save_function is not None:
            try:
                print("Saving model before exit...")
                print()
                self.save_function()
                print("Model saved successfully.")
                print()
            except Exception as e:
                print(f"Error saving model: {str(e)}")
                print()

        sys.exit(0)

def get_submodule(model, layer):
    """
    Get the submodule for a given layer.
    """
    if hasattr(model, 'gpt_neox'):
        return model.gpt_neox.layers[layer]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer]
    else:
        raise ValueError(
            f"Could not find layers in model {model.__class__.__name__}")

def hf_dataset_to_generator(dataset_name, split="train", streaming=True, field="text"):
    
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def generator():
        for x in iter(dataset):
            yield x[field]

    return generator()


def zst_to_generator(data_path, field="text"):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a given field.
    """
    compressed_file = open(data_path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def generator():
        for line in text_stream:
            yield json.loads(line)[field]

    return generator()

def load_dictionary(base_path: str, device: str) -> tuple:
    ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dict_class = config["trainer"]["dict_class"]

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "BatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "CrossCoder":
        dictionary = CrossCoder.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return dictionary, config
