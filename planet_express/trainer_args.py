from dataclasses import dataclass

@dataclass
class TrainerArgs:
    dtype: str = "float32"
    device: str = "0"
    num_epochs: int = 1000
    batch_size: int = 32
    max_grad_norm: float = 1.0
    log_interval: int = 10
    num_workers: int = 4
    seed: int = 42
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000
    decay_iterations: int = 100000
    min_lr_frac: float = 0.01
    output_path: str = "output"
    model_args_path: str = "model_args.json"
    dataset_args_path: str = "dataset_args.json"
