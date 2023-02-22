from dataclasses import dataclass
import torch


@dataclass
class Config:

    # setup
    model_path: str
    device: str
    dtype: str

    wm: int = 1

    # training
    num_epochs: int = 100
    batchsize: int = 256
    lr: float = 1e-1
    div_factor = 1e9
    final_div_factor = 1e1
    pct_start = 0.2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    bias_scaler: float = 1
    label_smoothing: float = 0.0

    # data
    num_train_samples: int = 50000
    cutout_size: int = 0
    pad_amount: int = 4
    crop_size: int = 32
