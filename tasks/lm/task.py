import ctypes
import os
from dataclasses import asdict
from pathlib import Path

import hivemind
import torch.optim
import transformers
from hivemind import Float16Compression, SizeAdaptiveCompression, Uniform8BitQuantization
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer

import utils
from arguments import BasePeerArguments, CollaborativeArguments, HFTrainerArguments
from huggingface_auth import authorize_with_huggingface
from lib.models import SimpleModelConfig, SimpleModelForPreTraining  

import multiprocessing as mp

from .base_data import make_training_dataset, SimpleDataCollator

hivemind.use_hivemind_log_handler("in_root_logger")
logger = hivemind.get_logger()


class LMTrainingTask:
    """A container for training config, model, tokenizer, optimizer, and other local training utilities"""
    
    _dht = _collaborative_optimizer = _training_dataset = _authorizer = None

    def __init__(
        self, peer_args: BasePeerArguments, trainer_args: HFTrainerArguments, collab_args: CollaborativeArguments
    ):
        self.peer_args, self.trainer_args, self.collab_args = peer_args, trainer_args, collab_args
        transformers.set_seed(trainer_args.seed)

        self.config = SimpleModelConfig.from_pretrained(peer_args.model_config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(peer_args.tokenizer_path, cache_dir=peer_args.cache_dir)

        output_dir = Path(trainer_args.output_dir)
        latest_checkpoint_dir = max(output_dir.glob("checkpoint*"), default=None, key=os.path.getctime)

        if latest_checkpoint_dir is None:
            self.model = SimpleModelForPreTraining(self.config)
        else:
            self.model = SimpleModelForPreTraining.from_pretrained(latest_checkpoint_dir)

        self.current_sequence_length = mp.Value(ctypes.c_int64, self.trainer_args.max_sequence_length)
        
    def _make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.trainer_args.learning_rate,
            betas=(self.trainer_args.adam_beta1, self.trainer_args.adam_beta2),
            eps=self.trainer_args.adam_epsilon,
            weight_decay=self.trainer_args.weight_decay
        )
    
    def _make_scheduler(self, optimizer: torch.optim.Optimizer) -> LambdaLR:
        num_warmup_steps = self.trainer_args.warmup_steps
        num_training_steps = self.trainer_args.total_steps
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            decaying = float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, decaying)
        
        return LambdaLR(optimizer, lr_lambda)
    
    @property
    def training_dataset(self):
        if self._training_dataset is None:
            current_length = self.current_sequence_length.value if self.current_sequence_length else self.trainer_args.max_sequence_length
            self._training_dataset = make_training_dataset(
                self.tokenizer,
                max_sequence_length=current_length
            )
        return self._training_dataset

    
    @property
    def data_collator(self):
        return SimpleDataCollator(
            tokenizer=self.tokenizer
        )
