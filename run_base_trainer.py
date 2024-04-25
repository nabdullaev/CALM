#!/usr/bin/env python

import os
from pathlib import Path

import transformers
from hivemind.utils.logging import get_logger, use_hivemind_log_handler
from transformers import HfArgumentParser

import callback
import utils
from arguments import CollaborativeArguments, HFTrainerArguments, BasePeerArguments
from lib.training.hf_trainer import CollaborativeHFTrainer
from tasks.lm.task import LMTrainingTask  # Assuming your LM training task is placed under tasks/lm/task

use_hivemind_log_handler("in_root_logger")
logger = get_logger()

def main():
    parser = HfArgumentParser((BasePeerArguments, HFTrainerArguments, CollaborativeArguments))
    peer_args, trainer_args, collab_args = parser.parse_args_into_dataclasses()

    logger.info(f"Trying {len(peer_args.initial_peers)} initial peers: {peer_args.initial_peers}")
    if len(peer_args.initial_peers) == 0:
        logger.warning("Specify at least one network endpoint in initial peers OR let others join your peer.")

    utils.setup_logging(trainer_args)
    task = LMTrainingTask(peer_args, trainer_args, collab_args)
    model = task.model.to(trainer_args.device)

    collaborative_callback = callback.CollaborativeCallback(task, peer_args)
    assert trainer_args.do_train and not trainer_args.do_eval

    # Create a trainer with customized callbacks and settings suitable for a collaborative training session
    trainer = CollaborativeHFTrainer(
        model=model,
        args=trainer_args,
        tokenizer=task.tokenizer,
        data_collator=task.data_collator,
        data_seed=hash(task.local_public_key),
        train_dataset=task.training_dataset,
        eval_dataset=None,
        collaborative_optimizer=task.collaborative_optimizer,
        callbacks=[collaborative_callback],
    )
    trainer.remove_callback(transformers.trainer_callback.PrinterCallback)
    trainer.remove_callback(transformers.trainer_callback.ProgressCallback)

    latest_checkpoint_dir = max(Path(trainer_args.output_dir).glob("checkpoint*"), key=os.path.getctime, default=None)
    trainer.train(model_path=latest_checkpoint_dir)

if __name__ == "__main__":
    main()
