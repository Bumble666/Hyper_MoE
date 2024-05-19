#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
import functools
sys.path = [os.path.abspath(os.path.join(os.getcwd(), "../.."))] + sys.path
from datasets.download.download_config import DownloadConfig
from typing import Optional, List
from datasets import concatenate_datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
#from data import AutoTask, TaskDataCollatorForSeq2Seq, AutoPostProcessor

from data.tasks import AutoTask
from data.postprocessors import AutoPostProcessor
from data.data_collator import TaskDataCollatorForSeq2Seq
#from third_party.trainers import Seq2SeqTrainer

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import numpy as np
import torch
from typing import Tuple
from data.tasks import TASK_MAPPING
from data.metrics import TASK_TO_METRICS
from data.metrics import build_compute_metrics_fn


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")



logger = logging.getLogger(__name__)

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    print_num_parameters: Optional[bool] = field(default=False, metadata={"help": "If set, print the parameters of "
                                                                                "the model."})
    do_test: Optional[bool] = field(default=False, metadata={"help": "If set, evaluates the test performance."})
    split_validation_test: Optional[bool] = field(default=False,metadata={})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If set measures the time."})
    compute_memory: Optional[bool] = field(default=True, metadata={"help": "if set, measures the memory"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    task_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    eval_dataset_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The name of the evaluation dataset to use (via the datasets library)."}
    )
    eval_dataset_config_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The configuration name of the evaluation dataset to use (via the datasets library)."}
    )
    test_dataset_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    test_dataset_config_name: Optional[List[str]] = field(
        default=None, metadata={"help": "The configuration name of the test dataset to use (via the datasets library)."}
    )
    lang_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                  "value if set."}
    )
    num_beams: Optional[int] = field(
        default=None, metadata={"help": "Number of beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    task_adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from task adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    #data_seed: Optional[int] = field(
    #    default=42, metadata={"help": "seed used to shuffle the data."})

    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )

    train_files: Optional[List[str]] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_files: Optional[List[str]] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_files: Optional[List[str]] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )

    def __post_init__(self):
        if self.task_name is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length
        if self.test_max_target_length is None:
            self.test_max_target_length = self.max_target_length



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    process_dim: int = field(
        default=128,
    )
    hypernet_input: int = field(
        default=128,
    )
    hypernetwork_bottleneck: int = field(
        default=64,
    )
    layer_emb_dim: int = field(
        default=10,
    )
    adapter_dim: int = field(
        default=64,
    )
    experts_embedding: bool = field(
        default=False,
    )
    experts_embedding_dim: int = field(
        default=128,
    )
    use_hypernet: bool = field(
        default=False,
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if "wandb" in training_args.report_to:  # todo remove wandb to avoid bug.
        training_args.report_to.remove("wandb")
    if "tensorboard" not in training_args.report_to:  # todo add tensorboard.
        training_args.report_to.append("tensorboard")

    if training_args.resume_from_checkpoint is not None:
        tail = training_args.resume_from_checkpoint.split('/')[-1]
        training_args.output_dir = os.path.join(training_args.output_dir, tail)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    training_args.save_total_limit = 1

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Tuning Strategy {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.



    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # put useful args into config: these arguments will be used in models, thus adding them to config
    #setattr(config, 'max_seq_length', data_args.max_seq_length)
    setattr(config, 'seed', training_args.seed)
    setattr(config, 'world_size', training_args.world_size)
    setattr(config, 'process_dim', model_args.process_dim)
    setattr(config, 'hypernet_input', model_args.hypernet_input)
    setattr(config, 'hypernetwork_bottleneck', model_args.hypernetwork_bottleneck)
    setattr(config, 'layer_emb_dim', model_args.layer_emb_dim)
    setattr(config, 'experts_embedding', model_args.experts_embedding)
    setattr(config, 'experts_embedding_dim', model_args.experts_embedding_dim)
    setattr(config, 'adapter_dim', model_args.adapter_dim)
    setattr(config, 'use_hypernet', model_args.use_hypernet)
    #setattr(training_args, 'split_validation_test', False)
    
    data_args.dataset_name = data_args.task_name

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # print(model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )



    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples, max_target_length, task_id=None):
        model_inputs = tokenizer(examples['source'], max_length=data_args.max_source_length,
                                 padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'], max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["extra_fields"] = examples['extra_fields']
        if task_id is not None:
            model_inputs["task_ids"] = [
                task_id for _ in examples['extra_fields']]

        return model_inputs

    column_names = ['source', 'target', 'extra_fields']
    performance_metrics = {}


    if training_args.do_train:
        train_datasets = [AutoTask.get(dataset_name,
                                       dataset_config_name,
                                       seed=training_args.data_seed).get(
            split="train",
            split_validation_test=training_args.split_validation_test,
            add_prefix= True,
            n_obs=data_args.max_train_samples, lang=data_args.lang_name, file_name=data_args.train_file)
            for dataset_name, dataset_config_name
            in zip(data_args.dataset_name, data_args.dataset_config_name)]

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length, )
            for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)]

        for i, train_dataset in enumerate(train_datasets):
            train_datasets[i] = train_datasets[i].map(
                functools.partial(preprocess_function,
                                max_target_length=max_target_lengths[i]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
            # if train_dataset != "superglue-record" else column_names+["answers"],
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        train_dataset = concatenate_datasets(train_datasets)

    if training_args.do_eval:
        eval_datasets = {eval_dataset: AutoTask.get(eval_dataset, eval_dataset_config,
                                                    seed=training_args.data_seed).get(
            split="test",
            split_validation_test=training_args.split_validation_test,
            add_prefix=True,
            n_obs=data_args.max_val_samples, lang=data_args.lang_name, file_name=data_args.validation_file)
            for eval_dataset, eval_dataset_config in
            zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)}

        max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
            tokenizer=tokenizer, default_max_length=data_args.max_target_length)
            for dataset_name, dataset_config_name in
            zip(data_args.eval_dataset_name, data_args.eval_dataset_config_name)]

        for k, name in enumerate(eval_datasets):
            eval_datasets[name] = eval_datasets[name].map(
                functools.partial(preprocess_function,
                              max_target_length=max_target_lengths[k]),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
            # if name != "superglue-record" else column_names+["answers"],
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

  #  if training_args.do_test:
  #      test_datasets = {test_dataset: AutoTask.get(test_dataset, test_dataset_config,
  #                                                  seed=training_args.data_seed).get(
  #          split="test",
  #          split_validation_test=training_args.split_validation_test,
  #          add_prefix=True,
  #          n_obs=data_args.max_test_samples, lang=data_args.lang_name, file_name=data_args.test_file)
  #          for test_dataset, test_dataset_config in
  #          zip(data_args.test_dataset_name, data_args.test_dataset_config_name)}

  #      max_target_lengths = [AutoTask.get(dataset_name, dataset_config_name).get_max_target_length(
  #          tokenizer=tokenizer, default_max_length=data_args.max_target_length)
  #          for dataset_name, dataset_config_name in
  #          zip(data_args.test_dataset_name, data_args.test_dataset_config_name)]

  #  for k, name in enumerate(test_datasets):
  #      test_datasets[name] = test_datasets[name].map(
  #          functools.partial(preprocess_function,
  #                            max_target_length=max_target_lengths[k]),
  #          batched=True,
  #          num_proc=data_args.preprocessing_num_workers,
  #          remove_columns=column_names,
  #          load_from_cache_file=not data_args.overwrite_cache,
  #      )

# ----------------------------------------------------------------------------

    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = TaskDataCollatorForSeq2Seq(
            tokenizer,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    eval_metrics = [AutoTask.get(dataset_name, dataset_config_name).metric
                    for dataset_name, dataset_config_name in zip(data_args.dataset_name, data_args.dataset_config_name)][0]

    print(data_args.eval_dataset_name)
    compute_metrics_fn = build_compute_metrics_fn(
        data_args.eval_dataset_name, tokenizer, data_args.ignore_pad_token_for_loss) if training_args.predict_with_generate else None
    print(compute_metrics_fn)

    data_info = {"eval": eval_datasets[data_args.eval_dataset_name[0]]['extra_fields'],
                 "train": train_dataset['extra_fields'] if training_args.do_train else None}

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        info = data_info['eval']
        post_processor = AutoPostProcessor.get(data_args.dataset_name[0], tokenizer,
                                               data_args.ignore_pad_token_for_loss)
        decoded_preds, decoded_labels = post_processor.process(
            preds, labels, info)
        result = {}
        for metric in eval_metrics:
            result.update(metric(decoded_preds, decoded_labels))
        return result

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.


    #print(train_dataset['task'])
    #exit()

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=list(eval_datasets.values())[
            0] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        evaluation_metrics = TASK_TO_METRICS[data_args.dataset_name[0]],
        data_collator=data_collator,
        data_info = data_info,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
        print(
            "Memory utilization",
            peak_memory,
            "GB"
        )
        performance_metrics.update({"peak_memory": peak_memory})
    if training_args.compute_memory or training_args.compute_time:
        trainer.save_metrics("performance", performance_metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        for task, eval_dataset in eval_datasets.items():
            metrics = trainer.evaluate(eval_dataset=eval_dataset, max_length=data_args.val_max_target_length, num_beams=data_args.num_beams,)
            #max_eval_samples = (
            #    data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            #)
            #metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

   # if training_args.do_test:
   #     logger.info("*** Test ***")
        # multi-task evaluations
   #     results = {}
   #     for task, test_dataset in test_datasets.items():
   #         metrics = trainer.evaluate(eval_dataset=test_dataset,
   #                                    max_length=data_args.test_max_target_length, num_beams=data_args.num_beams,
   #                                    metric_key_prefix="test"
   #                                    )
   #         trainer.log_metrics("test", metrics)
   #         trainer.save_metrics("test", metrics)

    if training_args.push_to_hub:

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
        if data_args.task_name is not None:
            kwargs["language"] = "en"
            kwargs["dataset_tags"] = "glue"
            kwargs["dataset_args"] = data_args.task_name
            kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    main()
