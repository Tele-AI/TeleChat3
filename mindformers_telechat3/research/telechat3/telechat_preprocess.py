# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""generate mindrecord script"""
import os
import argparse
import collections
import random
from random import shuffle
import datasets
import numpy as np
from tqdm import tqdm
from mindspore.mindrecord import FileWriter
from telechat_tokenizer import TelechatTokenizer
from mindformers.tools import logger

IGNORE_TOKEN_ID = -100


class TelechatDataset:
    """TelechatDataset"""
    def __init__(self, output_path, seed, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.raw_datasets = datasets.load_dataset(path="json", data_files=dataset_name)

    def get_train_data(self):
        """get train data"""
        if isinstance(self.raw_datasets, dict):
            dataset = self.raw_datasets["train"]
        elif isinstance(self.raw_datasets, list):
            dataset = self.raw_datasets
        else:
            raise ValueError("dataset type error")
        return dataset


def write_instance_to_file(writer, instance):
    """write the instance to file"""
    input_ids = instance["input_ids"]
    labels = instance["labels"]

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids).astype(np.int32)
    features["labels"] = np.asarray(labels).astype(np.int32)
    writer.write_raw_data([features])
    return features


def process_dataset(current_dataset, tokenizer, max_seq_len):
    """process dataset."""
    start_token_id = tokenizer.convert_tokens_to_ids(args.start_token) if args.start_token else None
    user_token_id = tokenizer.convert_tokens_to_ids(args.user_token)
    bot_token_id = tokenizer.convert_tokens_to_ids(args.bot_token)
    end_token_id = tokenizer.convert_tokens_to_ids(args.end_token)
    pad_token_id = tokenizer.convert_tokens_to_ids(args.pad_token)
    system_token_id = tokenizer.convert_tokens_to_ids(args.system_token)
    # dataset = []
    # all_lines = []

    tokens = []
    error_num = 0
    sentence_ids = []
    labels = []
    for idx in tqdm(range(len(current_dataset))):
        current_sentence_ids = []
        current_labels = []
        if 'system' not in current_dataset[idx]:
            current_dataset[idx]['system'] = ""
        sys_token = tokenizer(current_dataset[idx]["system"])["input_ids"]
        current_sentence_ids += [system_token_id] + sys_token
        current_labels += [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(sys_token)
        for sentence in current_dataset[idx]["dialog"]:
            role = sentence["role"]
            content = sentence["content"]
            if role == "user":
                user_token = tokenizer(content)["input_ids"]
                current_sentence_ids += [user_token_id] + user_token
                current_labels += [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(user_token)
            elif role == "bot":
                bot_token = tokenizer(content)["input_ids"]
                current_sentence_ids += [bot_token_id] + bot_token + [end_token_id] + tokenizer('\n')["input_ids"]
                current_labels += bot_token + [end_token_id] + [IGNORE_TOKEN_ID] + [IGNORE_TOKEN_ID] * len(tokenizer('\n')["input_ids"])

        if len(current_sentence_ids) > args.max_length:
            continue
        
        if len(sentence_ids) + len(current_sentence_ids) > args.max_length:
            sentence_ids = sentence_ids + (args.max_length - len(sentence_ids)) * [pad_token_id]
            labels = labels + (args.max_length - len(labels)) * [IGNORE_TOKEN_ID]
            tokens.append({"input_ids": sentence_ids, "labels": labels})
            sentence_ids = current_sentence_ids
            labels = current_labels
            continue
        
        sentence_ids += current_sentence_ids
        labels += current_labels
        # Last dialogue
        if idx == len(current_dataset) - 1:
            sentence_ids = sentence_ids + (args.max_length - len(sentence_ids)) * [pad_token_id]
            labels = labels + (args.max_length - len(labels)) * [IGNORE_TOKEN_ID]
            tokens.append({"input_ids": sentence_ids, "labels": labels})
            
    print("Number of concatenated samples: ", len(tokens))
    return tokens


def make_dataset():
    """make dataset."""
    raw_dataset = TelechatDataset(args.output_path, args.seed, args.input_dataset_file)
    train_dataset = raw_dataset.get_train_data()
    tokenizer = TelechatTokenizer(args.vocab_file_path, fast_tokenizer=True, padding_side="left")
    train_dataset = process_dataset(train_dataset, tokenizer, args.max_length)
    logger.info("***** Writing to output files *****")
    writer = FileWriter(args.output_dataset_file, 1)
    data_schema = {"input_ids": {"type": "int32", "shape": [-1]},
                   "labels": {"type": "int32", "shape": [-1]}}
    writer.add_schema(data_schema, "lm-schema")
    for dataset in tqdm(train_dataset):
        instance = {"input_ids": dataset["input_ids"], "labels": dataset["labels"]}
        write_instance_to_file(writer, instance=instance)
    writer.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_file", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--vocab_file_path", default="", type=str, help='which model to use.')
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--start_token", type=str, default="<_start>", help="start token")
    parser.add_argument("--user_token", type=str, default="<_user>", help="user token")
    parser.add_argument("--bot_token", type=str, default="<_bot>", help="bot token")
    parser.add_argument("--end_token", type=str, default="<_end>", help="end token")
    parser.add_argument("--pad_token", type=str, default="<_pad>", help="pad token")
    parser.add_argument("--system_token", type=str, default="<_system>", help="system token")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.output_path:
        if not args.output_path.endswith(".mindrecord"):
            os.makedirs(args.output_path, exist_ok=True)
            args.output_dataset_file = os.path.join(args.output_path, "dataset.mindrecord")
        else:
            args.output_dataset_file = args.output_path
    else:
        raise ValueError("output_path is needed.")

    make_dataset()
