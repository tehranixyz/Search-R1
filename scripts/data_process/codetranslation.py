# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Preprocess the nq dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def make_prefix(dp, template_type):
    source_lang = dp['source_lang']
    target_lang = dp['target_lang']
    source_code = dp['source_code']
    principle_content = dp['principle_content']


    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""Answer the following code translation question. Your answer must meet high-quality coding standards as expected in a professional code development.  
You must conduct reasoning inside <think> and </think> first every time you get new information and come up with a potential translation inside <potential_translation> and </potential_translation>.  
If you find you need some verification for your potential translation, you can call a judge by <judge> judge name </judge>, it will return its rationale between <judge_response> and </judge_response>.  
Available judges are: [‘Functional Equivalence’, ‘Language Conventions and Idiomatic’, ‘Readability and Structure’, ‘Error Handling’, ‘Dependency and API’].  

If a judge suggests improvements, update your reasoning inside <think>, update your potential translation inside <potential_translation> and you can call the judge again if necessary. Continue this loop until all verification aspects are addressed.  

If you find no further verification is needed, you can directly provide the answer inside <translation> and </translation> without detailed illustrations. For example, <translation> def hello_world():\nprint("Hello, World!")\n </translation>

Question: Translate the following {source_lang} code to {target_lang}, adhering to the user-defined principle.

Principle:
{principle_content}

{source_lang} Code:
```
{source_code}
```\n
"""
    else:
        raise NotImplementedError
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/codejudge')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'codejudge'

    dataset = datasets.load_dataset('json', data_files={'train': '/home/pkpr/classes/rs/Search-R1/scripts/C++-Python.jsonl'})
    # Split dataset into train and test sets
    dataset = dataset['train'].select(range(10000)).train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            # example['question'] = example['question'].strip()
            # if example['question'][-1] != '?':
            #     example['question'] += '?'
            question = make_prefix(example, template_type=args.template_type)
            src_uid = example['src_uid']
            source_lang = example['source_lang']
            target_lang = example['target_lang']

            solution = {
                "target": "",
            }

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "fact-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'src_uid': src_uid,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    
    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
