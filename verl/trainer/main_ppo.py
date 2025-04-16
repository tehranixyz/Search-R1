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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os
os.environ["HF_HUB_OFFLINE"]='1'
os.environ["WANDB_MODE"] = "offline"
from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, codejudge_reward
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np
from search_r1.llm_agent.retrievers import Retrievers


def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    elif data_source in ['codejudge']:
        return codejudge_reward.compute_score
    else:
        raise NotImplementedError(f"Reward scoring function not implemented for data source: '{data_source}'. Supported sources are: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle")


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, retriever_config_path, format_score=0., teacher_max_prompt_length=None, teacher_max_response_length=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.retrievers = Retrievers(json_path=retriever_config_path, test_mode=True)
        self.teacher_max_prompt_length = teacher_max_prompt_length
        self.teacher_max_response_length = teacher_max_response_length
        
        if self.teacher_max_prompt_length is None or self.teacher_max_response_length is None:
            print("Warning: teacher_max_prompt_length or teacher_max_response_length not provided. Using default values.")
            # Set reasonable defaults if not provided
            if self.teacher_max_prompt_length is None:
                self.teacher_max_prompt_length = 512
            if self.teacher_max_response_length is None:
                self.teacher_max_response_length = 512

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # Initialize lists to store judge queries and responses
        judge_queries = []
        judge_responses = []

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # Get score, judge_query, and judge_review from compute_score_fn
            score, judge_query, judge_review = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, format_score=self.format_score, retrievers=self.retrievers)

            reward_tensor[i, valid_response_length - 1] = score
            
            # Store judge query and response
            judge_queries.append(judge_query if judge_query is not None else "")
            judge_responses.append(judge_review if judge_review is not None else "")
            
            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        # Tokenize judge queries and responses
        judge_query_tokens = []
        judge_response_tokens = []
        judge_query_attention_masks = []
        judge_response_attention_masks = []
        
        for query, response in zip(judge_queries, judge_responses):
            # Tokenize judge query with proper truncation and padding
            query_encoding = self.tokenizer.encode(
                query, 
                add_special_tokens=True,
                max_length=self.teacher_max_prompt_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            query_tokens = query_encoding
            query_attention_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in query_tokens]
            
            # Tokenize judge response with proper truncation and padding
            if response is None or response == "":
                # For None or empty responses, create a dummy token and set attention mask to all zeros
                # This will effectively ignore these examples during loss calculation
                response_tokens = []
                query_attention_mask = [0] * self.teacher_max_prompt_length
            else:
                response_encoding = self.tokenizer.encode(
                    response, 
                    add_special_tokens=True,
                    max_length=self.teacher_max_response_length,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt'
                )
                response_tokens = response_encoding
                response_tokens = [label if label != self.tokenizer.pad_token_id else -100 for label in response_encoding]
            
            judge_query_tokens.append(query_tokens)
            judge_query_attention_masks.append(query_attention_mask)
            judge_response_tokens.append(response_tokens)
        
        # Add judge queries and responses to data
        data.batch['judge_queries'] = judge_query_tokens
        data.batch['judge_responses'] = judge_response_tokens
        data.batch['judge_query_attention_masks'] = judge_query_attention_masks

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0,
        retriever_config_path=config.retrievers.config_path,
        teacher_max_prompt_length=config.data.max_prompt_length,
        teacher_max_response_length=config.data.max_response_length
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=1,
        retriever_config_path=config.retrievers.config_path,
        teacher_max_prompt_length=config.data.max_prompt_length,
        teacher_max_response_length=config.data.max_response_length
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
