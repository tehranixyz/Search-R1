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
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _select_rm_score_fn(data_source):
    logger.info(f"[main_ppo.py] Selecting reward model score function for data source: {data_source}")
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    elif data_source in ['codejudge']:
        return codejudge_reward.compute_score
    else:
        logger.error(f"[main_ppo.py] Reward scoring function not implemented for data source: '{data_source}'")
        raise NotImplementedError(f"Reward scoring function not implemented for data source: '{data_source}'. Supported sources are: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa, musique, bamboogle")


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, retriever_config_path, format_score=0., teacher_max_prompt_length=None, teacher_max_response_length=None, test_mode=False) -> None:
        logger.info(f"[main_ppo.py] Initializing RewardManager with num_examine={num_examine}, format_score={format_score}")
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.retrievers = Retrievers(json_path=retriever_config_path, test_mode=test_mode)
        self.teacher_max_prompt_length = teacher_max_prompt_length
        self.teacher_max_response_length = teacher_max_response_length
        
        if self.teacher_max_prompt_length is None or self.teacher_max_response_length is None:
            logger.warning("[main_ppo.py] teacher_max_prompt_length or teacher_max_response_length not provided. Using default values.")
            # Set reasonable defaults if not provided
            if self.teacher_max_prompt_length is None:
                self.teacher_max_prompt_length = 512
            if self.teacher_max_response_length is None:
                self.teacher_max_response_length = 512

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        logger.info(f"[main_ppo.py] RewardManager.__call__ called with batch size: {len(data)}")

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            logger.info("[main_ppo.py] Using pre-computed rm_scores from batch")
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # Initialize lists to store judge queries and responses
        judge_queries = []
        judge_responses = []

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
            score, judge_query, judge_review = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth, retrievers=self.retrievers)

            reward_tensor[i, valid_response_length - 1] = score
            
            # Store judge query and response
            judge_queries.append(judge_query if judge_query is not None else "")
            judge_responses.append(judge_review if judge_review is not None else "")


        max_length = self.teacher_max_prompt_length + self.teacher_max_response_length
        full_texts = [p + a for p, a in zip(judge_queries, judge_responses)]

        batch = self.tokenizer(full_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = input_ids.clone()


        for i, answer in enumerate(judge_responses):
            if answer.strip() == "":
                labels[i, :] = -100
            else:
                prompt_ids = self.tokenizer(judge_queries[i], add_special_tokens=False)["input_ids"]
                prompt_len = len(prompt_ids)
                labels[i, :prompt_len] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

            
            
        return reward_tensor, input_ids, labels, attention_mask


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    logger.info("[main_ppo.py] Starting main function")
    if not ray.is_initialized():
        # this is for local ray cluster
        logger.info("[main_ppo.py] Initializing Ray cluster")
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    logger.info("[main_ppo.py] Launching main_task")
    ray.get(main_task.remote(config))
    logger.info("[main_ppo.py] Main function completed")


@ray.remote
def main_task(config):
    logger.info("[main_ppo.py] Starting main_task")
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    logger.info("[main_ppo.py] Configuration:")
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    logger.info(f"[main_ppo.py] Downloading checkpoint from HDFS: {config.actor_rollout_ref.model.path}")
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    logger.info(f"[main_ppo.py] Checkpoint downloaded to: {local_path}")

    # instantiate tokenizer
    logger.info("[main_ppo.py] Instantiating tokenizer")
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    logger.info(f"[main_ppo.py] Setting up worker classes with strategy: {config.actor_rollout_ref.actor.strategy}")
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
        logger.error(f"[main_ppo.py] Unsupported strategy: {config.actor_rollout_ref.actor.strategy}")
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
        logger.info(f"[main_ppo.py] Setting up reward model with strategy: {config.reward_model.strategy}")
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            logger.error(f"[main_ppo.py] Unsupported reward model strategy: {config.reward_model.strategy}")
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    logger.info("[main_ppo.py] Initializing reward functions")
    reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=0,
        retriever_config_path=config.retrievers.config_path,
        teacher_max_prompt_length=config.data.max_prompt_length,
        teacher_max_response_length=config.data.max_response_length,
        test_mode=config.retrievers.test_mode
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, 
        num_examine=1,
        retriever_config_path=config.retrievers.config_path,
        teacher_max_prompt_length=config.data.max_prompt_length,
        teacher_max_response_length=config.data.max_response_length,
        test_mode=config.retrievers.test_mode
    )

    logger.info("[main_ppo.py] Setting up resource pool manager")
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    logger.info("[main_ppo.py] Initializing RayPPOTrainer")
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    
    logger.info("[main_ppo.py] Initializing workers")
    trainer.init_workers()
    
    logger.info("[main_ppo.py] Starting training")
    trainer.fit()
    logger.info("[main_ppo.py] Training completed")


if __name__ == '__main__':
    logger.info("[main_ppo.py] Script started")
    main()
    logger.info("[main_ppo.py] Script completed")
