import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from .retrievers import Retrievers

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    retriever_config_path: str = None
class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.retrievers = Retrievers(config.retriever_config_path, test_mode=True)

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</judge>')[0] + '</judge>'
                 if '</judge>' in resp 
                 else resp.split('</translation>')[0] + '</translation>'
                 if '</translation>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Extract prompt information from the initial input
        prompt_info_list = self._extract_prompt_info(initial_input_ids)

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, prompt_info_list=prompt_info_list
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False, prompt_info_list=prompt_info_list
            )
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

    def _extract_prompt_info(self, initial_input_ids: torch.Tensor) -> List[Dict]:
        """
        Extract prompt information from the initial input.
        
        Args:
            initial_input_ids: Initial input token IDs (batch of samples)
            
        Returns:
            List of dictionaries containing prompt information for each sample
        """
        # Decode the initial input
        initial_input_str = self.tokenizer.batch_decode(initial_input_ids, skip_special_tokens=True)
        
        prompt_info_list = []
        
        # Process each sample in the batch
        for sample_str in initial_input_str:
            prompt_info = {
                'source_code': '',
                'source_language': '',
                'target_language': ''
            }
            
            # Extract source code - look for code blocks with any language
            source_code_pattern = r'```(?:.*?)\n(.*?)\n```'
            source_code_match = re.search(source_code_pattern, sample_str, re.DOTALL)
            if source_code_match:
                prompt_info['source_code'] = source_code_match.group(1).strip()
            
            # Extract source language
            source_lang_pattern = r'Translate the following (.*?) code to'
            source_lang_match = re.search(source_lang_pattern, sample_str)
            if source_lang_match:
                prompt_info['source_language'] = source_lang_match.group(1).strip()
            
            # Extract target language
            target_lang_pattern = r'to (.*?), adhering to'
            target_lang_match = re.search(target_lang_pattern, sample_str)
            if target_lang_match:
                prompt_info['target_language'] = target_lang_match.group(1).strip()
            
            prompt_info_list.append(prompt_info)
        
        return prompt_info_list

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True, prompt_info_list: List[Dict] = None) -> List[str]:
        """
        Execute predictions
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            predictions: List of action predictions
            pad_token: Token to use for padding
            active_mask: Mask indicating which predictions are active
            do_search: Whether to perform search operations
            prompt_info: List of dictionaries containing information from the prompt for each sample
            
        Returns:
            Tuple of (next observations, done flags, valid action flags, is search flags)
        """
        cur_actions, contents = self.postprocess_predictions(predictions, prompt_info_list)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        # Handle judge queries if needed
        judge_queries = []
        for i, (action, content) in enumerate(zip(cur_actions, contents)):
            if action == 'judge':
                # Construct judge query with all necessary information
                judge_query = {
                    'judge_name': content.get('judge_name', ''),
                    'potential_translation': content.get('potential_translation', ''),
                    'source_code': content.get('source_code', ''),
                    'source_language': content.get('source_language', ''),
                    'target_language': content.get('target_language', ''),
                }
                print("%" * 50)
                print("JUDGE_QUERY:", judge_query)
                judge_queries.append(judge_query)
        
        # Process judge queries if needed
        judge_results = []
        if judge_queries and do_search:
            # Call judge API or process judge queries
            judge_results = self.process_judge_queries(judge_queries)
            assert len(judge_results) == len(judge_queries)
        
        # Process each prediction
        judge_idx = 0
        for i, (action, content) in enumerate(zip(cur_actions, contents)):
            if active_mask is not None and not active_mask[i]:
                next_obs.append('')
                dones.append(True)
                valid_action.append(False)
                is_search.append(False)
                continue
            
            if action == 'judge':
                # Get judge result for this sample
                judge_result = judge_results[judge_idx] if judge_idx < len(judge_results) else 'INVALID'
                judge_idx += 1
                
                next_obs.append(judge_result)
                dones.append(True)
                valid_action.append(True)
                is_search.append(True)
            elif action == 'translation':
                next_obs.append(content.get('translation', ''))
                dones.append(True)
                valid_action.append(True)
                is_search.append(False)
            else:
                next_obs.append('')
                dones.append(True)
                valid_action.append(False)
                is_search.append(False)
        
        return next_obs, dones, valid_action, is_search

    def process_judge_queries(self, judge_queries: List[Dict]) -> List[str]:
        """
        Process judge queries and return judge responses.
        
        Args:
            judge_queries: List of judge query dictionaries
            
        Returns:
            List of judge responses
        """
        # This is a placeholder implementation
        # In a real implementation, this would call an API or process the queries
        
        judge_responses = []
        for query in judge_queries:
            judge_name = query.get('judge_name', '')
            potential_translation = query.get('potential_translation', '')
            if judge_name in self.retrievers.retrievers.keys() and potential_translation != '':
                source_code = query.get('source_code', '')
                source_language = query.get('source_language', '')
                target_language = query.get('target_language', '')
                response = self.retrievers.inquire(judge_name, source_code, potential_translation, source_language, target_language)
            else:
                response = 'INVALID'
            
            judge_responses.append(response)
        
        return judge_responses

    def postprocess_predictions(self, predictions: List[Any], prompt_info_list: List[Dict] = None) -> Tuple[List[str], List[Dict]]:
        """
        Process (text-based) predictions into actions and content.
        
        Args:
            predictions: List of raw predictions
            prompt_info: List of dictionaries containing information from the prompt for each sample, including:
                - source_code: The source code to translate
                - source_language: The source programming language
                - target_language: The target programming language
            
        Returns:
            Tuple of (actions list, content dictionaries list)
        """
        actions = []
        contents = []
        
        # Default prompt info if not provided
        if prompt_info_list is None:
            prompt_info_list = [{
                'source_code': '',
                'source_language': '',
                'target_language': ''
            } for _ in range(len(predictions))]
                
        for i, prediction in enumerate(predictions):
            # Get the corresponding prompt info for this sample
            sample_prompt_info = prompt_info_list[i]
            if isinstance(prediction, str): # for llm output
                # Check for judge action
                judge_pattern = r'<judge>(.*?)</judge>'
                judge_match = re.search(judge_pattern, prediction, re.DOTALL)
                
                # Check for translation action
                translation_pattern = r'<translation>(.*?)</translation>'
                translation_match = re.search(translation_pattern, prediction, re.DOTALL)
                
                # Check for potential translation
                potential_translation_pattern = r'<potential_translation>(.*?)</potential_translation>'
                potential_translation_match = re.search(potential_translation_pattern, prediction, re.DOTALL)
                
                content_dict = {}
                
                if judge_match:
                    action = 'judge'
                    content_dict['judge_name'] = judge_match.group(1).strip()
                    
                    # Extract potential translation if available
                    if potential_translation_match:
                        content_dict['potential_translation'] = potential_translation_match.group(1).strip()
                    
                    # Add prompt information to the content dictionary
                    content_dict['source_code'] = sample_prompt_info.get('source_code', '')
                    content_dict['source_language'] = sample_prompt_info.get('source_language', '')
                    content_dict['target_language'] = sample_prompt_info.get('target_language', '')
                
                elif translation_match:
                    action = 'translation'
                    content_dict['translation'] = translation_match.group(1).strip()
                else:
                    action = None
                    content_dict = {}
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content_dict)
        
        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }
        
        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
