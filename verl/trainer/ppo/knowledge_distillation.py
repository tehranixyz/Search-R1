import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
import re
import json
import logging
from search_r1.llm_agent.retrievers import Retrievers
from openai import OpenAI
from verl.protocol import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO:
# 1. Extract Information from prompt: source code, src_lang, trg_lang and principle
# 2. Create a query using the relevant assessment template
# 3. Call teacher and student model with the same query and obtain the response
# Question: What to do when <translation> does not exist in the response?
class KnowledgeDistillation:
    """
    A class for handling knowledge distillation between teacher and student models.
    This class manages the initialization of retrievers and provides methods for
    getting reviews from teacher and student models.
    """
    
    def __init__(self, retriever_config_path: str = None, max_combined_length: int = 4608):
        """
        Initialize the KnowledgeDistillation class.
        
        Args:
            retriever_config_path (str, optional): Path to the retriever configuration file.
            max_combined_length (int, optional): Maximum combined length for queries and teacher responses.
                Defaults to 4608.
        """
        self.retrievers = None
        if retriever_config_path:
            self.retrievers = Retrievers(json_path=retriever_config_path, test_mode=True)
        self.max_combined_length = max_combined_length
        logger.info(f"Initialized KnowledgeDistillation with max_combined_length={max_combined_length}")
    
    def extract_prompt_info(self, prompt: str) -> Dict[str, str]:
        """
        Extract source code, source language, target language, and principle from the prompt.
        
        Args:
            prompt (str): The input prompt.
            
        Returns:
            Dict[str, str]: Dictionary containing extracted information.
        """
        prompt_info = {
            'source_code': '',
            'source_language': '',
            'target_language': '',
            'principle': ''
        }
        
        # Extract source code - look for code blocks with any language
        source_code_pattern = r'```(?:.*?)\n(.*?)\n```'
        source_code_match = re.search(source_code_pattern, prompt, re.DOTALL)
        if source_code_match:
            prompt_info['source_code'] = source_code_match.group(1).strip()
        
        # Extract source language
        source_lang_pattern = r'Translate the following (.*?) code to'
        source_lang_match = re.search(source_lang_pattern, prompt)
        if source_lang_match:
            prompt_info['source_language'] = source_lang_match.group(1).strip()
        
        # Extract target language
        target_lang_pattern = r'to (.*?), adhering to'
        target_lang_match = re.search(target_lang_pattern, prompt)
        if target_lang_match:
            prompt_info['target_language'] = target_lang_match.group(1).strip()
        
        # Extract principle
        principle_pattern = r'Principle:\n(.*?)\n\n'
        principle_match = re.search(principle_pattern, prompt, re.DOTALL)
        if principle_match:
            prompt_info['principle'] = principle_match.group(1).strip()
        
        return prompt_info
    
    def extract_translation(self, response: str) -> str:
        """
        Extract the translation from the response.
        
        Args:
            response (str): The model's response.
            
        Returns:
            str: The extracted translation if found, empty string otherwise.
        """
        translation_pattern = r'<translation>(.*?)</translation>'
        match = re.search(translation_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return response
    
    def map_principle_to_preference(self, principle: str) -> str:
        """
        Map the principle to a preference name using the retrievers_config.json.
        
        Args:
            principle (str): The principle text.
            
        Returns:
            str: The mapped preference name.
        """
        if not self.retrievers:
            return "Functional Equivalence"  # Default preference
            
        # Load the retrievers configuration
        retriever_configs = self.retrievers.retrievers
        
        # Check each retriever's principles
        for retriever_name, retriever_config in retriever_configs.items():
            # Get the prompt template which contains the principles
            principles = retriever_config["principles"]
            
            # Check if the principle is in the prompt template
            if principle in principles:
                return retriever_name
        
        # If no match is found, return the default preference
        return "Functional Equivalence"
    
    def create_assessment_query(self, source_code: str, translated_code: str, 
                               src_lang: str, trg_lang: str, preference: str) -> str:
        """
        Create a query using the relevant assessment template.
        
        Args:
            source_code (str): The source code.
            translated_code (str): The translated code.
            src_lang (str): Source programming language.
            trg_lang (str): Target programming language.
            preference (str): The preference name.
            
        Returns:
            str: The formatted query.
        """
        if not self.retrievers or preference not in self.retrievers.retrievers:
            # Create a default query if retrievers are not available
            return f"Evaluate the following code translation from {src_lang} to {trg_lang}:\n\nSource code:\n{source_code}\n\nTranslated code:\n{translated_code}"
        
        # Get the prompt template for the preference
        prompt_template = self.retrievers.retrievers[preference]["prompt_template"]
        
        # Format the query using the template
        query = prompt_template.format(
            src_lang=src_lang,
            trg_lang=trg_lang,
            source_code=source_code,
            translated_code=translated_code
        )
        
        return query


        
    # New batch processing methods
    def extract_prompt_info_batch(self, prompts: List[str]) -> List[Dict[str, str]]:
        """
        Extract information from a batch of prompts.
        
        Args:
            prompts (List[str]): List of input prompts.
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing extracted information.
        """
        return [self.extract_prompt_info(prompt) for prompt in prompts]
    
    def extract_translation_batch(self, responses: List[str]) -> List[str]:
        """
        Extract translations from a batch of responses.
        
        Args:
            responses (List[str]): List of model responses.
            
        Returns:
            List[str]: List of extracted translations.
        """
        return [self.extract_translation(response) for response in responses]
    
    def get_teacher_reviews_batch(self, prompts: List[str], responses: List[str]) -> List[str]:
        """
        Get reviews from teacher model for a batch of prompt-response pairs.
        
        Args:
            prompts (List[str]): List of input prompts.
            responses (List[str]): List of model responses.
            
        Returns:
            List[str]: List of teacher reviews.
        """
        # Extract information from prompts
        prompt_infos = self.extract_prompt_info_batch(prompts)
        
        # Extract translations from responses
        translated_codes = self.extract_translation_batch(responses)
        
        # Process each example
        teacher_reviews = []
        for i, (prompt_info, translated_code) in enumerate(zip(prompt_infos, translated_codes)):
            # If no translation is found, return an error message
            if not translated_code:
                teacher_reviews.append("No translation found in the response. Please provide a translation between <translation> and </translation> tags.")
                continue
            
            # Map principle to preference name
            preference = self.map_principle_to_preference(prompt_info['principle'])
            
            # Get teacher review
            teacher_review = self.retrievers.inquire(
                retriever_name=preference, 
                source_code=prompt_info['source_code'], 
                translated_code=translated_code, 
                src_lang=prompt_info['source_language'], 
                trg_lang=prompt_info['target_language']
            )
            teacher_reviews.append(teacher_review)
        
        return teacher_reviews
    
    def process_batch(self, prompts: List[str], responses: List[str], student_model: Any, tokenizer: Any = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Process a batch of prompt-response pairs to get teacher and student reviews.
        
        Args:
            prompts (List[str]): List of input prompts.
            responses (List[str]): List of model responses.
            student_model (Any): The student model to use for review.
            tokenizer (Any, optional): The tokenizer to use. If None, will use student_model.tokenizer.
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: Tuple containing:
                - input_ids_batch: List of input ID tensors
                - labels_batch: List of label tensors
                - attention_mask_batch: List of attention mask tensors
        """
        logger.info(f"Getting teacher reviews for batch of {len(prompts)} examples...")
        teacher_reviews = self.get_teacher_reviews_batch(prompts, responses)
        logger.info(f"Completed getting {len(teacher_reviews)} teacher reviews")
        
        # Check for empty or invalid teacher reviews
        valid_indices = []
        for i, review in enumerate(teacher_reviews):
            if review and not review.startswith("No translation found"):
                valid_indices.append(i)
            else:
                logger.warning(f"Invalid or empty teacher review for example {i}: {review[:100]}...")
        
        # Extract information from prompts
        prompt_infos = self.extract_prompt_info_batch(prompts)
        
        # Extract translations from responses
        translated_codes = self.extract_translation_batch(responses)
        
        # Prepare batch data
        queries = []
        
        for i, (prompt_info, translated_code) in enumerate(zip(prompt_infos, translated_codes)):
            # Map principle to preference name
            preference = self.map_principle_to_preference(prompt_info['principle'])
            
            # Create assessment query
            query = self.create_assessment_query(
                source_code=prompt_info['source_code'],
                translated_code=translated_code,
                src_lang=prompt_info['source_language'],
                trg_lang=prompt_info['target_language'],
                preference=preference
            )
            
            queries.append(query)
        
        # Tokenize the queries with attention masks
        logger.info("Tokenizing queries...")
        prompt_enc = tokenizer(queries, padding="max_length", truncation=True, max_length=4096, 
                              return_attention_mask=True, add_special_tokens=False)
        logger.info("Query tokenization complete")
        
        # Tokenize the teacher reviews with attention masks
        logger.info("Tokenizing teacher reviews...")
        answer_enc = tokenizer(teacher_reviews, padding="max_length", truncation=True, max_length=512, 
                              return_attention_mask=True, add_special_tokens=False)
        logger.info("Teacher review tokenization complete")

        # Verify pad_token_id is set
        if tokenizer.pad_token_id is None:
            logger.warning("Tokenizer pad_token_id is None. Setting to eos_token_id.")
            tokenizer.pad_token_id = tokenizer.eos_token_id

        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for i, (prompt_ids, prompt_mask, answer_ids, answer_mask) in enumerate(zip(
            prompt_enc["input_ids"], prompt_enc["attention_mask"], 
            answer_enc["input_ids"], answer_enc["attention_mask"]
        )):
            # Combine input IDs and attention masks
            input_ids = prompt_ids + answer_ids
            attention_mask = prompt_mask + answer_mask
            
            # Create labels (ignore prompt part with -100)
            labels = [-100] * len(prompt_ids) + answer_ids

            # Truncate to max_combined_length and pad if needed
            input_ids = input_ids[:self.max_combined_length]
            labels = labels[:self.max_combined_length]
            attention_mask = attention_mask[:self.max_combined_length]

            # Calculate padding length
            pad_len = self.max_combined_length - len(input_ids)
            
            # Pad input_ids, labels, and attention_mask
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention_mask += [0] * pad_len  # 0 for padding tokens
            
            # For invalid samples, set attention mask to 0 to prevent them from contributing to loss
            if i not in valid_indices:
                # Set all attention mask values to 0 for invalid samples
                attention_mask = [0] * len(attention_mask)
                logger.info(f"Marking example {i} as invalid by setting attention mask to 0")

            # Convert to tensors
            input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))
            attention_mask_batch.append(torch.tensor(attention_mask, dtype=torch.long))
        
        logger.info(f"Processed {len(input_ids_batch)} examples for knowledge distillation, {len(valid_indices)} valid")
        return input_ids_batch, labels_batch, attention_mask_batch