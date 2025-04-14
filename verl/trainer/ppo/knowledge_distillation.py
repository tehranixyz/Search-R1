import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import numpy as np
import os
import re
import json
from search_r1.llm_agent.retrievers import Retrievers
from openai import OpenAI
from verl.protocol import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

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
    
    def __init__(self, retriever_config_path: str = None):
        """
        Initialize the KnowledgeDistillation class.
        
        Args:
            retriever_config_path (str, optional): Path to the retriever configuration file.
        """
        self.retrievers = None
        if retriever_config_path:
            self.retrievers = Retrievers(json_path=retriever_config_path, test_mode=True)
    
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
    
    def get_teacher_review(self, prompt: str, response: str, teacher_model: Any = None) -> str:
        """
        Get review from teacher model (GPT-4) for a given prompt-response pair.
        
        Args:
            prompt (str): The input prompt.
            response (str): The model's response.
            teacher_model (Any): The teacher model to use for review.
            
        Returns:
            str: The teacher's review of the response.
        """
        # Extract information from prompt
        prompt_info = self.extract_prompt_info(prompt)
        
        # Extract translation from response
        translated_code = self.extract_translation(response)

        # Print extracted prompt information and translation
        print("\nExtracted Prompt Information:")
        print(f"Source Language: {prompt_info.get('source_language', 'Not found')}")
        print(f"Target Language: {prompt_info.get('target_language', 'Not found')}")
        print(f"Principle: {prompt_info.get('principle', 'Not found')}")
        print(f"Source Code:\n{prompt_info.get('source_code', 'Not found')}")
        
        print("\nExtracted Translation:")
        print(f"Translated Code:\n{translated_code}")
        
        # If no translation is found, return an error message
        if not translated_code:
            return "No translation found in the response. Please provide a translation between <translation> and </translation> tags."
        
        # Map principle to preference name
        preference = self.map_principle_to_preference(prompt_info['principle'])
        
        # Create assessment query
        # query = self.create_assessment_query(
        #     source_code=prompt_info['source_code'],
        #     translated_code=translated_code,
        #     src_lang=prompt_info['source_language'],
        #     trg_lang=prompt_info['target_language'],
        #     preference=preference
        # )
        
        # Call teacher model with the query
        #teacher_review = teacher_model.generate(query)
        teacher_review = self.retrievers.inquire(retriever_name=preference, source_code=prompt_info['source_code'], translated_code=translated_code, src_lang=prompt_info['source_language'], trg_lang=prompt_info['target_language'])
        return teacher_review

    # def get_student_review(self, prompt: str, response: str, student_model: Any) -> str:
    #     """
    #     Get review from student model for the same prompt-response pair.
        
    #     Args:
    #         prompt (str): The input prompt.
    #         response (str): The model's response.
    #         student_model (Any): The student model to use for review.
            
    #     Returns:
    #         str: The student's review of the response.
    #     """
    #     # Extract information from prompt
    #     prompt_info = self.extract_prompt_info(prompt)
        
    #     # Extract translation from response
    #     translated_code = self.extract_translation(response)
        
    #     # If no translation is found, return an error message
    #     if not translated_code:
    #         return "No translation found in the response. Please provide a translation between <translation> and </translation> tags."
        
    #     # Map principle to preference name
    #     preference = self.map_principle_to_preference(prompt_info['principle'])
        
    #     # Create assessment query
    #     query = self.create_assessment_query(
    #         source_code=prompt_info['source_code'],
    #         translated_code=translated_code,
    #         src_lang=prompt_info['source_language'],
    #         trg_lang=prompt_info['target_language'],
    #         preference=preference
    #     )
        
    #     # Call student model with the query
    #     # Convert string query to DataProto
    #     tokenizer = student_model.tokenizer  # You need to access the tokenizer
    #     tokenized = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    #     input_ids = tokenized["input_ids"]
    #     attention_mask = tokenized["attention_mask"]
    #     position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)

    #     # Create DataProto
    #     query_data = DataProto.from_dict({
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "position_ids": position_ids
    #     })

    #     # Now call generate_sequences with the DataProto
    #     student_review_data = student_model.generate_sequences(query_data)
        
    #     # Convert the DataProto output to a string
    #     # Extract the response tokens and decode them
    #     response_tokens = student_review_data.batch['responses']
    #     student_review = tokenizer.decode(response_tokens[0], skip_special_tokens=True)
        
    #     return student_review

    def compute_review_similarity(self, teacher_review: str, student_review: str, tokenizer: Any) -> float:
        """
        Compute similarity between teacher and student reviews using embeddings.
        
        Args:
            teacher_review (str): The review from the teacher model.
            student_review (str): The review from the student model.
            tokenizer (Any): The tokenizer to use for embedding.
            
        Returns:
            float: The similarity score between the reviews.
        """
        # Tokenize and get embeddings
        teacher_tokens = tokenizer(teacher_review, return_tensors="pt", padding=True, truncation=True)
        student_tokens = tokenizer(student_review, return_tensors="pt", padding=True, truncation=True)
        
        # Get embeddings (assuming model has get_embeddings method)
        teacher_emb = teacher_tokens['input_ids'].mean(dim=1)  # Simple pooling
        student_emb = student_tokens['input_ids'].mean(dim=1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(teacher_emb, student_emb)
        return similarity.item()

    def compute_kd_reward(self, teacher_reviews: List[str], student_reviews: List[str], 
                         tokenizer: Any, kd_weight: float = 0.5) -> torch.Tensor:
        """
        Compute knowledge distillation reward based on review similarities.
        
        Args:
            teacher_reviews (List[str]): List of reviews from the teacher model.
            student_reviews (List[str]): List of reviews from the student model.
            tokenizer (Any): The tokenizer to use for embedding.
            kd_weight (float, optional): Weight for the knowledge distillation reward. Defaults to 0.5.
            
        Returns:
            torch.Tensor: The knowledge distillation reward tensor.
        """
        similarities = []
        for t_review, s_review in zip(teacher_reviews, student_reviews):
            sim = self.compute_review_similarity(t_review, s_review, tokenizer)
            similarities.append(sim)
        
        # Convert to tensor and scale by weight
        kd_reward = torch.tensor(similarities, dtype=torch.float32) * kd_weight
        return kd_reward
        
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
    
    def get_student_reviews_batch(self, prompts: List[str], responses: List[str], student_model: Any, tokenizer: Any = None) -> List[str]:
        """
        Get reviews from student model for a batch of prompt-response pairs.
        
        Args:
            prompts (List[str]): List of input prompts.
            responses (List[str]): List of model responses.
            student_model (Any): The student model to use for review.
            tokenizer (Any, optional): The tokenizer to use. If None, will use student_model.tokenizer.
            
        Returns:
            List[str]: List of student reviews.
        """
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
        
        # Tokenize all queries at once
        tokenized = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        position_ids = torch.arange(0, input_ids.shape[1]).unsqueeze(0).expand_as(input_ids)

        # Create DataProto for the batch
        query_data = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids
        })

        print("Padding query data to be divisible by world size...")
        query_data_padded, pad_size = pad_dataproto_to_divisor(query_data, student_model.world_size)
        print(f"Generating student reviews with padded data (pad_size={pad_size})...")
        student_review_data_batch_padded = student_model.generate_sequences(query_data_padded)
        print("Unpadding generated review data...")
        student_review_data = unpad_dataproto(student_review_data_batch_padded, pad_size=pad_size)
        print("Student review generation complete")
        
        # Extract the response tokens and decode them
        response_tokens = student_review_data.batch['responses']
        student_reviews_batch = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in response_tokens]
        
        return student_reviews_batch
    
    def process_batch(self, prompts: List[str], responses: List[str], student_model: Any, tokenizer: Any = None) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a batch of prompt-response pairs to get teacher and student reviews.
        
        Args:
            prompts (List[str]): List of input prompts.
            responses (List[str]): List of model responses.
            student_model (Any): The student model to use for review.
            tokenizer (Any, optional): The tokenizer to use. If None, will use student_model.tokenizer.
            
        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]: Tuple containing tokenized teacher and student reviews.
            Each dictionary contains 'input_ids' and 'attention_mask' tensors.
        """
        print(f"Getting teacher reviews for batch of {len(prompts)} examples...")
        teacher_reviews = self.get_teacher_reviews_batch(prompts, responses)
        print(f"Completed getting {len(teacher_reviews)} teacher reviews")
        
        print(f"Getting student reviews for batch of {len(prompts)} examples...")
        student_reviews = self.get_student_reviews_batch(prompts, responses, student_model, tokenizer)
        print(f"Completed getting {len(student_reviews)} student reviews")
        
        # Tokenize the reviews
        print("Tokenizing teacher and student reviews...")
        teacher_tokens = tokenizer(teacher_reviews, return_tensors="pt", padding=True, truncation=True)
        student_tokens = tokenizer(student_reviews, return_tensors="pt", padding=True, truncation=True)
        print("Tokenization complete")
        
        return teacher_tokens, student_tokens