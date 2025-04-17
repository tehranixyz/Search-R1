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

import re
import random
from typing import Dict

def extract_prompt_info(prompt: str) -> Dict[str, str]:
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

def map_principle_to_preference(principle: str, retrievers=None) -> str:
    """
    Map the principle to a preference name using the retrievers_config.json.
    
    Args:
        principle (str): The principle text.
        retrievers: The retrievers configuration object.
        
    Returns:
        str: The mapped preference name.
    """
    if not retrievers:
        return "Functional Equivalence"  # Default preference
        
    # Load the retrievers configuration
    retriever_configs = retrievers.retrievers
    
    # Check each retriever's principles
    for retriever_name, retriever_config in retriever_configs.items():
        # Get the prompt template which contains the principles
        principles = retriever_config["principles"]
        
        # Check if the principle is in the prompt template
        if principle in principles:
            return retriever_name
    
    # If no match is found, return the default preference
    return "Functional Equivalence"

def create_assessment_query(source_code: str, translated_code: str, 
                            src_lang: str, trg_lang: str, preference: str, retrievers=None) -> str:
    """
    Create a query using the relevant assessment template.
    
    Args:
        source_code (str): The source code.
        translated_code (str): The translated code.
        src_lang (str): Source programming language.
        trg_lang (str): Target programming language.
        preference (str): The preference name.
        retrievers: The retrievers configuration object.
        
    Returns:
        str: The formatted query.
    """
    if not retrievers or preference not in retrievers.retrievers:
        # Create a default query if retrievers are not available
        return f"Evaluate the following code translation from {src_lang} to {trg_lang}:\n\nSource code:\n{source_code}\n\nTranslated code:\n{translated_code}"
    
    # Get the prompt template for the preference
    prompt_template = retrievers.retrievers[preference]["prompt_template"]
    
    # Format the query using the template
    query = prompt_template.format(
        src_lang=src_lang,
        trg_lang=trg_lang,
        source_code=source_code,
        translated_code=translated_code
    )
    
    return query

def extract_solution(solution_str):
    """Extract the translation from the solution string.
    
    Args:
        solution_str: The solution text containing XML-like tags
        
    Returns:
        The extracted translation if found, None otherwise
    """
    # Look for content between <translation> and </translation> tags
    answer_pattern = r'<translation>(.*?)</translation>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def extract_judge_score(judge_review: str) -> int:
    """Extract the score from the judge review.
    
    Args:
        judge_review: The judge review text
        
    Returns:
        The extracted score if found, None otherwise
    """
    score_pattern = r'score[ :,\s]*(.*?)([1-5])'  # Case insensitive search for 'score' followed by any characters and then an integer 1-5
    match = re.search(score_pattern, judge_review, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return score
    return 0

def compute_score(solution_str, ground_truth, retrievers=None, score=1.0):
    """The scoring function for code translation task.
    
    Args:
        solution_str: The solution text
        ground_truth: The ground truth dictionary
        retrievers: The retrievers configuration object
        score: The score for the correct answer
        
    Returns:
        float: The computed score
    """
    # Extract the translation
    translation = extract_solution(solution_str)
    
    # If no translation found, return 0
    if translation is None:
        return 0, None, None
        
    prompt_info = extract_prompt_info(solution_str)
    preference = map_principle_to_preference(principle=prompt_info['principle'], retrievers=retrievers)
    
    # Print details about the inquiry
    print(f"Inquiry details:")
    print(f"- Retriever: {preference}")
    print(f"- Translated code: {translation}")
    print(f"- Source language: {prompt_info['source_language']}")
    print(f"- Target language: {prompt_info['target_language']}")
    
    judge_review = retrievers.inquire(
        retriever_name=preference, 
        source_code=prompt_info['source_code'], 
        translated_code=translation, 
        src_lang=prompt_info['source_language'], 
        trg_lang=prompt_info['target_language']
    )
    score = extract_judge_score(judge_review)

    judge_query = create_assessment_query(
                source_code=prompt_info['source_code'],
                translated_code=translation,
                src_lang=prompt_info['source_language'],
                trg_lang=prompt_info['target_language'],
                preference=preference,
                retrievers=retrievers
            )

    return score, judge_query, judge_review