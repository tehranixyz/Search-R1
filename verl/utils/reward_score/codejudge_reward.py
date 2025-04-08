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


def extract_solution(solution_str):
    """Extract the translation from the solution string.
    
    Args:
        solution_str: The solution text containing XML-like tags
        
    Returns:
        The extracted translation if found, None otherwise
    """
    # Look for content between <translation> and </translation> tags
    translation_pattern = r'<translation>(.*?)</translation>'
    match = re.search(translation_pattern, solution_str, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    return None


def check_format_compliance(solution_str):
    """Check if the solution follows the required format with proper tags.
    
    Args:
        solution_str: The solution text to check
        
    Returns:
        True if format is compliant, False otherwise
    """
    # Check for required tags in the correct order
    required_tags = [
        r'<think>.*?</think>',  # Reasoning
    ]
    
    # Optional judge tags that should be properly formatted if present
    judge_tags = [
        r'<judge>.*?</judge>',  # Judge call
    ]
    
    # Check if all required tags are present
    for tag_pattern in required_tags:
        if not re.search(tag_pattern, solution_str, re.DOTALL):
            return False
    
    # If judge tags are present, they should be properly formatted
    for tag_pattern in judge_tags:
        matches = re.findall(tag_pattern, solution_str, re.DOTALL)
        if matches and not all(match.strip() for match in matches):
            return False
    
    return True


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for code translation task.
    
    Args:
        solution_str: The solution text
        ground_truth: The ground truth dictionary
        method: The method to extract the solution (not used)
        format_score: The score for correct format but wrong answer
        score: The score for the correct answer
        
    Returns:
        float: The computed score
    """
    # Extract the translation
    translation = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth}")
        print(f"Extracted translation: {translation}")
        print(f"Solution string: {solution_str}")
    
    # If no translation found, return 0
    if translation is None:
        if do_print:
            print(f"No translation found")
        return 0
    
    # Check format compliance
    is_format_compliant = check_format_compliance(solution_str)
    if not is_format_compliant:
        if do_print:
            print(f"Format not compliant")
        return 0
    
    # If format is compliant but translation doesn't match ground truth
    if translation != ground_truth.get('target', ''):
        if do_print:
            print(f"Format compliant but wrong translation")
        return format_score
    
    # If everything is correct
    if do_print:
        print(f"Correct translation with proper format")
    return score
