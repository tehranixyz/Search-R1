import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import numpy as np

def get_teacher_review(prompt: str, response: str, teacher_model: Any) -> str:
    """Get review from teacher model (GPT-4) for a given prompt-response pair."""
    review_prompt = f"""Please review the following code translation:
Input: {prompt}
Output: {response}

Provide a detailed review focusing on:
1. Correctness of translation
2. Code quality and style
3. Potential issues or improvements

Review:"""
    
    teacher_review = teacher_model.generate(review_prompt)
    return teacher_review

def get_student_review(prompt: str, response: str, student_model: Any) -> str:
    """Get review from student model for the same prompt-response pair."""
    review_prompt = f"""Please review the following code translation:
Input: {prompt}
Output: {response}

Provide a detailed review focusing on:
1. Correctness of translation
2. Code quality and style
3. Potential issues or improvements

Review:"""
    
    student_review = student_model.generate(review_prompt)
    return student_review

def compute_review_similarity(teacher_review: str, student_review: str, tokenizer: Any) -> float:
    """Compute similarity between teacher and student reviews using embeddings."""
    # Tokenize and get embeddings
    teacher_tokens = tokenizer(teacher_review, return_tensors="pt", padding=True, truncation=True)
    student_tokens = tokenizer(student_review, return_tensors="pt", padding=True, truncation=True)
    
    # Get embeddings (assuming model has get_embeddings method)
    teacher_emb = teacher_tokens['input_ids'].mean(dim=1)  # Simple pooling
    student_emb = student_tokens['input_ids'].mean(dim=1)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(teacher_emb, student_emb)
    return similarity.item()

def compute_kd_reward(teacher_reviews: List[str], student_reviews: List[str], 
                     tokenizer: Any, kd_weight: float = 0.5) -> torch.Tensor:
    """Compute knowledge distillation reward based on review similarities."""
    similarities = []
    for t_review, s_review in zip(teacher_reviews, student_reviews):
        sim = compute_review_similarity(t_review, s_review, tokenizer)
        similarities.append(sim)
    
    # Convert to tensor and scale by weight
    kd_reward = torch.tensor(similarities, dtype=torch.float32) * kd_weight
    return kd_reward 