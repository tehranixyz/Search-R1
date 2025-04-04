import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file.")
parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Name of the retriever model.")

args = parser.parse_args()

#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# 1) Build a config (could also parse from arguments).
#    In real usage, you'd parse your CLI arguments or environment variables.
config = Config(
    retrieval_method = "e5",  # or "dense"
    index_path=args.index_path,
    corpus_path=args.corpus_path,
    retrieval_topk=args.topk,
    faiss_gpu=True,
    retrieval_model_path=args.retriever_model,
    retrieval_pooling_method="mean",
    retrieval_query_max_length=256,
    retrieval_use_fp16=True,
    retrieval_batch_size=512,
)

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    import random

    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Generate random results for testing purposes
    resp = []
    for query in request.queries:
        single_result = []
        for _ in range(request.topk):
            document = {
                "title": f"Random Title {random.randint(1, 100)}",
                "text": f"Random text content {random.randint(1, 1000)}."
            }
            score = random.uniform(0, 1) if request.return_scores else None
            if request.return_scores:
                single_result.append({"document": document, "score": score})
            else:
                single_result.append(document)
        resp.append(single_result)

    return {"result": resp}


if __name__ == "__main__":
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
