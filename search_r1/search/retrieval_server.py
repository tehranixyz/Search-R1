from typing import List, Optional


import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

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
        retrieval_topk: int = 10,
    ):
        self.retrieval_topk = retrieval_topk


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# 1) Build a config (could also parse from arguments).
#    In real usage, you'd parse your CLI arguments or environment variables.
config = Config()

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
                "contents": f"Random text content {random.randint(1, 1000)}."
            }
            score = random.uniform(0, 1) if request.return_scores else None
            if request.return_scores:
                single_result.append({"document": document, "score": score})
            else:
                single_result.append(document)
        resp.append(single_result)

    return {"result": resp}


import socket
if __name__ == "__main__":
    # Print the IP address of the machine
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host=ip_address, port=8000)