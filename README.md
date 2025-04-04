# Search-R1: Train your LLMs to reason and call a search engine with reinforcement learning

<strong>Search-R1</strong> is an extension of <strong>DeepSeek-R1(-Zero)</strong> methods for <em>training reasoning and searching (tool-call) interleaved LLMs</em>. We built upon [veRL](https://github.com/volcengine/verl).

Through RL (rule-based outcome reward), the 3B **base** LLM (both Qwen2.5-3b-base and Llama3.2-3b-base) develops reasoning and search engine calling abilities all on its own.

Paper: [link](https://arxiv.org/pdf/2503.09516); Model and data: [link](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5); Twitter thread: [link](https://x.com/BowenJin13/status/1895544294473109889); Full experiment log 1: [link](https://wandb.ai/peterjin/Search-R1-open); Full experiment log 2: [link](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)

You can refer to this [link](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa) for detailed instructions on reproducing the results from the paper.


![single-turn](public/single-turn.png)


## Links

- [Installation](#installation)
- [Quick start](#quick-start)
- [Preliminary results](#preliminary-results)
- [Inference](#inference)
- [Use your own dataset](#use-your-own-dataset)
- [Use your own search engine](#use-your-own-search-engine)
- [Ackowledge](#acknowledge)
- [Citations](#citations)

## Installation

### Search-r1 environment
```bash
conda create -n searchr1 python=3.9
conda activate searchr1
pip install vllm==0.5.4 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip install -e .
pip install flash-attn --no-build-isolation
pip install wandb
```

### Retriever environment
If you would like to call a local retriever as the search engine, you can install the environment as follows. (We recommend using a seperate environment.)
```bash
conda create -n retriever python=3.10
conda activate retriever

pip pip install fastapi uvicorn pydantic
```


## Quick start

(1) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) Run RL training (GRPO) with Qwen2.5-0.5B-Instruct.
```bash
conda activate searchr1
bash train_grpo.sh
```

## Preliminary results

(1) The base model (llama3.2-3b-base) learns to call the search engine and obtain improved performance.

![llama-3b](public/llama32-3b.png)


(2) The base model (Qwen2.5-7b-base) can learn to conduct multi-turn search engine calling and reasoning with RL.

![multi-turn](public/multi-turn.png)

## Inference
#### You can play with the trained Search-R1 model with your own question.
(1) Launch a local retrieval server.
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) Run inference.
```bash
conda activate searchr1
python infer.py
```
You can modify the ```question``` on line 7 to something you're interested in.

## Use your own dataset

### QA data
For each question-answer sample, it should be a dictionary containing the desired content as below:

```
data = {
        "data_source": data_source,
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": solution
        },
        "extra_info": {
            'split': split,
            'index': idx,
        }
    }
```

You can refer to ```scripts/data_process/nq_search.py``` for a concrete data processing example.

### Corpora

It is recommended to make your corpus a jsonl file, where each line (a dictionary with "id" key and "contents" key) corresponds to one passage. You can refer to ```example/corpus.jsonl``` for an example.

The "id" key corresponds to the passage id, while the "contents" key corresponds to the passage content.
For example:
```
{"id": "0", "contents": "Evan Morris Evan L. Morris (January 26, 1977 \u2013 July 9, 2015) was a lobbyist for Genentech and its parent corporation Roche in Washington."}
...
{"id": "100", "contents": "Three years later, when the United States Exploring Expedition to little-known portions of the globe was organised under Charles Wilkes, Hale was recommended, while yet an undergraduate."}
...
```

**Index your corpora (optional).**
If you would like to use a local retriever as the search engine, you can index your own corpus by:
```
bash search_r1/search/build_index.sh
```
You can change ```retriever_name``` and ```retriever_model``` to your interested off-the-shelf retriever.

## Use your own search engine

The main philosophy is to launch a local or remote search engine server separately from the main RL training pipeline. 

The LLM can call the search engine by calling the search API (e.g., "http://127.0.0.1:8000/retrieve").

You can refer to ```search_r1/search/retriever_server.py``` for an example of launching a local retriever server.

## To do
- Support google search / bing search / brave search API and others.
- Support LoRA tuning.
- Support supervised finetuning.
- Support off-the-shelf rerankers.

## Acknowledge

The concept of Search-R1 is inspired by [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero/tree/main).
Its implementation is built upon [veRL](https://github.com/volcengine/verl) and [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main). 
We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.
We thank Jinsung Yoon and Sercan Arik for insightful discussions.

## Citations

```bibtex
@article{jin2025search,
  title={Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```
