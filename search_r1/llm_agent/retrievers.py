import json
from openai import OpenAI
from typing import List, Dict, Any

class Retrievers:
    """
    A class that manages multiple LLM-based retrievers for code translation and analysis.
    
    This class loads retriever configurations from a JSON file and provides methods
    to interact with different LLM models for code-related queries.
    """
    
    def __init__(self, json_path: str):
        """
        Initialize the Retrievers class with configurations from a JSON file.
        
        Args:
            json_path (str): Path to the JSON configuration file containing retriever settings.
        """
        self.retrievers = {}
        with open(json_path, 'r') as f:
            retriever_configs = json.load(f)

        # Initialize each retriever with its configuration
        for retriever_config in retriever_configs:
            self.retrievers[retriever_config["name"]] = {
                "model_name": retriever_config["model_name"],
                "client":OpenAI(
                    api_key="EMPTY",
                    base_url=retriever_config["url"],
                ),
                "prompt_template": retriever_config["prompt_template"]
            }


    def inquire(self, retriever_name: str, source_code: str, translated_code: str, src_lang: str, trg_lang: str):
        """
        Send a query to a specific retriever and get its response.
        
        Args:
            retriever_name (str): Name of the retriever to use.
            source_code (str): The original source code.
            translated_code (str): The translated code.
            src_lang (str): Source programming language.
            trg_lang (str): Target programming language.
            
        Returns:
            str: The response from the LLM model.
        """
        retriever = self.retrievers[retriever_name]
        query = retriever["prompt_template"].format(src_lang=src_lang, trg_lang=trg_lang, source_code=source_code, translated_code=translated_code)
        completion = retriever["client"].completions.create(model=retriever["model_name"], prompt=query)
        return completion.choices[0].message.content



if __name__ == "__main__":
    # Example source code and translated code for testing
    source_code = """
    def factorial(n):
        if n == 0:
            return 1
        return n * factorial(n-1)
    """
    
    translated_code = """
    int factorial(int n) {
        if (n == 0) {
            return 1;
        }
        return n * factorial(n - 1);
    }
    """

    # Initialize retrievers with config file
    retrievers = Retrievers("retrievers_config.json")

    # Test each retriever
    for retriever_name in retrievers.retrievers:
        print(f"\nTesting {retriever_name} retriever:")
        try:
            response = retrievers.inquire(
                retriever_name=retriever_name,
                source_code=source_code,
                translated_code=translated_code,
                src_lang="Python",
                trg_lang="C++"
            )
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error testing {retriever_name}: {str(e)}")



