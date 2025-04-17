import json
import os
import random
from openai import OpenAI
from typing import List, Dict, Any

class Retrievers:
    """
    A class that manages multiple LLM-based retrievers for code translation and analysis.
    
    This class loads retriever configurations from a JSON file and provides methods
    to interact with different LLM models for code-related queries.
    """
    
    def __init__(self, json_path: str, test_mode: bool = False):
        """
        Initialize the Retrievers class with configurations from a JSON file.
        
        Args:
            json_path (str): Path to the JSON configuration file containing retriever settings.
            test_mode (bool, optional): If True, returns dummy responses without calling OpenAI API. Defaults to False.
            
        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
            json.JSONDecodeError: If the JSON file is not properly formatted.
        """
        self.retrievers = {}
        self.test_mode = test_mode
        
        # Check if the JSON file exists
        if not os.path.exists(json_path):
            print(f"Error: Configuration file not found at: {os.path.abspath(json_path)}")
            raise FileNotFoundError(f"Configuration file not found: {json_path}")
            
        try:
            with open(json_path, 'r') as f:
                retriever_configs = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in configuration file: {json_path}")
            raise json.JSONDecodeError(f"Invalid JSON format in configuration file: {e}", e.doc, e.pos)

        # Initialize each retriever with its configuration
        for retriever_config in retriever_configs:
            self.retrievers[retriever_config["name"]] = {
                "model_name": retriever_config["model_name"],
                "client":OpenAI(
                    api_key="EMPTY",
                    base_url=retriever_config["url"],
                ),
                "prompt_template": retriever_config["prompt_template"],
                "principles": retriever_config["principles"]
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
        
        if self.test_mode:
            # Return dummy response for testing purposes
            random_score = random.randint(1, 5)
            
            # Rationales for each score (4 per score)
            rationales = {
                1: [
                    "The translation has significant errors that affect functionality.",
                    "The code structure is completely different from the original.",
                    "Critical logic is missing or incorrectly implemented.",
                    "The translation fails to maintain the original algorithm's behavior.",
                    "The code contains fundamental conceptual errors that make it unusable."
                ],
                2: [
                    "The translation has several errors but maintains basic functionality.",
                    "Some important features are missing or incorrectly implemented.",
                    "The code structure differs significantly from the original.",
                    "The translation has syntax errors that would prevent compilation.",
                    "The code has logical errors that would cause incorrect results in most cases."
                ],
                3: [
                    "The translation is mostly correct but has some minor issues.",
                    "The code structure is similar but not identical to the original.",
                    "Some edge cases are not handled correctly.",
                    "The translation works but could be more idiomatic for the target language.",
                    "The code has minor performance inefficiencies compared to the original."
                ],
                4: [
                    "The translation is very good with only minor differences.",
                    "The code structure closely matches the original with proper adaptations.",
                    "All functionality is preserved with appropriate language-specific changes.",
                    "The translation is correct and follows good practices for the target language.",
                    "The code is well-optimized and handles all edge cases appropriately."
                ],
                5: [
                    "The translation is perfect and maintains all functionality.",
                    "The code structure is identical to the original with proper language adaptations.",
                    "All edge cases are handled correctly.",
                    "The translation is idiomatic and follows best practices for the target language.",
                    "The code is highly optimized and includes appropriate error handling and documentation."
                ]
            }
            
            # Select a random rationale for the generated score
            selected_rationale = random.choice(rationales[random_score])
            
            return f"{random_score} and\nRationale: {selected_rationale}"
        
        completion = retriever["client"].completions.create(model=retriever["model_name"], prompt=query)
        #print(f"Response from {retriever_name}:\n{completion}")
        return completion.choices[0].text



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
    retrievers = Retrievers("retrievers_config.json", test_mode=True)  # Enable test mode for this example

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



