# -*- coding: utf-8 -*-
"""
Created on Wed May 28 07:37:44 2025

@author: rameshUser
"""
# imports external and system packages
import os
from openai import OpenAI
import anthropic

# import internal packages
from rameshm.llmeng.utils import init_utils

class LLM_Instance:
    
    def __init__(self, llm_model_nm: str):
        self.llm_model_nm = llm_model_nm
        self.logger = init_utils.get_initialized_logger()
        self.llm_model_instance = self.__create_llm_model_instance()

    def __create_llm_model_instance(self):
        """ Creates an instance of the LLM based on the model name."""
        if "gpt" in self.llm_model_nm:
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "llama" in self.llm_model_nm:
            return OpenAI(base_url='http://localhost:11434/v1', api_key="ollama") #api_key=os.getenv("OLLAMA_API_KEY"))
        elif "claude" in self.llm_model_nm:
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise Exception(f"LLM Model: {self.llm_model_nm} is not yet supported")

    def get_llm_model_instance(self):
        """ Returns the LLM model instance."""
        return self.llm_model_instance

    def get_llm_model_name(self) -> str:
        """ Returns the LLM model name."""
        return self.llm_model_nm

if __name__ == "__main__":
    llm_model_list = ["llama3.2", "gpt-4o-mini", "claude-sonnet-4-20250514"]
    for llm_model_nm in llm_model_list:
       print(f"Creating LLM Model: {llm_model_nm}")
       llm_instance = LLM_Instance(llm_model_nm)
       llm_model_instance = llm_instance.get_llm_model_instance()
       if ("llama" in llm_model_nm or "gpt" in llm_model_nm):
           messages = [
                {"role": "system", "content": "You are a funny assistant"},
                {"role": "user", "content": "What is 2 + 2*3?"}
                ]
           response = llm_model_instance.chat.completions.create(
               model = llm_model_nm,  messages = messages
               )
           print(f"Response from Model: {llm_model_nm}: {response.choices[0].message.content} \n\n")
       elif ("claude" in llm_model_nm):
           message = llm_model_instance.messages.create(
                model=llm_model_nm,   #"claude-3-7-sonnet-latest",
                max_tokens=200,
                temperature=0.7,
                system= "You are a funny assistant",
                messages=[{"role": "user", "content": "what is 2 + 2*3?"}]
                )
           print(f"Response from Model: {llm_model_nm} : {message.content[0].text} \n")
       else:
           raise Exception(f"LLM Model: {llm_model_nm} is not yet supported")
       