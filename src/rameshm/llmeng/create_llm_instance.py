# -*- coding: utf-8 -*-
"""
Created on Wed May 28 07:37:44 2025

@author: rameshUser
"""
# imports external and system packages
import os
from openai import OpenAI
import anthropic
#import google.generativeai as genai
from google import genai

# import internal packages
from rameshm.llmeng.utils import init_utils

class LLM_Instance:
    
    def __init__(self, llm_model_nm: str):
        self.llm_model_nm = llm_model_nm
        self.logger = init_utils.set_environment_logger()
        self.llm_model_instance = self.__create_llm_model_instance()

    def __create_llm_model_instance(self):
        """ Creates an instance of the LLM based on the model name."""
        if "gpt" in self.llm_model_nm:
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "llama" in self.llm_model_nm or "gemma" in self.llm_model_nm:
            return OpenAI(base_url='http://localhost:11434/v1', api_key="ollama")
        elif "claude" in self.llm_model_nm:
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif "gemini" in self.llm_model_nm:
            return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            #genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            #return genai.GenerativeModel(model_name=self.llm_model_nm)
        else:
            raise Exception(f"LLM Model: {self.llm_model_nm} is not yet supported")

    def get_llm_model_instance(self):
        """ Returns the LLM model instance."""
        return self.llm_model_instance

    def get_llm_model_name(self) -> str:
        """ Returns the LLM model name."""
        return self.llm_model_nm

if __name__ == "__main__":
    #llm_model_list = ["llama3.2", "gpt-4o-mini", "claude-sonnet-4-20250514", "gemini-2.0-flash","gemini-2.0-flash-lite"]
    #llm_model_list = ["llama3.2", "gpt-4o-mini", "claude-sonnet-4-20250514","gemini-1.5-flash"]
    llm_model_list = ["llama3.2", "gemma3:1b", "gpt-4o-mini", "claude-sonnet-4-20250514", "gemini-1.5-flash"]

    for llm_model_nm in llm_model_list:
       print(f"Creating LLM Model: {llm_model_nm}")
       llm_instance = LLM_Instance(llm_model_nm)
       llm_model_instance = llm_instance.get_llm_model_instance()
       if any(model in llm_model_nm for model in ["llama", "gpt", "gemma"]): #("llama" in llm_model_nm or "gpt" in llm_model_nm or "gemma" in llm_model_nm):
           messages = [
                {"role": "system", "content": "You are a funny assistant"},
                {"role": "user", "content": "What is 2 + 2*3?"}
                ]
           response = llm_model_instance.chat.completions.create(
               model = llm_model_nm,  messages = messages
               )
           print(f"Response from Model: {llm_model_nm}: Model Name in Response: {response.model} {response.choices[0].message.content} \n\n")
       elif ("claude" in llm_model_nm):
           message = llm_model_instance.messages.create(
                model=llm_model_nm,   #"claude-3-7-sonnet-latest",
                max_tokens=200,
                temperature=0.7,
                system= "You are a funny assistant",
                messages=[{"role": "user", "content": "what is 2 + 2*3?"}]
                )
           print(f"Response from Model: {llm_model_nm} : {message.content[0].text} \n")
       elif ("gemini" in llm_model_nm):
            response = llm_model_instance.models.generate_content(
                model="gemini-1.5-flash",     #gemini-2.0-flash",
                contents="you are a funny assistant, what is 2+2*3?"
            )
            print(f"Google Response: {response.text} \n\n")
           #genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
           #model = genai.GenerativeModel(llm_model_nm)
           #print(f"Gemini Model is: {model}")
           #response = model.generate_content("What is 2+2*3")
           #client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
           #print(f"My Google Client is: {client}")
           #response = client.models.generate_content(
           #    model="gemini-2.0-flash",
           #    contents="Explain how AI works in a few words",
           #)

           #response = llm_model_instance.generate_content("What is 2+2*3")
           #print(f"Finished generating content for model: {llm_model_nm}")
           #print(f"Response from Model: {llm_model_nm}: {response.text} \n\n")
       else:
           raise Exception(f"LLM Model: {llm_model_nm} is not yet supported")

    """
    logger = init_utils.set_environment_logger()
    print(f"Google API Key: {os.getenv('GOOGLE_API_KEY')}")
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    print(f"My Google Client is: {client}")
    response = client.models.generate_content(
           model="gemini-1.5-flash",     #gemini-2.0-flash",
           contents="you are a funny assistant, what is 2+2*3?"
    )
    print(f"Google Response: {response.text} \n\n")
    """