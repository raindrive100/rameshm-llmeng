# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:40:30 2025

@author: rameshUser

HACK(s):
    1. Works only with OPenAI and Llama. Modify to use langchain or add clauses for the code to work with other LLM
    2. The .env file from which the LLM Connection Key is sourced is set in KEY_FILE environment variable. Modify to read as argument.
    
"""

# imports external and system packages
#from rich.console import Console
#from IPython.display import Markdown #, display, update_display

# import internal packages
from rameshm.llmeng.website import Website
from rameshm.llmeng.create_llm_instance import LLM_Instance
from rameshm.llmeng.utils import init_utils

# Initialize the logger and sets environment variables
logger = init_utils.get_initialized_logger()
 
def get_system_prompt() -> str:
    system_prompt = (   "You are an assistant that analyzes the contents of a website and provides a short summary,"
                        " ignoring text that might be navigation related."
                        " If the website contains news or announcements, summarize these too."
                        " Respond in markdown."
                    )
                     
    return system_prompt
        
def get_user_prompt(website: Website) -> str:
    return  (   f"You are looking at a website titled {website.title}"
                "\nThe contents of this website is as follows;"
                " please provide a short summary of this website in markdown."
                " If it includes news or announcements, then summarize these too.\n\n"
                f"{website.get_contents()}"
            )

def get_message(website: Website) -> list[dict[str, str]]:
    return [
        {"role": "system", "content" : get_system_prompt()},
        {"role": "user", "content": get_user_prompt(website)}
        ]
    
def summarize_site(url: str, llm_model_nm: str) -> str:
    website = Website(url)
    llm_instance = LLM_Instance(llm_model_nm)

    messages = get_message(website)
    logger.debug(f"System and User Message for URL: {website.url} is: {messages}")
    response = llm_instance.get_llm_model_instance().chat.completions.create(
        model = llm_model_nm, 
        messages = messages
        )
    
    return response.choices[0].message.content

def create_summary(url: str, llm_model_nm: str) -> str:
    #console = Console()
    summary = summarize_site(url,llm_model_nm)
    logger.debug(f"Website Summary for URL: {url} Summary: {summary}")
    #console.print(Markdown(summary))
    return summary

def usage(url: str, llm_model_nm: str):
    """Print usage instructions."""
    print("Usage: python llm_website_summary.py <website URL> [LLM Model Name]")
    print("Example: python llm_website_summary.py https://example.com gpt-4o-mini")
    print(f"Values supplied are: URL: {url} LLM Model Name: {llm_model_nm}")

def main():
    """Main function to run the LLM Web Scraper """
    """ Models currently supported are llama3.2, gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo"""
    use_defaults = input("Use defaults? (Y/N): ").upper() == "Y"
    llm_model_nm = "llama3.2" if use_defaults else input("Enter LLM Model Name: ")
    url = "https://huggingface.com" if use_defaults else input("Enter the URL: ")
    logger.debug(f"Creating website summary for URL: {url} using LLM Model: {llm_model_nm} \n")
    summary = create_summary(url, llm_model_nm)
    logger.info(f"Here is the summary for URL: {url} using LLM Model: {llm_model_nm}:\n\n {summary}")

if __name__ == "__main__":
    main()
