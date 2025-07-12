# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:40:30 2025

@author: rameshUser

HACK(s):
    1. Works only with OPenAI and Llama. Modify to use langchain or add clauses for the code to work with other LLM
    2. The .env file from which the LLM Connection Key is sourced is set in KEY_FILE environment variable. Modify to read as argument.
    
"""

from google.genai import types
import gradio as gr

# import internal packages
from rameshm.llmeng.webpage_brochure.website import Website
from rameshm.llmeng.webpage_brochure.create_llm_instance import LlmInstance
from rameshm.llmeng.utils import init_utils

# Initialize the logger and sets environment variables
logger = init_utils.set_environment_logger()
 
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
                " If it includes news or announcements, then summarize these too.\n"
                " Respond in Markdown\n\n"
                f"{website.get_contents()}"
            )
def get_message(website: Website) -> list[dict[str, str]]:
    return [
        {"role": "system", "content" : get_system_prompt()},
        {"role": "user", "content": get_user_prompt(website)}
        ]

def gpt_response(user_prompt: str, llm_model_nm: str) -> str:
    llm_instance = LlmInstance(llm_model_nm)
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": "Is Java a popular programming language"}
    ]
    response = llm_instance.get_llm_model_instance().chat.completions.create(
        model=llm_instance.get_llm_model_name(),
        messages=messages
        #response_format={"type": "json_object"}
    )
    return response.choices[0].message.content

def claude_response(user_prompt: str, llm_model_nm: str) -> str:
    llm_instance = LlmInstance(llm_model_nm)
    print(llm_instance.get_llm_model_instance())
    response = llm_instance.get_llm_model_instance().messages.create(
        model=llm_model_nm,
        max_tokens=1000,
        temperature=0.7,
        system=get_system_prompt(),
        messages=[
                {"role": "user", "content": user_prompt}
        ]
    )
    return response.content[0].text

def gemini_response(user_prompt: str, llm_model_nm: str) -> str:
    llm_instance = LlmInstance(llm_model_nm)

    response = llm_instance.get_llm_model_instance().models.generate_content(
        model=llm_model_nm,
        config=types.GenerateContentConfig(
            system_instruction= get_system_prompt(),
            temperature=0.7),
        contents=user_prompt
    )

    return response.text

def get_response(url: str, llm_model_nm: str) -> str:
    website = Website(url)
    user_prompt = get_user_prompt(website)
    llm_instance = LlmInstance(llm_model_nm)
    logger.debug(f"Model Name: {llm_model_nm} Processing URL: {url}\n\t User Prompt: {user_prompt}")

    if "gpt" in llm_model_nm or "llama" in llm_model_nm:
        result = gpt_response(user_prompt, llm_model_nm)
    elif "claude" in llm_model_nm:
        result = claude_response(user_prompt, llm_model_nm)
    elif "gemini" in llm_model_nm:
        result = gemini_response(user_prompt, llm_model_nm)
    else:
        raise ValueError(f"{llm_model_nm} is not supported")
    return result

def summarize_site(url: str, llm_model_nm: str) -> str:
    summary = get_response(url, llm_model_nm)
    logger.debug(f"Response: {summary}")
    print(f"Done for URL: {url}")
    return summary

# def create_summary(url: str, llm_model_nm: str) -> str:
#     #console = Console()
#     summary = summarize_site(url,llm_model_nm)
#     logger.debug(f"Website Summary for URL: {url} Summary: {summary}")
#     #console.print(Markdown(summary))
#     return summary

url_summary_demo = gr.Interface(
    fn=summarize_site,
    inputs=[
        gr.Textbox(label="URL", placeholder="Type the URL here..."),
        gr.Dropdown(label="Model", choices=["gpt-4o", "gpt-4o-mini", "llama3.2", "claude-sonnet-4-0", "gemini-1.5-flash"],
                    value="llama3.2")
        ],
    outputs=[
        gr.Markdown(label="Brochure")
        ],
    flagging_mode="never"
    )

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
    summary = summarize_site(url, llm_model_nm)
    logger.info(f"Here is the summary for URL: {url} using LLM Model: {llm_model_nm}:\n\n {summary}")

if __name__ == "__main__":
    # Run main()
    #main()

    # Run Gradio.
    # Initialize the Gradio app
    try:
        url_summary_demo.close()
    except Exception as e:
        logger.error(f"Error in the App: {e}", exc_info=True)

    url_summary_demo.launch(inbrowser=True)
