# -*- coding: utf-8 -*-
"""
Created on Wed May 28 07:17:19 2025

@author: rameshUser

Generates a Brochure by traversing the URL and other important links within the supplied URL

"""
# imports external and system packages
import json
from typing import Optional
import re

# import internal packages
from website import Website
from create_llm_instance import LLM_Instance
from rameshm.llmeng.utils import init_utils

""" Initialize the logger and sets environment variables"""
logger = init_utils.get_initialized_logger()

def get_links_system_prompt() -> str:
    """ Returns System Prompt for getting all links within a URL. """
    links_system_prompt = ( "You are provided with a list of links found on a webpage."
                            " You are able to decide which of the links would be most relevant to include in a brochure about the company,"
                            " such as links to an About page, or a Company page, or Careers/Jobs pages.\n"
                            "You should ignore links to Terms of Service, Privacy Policy, email links, etc.\n"
                            )
    links_system_prompt += "You should respond in JSON as in this example:"
    links_system_prompt += """
    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page": "url": "https://another.full.url/careers"}
        ]
    }
    """
    logger.debug(f"Link System Prompt: {links_system_prompt}")
    return links_system_prompt

def get_links_user_prompt(website: Website) -> str:
    """ Returns User Prompt for getting all links within a URL. """
    links_user_prompt = f"Here is the list of links on the website of {website.url}.\n"
    links_user_prompt += (  "Please decide which of these are relevant web links for a brochure about the company,"
                            " respond with the full https URL in JSON format."
                            " Do not include Terms of Service, Privacy, email links.\n"
                        )
    links_user_prompt += "Links (some might be relative links):\n"
    links_user_prompt += "\n".join(website.links)
    logger.debug(f"Link User Prompt: {links_user_prompt}")
    return links_user_prompt

def get_links_message(website: Website) -> list[dict[str, str]]:
    """ Returns the string to be used in messages part of the create call. """
    links_message = [
        {"role": "system", "content" : get_links_system_prompt()},
        {"role": "user", "content": get_links_user_prompt(website)}
        ]
    logger.debug(f"Link System Prompt: {links_message}")
    return links_message

def get_links(website: Website, llm_instance: LLM_Instance) -> str:
    """ Using LLL returns all relevant links contained in URL used in creating the Website instance. """
    response = llm_instance.get_llm_model_instance().chat.completions.create(
        model = llm_instance.get_llm_model_name(),
        messages = get_links_message(website),
        response_format={"type": "json_object"}
        )
    result = response.choices[0].message.content
    content = json.loads(result)
    
    logger.debug(f"get_links Content is: {content}")
    return content
    
def get_all_linked_details(website: Website, llm_instance: LLM_Instance) -> str:
    """ Returns contents for list of URLs within the original landing page URL passed in. """
    all_content = f"Content for Landing Page URL: {website.url} is: {website.get_contents()}\n"
    all_content += "\nContent from embedded links:\n"
    links = get_links(website, llm_instance)
    for link in links["links"]:
        all_content += f"\n{link['type']}\n"        
        try:
                # Encountering some dummy URLs etc., hence ignoring irrelevant ones
            lnk_url = link.get("url") # remove spaces in URL
            logger.debug(f"Processing Link: {lnk_url}\n")
            if lnk_url:
                lnk_url = re.sub(r" +", "", lnk_url) # remove spaces in URL
            lnk_website = Website(lnk_url)
            all_content += f"Link {lnk_website.url} is of type: '{link['type']}', and its content is as follows: "
            all_content += f"{lnk_website.get_contents()} \n"
        except Exception as e:
            logger.warning(f"Exception {e} happened while processing URL: {lnk_url}. Error ignored and  processing continued\n")
    return all_content
 
def get_brochure_system_prompt() -> str:
    """ Returns system prompt for preparing Brochure based on the data from all the links. """
    brochure_system_prompt = (  "You are an assistant that analyzes the contents of several"
                                " relevant pages from a company website and creates a short"
                                " brochure about the company for prospective customers"
                                " investors and recruits. Respond in markdown."
                                "Include details of company culture, customers and careers/jobs if you have the information."
                             )
    
                # Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':
                
                # system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
                # and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
                # Include details of company culture, customers and careers/jobs if you have the information."pass
    logger.debug(f"Link System Prompt: {brochure_system_prompt}")
    return brochure_system_prompt

def get_brochure_user_prompt(website: Website, llm_instance: LLM_Instance, prompt_lmt: Optional[int] = None) -> str:
    """ Returns User prompt for preparing Brochure based on the data from all the links. """
    company_nm = website.url.split("//")[1].split(".")[0].capitalize() # Get's company name immediately following http:// and before .com etc.
    
    brochure_user_prompt = f"You are looking at a company called: {company_nm}\n"
    brochure_user_prompt += (   "Here are the contents of its landing page and other relevant"
                                " pages; use this information to build a short brochure of the"
                                " company in markdown.\n"
                            )
    brochure_user_prompt += get_all_linked_details(website, llm_instance)
    brochure_user_prompt = brochure_user_prompt[:prompt_lmt] # Truncate if more than 20,000 characters
    
    logger.debug(f"Truncated Brochure User Prompt: {brochure_user_prompt}")
    return brochure_user_prompt

def create_brochure(url: str, llm_model_nm: str) -> str:
    """ Returns the text of Brochure being returned by the LLM. """
    website = Website(url)
    llm_instance = LLM_Instance(llm_model_nm)
    system_prompt = get_brochure_system_prompt()
    user_prompt = get_brochure_user_prompt(website, llm_instance, 20_000)
    logger.debug(f"Brochure Truncated User Prompt: {user_prompt}")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
        ]
    logger.debug(f"Brochure Messages Prompt: {messages}")
    response = llm_instance.get_llm_model_instance().chat.completions.create(
        model = llm_instance.get_llm_model_name(),
        messages = messages
        )
    result = response.choices[0].message.content
    
    return result    

def main():
    """ Main function to create a brochure from a URL. """
    """ Models currently supported are llama3.2, gpt-4o-mini, gpt-4o, gpt-4-turbo, gpt-3.5-turbo"""
    logger.info("Starting Brochure Creation Process")
    use_defaults = input("Use defaults? (Y/N): ").upper() == "Y"
    llm_model_nm = "llama3.2" if use_defaults else input("Enter LLM Model Name: ")
    url = "https://huggingface.com" if use_defaults else input("Enter the URL: ")
    logger.debug(f"Creating brochure for URL: {url} using LLM Model: {llm_model_nm} \n")
    brochure = create_brochure(url, llm_model_nm)
    logger.info(f"Here is the brochure for URL: {url} using LLM Model: {llm_model_nm}:\n\n {brochure}")

if __name__ == "__main__":
   main()