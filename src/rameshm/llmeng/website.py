# -*- coding: utf-8 -*-
"""
Created on Wed May 28 06:52:22 2025
@author: rameshUser

Scrapes a URL and returns its contents.

"""
#imports external libs and packages
from bs4 import BeautifulSoup
import requests

#imports my modules and packages
from rameshm.llmeng.utils import init_utils


class Website:

    def __init__(self, url):
        """Initialize Website object and scrape content using BeautifulSoup."""
        self.url = url
        headers = {"User-Agent":
                        ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                        " (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
                        )
                   }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        links = [link.get('href') for link in soup.find_all('a')]
        self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

def main():
    """Main function to demonstrate the Website class."""
    logger = init_utils.get_initialized_logger()
    url = "https://huggingface.co"
    website = Website(url)
    logger.info(f"Website URL: {website.url} with Title: {website.title}")
    logger.info(f"Website Contents:\n{website.get_contents()}")
    logger.debug(f"Website Links: {website.links}")

if __name__ == "__main__":
    main()