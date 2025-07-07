import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import wikipediaapi
import os
import time
from llm_interface import call_fast_llm

# Configure Google AI Studio API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY and 'OPENAI_API_KEY' in os.environ:
    GOOGLE_API_KEY = os.environ['OPENAI_API_KEY']

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

def get_baidu_baike_content(keyword):
    # design api by the baidubaike
    url = f'https://baike.baidu.com/item/{keyword}'
    # post request
    response = requests.get(url)

    # Beautiful Soup part for the html content
    soup = BeautifulSoup(response.content, 'html.parser')
    # find the main content in the page
    # main_content = soup.find('div', class_='lemma-summary')
    main_content = soup.contents[-1].contents[0].contents[4].attrs['content']
    # find the target content
    # content_text = main_content.get_text().strip()
    return main_content


def get_wiki_content(keyword):
    #  Wikipedia API ready
    wiki_wiki = wikipediaapi.Wikipedia('MyProjectName (merlin@example.com)', 'en')
    #the topic content which you want to spider
    search_topic = keyword
    # get the page content
    page_py = wiki_wiki.page(search_topic)
    # check the existence of the content in the page
    if page_py.exists():
        print("Page - Title:", page_py.title)
        print("Page - Summary:", page_py.summary)
    else:
        print("Page not found.")
    return page_py.summary



def modal_trans(task_dsp):
    try:
        task_in = "'" + task_dsp + \
               "'Just give me the most important keyword about this sentence without explaining it and your answer should be only one keyword."

        # Use centralized LLM interface to get keyword
        response = call_fast_llm(task_in)
        response_text = response["choices"][0]["message"]["content"]

        spider_content = get_wiki_content(response_text)

        task_in = "'" + spider_content + \
               "',Summarize this paragraph and return the key information."

        # Use centralized LLM interface to summarize content
        response = call_fast_llm(task_in)
        result = response["choices"][0]["message"]["content"]
        print("web spider content:", result)
    except Exception as e:
        result = ''
        print(f"web spider error: {e}")
    return result