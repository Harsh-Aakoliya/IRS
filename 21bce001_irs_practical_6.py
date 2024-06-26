# -*- coding: utf-8 -*-
"""21BCE111_IRS_Practical_6.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Ih_2aEIquZtVD9U1SvK9fbC38ZbMRvE8
"""

import requests
from urllib.parse import urlparse, urljoin
from collections import deque
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser

def is_allowed(url):
    rp = RobotFileParser()
    rp.set_url(urljoin(url, "/robots.txt"))
    rp.read()
    return rp.can_fetch("*", url)

def is_valid(url, base_url):
    parsed_url = urlparse(url)
    parsed_base_url = urlparse(base_url)
    return parsed_url.scheme in {'http', 'https'} and parsed_url.netloc == parsed_base_url.netloc

def get_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []
    for a_tag in soup.find_all('a', href=True):
        link = a_tag['href']
        if '#' not in link:  # Ignore links with fragments
            links.append(link)
    return links

def crawl(start_url, limit=100):
    queue = deque([(start_url, 0)])  # Use tuple to store (url, depth)
    visited = set()
    base_url = urlparse(start_url).netloc
    count = 0

    while queue and count < limit:
        url, depth = queue.popleft()
        if url in visited:
            continue

        if not is_allowed(url):
            print("Not allowed to crawl:", url)
            continue

        try:
            response = requests.get(url)
            if response.status_code == 200:
                visited.add(url)
                print("Crawling:", url)
                count += 1

                links = get_links(url)
                for link in links:
                    full_url = urljoin(url, link)
                    print("Found URL:", full_url)
                    if is_valid(full_url, base_url) and full_url not in visited and depth < 3:  # Limit depth to 3
                        queue.append((full_url, depth + 1))
        except Exception as e:
            print("Error crawling:", url, e)

# Sample usage:
crawl("https://en.wikipedia.org/wiki/Web_scraping", limit=100)