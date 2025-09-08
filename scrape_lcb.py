import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time

BASE_URL = "https://lcbfertilizers.com"
SCRAPED_DIR = "scraped_data"
os.makedirs(SCRAPED_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
}

CRAWL_DEPTH_LIMIT = 2   # Change as needed for deeper crawl
REQUEST_DELAY = 1.5     # Seconds between requests
TIMEOUT = 15            # Seconds for each request

def is_valid_link(link):
    parsed = urlparse(link)
    if parsed.scheme in ["http", "https"]:
        return link.startswith(BASE_URL)
    if parsed.scheme == "":
        return not link.startswith("#") and not link.startswith("mailto:")
    return False

def extract_visible_text(soup):
    # Exclude scripts, styles, and navs
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    clean_lines = [line for line in lines if line]
    return "\n".join(clean_lines)

def save_text(text, url):
    safe_name = url.replace(BASE_URL, "").strip("/")
    if not safe_name: safe_name = "homepage"
    safe_name = safe_name.replace("/", "_")
    file_path = os.path.join(SCRAPED_DIR, f"{safe_name}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def scrape_page(url, depth, visited):
    if url in visited or depth > CRAWL_DEPTH_LIMIT:
        return
    print(f"Scraping: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if response.status_code != 200:
            print(f"Failed ({response.status_code}): {url}")
            return
        soup = BeautifulSoup(response.text, "html.parser")
        text = extract_visible_text(soup)
        if len(text.strip()) > 40:  # avoid empty/dud pages
            save_text(text, url)
        visited.add(url)
        # Extract all internal links
        for a_tag in soup.find_all("a", href=True):
            next_url = urljoin(BASE_URL, a_tag["href"])
            if is_valid_link(next_url) and urlparse(next_url).netloc == urlparse(BASE_URL).netloc:
                scrape_page(next_url.split("#"), depth + 1, visited)
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        print(f"Error scraping {url}: {e}")

def main():
    visited = set()
    scrape_page(BASE_URL, depth=0, visited=visited)
    print("\nScraping done. Files saved in ./scraped_data/\nPages scraped:", len(visited))

if __name__ == "__main__":
    main()
