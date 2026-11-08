import urllib.request
import re

url = "https://astroweb.case.edu/SPARC/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    links = re.findall(r'href=["\']([^"\']+)["\']', html)
    data_links = [
        link for link in links
        if any(link.endswith(ext) for ext in [".zip", ".gz", ".mrt", ".txt", ".tar", ".csv"])
    ]
    print("Found links:", data_links)
except Exception as e:
    print("Scraping error:", e)
