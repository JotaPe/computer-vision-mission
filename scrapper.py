import shutil
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm
urllib3.disable_warnings()

"""
PoolManager with disabled certs
Not ideal, but the site for my connection was not having updated cert
"""
HTTP = urllib3.PoolManager(cert_reqs="CERT_NONE")


# Some example URLs
# Not ideal too, if we change site we're going to have problems
URL_TAM = "https://www.airliners.net/search?airline=54645&photoCategory=23&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail"
URL_UNITED_AIRLINE = "https://www.airliners.net/search?airline=58361&photoCategory=23&sortBy=dateAccepted&sortOrder=desc&perPage=84&display=detail"
URL_DELTA = "https://www.airliners.net/search?airline=18647&photoCategory=23&sortBy=dateAccepted&sortOrder=desc&perPage=84&display=detail"
URL_UNTITLED = "https://www.airliners.net/search?airline=58641&photoCategory=23&sortBy=dateAccepted&sortOrder=desc&perPage=36&display=detail"


def airliner_scrapper(url, company_name, folder):
    """
    Receiver a URL(string) and a Airliner Name(string)
    the airliner name is used to be the name in the start of the file
    for example airliner_name = "tam" the output name will be tamX.jpeg
    X being a counter variable
    """
    print("Starting scrapper")
    page = HTTP.request("GET", url)
    soup = BeautifulSoup(page.data, "html.parser")
    images = soup.findAll("img", {"class": "lazy-load"})
    i = 0
    pbar = tqdm(total=len(images))
    for image in images:
        i += 1
        image_url = image['src']
        pbar.update(1)
        file_path = f"{folder}/{company_name}{i}.jpeg"
        with HTTP.request("GET", image_url, preload_content=False) as res:
            with open(file_path, "wb") as out_file:
                shutil.copyfileobj(res, out_file)
    pbar.close()
