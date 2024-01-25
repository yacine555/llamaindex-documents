import requests
from bs4 import BeautifulSoup
import os
import urllib


def download_site(url:str,output_dir:str,html_filter:str=None):
    print(f"Downlad site {url} in {output_dir}")
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Fetch the page
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    # Find all links to .html files
    links = soup.find_all("a", href=True)
    print(f"{len(links)} links found")

    #create list of pages to avoid duplicate
    page_download= []

    for link in links:
        href = link["href"]
        href_abs = ""
        href_rel = ""
        print(f"link to download {href}")

        # If it's a .html file
        if href.endswith(".html") or href.endswith("/"):
            page_name = href
            if href.endswith("/"):
                page_name = href + "index.html"

            # Make a full URL if necessary
            if not href.startswith("http"):
                href = urllib.parse.urljoin(url, href)

            if href not in page_download:
                page_download.append(href)

                # Fetch the .html file
                print(f"downloading {href} into {page_name}...")
                response = requests.get(href)
                response_text = ""
                

                if html_filter is not None:
                    soup = BeautifulSoup(response.text, "html.parser")
                    list_items = soup.select(html_filter)
                    for item in list_items:
                        response_text = response_text + str(item)
                else:
                    response_text = response.text

                # print(f"response code {response_text}")

                # Write it to a file
                
                file_name = os.path.join(output_dir, os.path.basename(page_name))
                directory = output_dir + href.replace(url, "").replace("https://","/")
                file_name = output_dir + page_name.replace("https://","/")
                
                if not os.path.exists(directory):
                    print(f"create directory {directory} ")
                    os.makedirs(directory)

                print(f"save file into {file_name}")
                with open(file_name, "w", encoding="utf-8") as file:
                    file.write(response_text)

            else:
                print(f"Page {href} already downloaded.")


if __name__ == "__main__":

    # download_site("https://docs.llamaindex.ai/en/stable/","./docs/llamindex-docs/")
    download_site("https://mistral.ai","./docs/mistral_ai/")
