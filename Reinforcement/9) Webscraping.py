import requests
from bs4 import BeautifulSoup        #BeautifulSoup reads and understands the structure of HTML.

url = "https://en.wikipedia.org/wiki/Raigad_district"

headers = {
    "User-Agent": "KiranPawalPythonApp/1.0 (vaishalipawal45@gmail.com)"                #A User-Agent tells the website information about the program making the request.
}

response = requests.get(                #This is the most important line.        It sends a GET request to Wikipedia.
    url,
    headers=headers
)

print("Status:", response.status_code)          #The server sends an HTTP status code.

#if responce successful
if response.ok:
    soup = BeautifulSoup(response.text, "html.parser")  #response.txt contains the HTML returned by Wikipedia.        #This tells BeautifulSoup:Treat this response as HTML and parse its structure.
    print("Title:", soup.title.get_text(strip=True))            #Extract the webpage title

    print("\nParagraphs:\n")
    paragraphs = soup.find_all("p")              #This is where actual HTML data extraction starts.

    for paragraph in paragraphs:                     
        text = paragraph.get_text(" ", strip=True)          #The " " means that if there are multiple pieces of text inside the paragraphs

        if text:
            print(text)
            print()
else:
    print("Request failed:")
    print(response.text)
