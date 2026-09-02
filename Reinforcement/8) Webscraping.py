
'''     response = requests.get(url, timeout=10)
                    Python Program
                        |
                        | Request webpage
                        ↓
                    Website
                        |
                        | Sends HTML
                        ↓
                    Python Program
'''


import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_books(pages=5):                              # This means by default it will scrape 5 pages.
    all_books = []
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"                  # This asks the website: "Please send me the content of this webpage."

    for page in range(1, pages + 1):
        url = base_url.format(page)                  # Create the URL for Each Page
        response = requests.get(url, timeout=10)         # This sends an HTTP GET request.   Wait for a maximum of 10 seconds.


        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, "html.parser")         #response.text: contains the HTML of the webpage.
        books = soup.find_all("article", class_="product_pod")      # Find All Books: Each book on the website is inside an HTML structure similar to:

        for book in books:
            title = book.h3.a["title"]                             # Extract the Book Title
            price = book.find("p", class_="price_color").text.strip()
            availability = book.find("p", class_="instock availability").text.strip()
            rating = book.p["class"][1]  # e.g. "Three"

            all_books.append({                      # Store Book Data: This creates a Python dictionary:
                "title": title,
                "price": price,
                "availability": availability,
                "rating": rating
            })

        print(f"Scraped page {page}")
        time.sleep(1)  # be polite — don't hammer the server

    return pd.DataFrame(all_books)


df = scrape_books(pages=5)
df.to_csv("books_data.csv", index=False)
print(df.head())