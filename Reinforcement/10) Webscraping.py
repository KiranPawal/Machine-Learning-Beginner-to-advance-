import requests
from bs4 import BeautifulSoup
import pandas as pd
import time 

def scrape_books(pages=5):
    all_books = []
    base_url = "http://books.toscrape.com/catalogue/page-{}.html"

    for page in range(1, pages+1):
        url = base_url.format(page)
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        books = soup.find_all("article", class_ ="product_pod")

        for book in books:
            title = book.h3.a["title"]
            price = book.find("p", class_="price_color").text.strip()
            availability = book.find("p", class_="instock availability").text.strip()
            rating = book.p["class"][1]

            all_books.append({
                "title":title,
                "price":price,
                "availability":availability,
                "rating":rating,
            })

        print(f"Scraped page {page}")
        time.sleep(1)

    return pd.DataFrame(all_books)

df = scrape_books(pages=5)
df.to_csv("books_data.csv",index=False)
print(df)