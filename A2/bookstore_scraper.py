"""
Assignment 2 - The Bookstore Intelligence Mission
Scrapes https://books.toscrape.com/ (all catalogue pages), builds a clean
DataFrame, saves competitor_books.csv, answers the 7 analysis questions,
solves the CEO GBP 100 challenge, and draws the rating-count bar chart.

Run:  pip install requests beautifulsoup4 pandas matplotlib numpy
      python bookstore_scraper.py            # all pages (BOSS LEVEL)
      python bookstore_scraper.py --pages 3  # only first 3 pages
"""
import argparse
import re
import time
from urllib.parse import urljoin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/catalogue/page-1.html"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
BUDGET_GBP = 100


# ---------------------------------------------------------------- Mission 1/2
def parse_book(card, page_url):
    """Extract one record from an <article class="product_pod"> element."""
    link = card.h3.a
    title = link["title"]                                    # full title (text is truncated)
    price_text = card.find("p", class_="price_color").text   # e.g. "£51.77"
    price = float(re.search(r"\d+\.\d+", price_text).group())
    rating_word = card.find("p", class_="star-rating")["class"][1]  # ["star-rating", "Three"]
    rating = RATING_MAP[rating_word]
    availability = card.find("p", class_="instock availability").text.strip()
    return {
        "Title": title,
        "Price": price,
        "Rating": rating,
        "Availability": availability,
        "Product_URL": urljoin(page_url, link["href"]),      # BONUS 2
    }


def parse_page(html, page_url):
    """Return (records, next_page_url or None) for one catalogue page."""
    soup = BeautifulSoup(html, "html.parser")
    records = [parse_book(c, page_url) for c in soup.select("article.product_pod")]
    nxt = soup.select_one("li.next > a")
    return records, (urljoin(page_url, nxt["href"]) if nxt else None)


# -------------------------------------------------------------------- Mission 3
def fetch(url, session):
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    resp.encoding = "utf-8"          # avoids the "Â£" encoding glitch
    return resp.text


def scrape(max_pages=None, delay=0.3, fetch_fn=fetch):
    """One reusable loop: follows the 'next' button until the end
    (or until max_pages). Same code for 3 pages, 5 pages or all 50."""
    session = requests.Session()
    url, all_records, page_no = BASE_URL, [], 0
    while url and (max_pages is None or page_no < max_pages):
        html = fetch_fn(url, session)
        records, url = parse_page(html, url)
        all_records.extend(records)
        page_no += 1
        print(f"Page {page_no}: {len(records)} books (total {len(all_records)})")
        time.sleep(delay)            # be polite to the server
    return pd.DataFrame(all_records)


# -------------------------------------------------------------------- Mission 4
def ceo_challenge(df, budget=BUDGET_GBP):
    """Exact 0/1 knapsack: maximise total stars within budget.
    Tie-break: lowest total cost. Prices handled in pence (integers)."""
    cap = int(round(budget * 100))
    prices = (df["Price"] * 100).round().astype(int).to_numpy()
    values = df["Rating"].to_numpy() * 1_000_000 - prices   # stars first, then cheaper
    n = len(df)
    dp = np.zeros(cap + 1, dtype=np.int64)
    keep = np.zeros((n, cap + 1), dtype=bool)
    for i in range(n):
        p, v = prices[i], values[i]
        if p > cap:
            continue
        cand = dp[: cap + 1 - p] + v
        better = cand > dp[p:]
        keep[i, p:] = better
        dp[p:] = np.where(better, cand, dp[p:])
    chosen, c = [], cap
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            chosen.append(i)
            c -= prices[i]
    return df.iloc[sorted(chosen)]


def analyse(df):
    out = []
    in_stock = df["Availability"].str.contains("In stock", case=False)
    p_max, p_min = df.loc[df["Price"].idxmax()], df.loc[df["Price"].idxmin()]
    five = df[df["Rating"] == 5]
    top5 = df.sort_values(["Rating", "Price"], ascending=[False, True]).head(5)
    cheap5 = five.loc[five["Price"].idxmin()] if len(five) else None

    out.append("## Intelligence Analysis\n")
    out.append(f"Books scraped: {len(df)}\n")
    out.append(f"1. Most expensive: {p_max.Title} - £{p_max.Price:.2f}")
    out.append(f"2. Cheapest: {p_min.Title} - £{p_min.Price:.2f}")
    out.append(f"3. Average price: £{df.Price.mean():.2f}")
    out.append(f"4. 5-star books: {len(five)}")
    out.append(f"5. In stock: {in_stock.mean() * 100:.1f}%")
    out.append("6. Top 5 highest-rated (all 5-star; ties broken by lowest price):")
    for _, r in top5.iterrows():
        out.append(f"   - {r.Title} ({r.Rating}★, £{r.Price:.2f})")
    if cheap5 is not None:
        out.append(f"7. Cheapest 5-star book: {cheap5.Title} - £{cheap5.Price:.2f}")

    counts = df["Rating"].value_counts().sort_index()          # BONUS 3
    out.append("\n## Books per rating level\n" + counts.to_string())
    avg_by_rating = df.groupby("Rating")["Price"].mean().round(2)
    out.append("\n## Average price by rating\n" + avg_by_rating.to_string())

    sel = ceo_challenge(df)                                     # FINAL CHALLENGE
    out.append(f"\n## CEO £{BUDGET_GBP} Challenge")
    for _, r in sel.iterrows():
        out.append(f"   - {r.Title} | £{r.Price:.2f} | {r.Rating}★")
    out.append(f"Books selected: {len(sel)}")
    out.append(f"Total cost: £{sel.Price.sum():.2f}")
    out.append(f"Total rating score: {sel.Rating.sum()}")

    ax = counts.plot(kind="bar", color="steelblue", edgecolor="black")  # BONUS 4
    ax.set_title("Number of Books by Star Rating")
    ax.set_xlabel("Star rating"); ax.set_ylabel("Number of books")
    plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig("rating_counts.png", dpi=150); plt.close()
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", type=int, default=None,
                    help="number of pages (default: all available pages)")
    args = ap.parse_args()

    df = scrape(max_pages=args.pages)
    df = df.drop_duplicates(subset="Product_URL").reset_index(drop=True)
    df.to_csv("competitor_books.csv", index=False, encoding="utf-8")
    print("\n", df.head(), "\n")
    print(df.dtypes, "\n")
    report = analyse(df)
    print(report)
    with open("analysis_answers.md", "w", encoding="utf-8") as f:
        f.write(report)
    print


if __name__ == "__main__":
    main()
