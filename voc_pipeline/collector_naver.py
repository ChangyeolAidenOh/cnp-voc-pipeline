import requests
import pandas as pd
import os
import time
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

OUTPUT_DIR = "voc_pipeline/data/raw"


def search_naver(query, source="blog", display=100, start=1):
    """
    Query Naver Search API for blog or cafe posts.

    Args:
        query: search keyword
        source: 'blog' or 'cafearticle'
        display: number of results per request (max 100)
        start: start index for pagination
    """
    url = f"https://openapi.naver.com/v1/search/{source}"
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET
    }
    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": "date"
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


def collect_naver_data(queries, source="blog"):
    """
    Collect posts from Naver blog or cafe for a list of queries.

    Args:
        queries: list of search keywords
        source: 'blog' or 'cafearticle'

    Returns:
        DataFrame of collected posts
    """
    all_items = []
    for query in queries:
        print(f"Collecting [{source}]: {query}")
        result = search_naver(query, source=source, display=100)
        if result and "items" in result:
            for item in result["items"]:
                item["query"] = query
                item["source"] = source
                all_items.append(item)
        time.sleep(0.5)  # rate limiting between API calls
    return pd.DataFrame(all_items)


if __name__ == "__main__":
    # CNP-related search queries
    cnp_queries = [
        # Product-specific queries
        "차앤박 프로폴리스 앰플",
        "CNP PDRN 앰플",
        "차앤박 안티포어",
        "CNP 트러블 카밍",
        "차앤박 클렌징 폼",

        # Churn / dissatisfaction signals
        "차앤박 별로",
        "차앤박 안맞는",
        "차앤박 트러블",
        "차앤박 효과없음",

        # Competitive comparison queries
        "더마코스메틱 추천",
        "PDRN 앰플 비교",
        "차앤박 대신",
        "CNP 아누아 비교",
    ]

    # Collect blog posts
    blog_df = collect_naver_data(cnp_queries, source="blog")
    print(f"blog {len(blog_df)} records")

    # Collect cafe posts
    cafe_df = collect_naver_data(cnp_queries, source="cafearticle")
    print(f"cafe {len(cafe_df)} records")

    # Save to raw data directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    blog_df.to_csv(f"{OUTPUT_DIR}/naver_blog_cnp.csv", index=False, encoding="utf-8-sig")
    cafe_df.to_csv(f"{OUTPUT_DIR}/naver_cafe_cnp.csv", index=False, encoding="utf-8-sig")
    print(f"{OUTPUT_DIR}/naver_blog_cnp.csv")
    print(f"{OUTPUT_DIR}/naver_cafe_cnp.csv")