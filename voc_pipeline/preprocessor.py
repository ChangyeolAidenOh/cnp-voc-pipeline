import re
import ast
import pandas as pd
from kiwipiepy import Kiwi

kiwi = Kiwi()

# Set stop wards
# Korean particles, conjunctions, and domain-specific noise terms
STOPWORDS = {
    # Korean particles and conjunctions
    "이", "가", "을", "를", "은", "는", "에", "의", "도", "으로", "로", "와", "과",
    "하다", "있다", "되다", "않다", "없다", "같다", "보다", "그", "이것", "저것",
    "그것", "여기", "저기", "거기", "때", "더", "또", "및", "등", "즉", "그리고",
    "하지만", "그런데", "그래서", "그러나", "근데",

    # Exclamations and reaction words (YouTube noise)
    "좀", "잘", "너무", "진짜", "정말", "완전", "약간", "조금", "많이", "빨리",
    "항상", "매일", "계속", "대박", "레전드", "꿀팁", "ㅠㅠ", "ㅋㅋ", "헐",

    # YouTube channel noise
    "감사", "구독", "영상", "언니", "좋아요", "알림", "시청", "채널",
    "방송", "유튜브", "shorts", "정보", "도움", "방법", "설명", "이름",
    "목소리", "싸가지", "재수", "중독", "팩트", "브랜드",

    # Shopping platform noise
    "코스트코", "다이소", "쿠팡", "컬러그램", "최저", "최대",
    "적립", "연락처", "회원", "연락", "상품", "옵션", "기한",

    # Dermatology context noise (confounds CNP brand positioning)
    "병원", "시술", "레이저", "의사", "서울", "성북구", "치료", "환자",
    "리프팅", "피부과",

    # General beauty stopwords
    "화장품", "스킨케어", "뷰티", "추천", "광고", "협찬", "내돈내산",
    "할인", "세일", "이벤트", "기획", "세트", "알바",
}


# Text cleaning step
def clean_text(text):
    """
    Remove HTML tags, URLs, and special characters from raw text.
    Args:
        text: raw input string
    Returns:
        cleaned string
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Morphological analysis
def extract_tokens(text):
    """
    Extract nouns and adjective-noun pairs using kiwipiepy.
    Args:
        text: cleaned Korean text
    Returns:
        dict with keys:
            'nouns': list of noun tokens
            'adj_noun_pairs': list of (adjective, noun) tuples
    """
    if not text or len(text) < 2:
        return {"nouns": [], "adj_noun_pairs": []}
    try:
        result = kiwi.analyze(text)
        tokens = result[0][0]
        nouns, pairs = [], []
        for i, token in enumerate(tokens):
            # extract nouns: minimum 2 characters, exclude stopwords
            if token.tag in ("NNG", "NNP") and len(token.form) >= 2:
                if token.form not in STOPWORDS:
                    nouns.append(token.form)
            # extract adjective-noun pairs for sentiment-bearing phrases
            if token.tag in ("VA", "XR") and i + 1 < len(tokens):
                next_token = tokens[i + 1]
                if next_token.tag in ("NNG", "NNP") and len(next_token.form) >= 2:
                    if next_token.form not in STOPWORDS:
                        pairs.append((token.form, next_token.form))
        return {"nouns": nouns, "adj_noun_pairs": pairs}
    except:
        return {"nouns": [], "adj_noun_pairs": []}


# Preprocessing step
def make_unigram(nouns):
    """Single noun tokens filtered by stopwords."""
    return [n for n in nouns if n not in STOPWORDS]


def make_bigram(nouns):
    """Consecutive noun pairs (e.g. 차앤박_앰플)."""
    bigrams = []
    for i in range(len(nouns) - 1):
        bigrams.append(f"{nouns[i]}_{nouns[i+1]}")
    return bigrams


def make_unibi_mix(nouns):
    """Union of unigram and bigram tokens."""
    return make_unigram(nouns) + make_bigram(nouns)


def make_adj_noun(pairs):
    """Adjective-noun compound phrases (e.g. 순한_클렌저)."""
    return [f"{adj}_{noun}" for adj, noun in pairs]


# Preprocessing per-sauce
def process_row(row, text_cols):
    """
    Process a single row by combining specified text columns
    and applying all preprocessing modes.
    Args:
        row: DataFrame row
        text_cols: list of column names to combine as input text
    Returns:
        dict with raw_text and four token mode lists
    """
    combined = " ".join([str(row.get(col, "")) for col in text_cols])
    cleaned = clean_text(combined)
    tokens = extract_tokens(cleaned)
    nouns = tokens["nouns"]
    pairs = tokens["adj_noun_pairs"]
    return {
        "raw_text": cleaned,
        "unigram": nouns,
        "bigram": make_bigram(nouns),
        "unibi_mix": make_unibi_mix(nouns),
        "adj_noun": make_adj_noun(pairs),
    }


def preprocess_naver(df, source_name):
    """
    Preprocess Naver blog or cafe posts.
    Combines title and description columns as input text.
    """
    records = []
    for _, row in df.iterrows():
        processed = process_row(row, ["title", "description"])
        records.append({
            "source": source_name,
            "date": str(row.get("postdate", "")),
            "query": row.get("query", ""),
            **processed
        })
    return pd.DataFrame(records)


def preprocess_youtube(df):
    """
    Preprocess YouTube comments.
    Uses comment text as input.
    """
    records = []
    for _, row in df.iterrows():
        processed = process_row(row, ["comment"])
        records.append({
            "source": "youtube",
            "date": str(row.get("published_at", ""))[:10],
            "query": row.get("query", ""),
            "video_title": row.get("video_title", ""),
            "likes": row.get("likes", 0),
            **processed
        })
    return pd.DataFrame(records)


# Entry work
if __name__ == "__main__":
    RAW_DIR = "voc_pipeline/data/raw"
    PROCESSED_DIR = "voc_pipeline/data/processed"

    print("Preprocessing Naver blog posts...")
    blog_df = pd.read_csv(f"{RAW_DIR}/naver_blog_cnp.csv")
    blog_processed = preprocess_naver(blog_df, "blog")

    print("Preprocessing Naver cafe posts...")
    cafe_df = pd.read_csv(f"{RAW_DIR}/naver_cafe_cnp.csv")
    cafe_processed = preprocess_naver(cafe_df, "cafearticle")

    print("Preprocessing YouTube comments...")
    yt_df = pd.read_csv(f"{RAW_DIR}/youtube_comments_cnp.csv")
    yt_processed = preprocess_youtube(yt_df)

    all_df = pd.concat([blog_processed, cafe_processed, yt_processed], ignore_index=True)

    # save processed output
    all_df.to_csv(f"{PROCESSED_DIR}/cnp_processed.csv", index=False, encoding="utf-8-sig")
    print(f"\nPreprocessing complete: {len(all_df)} records")
    print(f"Saved: {PROCESSED_DIR}/cnp_processed.csv")

    # sample output for verification
    sample = all_df[
        all_df["unigram"].apply(
            lambda x: len(ast.literal_eval(str(x)) if isinstance(x, str) else x) > 3
        )
    ].iloc[0]
    print("\n# Sample token #")
    print(f"Raw text: {sample['raw_text'][:60]}")
    for mode in ["unigram", "bigram", "unibi_mix", "adj_noun"]:
        val = sample[mode]
        if isinstance(val, str):
            val = ast.literal_eval(val)
        print(f"[{mode}]: {val[:6]}")