import ast
import os
import sys
import pandas as pd
from collections import Counter
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import warnings
warnings.filterwarnings("ignore")

# allow imports from sibling modules
sys.path.append(os.path.dirname(__file__))
from preprocessor import clean_text, extract_tokens

PROCESSED_DIR = "voc_pipeline/data/processed"


def parse_list(val):
    """Parse a stringified Python list back to a list object."""
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return str(val).split() if isinstance(val, str) else []


def preprocess_anua_translated(df):
    """
    Apply Korean preprocessing pipeline to translated ANUA reviews.
    Reuses the same clean_text and extract_tokens functions as the
    CNP pipeline to ensure comparable token representations.
    Args:
        df: DataFrame with 'text_ko' column (translated review text)
    Returns:
        preprocessed DataFrame with unigram and bigram columns
    """
    records = []
    for _, row in df.iterrows():
        text = str(row.get("text_ko", ""))
        cleaned = clean_text(text)
        tokens = extract_tokens(cleaned)
        nouns = tokens["nouns"]

        # generate consecutive noun bigrams
        bigrams = [f"{nouns[i]}_{nouns[i+1]}" for i in range(len(nouns) - 1)]

        records.append({
            "source": "anua_translated",
            "product": row.get("variant", ""),
            "star_rating": row.get("Star-rating", ""),
            "date": str(row.get("Date", ""))[:10],
            "raw_text": cleaned,
            "unigram": nouns,
            "bigram": bigrams,
        })
    return pd.DataFrame(records)


def get_top_keywords(texts, top_n=100):
    """
    Compute top-n keyword frequencies across a list of token lists.
    Args:
        texts: list of token lists
        top_n: number of top keywords to return
    Returns:
        dict of {keyword: count}
    """
    counter = Counter()
    for text in texts:
        counter.update(text)
    return dict(counter.most_common(top_n))


def keyword_divergence(cnp_freq, anua_freq):
    """
    Compute brand-specific keyword divergence between CNP and ANUA.
    Bias score: (cnp_count - anua_count) / total
        > 0.3  → CNP-specific
        < -0.3 → ANUA-specific
        else   → shared

    Args:
        cnp_freq: keyword frequency dict for CNP
        anua_freq: keyword frequency dict for ANUA
    Returns:
        DataFrame sorted by bias score (descending)
    """
    all_keys = set(cnp_freq.keys()) | set(anua_freq.keys())
    rows = []
    for kw in all_keys:
        c = cnp_freq.get(kw, 0)
        a = anua_freq.get(kw, 0)
        total = c + a
        if total < 5:
            continue
        bias = (c - a) / total
        rows.append({
            "keyword": kw,
            "cnp_count": c,
            "anua_count": a,
            "bias": round(bias, 3),
            "brand": "CNP 특화" if bias > 0.3
                     else "ANUA 특화" if bias < -0.3
                     else "공통"
        })
    return pd.DataFrame(rows).sort_values("bias", ascending=False)


def run_lda(texts, label, num_topics=5):
    """
    Run LDA topic modeling and print topic keywords with coherence score.
    Args:
        texts: list of token lists
        label: display label for logging
        num_topics: number of topics
    Returns:
        trained LdaModel or None if data is insufficient
    """
    texts = [t for t in texts if len(t) >= 2]
    if len(texts) < 20:
        print(f"  {label} Insufficient data ")
        return None
    d = corpora.Dictionary(texts)
    d.filter_extremes(no_below=2, no_above=0.85)
    c = [d.doc2bow(t) for t in texts]
    model = models.LdaModel(
        corpus=c, id2word=d,
        num_topics=num_topics, passes=15, random_state=42
    )
    score = CoherenceModel(
        model=model, texts=texts,
        dictionary=d, coherence="c_v"
    ).get_coherence()
    print(f"\n  {label} coherence={score:.4f}")
    for idx, topic in model.print_topics(num_words=8):
        print(f"    Topic {idx}: {topic}")
    return model


# Entry step
if __name__ == "__main__":
    # load data
    cnp_df = pd.read_csv(f"{PROCESSED_DIR}/cnp_causal_signals.csv")
    anua_raw = pd.read_csv(f"{PROCESSED_DIR}/anua_translated.csv")
    print(f"CNP: {len(cnp_df)} ")
    print(f"ANUA translated: {len(anua_raw)} \n")

    # preprocess translated ANUA reviews
    anua_df = preprocess_anua_translated(anua_raw)
    anua_df.to_csv(f"{PROCESSED_DIR}/anua_processed_ko.csv",
                   index=False, encoding="utf-8-sig")
    print(f" {len(anua_df)} records\n")

    # keyword frequency comparison
    print(" Keyword Frequency Comparison ")
    cnp_texts = cnp_df["unigram"].apply(parse_list).tolist()
    anua_texts = anua_df["unigram"].tolist()

    cnp_freq = get_top_keywords(cnp_texts, top_n=150)
    anua_freq = get_top_keywords(anua_texts, top_n=150)

    print("\nCNP Top 15:")
    for kw, cnt in list(cnp_freq.items())[:15]:
        print(f"  {kw}: {cnt}")

    print("\nANUA Top 15:")
    for kw, cnt in list(anua_freq.items())[:15]:
        print(f"  {kw}: {cnt}")

    # keyword divergence analysis
    print("\n Brand-specific Keyword Divergence ")
    div_df = keyword_divergence(cnp_freq, anua_freq)

    print("\n CNP-specific keywords (bias > 0.3) ")
    cnp_spec = div_df[div_df["brand"] == "CNP specialization"].head(15)
    print(cnp_spec[["keyword", "cnp_count", "anua_count", "bias"]].to_string(index=False))

    print("\n ANUA-specific keywords (bias < -0.3) ")
    anua_spec = div_df[div_df["brand"] == "ANUA specialization"].head(15)
    print(anua_spec[["keyword", "cnp_count", "anua_count", "bias"]].to_string(index=False))

    print("\n Shared keywords ")
    common = div_df[div_df["brand"] == "Common"].head(15)
    print(common[["keyword", "cnp_count", "anua_count", "bias"]].to_string(index=False))

    # LDA topic comparison (bigram mode)
    print("\n LDA Topic Comparison (bigram) ")
    cnp_bigram = cnp_df["bigram"].apply(parse_list).tolist()
    anua_bigram = anua_df["bigram"].tolist()

    print("\n CNP Topics ")
    run_lda(cnp_bigram, "CNP")

    print("\n ANUA Topics ")
    run_lda(anua_bigram, "ANUA")

    # ANUA keyword analysis by star rating
    print("\n ANUA Keywords by Star Rating ")
    for star in [1, 2, 4, 5]:
        sub = anua_df[anua_df["star_rating"].astype(str).str.startswith(str(star))]
        if len(sub) < 10:
            continue
        texts = sub["unigram"].tolist()
        freq = get_top_keywords(texts, top_n=10)
        print(f"\n  {star} ({len(sub)} reviews): "
              f"{', '.join(list(freq.keys())[:8])}")

    # save divergence results
    div_df.to_csv(f"{PROCESSED_DIR}/stream_b_keyword_divergence.csv",
                  index=False, encoding="utf-8-sig")
    print(f"\n {PROCESSED_DIR}/stream_b_keyword_divergence.csv")