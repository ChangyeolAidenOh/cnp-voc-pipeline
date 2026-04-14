import ast
import pandas as pd
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import warnings
warnings.filterwarnings("ignore")

# CNP-related keywords for relevance filtering
# documents not containing any of these are excluded
CNP_KEYWORDS = {
    "차앤박", "CNP", "프로폴리스", "앰플", "PDRN",
    "안티포어", "트러블", "카밍", "클렌징", "세럼",
    "더마", "피부과"
}

PROCESSED_DIR = "voc_pipeline/data/processed"


def is_relevant(text):
    """
    Check if a document contains at least one CNP-related keyword.
    Args:
        text: raw document string
    Returns:
        bool
    """
    if not isinstance(text, str):
        return False
    return any(kw in text for kw in CNP_KEYWORDS)


def parse_list(val):
    """Parse a stringified Python list back to a list object."""
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return []


def build_lda(texts, num_topics, passes=15):
    """
    Build an LDA model on a given corpus.
    Args:
        texts: list of token lists
        num_topics: number of topics
        passes: number of training passes
    Returns:
        tuple of (model, corpus, dictionary)
    """
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=3, no_above=0.85)
    corpus = [dictionary.doc2bow(t) for t in texts]
    model = models.LdaModel(
        corpus=corpus, id2word=dictionary,
        num_topics=num_topics, passes=passes, random_state=42
    )
    return model, corpus, dictionary


def find_best_k(texts, k_range=range(2, 8)):
    """
    Find the optimal number of topics by maximizing c_v coherence score.
    Args:
        texts: list of token lists
        k_range: range of topic counts to evaluate
    Returns:
        tuple of (best_model, best_k, best_score)
    """

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=3, no_above=0.85)
    corpus = [dictionary.doc2bow(t) for t in texts]
    best_k, best_score, best_model = 2, -1, None
    for k in k_range:
        model = models.LdaModel(
            corpus=corpus, id2word=dictionary,
            num_topics=k, passes=15, random_state=42
        )
        score = CoherenceModel(
            model=model, texts=texts,
            dictionary=dictionary, coherence="c_v"
        ).get_coherence()
        print(f"    k={k}: {score:.4f}")
        if score > best_score:
            best_score, best_k, best_model = score, k, model
    return best_model, best_k, best_score


def run_analysis(df, mode_col, label):
    """
    Run LDA topic modeling for a specific preprocessing mode.
    Args:
        df: filtered DataFrame
        mode_col: column name for token lists (e.g. 'bigram')
        label: display label for logging
    Returns:
        tuple of (model, best_k, coherence_score)
    """
    texts = df[mode_col].apply(parse_list).tolist()
    texts = [t for t in texts if len(t) >= 2]
    if len(texts) < 30:
        print(f"  [{label}] Insufficient data {len(texts)} docs")
        return None, None, None
    print(f"  [{label}] Analyzing {len(texts)} ")
    model, k, score = find_best_k(texts)
    return model, k, score


# Entry
if __name__ == "__main__":
    df = pd.read_csv(f"{PROCESSED_DIR}/cnp_processed.csv")
    print(f"Total documents: {len(df)}")

    # filter to CNP-relevant documents only
    df_filtered = df[df["raw_text"].apply(is_relevant)].copy()
    print(f"After CNP relevance filter: {len(df_filtered)} \n")

    results = []
    modes = ["unigram", "bigram", "unibi_mix", "adj_noun"]

    # full corpus for 4 modes
    print(" Full Corpus ")
    for mode in modes:
        model, k, score = run_analysis(df_filtered, mode, mode)
        if model:
            print(f"  best k={k}, coherence={score:.4f}")
            print(f"  Topic keywords:")
            for idx, topic in model.print_topics(num_words=8):
                kws = topic
                print(f"    Topic {idx}: {kws}")
                results.append({
                    "scope": "all",
                    "source": "all",
                    "mode": mode,
                    "topic_id": idx,
                    "optimal_k": k,
                    "coherence": round(score, 4),
                    "keywords": kws
                })
            print()

    # per-source for 4 modes
    for source in ["blog", "cafearticle", "youtube"]:
        sub = df_filtered[df_filtered["source"] == source]
        print(f"\n [{source}] {len(sub)} ")
        for mode in modes:
            model, k, score = run_analysis(sub, mode, mode)
            if model:
                print(f"  best k={k}, coherence={score:.4f}")
                for idx, topic in model.print_topics(num_words=8):
                    results.append({
                        "scope": source,
                        "source": source,
                        "mode": mode,
                        "topic_id": idx,
                        "optimal_k": k,
                        "coherence": round(score, 4),
                        "keywords": topic
                    })

    # save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{PROCESSED_DIR}/cnp_lda_results.csv",
                      index=False, encoding="utf-8-sig")

    # coherence summary
    print("\n Coherence Summary ")
    summary = results_df.groupby(["scope", "mode"])["coherence"].first().unstack()
    print(summary.to_string())
    print(f"\n {PROCESSED_DIR}/cnp_lda_results.csv")