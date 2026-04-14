import ast
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

## Causal signal keyword dictionaries
# Churn signals: keywords indicating dissatisfaction or brand switching
CHURN_KEYWORDS = {
    "direct_churn": ["환불", "반품", "후회", "실망", "별로", "최악", "불만", "환불요청"],
    "efficacy_failure": ["효과없음", "효과없어", "효과모르겠", "변화없", "그대로", "소용없"],
    "skin_reaction": ["트러블", "여드름", "뾰루지", "올라왔", "올라오", "자극", "빨개", "따가"],
    "competitor_switch": ["아누아", "anua", "다른거", "다른브랜드", "갈아탔", "갈아탈", "대신"],
    "formula_change": ["예전이", "예전엔", "바뀐것", "바뀐거", "달라진", "리뉴얼후", "구버전"],
}

# Positive signals: keywords indicating satisfaction and repurchase intent
POSITIVE_KEYWORDS = {
    "repurchase": ["재구매", "또샀", "계속쓸", "인생템", "정착"],
    "efficacy_positive": ["좋아졌", "나아졌", "효과있", "효과봤", "차이나"],
    "recommendation": ["추천", "강추", "사세요", "사길잘했"],
}

PROCESSED_DIR = "voc_pipeline/data/processed"


def parse_list(val):
    """Parse a stringified Python list back to a list object."""
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return []


def score_document(text, noun_list):
    """
    Score a single document for causal signals.
    Matches churn and positive keywords against both raw text
    and extracted noun tokens. Returns signal scores and
    a classified signal type.
    Args:
        text: raw document string
        noun_list: list of extracted noun tokens
    Returns:
        dict with churn_score, positive_score, net_signal,
        signal_type, churn_signals, positive_signals, anua_mentioned
    """
    if not isinstance(text, str):
        text = ""
    text_lower = text.lower()
    nouns_set = set(noun_list)
    scores = {}

    # score churn signals
    churn_score = 0
    churn_signals = []
    for category, keywords in CHURN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower or kw in nouns_set:
                churn_score += 1
                churn_signals.append(kw)
    scores["churn_score"] = churn_score
    scores["churn_signals"] = churn_signals

    # score positive signals
    pos_score = 0
    pos_signals = []
    for category, keywords in POSITIVE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower or kw in nouns_set:
                pos_score += 1
                pos_signals.append(kw)
    scores["positive_score"] = pos_score
    scores["positive_signals"] = pos_signals

    # classify net signal direction
    scores["net_signal"] = pos_score - churn_score
    scores["signal_type"] = (
        "이탈위험" if churn_score > pos_score
        else "긍정" if pos_score > churn_score
        else "중립"
    )

    # flag documents mentioning ANUA as a competitor reference
    scores["anua_mentioned"] = any(
        kw in text_lower for kw in ["아누아", "anua", "anu-a"]
    )

    return scores


def detect_causal_signals(df):
    """
    Apply causal signal scoring to all documents in the DataFrame.
    Args:
        df: preprocessed DataFrame with raw_text and unigram columns
    Returns:
        DataFrame with appended signal score columns
    """
    print("Detecting causal signals...")
    results = []
    for _, row in df.iterrows():
        noun_list = parse_list(row.get("unigram", "[]"))
        scores = score_document(row.get("raw_text", ""), noun_list)
        results.append(scores)
    signal_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), signal_df], axis=1)


def analyze_signals(df):
    """
    Print a summary analysis of detected causal signals.
    Covers overall signal distribution, per-source churn rates,
    top churn keywords, ANUA mention samples, and high-risk documents.
    Args:
        df: DataFrame with signal score columns
    Returns:
        df (unchanged)
    """
    print("\n Causal Signal Analysis \n")

    # overall signal distribution
    total = len(df)
    signal_counts = df["signal_type"].value_counts()
    print(" Signal Distribution ")
    for sig, cnt in signal_counts.items():
        print(f"  {sig}: {cnt} ({cnt/total*100:.1f}%)")

    # per-source churn rate
    print("\n Churn Rate by Source ")
    for source in df["source"].unique():
        sub = df[df["source"] == source]
        churn_rate = (sub["signal_type"] == "이탈위험").mean() * 100
        anua_rate = sub["anua_mentioned"].mean() * 100
        print(f"  [{source}] churn risk: {churn_rate:.1f}%  ==>  ANUA mention: {anua_rate:.1f}%")

    # top churn keywords
    print("\n Top Churn Keywords ")
    all_churn = []
    for signals in df["churn_signals"]:
        if isinstance(signals, list):
            all_churn.extend(signals)
    if all_churn:
        top_churn = Counter(all_churn).most_common(10)
        for kw, cnt in top_churn:
            print(f"  '{kw}': {cnt}")

    # ANUA mention sample documents
    anua_docs = df[df["anua_mentioned"] == True]
    print(f"\n ANUA Mention Documents: {len(anua_docs)} ")
    for _, row in anua_docs.head(5).iterrows():
        text = row["raw_text"][:80] if isinstance(row["raw_text"], str) else ""
        print(f"  {row['source']} {text}")

    # high-risk churn document samples
    churn_docs = df[df["signal_type"] == "이탈위험"].nlargest(5, "churn_score")
    print(f"\n Top Churn Risk Documents ")
    for _, row in churn_docs.iterrows():
        text = row["raw_text"][:80] if isinstance(row["raw_text"], str) else ""
        print(f"  {row['source']} score={row['churn_score']} ==> {text}")

    return df


def temporal_analysis(df):
    """
    Analyze monthly trends in churn and positive signal rates.
    Args:
        df: DataFrame with signal_type, date, and anua_mentioned columns
    Returns:
        df (unchanged)
    """
    print("\n Temporal Signal Analysis \n")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_dated = df.dropna(subset=["date"])

    if len(df_dated) < 10:
        print("Insufficient date data — skipping temporal analysis")
        return df
    # aggregate monthly signal rates
    df_dated["month"] = df_dated["date"].dt.to_period("M")
    monthly = df_dated.groupby("month").agg(
        total=("signal_type", "count"),
        churn=("signal_type", lambda x: (x == "이탈위험").sum()),
        positive=("signal_type", lambda x: (x == "긍정").sum()),
        anua=("anua_mentioned", "sum")
    ).reset_index()
    monthly["churn_rate"] = monthly["churn"] / monthly["total"]
    monthly["positive_rate"] = monthly["positive"] / monthly["total"]

    print(" Monthly Churn / Positive Signal Trend ")
    print(monthly[["month", "total", "churn_rate", "positive_rate", "anua"]].to_string(index=False))

    monthly.to_csv(f"{PROCESSED_DIR}/cnp_temporal_signals.csv",
                   index=False, encoding="utf-8-sig")
    print(f"\nSaved: {PROCESSED_DIR}/cnp_temporal_signals.csv")

    return df


# Entry step
if __name__ == "__main__":
    # load preprocessed data
    df = pd.read_csv(f"{PROCESSED_DIR}/cnp_processed.csv")

    # filter to CNP-relevant documents
    CNP_KEYWORDS = {"차앤박", "CNP", "프로폴리스", "앰플", "PDRN",
                    "안티포어", "트러블", "카밍", "클렌징", "세럼", "더마"}
    df = df[df["raw_text"].apply(
        lambda x: any(kw in str(x) for kw in CNP_KEYWORDS)
    )].copy()
    print(f"{len(df)}\n")

    # run causal signal detection
    df_scored = detect_causal_signals(df)

    # print analysis summary
    df_scored = analyze_signals(df_scored)

    # run temporal analysis
    df_scored = temporal_analysis(df_scored)

    # save results
    df_scored.to_csv(f"{PROCESSED_DIR}/cnp_causal_signals.csv",
                     index=False, encoding="utf-8-sig")
    print(f"\n {PROCESSED_DIR}/cnp_causal_signals.csv")