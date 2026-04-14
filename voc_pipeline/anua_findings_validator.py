import ast
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

PROCESSED_DIR = "voc_pipeline/data/processed"


def parse_list(val):
    """Parse a stringified Python list back to a list object."""
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except:
        return str(val).split() if isinstance(val, str) else []


"""
Finding definition for ANUA
Each finding maps to a set of Korean keywords detected in the original
ANUA Amazon review analysis, now validated against CNP Korean VoC data.
"""

ANUA_FINDINGS = {
    "Finding3_FormulaChange": {
        "keywords": ["예전이", "예전엔", "바뀐것", "바뀐거", "달라진",
                     "리뉴얼후", "구버전", "전버전", "이전이", "이전버전",
                     "예전거", "예전꺼", "리뉴얼"],
        "description": "Formula change → churn pattern"
    },
    "Finding4_ChannelEffect": {
        "keywords": ["유튜브", "영상보고", "영상봤", "쇼츠", "인플루언서",
                     "추천받아", "추천받고", "보고샀", "보고구매"],
        "description": "Channel discovery (YouTube) → purchase → reaction pattern"
    },
    "Finding5_SensitiveSkinSegmentation": {
        "keywords": ["민감피부", "민감성", "트러블", "자극", "예민",
                     "수분", "건성", "건조"],
        "description": "Sensitive skin segment bifurcation pattern"
    }
}


def verify_anua_findings(df):
    """
    Validate ANUA findings against CNP Korean VoC data.
    For each finding, detects matching documents, cross-tabulates
    with signal type, and prints representative samples.
    Args:
        df: CNP causal signals DataFrame
    Returns:
        DataFrame summarizing detection rate and churn rate per finding
    """
    print("=== Stream A-1: ANUA Findings Validation on CNP Data ===\n")

    results = []
    for finding, config in ANUA_FINDINGS.items():
        matched = df[df["raw_text"].apply(
            lambda x: any(kw in str(x) for kw in config["keywords"])
        )]
        total = len(df)
        count = len(matched)
        rate = count / total * 100

        print(f" {finding}  {config['description']}")
        print(f" {count}  {rate:.1f}%")

        if count > 0:
            # cross-tabulate with signal type
            sig_dist = matched["signal_type"].value_counts()
            for sig, cnt in sig_dist.items():
                print(f"   {sig}: {cnt} ({cnt/count*100:.1f}%")

            # sample documents
            print(f"  Samples:")
            for _, row in matched.head(3).iterrows():
                print(f"    {row['source']} {str(row['raw_text'])[:80]}")

        results.append({
            "finding": finding,
            "description": config["description"],
            "count": count,
            "rate": round(rate, 1),
            "churn_rate": round(
                (matched["signal_type"] == "이탈위험").mean() * 100, 1
            ) if count > 0 else 0
        })
        print()

    return pd.DataFrame(results)


def formula_change_analysis(df):
    """
    Deep-dive temporal analysis for Finding 3: formula change signals.
    Tracks monthly distribution of formula-change mentions and their churn rate.
    """
    print(" Finding 3 Deep-dive perspective: Formula Change Timeline ")
    formula_kws = ANUA_FINDINGS["Finding3_FormulaChange"]["keywords"]
    df_formula = df[df["raw_text"].apply(
        lambda x: any(kw in str(x) for kw in formula_kws)
    )].copy()

    df_formula["date"] = pd.to_datetime(df_formula["date"], errors="coerce")
    df_dated = df_formula.dropna(subset=["date"])

    if len(df_dated) > 0:
        df_dated["month"] = df_dated["date"].dt.to_period("M")
        monthly = df_dated.groupby("month").agg(
            count=("signal_type", "count"),
            churn=("signal_type", lambda x: (x == "이탈위험").sum())
        ).reset_index()
        monthly["churn_rate"] = monthly["churn"] / monthly["count"]
        print(monthly[["month", "count", "churn_rate"]].to_string(index=False))
    print()


def channel_effect_analysis(df):
    """
    Deep-dive analysis for Finding 4: channel discovery effect.
    Compares churn and positive rates between channel-driven
    and non-channel-driven purchase documents.
    """
    print(" Finding 4 Deep-dive: Channel Effect on Churn vs Positive Rate ")
    channel_kws = ANUA_FINDINGS["Finding4_ChannelEffect"]["keywords"]

    df_channel = df[df["raw_text"].apply(
        lambda x: any(kw in str(x) for kw in channel_kws)
    )]
    df_no_channel = df[~df.index.isin(df_channel.index)]

    for label, subset in [("Channel-driven purchase", df_channel),
                           ("Non-channel purchase", df_no_channel)]:
        if len(subset) == 0:
            continue
        churn = (subset["signal_type"] == "이탈위험").mean() * 100
        pos = (subset["signal_type"] == "긍정").mean() * 100
        print(f"  {label} n={len(subset)} | "
              f"churn risk: {churn:.1f}% || positive: {pos:.1f}%")
    print()


def sensitive_skin_segmentation(df):
    """
    Deep-dive analysis for Finding 5: sensitive skin segment bifurcation.
    Splits sensitive skin documents into trouble-prone vs moisture-focused
    sub-segments and compares their churn rates.
    """
    print("[ Finding 5 Deep-dive perspective: Sensitive Skin Segment Bifurcation ]")

    sensitive_kws = ["민감피부", "민감성", "예민한피부", "민감한피부"]
    trouble_kws = ["트러블", "여드름", "뾰루지", "자극", "올라오"]
    moisture_kws = ["수분", "건조", "건성", "촉촉", "보습"]

    df_sensitive = df[df["raw_text"].apply(
        lambda x: any(kw in str(x) for kw in sensitive_kws)
    )]

    if len(df_sensitive) == 0:
        print("  No sensitive skin documents found")
        return

    # bifurcate into trouble-prone vs moisture-focused sub-segments
    df_trouble = df_sensitive[df_sensitive["raw_text"].apply(
        lambda x: any(kw in str(x) for kw in trouble_kws)
    )]
    df_moisture = df_sensitive[df_sensitive["raw_text"].apply(
        lambda x: any(kw in str(x) for kw in moisture_kws)
    )]

    print(f"  Sensitive skin total: {len(df_sensitive)}")
    print(f"  Trouble-prone segment: {len(df_trouble)} "
          f"(churn risk: {(df_trouble['signal_type']=='이탈위험').mean()*100:.1f}%)")
    print(f"  Moisture-focused segment: {len(df_moisture)} "
          f"(churn risk: {(df_moisture['signal_type']=='이탈위험').mean()*100:.1f}%)")

    print("\n  Trouble-prone samples:")
    for _, row in df_trouble.head(2).iterrows():
        print(f"    [{row['source']}] {str(row['raw_text'])[:80]}...")
    print("\n  Moisture-focused samples:")
    for _, row in df_moisture.head(2).iterrows():
        print(f"    [{row['source']}] {str(row['raw_text'])[:80]}...")
    print()


def anua_mention_deep_analysis(df):
    """
    Deep-dive analysis of documents that explicitly mention ANUA.
    Compares churn rate of ANUA-mention documents vs overall CNP corpus,
    identifies top keywords in churn+ANUA documents, and surfaces
    brand-switching signals.
    Args:
        df: CNP causal signals DataFrame
    Returns:
        filtered DataFrame of ANUA-mention documents
    """
    print(" Stream A-2: ANUA Mention Deep Analysis \n")

    anua_df = df[df["anua_mentioned"] == True].copy()
    print(f"ANUA mention documents: {len(anua_df)}\n")

    # signal distribution within ANUA-mention documents
    print(" Signal Distribution ")
    sig_dist = anua_df["signal_type"].value_counts()
    for sig, cnt in sig_dist.items():
        print(f"  {sig}: {cnt} {cnt/len(anua_df)*100:.1f}%")

    # compare churn rate vs overall CNP corpus
    overall_churn = (df["signal_type"] == "이탈위험").mean() * 100
    anua_churn = (anua_df["signal_type"] == "이탈위험").mean() * 100
    print(f"\n  Overall CNP churn risk: {overall_churn:.1f}%")
    print(f"  ANUA-mentioned churn risk: {anua_churn:.1f}%")
    print(f"  Delta: {anua_churn - overall_churn:+.1f}%p")

    # source distribution
    print("\n ANUA Mentions by Source ")
    source_dist = anua_df["source"].value_counts()
    for src, cnt in source_dist.items():
        print(f"  {src}: {cnt}")

    # top keywords in ANUA-mention + churn documents
    print("\n Top Keywords in ANUA-mention + Churn Documents ")
    churn_anua = anua_df[anua_df["signal_type"] == "이탈위험"]
    all_nouns = []
    for val in churn_anua["unigram"]:
        all_nouns.extend(parse_list(val))
    top_kws = Counter(all_nouns).most_common(15)
    for kw, cnt in top_kws:
        if kw not in {"차앤박", "앰플", "피부", "아누아", "제품", "사용"}:
            print(f"  {kw}: {cnt}")

    # brand-switching signal documents
    print("\n Brand-switching Signal Documents ")
    switch_kws = ["갈아탔", "갈아탈", "대신", "바꿨", "바꿔", "넘어갔"]
    switch_docs = anua_df[anua_df["raw_text"].apply(
        lambda x: any(kw in str(x) for kw in switch_kws)
    )]
    print(f"  Brand-switching documents: {len(switch_docs)}")
    for _, row in switch_docs.head(5).iterrows():
        print(f"  {row['source']} {str(row['raw_text'])[:100]}")

    return anua_df


# Entry step
if __name__ == "__main__":
    df = pd.read_csv(f"{PROCESSED_DIR}/cnp_causal_signals.csv")
    print(f"For analysis: {len(df)}\n")

    # Stream A-1: validate ANUA findings on CNP data
    findings_df = verify_anua_findings(df)
    formula_change_analysis(df)
    channel_effect_analysis(df)
    sensitive_skin_segmentation(df)

    # Stream A-2: deep analysis of ANUA-mention documents
    anua_deep = anua_mention_deep_analysis(df)

    # save results
    findings_df.to_csv(f"{PROCESSED_DIR}/stream_a_findings.csv",
                       index=False, encoding="utf-8-sig")
    print(f"\n {PROCESSED_DIR}/stream_a_findings.csv")