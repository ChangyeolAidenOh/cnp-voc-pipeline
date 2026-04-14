import os
import time
import pandas as pd
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = "voc_pipeline/data/raw"
PROCESSED_DIR = "voc_pipeline/data/processed"


def translate_batch(texts, src="en", dest="ko", batch_size=50):
    """
    Translate a list of texts in batches using Google Translate.
    Includes rate limiting between requests to avoid API throttling.
    Texts exceeding 500 characters are truncated before translation.
    Args:
        texts: list of source strings
        src: source language code (default: 'en')
        dest: target language code (default: 'ko')
        batch_size: number of texts per batch
    Returns:
        list of translated strings (empty string on failure)
    """
    translated = []
    translator = GoogleTranslator(source=src, target=dest)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Translate now {i+1}~{min(i+batch_size, len(texts))} / {len(texts)}")
        for text in batch:
            try:
                if not isinstance(text, str) or len(text.strip()) < 2:
                    translated.append("")
                    continue
                # truncate to 500 characters to stay within API limits
                text_trimmed = text[:500]
                result = translator.translate(text_trimmed)
                translated.append(result if result else "")
            except Exception as e:
                print(f"    Translation error: {e}")
                translated.append("")
            time.sleep(0.1)  # rate limiting between individual requests
        time.sleep(1)  # rate limiting between batches

    return translated


# ── Entry point ───────────────────────────────────────
if __name__ == "__main__":
    # verify deep_translator is installed
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        exit()

    # load ANUA English review files
    anua_files = [
        f"{RAW_DIR}/anua_cleansing_oil_reviews.csv",
        f"{RAW_DIR}/anua_toner_reviews.csv"
    ]

    dfs = []
    for path in anua_files:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f" {path} ({len(df)} records)")
            dfs.append(df)

    if not dfs:
        print(f"ANUA CSV files not found in {RAW_DIR}/")
        exit()

    anua_df = pd.concat(dfs, ignore_index=True)
    print(f"Total ANUA reviews: {len(anua_df)}\n")

    # detect review text column
    print("Columns:", list(anua_df.columns))
    text_col = next(
        (c for c in ["Review Content", "review", "text", "body",
                     "Review", "Text", "content"]
         if c in anua_df.columns),
        anua_df.columns[0]
    )
    print(f"Text column: {text_col}")
    print(f"Sample: {anua_df[text_col].iloc[0][:100]}\n")

    # run translation
    texts = anua_df[text_col].fillna("").tolist()
    translated = translate_batch(texts, src="en", dest="ko")

    anua_df["text_ko"] = translated
    output_path = f"{PROCESSED_DIR}/anua_translated.csv"
    anua_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n {output_path}")

    # translation quality check
    print("\n= Translation Sample ")
    sample = anua_df[anua_df["text_ko"] != ""].head(5)
    for _, row in sample.iterrows():
        orig = str(row[text_col])[:80]
        trans = str(row["text_ko"])[:80]
        print(f"Original : {orig}")
        print(f"After translation: {trans}")
        print()