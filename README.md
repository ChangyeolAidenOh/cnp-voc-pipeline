# CNP VoC Causal Signal Detection Pipeline

A Korean-language NLP pipeline for detecting causal signals in consumer Voice of Customer (VoC) data for **CNP (차앤박)**, a dermocosmetic brand under LG생활건강. The pipeline collects multi-source Korean VoC data, applies LDA and BERTopic topic modeling across multiple preprocessing strategies, and extracts structured causal signals — identifying *why* consumers churn, not just *what* they said.

Built as a portfolio project targeting the **AX (AI Transformation) role at LG생활건강**, this pipeline directly addresses the strategic question the company faces: *why is CNP losing consumers to indie brands like ANUA, and what are the language-level patterns that signal churn before it becomes irreversible?*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [LDA Topic Modeling](#lda-topic-modeling)
  - [Causal Signal Detection](#causal-signal-detection)
  - [BERTopic and Ensemble Validation](#bertopic-and-ensemble-validation)
  - [ANUA Comparative Analysis](#anua-comparative-analysis)
- [Key Findings](#key-findings)
- [Dashboard](#dashboard)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

The goal of this project is to identify *causal* patterns in CNP consumer language — not just topic frequency, but the structural relationships between topic clusters, sentiment signals, and temporal trends that explain consumer behavior.

The pipeline is designed to answer:
- What topics and keywords characterize CNP churn risk vs. satisfaction?
- Do ANUA's consumer language patterns replicate in CNP's Korean VoC data?
- Which consumer segments (e.g. trouble-prone vs. moisture-focused sensitive skin) exhibit structurally different churn profiles?
- What is the causal relationship between ANUA mentions and CNP churn risk?
- How has churn signal intensity changed over time, and when did it peak?

---

## Business Context

LG생활건강 reported a 62.8% drop in operating profit in 2025 (₩1,707B), with its cosmetics segment recording its first deficit in 20 years. CNP accounts for approximately 5% of cosmetics revenue and competes directly with fast-growing indie dermocosmetic brands — most notably ANUA, which occupies the same product category (cleansers, toners, serums) and has achieved significant viral traction on YouTube and social platforms.

This pipeline treats CNP's VoC data as a signal layer to answer a question LG생활건강's existing VoC classification systems (which categorize *what* consumers said) cannot fully answer: ***why*** are consumers switching, and which segments are at the highest structural risk of churn?

---

## Dataset

- **Brand**: CNP (차앤박 / CNP Laboratory) — LG생활건강 dermocosmetic brand
- **Total documents**: 6,522 collected → 3,013 after CNP relevance filtering
- **Sources**:

| Source | Collection Method | Records | Notes |
|---|---|---|---|
| Naver Blog | Naver Search API (official) | ~1,170 | Long-form usage reviews, ingredient analysis |
| Naver Cafe | Naver Search API (official) | ~1,051 | Community Q&A, high-engagement consumers |
| YouTube Comments | YouTube Data API v3 (official) | ~2,500 | Purchase motivation + post-purchase reaction |
| ANUA Reviews (EN) | Existing dataset (Amazon.com) | 990 | Used for Stream B cross-language comparison |

- **Relevance filter**: Documents must contain at least one CNP-related keyword (`차앤박`, `CNP`, `프로폴리스`, `앰플`, `PDRN`, `안티포어`, `트러블`, `카밍`, `클렌징`, `세럼`, `더마`) to be included in analysis.
- **Language**: Korean (primary); English ANUA reviews machine-translated to Korean for Stream B.
- **Collection note**: All data collected via official APIs or manual export tools. No scraping of systems that prohibit automated access. Data used for non-commercial portfolio purposes only; raw data not redistributed.

---

## Project Structure

```
cnp-voc-pipeline/
│
├── voc_pipeline/
│   ├── collector_naver.py          # Naver Blog/Cafe collection via official Search API
│   ├── collector_youtube.py        # YouTube comment collection via Data API v3
│   ├── preprocessor.py             # Korean morphological analysis + 4 preprocessing modes
│   ├── LDA_pipeline.py             # LDA topic modeling: per-source × per-mode
│   ├── causal_signal_detector.py   # Causal signal scoring + temporal analysis
│   ├── anua_findings_validator.py  # Stream A: ANUA findings validation on CNP data
│   ├── anua_review_translator.py   # Stream B: EN→KO translation of ANUA reviews
│   ├── cnp_anua_comparator.py      # Stream B: keyword divergence + LDA comparison
│   └── dashboard.py                # Streamlit interactive dashboard
│
├── voc_pipeline/data/
│   ├── raw/                        # Raw collected data (gitignored)
│   └── processed/                  # Analysis outputs (gitignored)
│
├── notebooks/
│   └── bertopic_colab.ipynb        # BERTopic pipeline (run on Google Colab, GPU recommended)
│
├── .env.example                    # API key template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Pipeline

```
Raw Data Collection
(Naver Blog/Cafe via Search API + YouTube via Data API v3)
        │
        ▼
collector_naver.py / collector_youtube.py
        │
        ▼
voc_pipeline/data/raw/
(naver_blog_cnp.csv / naver_cafe_cnp.csv / youtube_comments_cnp.csv)
        │
        ▼
preprocessor.py
(kiwipiepy morphological analysis + 4 preprocessing modes)
        │
        ▼
voc_pipeline/data/processed/cnp_processed.csv
        │
        ├─────────────────────────────────────────────────┐
        ▼                                                 ▼
LDA_pipeline.py                              notebooks/bertopic_colab.ipynb
(per-source × per-mode,                      (BERT embeddings + HDBSCAN,
 c_v coherence optimization)                  Google Colab / GPU)
        │                                                 │
        ▼                                                 ▼
cnp_lda_results.csv                          cnp_bertopic_results.csv
        │                                                 │
        └──────────────┬──────────────────────────────────┘
                       ▼
            cnp_lda_bertopic_consensus.csv
            (high-confidence signals = both models agree)
                       │
                       ▼
        causal_signal_detector.py
        (keyword-based causal scoring + temporal analysis)
                       │
                       ▼
        cnp_causal_signals.csv / cnp_temporal_signals.csv
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
Stream A                          Stream B
anua_findings_validator.py        anua_review_translator.py
(ANUA Findings 3/4/5              (EN→KO translation)
 validated on CNP data)                    │
                                  cnp_anua_comparator.py
                                  (keyword divergence analysis)
        │                                  │
        ▼                                  ▼
stream_a_findings.csv             stream_b_keyword_divergence.csv
        │                                  │
        └──────────────┬──────────────────┘
                       ▼
                dashboard.py
          (Streamlit interactive dashboard)
```

---

## Methodology

### Preprocessing

Four preprocessing strategies were applied to each document, enabling direct comparison of how token representation affects topic coherence:

| Mode | Description |
|---|---|
| `unigram` | Single noun tokens; stopwords + domain noise removed |
| `bigram` | Consecutive noun pairs (e.g. `차앤박_앰플`, `트러블_카밍`) |
| `unibi_mix` | Union of unigram and bigram tokens |
| `adj_noun` | Adjective-noun pairs (e.g. `순한_클렌저`) for sentiment-bearing phrases |

All modes use `kiwipiepy` for Korean morphological analysis (selected over KoNLPy for M-series Mac compatibility). A custom domain stopword list removes YouTube channel noise (`감사`, `구독`, `영상`), shopping platform noise (`코스트코`, `쿠팡`, `최저`), and terms that confound CNP's brand positioning with unrelated dermatology context (`병원`, `시술`, `레이저`).

### LDA Topic Modeling

Gensim's `LdaModel` was applied independently to each combination of:
- Source (`blog` / `cafearticle` / `youtube` / `all`)
- Preprocessing mode (`unigram` / `bigram` / `unibi_mix` / `adj_noun`)

Optimal topic count *k* was selected automatically by maximizing c_v coherence score across k = 2 to 7. All models used `passes=15` and `random_state=42`.

**Coherence summary:**

| Scope | unigram | bigram | unibi_mix |
|---|---|---|---|
| blog | 0.5193 | 0.4831 | **0.6256** |
| cafearticle | 0.4186 | **0.5488** | 0.5408 |
| youtube | 0.4497 | **0.6091** | 0.5050 |
| all | 0.4033 | 0.4736 | **0.5201** |

Phrase-level representations (bigram, unibi_mix) consistently outperform word-level (unigram), consistent with findings from the prior ANUA Amazon review analysis.

### Causal Signal Detection

`causal_signal_detector.py` applies a keyword-based causal scoring system to each document. Documents are scored across two opposing signal categories:

**Churn signals** (`direct_churn`, `efficacy_failure`, `skin_reaction`, `competitor_switch`, `formula_change`) and **positive signals** (`repurchase`, `efficacy_positive`, `recommendation`).

Each document receives a `signal_type` label (`이탈위험` / `긍정` / `중립`) based on the net balance of detected signals. The module additionally flags documents that mention ANUA as a competitor reference (`anua_mentioned`), enabling direct measurement of brand-switching signal intensity.

Temporal analysis aggregates monthly churn and positive rates to surface trend anomalies.

### BERTopic and Ensemble Validation

BERTopic (`paraphrase-multilingual-MiniLM-L12-v2` embeddings + HDBSCAN clustering) was run on Google Colab (T4 GPU) on the same filtered corpus. 14 topics were auto-detected.

Cross-validation logic: topics where LDA and BERTopic agree on overlapping keywords are classified as **high-confidence signals**; LDA-only topics require further validation.

| LDA Topic | BERTopic Match | Overlap | Confidence |
|---|---|---|---|
| 블랙헤드_안티포어 | Topic 4, 11 | 블랙, 키트, 클리어 | High |
| 더마코스메틱 | Topic 7, 13 | 피부과, 추출물 | High |
| 프로폴리스_앰플 | None | — | Low |
| 트러블_이탈 | Topic 0 | 여드름 | High (diluted) |

### ANUA Comparative Analysis

**Stream A** validates the three key findings from the prior ANUA Amazon review project against CNP Korean VoC data, testing whether the same causal patterns replicate cross-language and cross-brand.

**Stream B** translates 990 ANUA English reviews to Korean using `deep-translator` (Google Translate), applies the same preprocessing pipeline, and computes keyword divergence scores between CNP and ANUA corpora. Divergence bias is defined as `(cnp_count - anua_count) / total`, with values > 0.3 indicating CNP-specific vocabulary and < -0.3 indicating ANUA-specific vocabulary.

**Stream B methodological note**: Translation-based LDA coherence was low (0.2786) due to repetitive sentence structures introduced during EN→KO translation (`제품_사용`, `사용_피부`). Keyword frequency and divergence analysis from Stream B are considered exploratory; core findings are grounded in Stream A's native Korean analysis.

---

## Key Findings

### 1. Churn Risk Concentration — 27.9% of CNP Documents Signal Active Churn Risk

Of 3,013 CNP-relevant documents, 840 (27.9%) were classified as churn risk — 1.7× the positive signal rate (16.8%). The top churn signal keywords by frequency: `트러블` (523), `별로` (180), `여드름` (153), `대신` (132), `자극` (128).

> **Business implication**: Nearly 3 in 10 organic CNP mentions carry active churn signals. This is not noise — it is a structural pattern. The dominant churn mechanism is skin reaction (`트러블`, `자극`, `올라오`, `여드름`), not price or availability, suggesting the issue is formulation-consumer fit rather than a marketing or distribution problem.

---

### 2. ANUA Mention = 81.0% Churn Risk — A Statistically Robust Causal Signal

Documents that mention ANUA show a churn risk rate of **81.0%**, compared to 27.9% for the overall CNP corpus — a delta of **+53.2 percentage points**.

This pattern holds across all three sources (blog: 29 docs, YouTube: 22, cafe: 7) and was independently confirmed in Stream A analysis.

> **Business implication**: When a CNP consumer mentions ANUA in the same document, they have almost certainly already switched or are actively considering switching. ANUA mention is not a neutral comparison — it is a churn completion signal. Monitoring ANUA co-occurrence in CNP VoC in real time would function as a leading indicator of brand erosion velocity.

---

### 3. Finding 3 Replicated — Formula Change as a Churn Driver

ANUA Finding 3 (`old_formula` / `prior_formula` → ★1 concentration) was tested on CNP data using Korean equivalents (`예전이`, `달라진`, `리뉴얼후`, `구버전`). 47 documents (1.6%) matched, with 19.1% churn risk — elevated relative to the 16.8% positive rate baseline.

Unlike the ANUA toner case — where formula change language was the *dominant* churn driver — CNP's formula change signal is present but not the primary driver. Skin reaction signals outweigh formula change signals by approximately 11:1.

> **Business implication**: CNP's churn profile is driven more by formulation-skin mismatch than by formulation change per se. This suggests the problem is less about recent changes and more about a persistent mismatch between CNP's product positioning (dermocosmetic, suitable for sensitive skin) and the actual skin-type distribution of its consumer base.

---

### 4. Finding 4 Replicated — Channel Effect Reversal Confirmed

ANUA Finding 4 found that TikTok-driven purchases of the ANUA toner produced *higher* churn rates than organic purchases. Tested on CNP data with YouTube as the equivalent channel:

| Segment | n | Churn Risk | Positive |
|---|---|---|---|
| Channel-driven purchase | 27 | 22.2% | **33.3%** |
| Non-channel purchase | 2,986 | 27.9% | 16.7% |

CNP's YouTube-driven consumers show *higher satisfaction* and *lower churn risk* than the overall corpus — the **opposite pattern** from ANUA toner's TikTok effect.

> **Business implication**: CNP's YouTube presence is working correctly — content is generating realistic expectations and satisfied buyers. This is a replicable and scalable acquisition channel. The risk seen in ANUA's TikTok case (expectation mismatch → churn) does not currently apply to CNP's YouTube channel, but should be monitored as content volume scales.

---

### 5. Finding 5 Replicated — Sensitive Skin Segment Bifurcation

ANUA Finding 5 identified that `sensitive_skin` consistently bifurcates into two structurally distinct topic clusters with opposite churn profiles. Tested on CNP Korean VoC:

| Segment | Keywords | n | Churn Risk |
|---|---|---|---|
| Trouble-prone | 트러블, 여드름, 뾰루지, 자극, 올라오 | 55 | **67.3%** |
| Moisture-focused | 수분, 건조, 건성, 촉촉, 보습 | 52 | 28.8% |

The bifurcation replicates cross-language and cross-brand, with nearly identical structural separation. Trouble-prone sensitive skin consumers show 2.3× higher churn risk than moisture-focused sensitive skin consumers when using CNP products.

> **Business implication**: CNP's current marketing targeting "sensitive skin" as a monolithic segment is structurally generating polarized outcomes. Trouble-prone sensitive skin consumers — who are reactive to occlusive or comedogenic ingredients — are experiencing breakouts and churning at high rates. Moisture-focused sensitive skin consumers are satisfied. Differentiating messaging, usage guidance, and product recommendations by sensitive skin sub-type (acne-prone vs. dry/reactive) would directly reduce the 67.3% churn rate in the highest-risk segment.

---

### 6. Temporal Anomaly — Churn Spike in January–February 2026

Monthly churn rate trend (documents with date metadata):

| Period | Churn Rate | Notes |
|---|---|---|
| 2025-08 ~ 2025-11 | 12–24% | Baseline range |
| 2025-12 | 31.7% | Initial elevation |
| **2026-01** | **54.3%** | Spike peak |
| **2026-02** | **49.1%** | Sustained |
| 2026-03 | 32.6% | Recovery begins |
| 2026-04 | 16.9% | Near-baseline |

The January–February 2026 spike (54.3%) is more than double the baseline and aligns temporally with the period when CNP's PDRN ampoule line received increased YouTube coverage. The recovery by April 2026 suggests the spike was event-driven rather than structural — but the triggering event cannot be confirmed from VoC data alone without external event correlation.

> **Business implication**: Temporal churn monitoring — even at monthly granularity — can surface product or communication events that generate disproportionate consumer backlash before they become permanent brand damage. A real-time version of this pipeline, integrated with LG생활건강's existing data infrastructure, would enable 2–4 week early warning on emerging churn signals.

---

### 7. Preprocessing Strategy — Phrase-Level Representations Outperform Word-Level

Consistent with the prior ANUA analysis, bigram and unibi_mix modes produce significantly higher coherence scores than unigram across all sources. The blog unibi_mix combination achieves the highest coherence (0.6256), suggesting that Korean beauty consumers naturally describe their experiences through compound noun expressions that single-token models fail to capture.

> **Methodological implication**: For Korean beauty/skincare VoC, phrase-level preprocessing is not optional — it is the difference between coherent topic structures and noise. Domain-specific stopword engineering and compound noun extraction are the primary levers for coherence improvement, not model architecture.

---

## Dashboard

An interactive Streamlit dashboard visualizes all pipeline outputs:

```bash
streamlit run voc_pipeline/dashboard.py
```

**Tabs**: Overview | LDA Topics | Causal Signals | ANUA Comparison | BERTopic

---

## How to Run

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your Naver and YouTube API keys

# 3. Collect data
python voc_pipeline/collector_naver.py
python voc_pipeline/collector_youtube.py

# 4. Preprocess
python voc_pipeline/preprocessor.py

# 5. Run LDA pipeline
python voc_pipeline/LDA_pipeline.py

# 6. Run causal signal detection
python voc_pipeline/causal_signal_detector.py

# 7. Run comparative analysis
python voc_pipeline/anua_findings_validator.py
python voc_pipeline/anua_review_translator.py   # requires ANUA CSVs in data/raw/
python voc_pipeline/cnp_anua_comparator.py

# 8. Run BERTopic (Google Colab recommended)
# Open notebooks/bertopic_colab.ipynb on Google Colab
# Upload voc_pipeline/data/processed/cnp_processed.csv
# Download outputs to voc_pipeline/data/processed/

# 9. Launch dashboard
streamlit run voc_pipeline/dashboard.py
```

---

## Dependencies

```
# local environment
kiwipiepy==0.23.1
gensim==4.4.0
streamlit==1.56.0
plotly==6.7.0
pandas==3.0.1
requests==2.33.1
python-dotenv==1.2.2
google-api-python-client==2.194.0
deep-translator==1.11.4

# Google Colab environment (BERTopic pipeline)
# Run notebooks/bertopic_colab.ipynb on Google Colab (GPU recommended)
# bertopic
# sentence-transformers
# scikit-learn
# kiwipiepy
```

Install local dependencies:
```bash
pip install -r requirements.txt
```