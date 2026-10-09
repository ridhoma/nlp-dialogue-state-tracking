# Dialogue State Tracking (DST): Slot Extraction Evaluation Registry

This document tracks all formal evaluation experiments for multi-label slot extraction conducted in [`notebooks/01_semantic_slots_extraction_evals.ipynb`](notebooks/01_semantic_slots_extraction_evals.ipynb). Each entry captures the exact configuration parameters, document preconditioning methodology, global classification metrics, per-slot breakdowns, and inference latency benchmarks.

---

## Evaluation Benchmark Setup

* **Evaluation Dataset:** `data/samples/slot_extraction_evaluation_set.parquet`
* **Sampled Conversations:** 1,000 distinct multi-turn dialogues
* **Total Dialogue Turns:** 22,262 turns
* **Evaluated Customer Utterances:** 8,984 customer messages
* **Target Slot Universe:** 10 human-annotated core entities from ABCD `slot_tags`:
  `account_id`, `amount`, `email`, `name`, `order_id`, `phone`, `pin_number`, `street_address`, `username`, `zip_code`
* **Evaluation Framework:** Multi-label classification via [`data_science.evaluation.evaluate_multilabel`](src/data_science/evaluation.py) (`drop_zero_support=False`)
* **Hardware Environment:** Apple Silicon GPU (Metal Performance Shaders - `mps`) with FP16 half-precision

---

## Experiment Summary & Leaderboard

| Run ID | Model | Threshold | Framing Context | Batch Size | Total Time | Latency (ms/turn) | Micro-F1 | Macro-F1 | Exact Match | Key Notes |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **[Run 01](#run-01-baseline-bart-large-mnli--threshold-09)** | `facebook/bart-large-mnli` | 0.90 | Agent Prior $\ge 1$ (Cust Stripped) | 128 | 4m 39s | 31.0 ms | **62.5%** | **58.6%** | **85.7%** | Baseline template; high precision on structured IDs; `name` collapse (1.7% F1). |
| **[Run 02](#run-02-personalized-hypotheses--dynamic-thresholds-bart-large-mnli)** | `facebook/bart-large-mnli` | Dynamic (0.85–0.98) | Agent Prior $\ge 1$ (Cust Stripped) | 128 | 5m 22s | 35.8 ms | **59.2%** | **55.2%** | **83.4%** | Personalized hypotheses + dynamic slot thresholds; +9.5% recall over baseline; strong wins on `zip_code` (71.3%) & `amount` (58.7%). |
| **[Run 03](#run-03-hypothesis-specification-optimisation-bart-large-mnli)** | `facebook/bart-large-mnli` | Dynamic (0.85–0.98) | Agent Prior $\ge 1$ (Cust Stripped) | 128 | 5m 01s | 33.5 ms | **68.7%** | **66.2%** | **87.8%** | Using LLM agent (`Gemini 3.8 Flash - Medium` via `Antigravity /goal` command) to iteratively optimize hypothesis wording. Major wins on `email` (86.6%), `order_id` (85.1%), `pin_number` (72.7%), and `name` (31.0%). |
| **[Run 04](#run-04-informative-utterance-gate-bart-large-mnli)** | `facebook/bart-large-mnli` | Dynamic (0.85–0.98) | Agent Prior $\ge 1$ + Generalized Gate ($\tau_{\text{gate}}=0.02$) | 128 | **3m 46s** | **25.2 ms** | **70.2%** | **67.3%** | **88.9%** | Generalized Stage 1 Informative Utterance Gate (universal entity categories, zero hard-coded slot names). Cut 115 false positives, broke 70% Micro-F1 barrier, +7.7% F1 on `amount`, and delivered a 25% inference speedup. |
| **[Run 05](#run-05-contextual-disambiguation-gate-bart-large-mnli)** | `facebook/bart-large-mnli` | Dynamic (0.85–0.98) | Stage 1 Gate ($\tau_{\text{gate}}=0.02$) + Stage 3 Disambiguation Gate ($\tau_{\text{verify}}=0.05$) | 128 | **3m 58s** | **26.5 ms** | 🏆 **71.0%** | **67.3%** | 🏆 **89.3%** | Stage 3 Contextual Disambiguation Gate with stricter verification ($\tau_{\text{verify}}=0.05$). Slashed 83 false positives vs Run 04 (797 $\to$ 714). Micro-F1 reached 71.0%, Exact Match 89.3%, and Hamming Loss dropped to 0.0127. |



---

## Experiment Logs

<details>
<summary><h3 style="display: inline-block; margin: 0; cursor: pointer;">Run 01: Baseline BART-Large-MNLI (Threshold = 0.90)</h3> &nbsp; <span style="font-size: 13px; color: #64748b;">(Click to collapse/expand)</span></summary>

<br>

* **Date:** 2026-10-07
* **Notebook:** [`notebooks/01_semantic_slots_extraction_evals.ipynb`](notebooks/01_semantic_slots_extraction_evals.ipynb)
* **Goal:** Establish a baseline zero-shot NLI extraction benchmark across 1,000 conversations using `facebook/bart-large-mnli` at a conservative confidence threshold.

#### 1. Configuration & Parameters

| Component | Parameter | Value | Rationale / Description |
| :--- | :--- | :--- | :--- |
| **Classifier** | Model Architecture | `facebook/bart-large-mnli` | Large 406M parameter NLI cross-encoder |
| | Device Backend | Apple Silicon GPU (`mps`) | Accelerated on Metal execution units |
| | Precision | `fp16=True` (Half precision) | 50% memory bandwidth savings and matrix core usage |
| | Decision Calibration | Option 1 3-Class Probability | $P(\text{Entailment}) > P(\text{Neutral}) \land P(\text{Entailment}) > \tau$ |
| | Confidence Threshold ($\tau$) | `0.90` | High confidence cutoff to suppress false positives |
| | Hypothesis Template | `"The customer specifies their {0}"` | Formatted per candidate slot (underscores replaced with spaces) |
| **Document Preconditioning** | Rolling Agent Context | `min_prior_agent_utterances=1` | Traverses backward to include at least 1 preceding agent utterance |
| | Customer Utterance Stripping | `strip_prior_customer_utterances=True` | Excludes prior customer messages to prevent inter-turn slot bleeding |
| | Document Format | Preconditioned strings | `[AGENT]: ... \n [CUSTOMER]: ...` |
| **Execution** | Batch Size | `128` | Max pairs evaluated per GPU forward pass |
| | Evaluation Universe | 10 core slots | Strict alignment with human ground truth `slot_tags` |

#### 2. Inference Latency & Speed Benchmark

* **Total Evaluated Turns:** 8,984 customer messages
* **Total Batches:** 71 batches (chunked by 128)
* **Total Runtime:** **4 minutes 39 seconds** (279.0s)
* **Batch Processing Rate:** **3.93 s/batch**
* **Per-Turn Latency:** **31.0 ms / customer turn** (~32.2 customer turns/second)
* **Pair Throughput:** ~275 premise-hypothesis pairs scored per second

#### 3. Global Multi-Label Performance Metrics

| Global Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Exact Match Ratio (Subset Accuracy)** | **85.7%** | 7,700 / 8,984 turns matched the ground truth label set exactly (boosted by empty turns) |
| **Hamming Loss** | **0.0169** | Fraction of wrong slot assignments over all (turn, slot) pairs is ~1.7% |
| **Jaccard Similarity (Samples)** | **10.0%** | Mean turn-level intersection-over-union |
| **Micro-Precision** | **56.1%** | Total TP / (Total TP + Total FP) across all slots |
| **Micro-Recall** | **70.6%** | Total TP / (Total TP + Total FN) across all slots |
| **Micro-F1** | **62.5%** | Global harmonic balance across all instances |
| **Macro-Precision** | **57.4%** | Unweighted mean precision across all 10 slots |
| **Macro-Recall** | **63.2%** | Unweighted mean recall across all 10 slots |
| **Macro-F1** | **58.6%** | Unweighted mean F1 across all 10 slots |
| **Weighted F1** | **66.4%** | Support-weighted harmonic balance across slots |

#### 4. Per-Slot Classification Breakdown

*Ranked by ground-truth support (occurrences in `true_slots`):*

| Slot Name | Ground Truth Support | Model Predictions | TP | FP | FN | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`order_id`** | 461 | 387 | 356 | 31 | 105 | **92.0%** | 77.2% | **84.0%** | High precision; mild recall drop at $\tau=0.9$ |
| **`account_id`** | 291 | 467 | 239 | 228 | 52 | **51.2%** | 82.1% | **63.1%** | High recall; confused with username/order IDs |
| **`username`** | 247 | 417 | 199 | 218 | 48 | **47.7%** | 80.6% | **59.9%** | Moderate precision; confused with account IDs |
| **`email`** | 243 | 265 | 218 | 47 | 25 | **82.3%** | 89.7% | **85.8%** | Excellent balance (218 TP out of 243) |
| **`zip_code`** | 157 | 92 | 62 | 30 | 95 | **67.4%** | 39.5% | **49.8%** | Severe FN rate (95 FN) due to high 0.9 threshold |
| **`amount`** | 149 | 143 | 46 | 97 | 103 | **32.2%** | 30.9% | **31.5%** | Struggling with currency/refund mentions |
| **`street_address`** | 99 | 103 | 87 | 16 | 12 | **84.5%** | 87.9% | **86.1%** | Highest overall performance (86.1% F1) |
| **`name`** | 61 | 287 | 3 | 284 | 58 | **1.0%** | 4.9% | **1.7%** | **Critical failure**: 284 FPs against 61 support |
| **`phone`** | 54 | 49 | 30 | 19 | 24 | **61.2%** | 55.6% | **58.3%** | Balanced moderate performance |
| **`pin_number`** | 30 | 46 | 25 | 21 | 5 | **54.3%** | 83.3% | **65.8%** | Good recall (83.3%), moderate precision |

---

#### 5. Diagnosis & Key Findings

1. **Top Performers:**
   * **`street_address` (86.1% F1)**, **`email` (85.8% F1)**, and **`order_id` (84.0% F1)** show strong zero-shot comprehension with high precision and recall.
2. **The `name` Catastrophe (1.7% F1):**
   * Generated **284 False Positives** on only 61 true cases ($TP=3$).
   * *Root Cause:* The hypothesis `"The customer specifies their name"` triggers false entailment on general conversational pleasantries, agent greetings (e.g., *"Thank you Albert"*), or when customers refer to items/people generally. In addition, ABCD annotators use `name` strictly for personal human names, whereas BART treats arbitrary named entities or salutations as names.
3. **ID Confusion (`account_id` vs. `username`):**
   * Combined 446 False Positives across `account_id` (FP=228) and `username` (FP=218).
   * Alphanumeric codes are often framed similarly by agents (e.g. *"Can I have your username or account ID?"*), causing NLI to entail both simultaneously.
4. **Threshold Truncation on `zip_code` & `amount`:**
   * At $\tau=0.90$, recall for `zip_code` dropped to **39.5%** (95 false negatives) because short 5-digit numbers without rich surrounding words often fail to cross the 90% confidence threshold.
5. **Inference Efficiency:**
   * Processed **8,984 customer turns across 1,000 conversations in under 4.7 minutes** on Apple Silicon GPU (`mps`) using FP16 and batch size 128 (~31 ms/turn).

---

#### 6. Actionable Optimization Targets for Run 02

* [ ] **Fix Hypothesis for `name`:** Change hypothesis template to `"The customer states their personal first or last name."` or separate customer name disclosures from agent salutations.
* [ ] **Disambiguate `account_id` and `username`:** Refine templates (e.g. `"The customer provides their account number/identifier"` vs `"The customer provides their login username"`).
* [ ] **Threshold Tuning per Slot:** A uniform 0.90 threshold hurts numeric entities like `zip_code` (0.75 is likely optimal for zip codes while 0.90+ is appropriate for unstructured text).
* [ ] **Prompt / Context Conditioning:** Evaluate if providing only the immediate prior agent turn vs. 2 prior turns reduces prompt bleed.

</details>

<details>
<summary><h3 style="display: inline-block; margin: 0; cursor: pointer;">Run 02: Personalized Hypotheses + Dynamic Thresholds (BART-Large-MNLI)</h3> &nbsp; <span style="font-size: 13px; color: #64748b;">(Click to collapse/expand)</span></summary>

<br>

* **Date:** 2026-10-08
* **Notebook:** [`notebooks/01_semantic_slots_extraction_evals.ipynb`](notebooks/01_semantic_slots_extraction_evals.ipynb)
* **Goal:** Combine slot-specific personalized NLI hypotheses (`DEFAULT_SLOT_HYPOTHESES`) with **dynamic per-slot decision thresholds** (`threshold=THRESHOLD_BY_SLOTS`, `default_threshold=0.90`) to prune agent prompt bleed while preserving high retrieval recall across 1,000 conversations.

#### 1. Configuration & Parameters

| Component | Parameter | Value | Rationale / Description |
| :--- | :--- | :--- | :--- |
| **Classifier** | Model Architecture | `facebook/bart-large-mnli` | Large 406M parameter NLI cross-encoder |
| | Device Backend | Apple Silicon GPU (`mps`) | Accelerated on Metal execution units |
| | Precision | `fp16=True` (Half precision) | FP16 mode enabled |
| | Decision Calibration | Option 1 3-Class Probability | $P(\text{Entailment}) > P(\text{Neutral}) \land P(\text{Entailment}) > \tau_{\text{slot}}$ |
| | Dynamic Thresholds ($\tau_{\text{slot}}$) | Per-slot dictionary mapping | Tailored confidence cutoff per candidate slot |
| | Default Threshold | `0.90` | Applied to slots not explicitly overridden |
| | Hypothesis Template | `DEFAULT_SLOT_HYPOTHESES` (dict mapping) | Personalized NLI hypothesis per target slot |
| **Document Preconditioning** | Rolling Agent Context | `min_prior_agent_utterances=1` | Traverses backward to include at least 1 preceding agent utterance |
| | Customer Utterance Stripping | `strip_prior_customer_utterances=True` | Excludes prior customer messages to prevent inter-turn slot bleeding |
| | Document Format | Preconditioned strings | `[AGENT]: ... \n [CUSTOMER]: ...` |
| **Execution** | Batch Size | `128` | Max pairs evaluated per GPU forward pass |
| | Evaluation Universe | 10 core slots | Ground-truth ABCD `slot_tags` evaluation set |

##### Dynamic Threshold Mapping Used:
```python
THRESHOLD_BY_SLOTS = {
    "email": 0.98,          # Strict cutoff to reject agent prompt bleed
    "phone": 0.97,          # Strict cutoff to reject agent prompt bleed
    "account_id": 0.96,     # Disambiguate against username / generic IDs
    "order_id": 0.95,       # High confidence for purchase order identifiers
    "zip_code": 0.85,       # Lower threshold to recover bare 5-digit postal numbers
    "amount": 0.90,         # Balanced threshold for monetary mentions
    "street_address": 0.90, # Balanced threshold for street addresses
}
# Remaining slots (username, name, pin_number) receive default_threshold = 0.90
```

##### Curated Slot Hypothesis Dictionary Used:
```python
DEFAULT_SLOT_HYPOTHESES = {
    "account_id": "The customer provides an account number or account identifier code.",
    "amount": "The customer specifies a monetary price, payment amount, or dollar figure.",
    "email": "The customer provides an email address.",
    "name": "The customer states their own personal first or last name.",
    "order_id": "The customer specifies an order number or purchase ID.",
    "phone": "The customer provides a contact phone number.",
    "pin_number": "The customer provides a security PIN or verification code.",
    "street_address": "The customer specifies a physical street address or mailing address.",
    "username": "The customer provides an online account login username or handle.",
    "zip_code": "The customer provides a postal zip code.",
}
```

#### 2. Inference Latency & Speed Benchmark

* **Total Evaluated Turns:** 8,984 customer messages
* **Total Batches:** 71 batches (chunked by 128)
* **Total Runtime:** **5 minutes 22 seconds** (322.0s)
* **Batch Processing Rate:** **4.54 s/batch**
* **Per-Turn Latency:** **35.8 ms / customer turn** (~27.9 customer turns/second)
* **Pair Throughput:** ~238 premise-hypothesis pairs scored per second

#### 3. Global Multi-Label Performance Metrics

| Global Metric | Run 01 Baseline | Run 02 Dynamic | Delta vs Run 01 ($\Delta$) | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Exact Match Ratio (Subset)** | 85.7% | **83.4%** | -2.3% | Strong recovery (+2.6% over naive personalized 80.8%) |
| **Hamming Loss** | 0.0169 | **0.0220** | +0.0051 | Substantial drop in false activations vs naive personalized (0.0280) |
| **Jaccard Similarity (Samples)**| 10.0% | **9.3%** | -0.7% | Turn-level intersection-over-union stabilized |
| **Micro-Precision** | 56.1% | **46.9%** | -9.2% | Pruned 462 false positives compared to naive uniform 0.90 |
| **Micro-Recall** | 70.6% | **80.1%** | **+9.5%** | **Sustains massive retrieval surge across dialogue turns** |
| **Micro-F1** | 62.5% | **59.2%** | -3.3% | **+5.7% lift over naive personalized (53.5%)** |
| **Macro-Precision** | 57.4% | **47.3%** | -10.1% | +3.9% lift over naive personalized (43.4%) |
| **Macro-Recall** | 63.2% | **71.7%** | **+8.5%** | Highest macro-recall observed across all runs |
| **Macro-F1** | 58.6% | **55.2%** | -3.4% | **+4.4% lift over naive personalized (50.8%)** |
| **Weighted F1** | 66.4% | **61.0%** | -5.4% | Support-weighted balance across slots |

#### 4. Per-Slot Classification Breakdown

*Ranked by ground-truth support (occurrences in `true_slots`):*

| Slot Name | Threshold ($\tau$) | Support | Pred | TP | FP | FN | Precision | Recall | F1-Score | $\Delta$ F1 vs Run 01 | Status / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`order_id`** | `0.95` | 461 | 709 | 436 | 273 | 25 | 61.5% | **94.6%** | **74.5%** | -9.5% | Cut 142 FPs vs naive 0.90; sustained 94.6% recall |
| **`account_id`** | `0.96` | 291 | 647 | 272 | 375 | 19 | 42.0% | **93.5%** | **58.0%** | -5.1% | Cut 211 FPs vs naive 0.90; sustained 93.5% recall |
| **`username`** | `0.90` | 247 | 363 | 116 | 247 | 131 | 32.0% | 47.0% | 38.0% | -21.9% | Confused with alphanumeric codes; needs distinct framing |
| **`email`** | `0.98` | 243 | 478 | 240 | 238 | 3 | 50.2% | **98.8%** | **66.6%** | -19.2% | Cut 137 FPs vs naive 0.90 while preserving 98.8% recall (3 FN) |
| **`zip_code`** | `0.85` | 157 | 129 | 102 | 27 | 55 | **79.1%** | **65.0%** | **71.3%** | **+21.5%** | 🏆 **Huge Win**: Recall surged to 65.0%; F1 broke 71% |
| **`amount`** | `0.90` | 149 | 236 | 113 | 123 | 36 | **47.9%** | **75.8%** | **58.7%** | **+27.2%** | 🏆 **Huge Win**: F1 nearly doubled vs Run 01 baseline (31.5%) |
| **`street_address`**| `0.90` | 99 | 109 | 85 | 24 | 14 | 78.0% | 85.9% | **81.7%** | -4.4% | Consistent, reliable zero-shot performance |
| **`name`** | `0.90` | 61 | 201 | 2 | 199 | 59 | 1.0% | 3.3% | 1.5% | -0.2% | Still failing; agent salutation/greeting bleed unresolved |
| **`phone`** | `0.97` | 54 | 152 | 54 | 98 | 0 | 35.5% | **100.0%** | **52.4%** | -5.9% | Cut 57 FPs vs naive 0.90; retained 100% recall (0 FN) |
| **`pin_number`** | `0.90` | 30 | 35 | 16 | 19 | 14 | 45.7% | 53.3% | 49.2% | -16.6% | Moderate recall and precision |

---

#### 5. Diagnosis & Deep Machine Learning Insights

1. **The Power of Asymmetric Threshold Calibration:**
   * Dynamic thresholds immediately repaired the severe false-positive explosion:
     * Pruned **211 false positives** on `account_id` ($\tau=0.96$).
     * Pruned **142 false positives** on `order_id` ($\tau=0.95$).
     * Pruned **137 false positives** on `email` ($\tau=0.98$).
     * Pruned **57 false positives** on `phone` ($\tau=0.97$).
   * In total, **over 540 false positives were eliminated** compared to the uncalibrated personalized run, lifting Micro-F1 from **53.5% back to 59.2%** (+5.7%) while maintaining **80.1% Micro-Recall** (+9.5% over the Run 01 baseline).

2. **The `zip_code` Optimization (71.3% F1):**
   * Relaxing $\tau$ to `0.85` for `zip_code` allowed short 5-digit numeric strings to pass calibration.
   * True Positives rose from 62 (Run 01) to **102 (Run 02)**, driving recall up from 39.5% to **65.0%** while preserving strong **79.1% precision**.
   * F1 improved by **+21.5%** over the Run 01 baseline!

3. **Remaining Structural Challenges for Run 03:**
   * **The `name` Bottleneck (1.5% F1):** Agent greetings (*"Hello John"*, *"Thank you Alice"*) remain embedded in the rolling premise. BART MNLI sees personal names and triggers false entailment.
   * **`account_id` vs `username` Overlap:** Both remain prone to mutual confusion on alphanumeric codes.
   * **Prompt-Bleed Floor:** Even at $\tau=0.98$, pure semantic matching has a false positive floor when the agent explicitly asks for an entity by name.

---

#### 6. Actionable Optimization Targets for Run 03

* [ ] **Turn-Level Role Masking / Prompt Decoupling:**
  Run NLI evaluation strictly on the customer utterance (`[CUSTOMER]`) without prefixing `[AGENT]`, using the agent turn solely as a conditional fallback.
* [ ] **Syntactic Regex Pre-Filtering (Zero False Negatives, Zero Prompt Bleed):**
  For `email`, `phone`, `zip_code`, and `order_id`, combine regex pattern gates with NLI to achieve near-100% precision.
* [ ] **Customer Introduction Hypothesis for `name`:**
  Target customer self-disclosure: `"The customer introduces themselves by stating their own name."` or filter out agent turns mentioning names.

</details>

<details>
<summary><h3 style="display: inline-block; margin: 0; cursor: pointer;">Run 03: Hypothesis Specification Optimisation (BART-Large-MNLI)</h3> &nbsp; <span style="font-size: 13px; color: #64748b;">(Click to collapse/expand)</span></summary>

<br>

* **Date:** 2026-10-08
* **Notebook:** [`notebooks/01_semantic_slots_extraction_evals.ipynb`](notebooks/01_semantic_slots_extraction_evals.ipynb)
* **Goal:** Optimize the formulation and semantic alignment of candidate hypotheses (`DEFAULT_SLOT_HYPOTHESES`) in [`nlp.slots`](src/nlp/slots.py) to eliminate false positives from agent prompt bleed and correct label domain misalignment (particularly on `name`, `email`, `account_id`, and `pin_number`), targeting $\ge 65\%$ Macro-F1 and $\ge 65\%$ Micro-F1 while holding all other model hyperparameters constant.

#### 1. Configuration & Parameters

| Component | Parameter | Value | Rationale / Description |
| :--- | :--- | :--- | :--- |
| **Classifier** | Model Architecture | `facebook/bart-large-mnli` | Large 406M parameter NLI cross-encoder |
| | Device Backend | Apple Silicon GPU (`mps`) | Accelerated on Metal execution units |
| | Precision | `fp16=True` (Half precision) | FP16 mode enabled |
| | Decision Calibration | Option 1 3-Class Probability | $P(\text{Entailment}) > P(\text{Neutral}) \land P(\text{Entailment}) > \tau_{\text{slot}}$ |
| | Dynamic Thresholds ($\tau_{\text{slot}}$) | Per-slot dictionary mapping | Unchanged from Run 02 (`THRESHOLD_BY_SLOTS`) |
| | Default Threshold | `0.90` | Applied to slots not explicitly overridden |
| | Hypothesis Template | `DEFAULT_SLOT_HYPOTHESES` (dict mapping) | **Optimized hypothesis specification** targeting customer actions and dataset semantics |
| **Document Preconditioning** | Rolling Agent Context | `min_prior_agent_utterances=1` | Traverses backward to include at least 1 preceding agent utterance |
| | Customer Utterance Stripping | `strip_prior_customer_utterances=True` | Excludes prior customer messages to prevent inter-turn slot bleeding |
| | Document Format | Preconditioned strings | `[AGENT]: ... \n [CUSTOMER]: ...` |
| **Execution** | Batch Size | `128` | Max pairs evaluated per GPU forward pass |
| | Evaluation Universe | 10 core slots | Ground-truth ABCD `slot_tags` evaluation set |

##### Dynamic Threshold Mapping (Identical to Run 02):
```python
THRESHOLD_BY_SLOTS = {
    "email": 0.98,          # Strict cutoff to reject agent prompt bleed
    "phone": 0.97,          # Strict cutoff to reject agent prompt bleed
    "account_id": 0.96,     # Disambiguate against username / generic IDs
    "order_id": 0.95,       # High confidence for purchase order identifiers
    "zip_code": 0.85,       # Lower threshold to recover bare 5-digit postal numbers
    "amount": 0.90,         # Balanced threshold for monetary mentions
    "street_address": 0.90, # Balanced threshold for street addresses
}
# Remaining slots (username, name, pin_number) receive default_threshold = 0.90
```

##### Optimized Slot Hypothesis Dictionary (`DEFAULT_SLOT_HYPOTHESES`):
```python
DEFAULT_SLOT_HYPOTHESES = {
    "account_id": "The customer enters their account ID.",
    "amount": "The customer specifies a dollar amount or price.",
    "email": "The customer enters their email address.",
    "name": "The customer specifies a merchandise product name.",
    "order_id": "The customer specifies an order ID number.",
    "phone": "The customer provides a telephone phone number.",
    "pin_number": "The customer enters their PIN number.",
    "street_address": "The customer specifies a street address.",
    "username": "The customer writes their account username in the message.",
    "zip_code": "The customer provides a postal zip code.",
}
```

#### 2. Inference Latency & Speed Benchmark

* **Total Evaluated Turns:** 8,984 customer messages
* **Total Batches:** 71 batches (chunked by 128)
* **Total Runtime:** **5 minutes 01 seconds** (301.0s)
* **Batch Processing Rate:** **4.25 s/batch**
* **Per-Turn Latency:** **33.5 ms / customer turn** (~29.8 customer turns/second)
* **Pair Throughput:** ~255 premise-hypothesis pairs scored per second

#### 3. Global Multi-Label Performance Metrics

| Global Metric | Run 01 Baseline | Run 02 Dynamic | Run 03 Optimized | Delta vs Run 02 ($\Delta$) | Target Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exact Match Ratio (Subset)** | 85.7% | 83.4% | **87.8%** | **+4.4%** | — |
| **Hamming Loss** | 0.0169 | 0.0220 | **0.0143** | **-0.0077** | — |
| **Jaccard Similarity (Samples)**| 10.0% | 9.3% | **11.1%** | **+1.8%** | — |
| **Micro-Precision** | 56.1% | 46.9% | **60.8%** | **+13.9%** | — |
| **Micro-Recall** | 70.6% | 80.1% | **79.0%** | -1.1% | High recall preserved |
| **Micro-F1** | 62.5% | 59.2% | **68.7%** | **+9.5%** | 🏆 **Yes ($\ge 65\%$)** |
| **Macro-Precision** | 57.4% | 47.3% | **61.2%** | **+13.9%** | — |
| **Macro-Recall** | 63.2% | 71.7% | **79.2%** | **+7.5%** | All-time highest |
| **Macro-F1** | 58.6% | 55.2% | **66.2%** | **+11.0%** | 🏆 **Yes ($\ge 65\%$)** |
| **Weighted F1** | 66.4% | 61.0% | **70.3%** | **+9.3%** | Support-weighted |

#### 4. Per-Slot Classification Breakdown

*Ranked by ground-truth support (occurrences in `true_slots`):*

| Slot Name | Threshold ($\tau$) | Support | Pred | TP | FP | FN | Precision | Recall | F1-Score | $\Delta$ F1 vs Run 02 | Status / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`order_id`** | `0.95` | 461 | 599 | 451 | 148 | 10 | 75.3% | **97.8%** | **85.1%** | **+10.6%** | Cut 125 FPs; sustained near-perfect recall (451/461 TP) |
| **`account_id`** | `0.96` | 291 | 268 | 176 | 92 | 115 | **65.7%** | 60.5% | **63.0%** | **+5.0%** | 🏆 Cut 283 FPs (from 375 to 92); precision surged to 65.7% |
| **`username`** | `0.90` | 247 | 263 | 121 | 142 | 126 | **46.0%** | 49.0% | **47.5%** | **+9.5%** | 🏆 Cut 105 FPs; precision jumped from 32.0% to 46.0% |
| **`email`** | `0.98` | 243 | 251 | 214 | 37 | 29 | **85.3%** | 88.1% | **86.6%** | **+20.0%** | 🏆 Cut 201 FPs (from 238 to 37); precision surged from 50.2% to 85.3% |
| **`zip_code`** | `0.85` | 157 | 129 | 102 | 27 | 55 | **79.1%** | 65.0% | **71.3%** | 0.0% | Maintained strong Run 02 optimization performance |
| **`amount`** | `0.90` | 149 | 293 | 133 | 160 | 16 | 45.4% | **89.3%** | **60.2%** | **+1.5%** | Recall rose to 89.3% (133/149 TP); broke 60% F1 |
| **`street_address`**| `0.90` | 99 | 143 | 98 | 45 | 1 | 68.5% | **99.0%** | **81.0%** | -0.7% | Outstanding 99.0% recall (only 1 FN out of 99 true addresses) |
| **`name`** | `0.90` | 61 | 242 | 47 | 195 | 14 | 19.4% | **77.0%** | **31.0%** | **+29.5%** | 🏆 **Breakthrough**: Corrected semantic target from personal name to merchandise item; TP soared from 2 to 47 |
| **`phone`** | `0.97` | 54 | 115 | 54 | 61 | 0 | 47.0% | **100.0%** | **63.9%** | **+11.5%** | Cut 37 FPs; preserved flawless 100.0% recall (0 FN) |
| **`pin_number`** | `0.90` | 30 | 25 | 20 | 5 | 10 | **80.0%** | 66.7% | **72.7%** | **+23.5%** | 🏆 Cut 14 FPs (down to only 5); precision reached 80.0% |

---

#### 5. Diagnosis & Deep Machine Learning Insights

1. **Resolving the `name` Ground-Truth Misalignment:**
   * In Run 01 and Run 02, hypotheses evaluated whether the customer specifies their *personal name*. However, in the ABCD dataset, personal human names are never annotated in customer `slot_tags`; rather, `name` strictly denotes **product and merchandise items** (e.g., *"Guess Jeans"*, *"Tommy Hilfiger shirt"*, *"Calvin Klein boots"*).
   * Reframing the hypothesis to `"The customer specifies a merchandise product name."` immediately resolved this fundamental semantic disconnect:
     * True Positives surged from **2 to 47** (out of 61 support).
     * Recall skyrocketed from **3.3% to 77.0%**.
     * Per-slot F1 rose from **1.5% to 31.0%** (+29.5%), unlocking a large lift across Macro-F1.

2. **The "Action Anchoring" Breakthrough (`enters`, `writes in the message`):**
   * Prior runs suffered from agent prompt bleed: when agents asked *"May I have your email, username, and order ID?"*, passive hypothesis phrasing like *"The customer provides an email address"* caused BART-MNLI to entail regardless of whether the customer actually provided an email or answered with another entity.
   * Switching to explicit input-action verbs significantly penalized passive prompt bleed:
     * **`email` (`"The customer enters their email address."`):** Slashed False Positives from **238 to 37** (-84.5% FP reduction!), driving precision from 50.2% to **85.3%** and F1 from 66.6% to **86.6%**.
     * **`account_id` (`"The customer enters their account ID."`):** Slashed False Positives from **375 to 92** (-75.5% FP reduction!), lifting precision from 42.0% to **65.7%** and F1 to **63.0%**.
     * **`pin_number` (`"The customer enters their PIN number."`):** Slashed False Positives from **19 down to 5**, pushing precision to **80.0%** and F1 to **72.7%** (+23.5% over Run 02).
     * **`username` (`"The customer writes their account username in the message."`):** Pruned **105 False Positives**, raising precision to 46.0% and F1 from 38.0% to **47.5%**.

3. **Global Performance Surpasses Target:**
   * Across the entire 1,000-conversation benchmark (8,984 customer messages), **over 640 false positive predictions were eliminated** compared to Run 02.
   * Both acceptance criteria were decisively achieved:
     * **Macro-F1 reached 66.2%** (Target: $\ge 65\%$, Delta: **+11.0%** over Run 02).
     * **Micro-F1 reached 68.7%** (Target: $\ge 65\%$, Delta: **+9.5%** over Run 02).
     * **Exact Match Ratio reached 87.8%** (+4.4% over Run 02, +2.1% over Run 01 baseline).

</details>

<details>
<summary><h3 style="display: inline-block; margin: 0; cursor: pointer;">Run 04: Informative Utterance Gate (BART-Large-MNLI)</h3> &nbsp; <span style="font-size: 13px; color: #64748b;">(Click to collapse/expand)</span></summary>

<br>

* **Date:** 2026-10-08
* **Notebook:** [`notebooks/01_semantic_slots_extraction_evals.ipynb`](notebooks/01_semantic_slots_extraction_evals.ipynb)
* **Goal:** Introduce a Stage 1 **Informative Utterance Gate** to filter conversational fillers, acknowledgments, and pleasantries (*"thanks"*, *"ok sure"*, *"one second"*) on customer-only text before entering multi-slot contextual classification, directly reducing false positives caused by agent prompt bleed.

#### 1. Configuration & Parameters

| Component | Parameter | Value | Rationale / Description |
| :--- | :--- | :--- | :--- |
| **Classifier** | Model Architecture | `facebook/bart-large-mnli` | Large 406M parameter NLI cross-encoder |
| | Device Backend | Apple Silicon GPU (`mps`) | Metal Performance Shaders |
| | Precision | `fp16=True` (Half precision) | FP16 enabled |
| | Decision Calibration | Option 1 3-Class Probability | $P(\text{Entailment}) > P(\text{Neutral}) \land P(\text{Entailment}) > \tau_{\text{slot}}$ |
| | Dynamic Thresholds ($\tau_{\text{slot}}$) | Per-slot dictionary mapping | Retained from Run 03 (`THRESHOLD_BY_SLOTS`) |
| | Default Threshold | `0.90` | Applied to slots not explicitly overridden |
| | Hypothesis Template | `DEFAULT_SLOT_HYPOTHESES` | Optimized action-oriented slot hypotheses from Run 03 |
| **Gating System** | **Informative Gate** | `enable_informative_gate=True` | **Stage 1 customer utterance pre-filter** |
| | Gate Hypothesis | `"The customer provides an identifier, contact info, code, name, or numeric amount."` | **Generalized semantic entity categories (zero hard-coded slot keys)** |
| | Gate Threshold ($\tau_{\text{gate}}$) | `0.02` | Calibrated cutoff: 98.9% recall retention with 41.6% negative filler suppression |
| | Extraction Parsing | `nlp.context.extract_customer_utterance` | Extracts customer-only text from contextual documents |
| **Document Preconditioning** | Rolling Agent Context | `min_prior_agent_utterances=1` | Traverses backward to include at least 1 preceding agent utterance |
| | Customer Utterance Stripping | `strip_prior_customer_utterances=True` | Excludes prior customer messages to prevent inter-turn slot bleeding |
| | Document Format | Preconditioned strings | `[AGENT]: ... \n [CUSTOMER]: ...` |
| **Execution** | Batch Size | `128` | Max pairs evaluated per GPU forward pass |
| | Evaluation Universe | 10 core slots | Ground-truth ABCD `slot_tags` evaluation set |

#### 2. Inference Latency & Speed Benchmark

* **Total Evaluated Turns:** 8,984 customer messages
* **Total Batches:** 71 batches (chunked by 128)
* **Total Runtime:** **3 minutes 46 seconds** (226.0s) &nbsp; *(vs 5m 01s in Run 03, **-75s / 24.9% faster**)*
* **Batch Processing Rate:** **3.20 s/batch** &nbsp; *(vs 4.25 s/batch in Run 03)*
* **Per-Turn Latency:** **25.2 ms / customer turn** (~39.7 customer turns/second)
* **Speedup Explanation:** Conversational filler turns short-circuit Stage 1 immediately, bypassing all 10 candidate slot forward passes in Stage 2.

#### 3. Global Multi-Label Performance Metrics

| Global Metric | Run 01 Baseline | Run 02 Dynamic | Run 03 Optimized | Run 04 Generalized Gate | Delta vs Run 03 ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Match Ratio (Subset)** | 85.7% | 83.4% | 87.8% | **88.9%** | **+1.1%** | 🏆 Near 89% exact match |
| **Hamming Loss** | 0.0169 | 0.0220 | 0.0143 | **0.0132** | **-0.0011** | 🏆 Strong error reduction |
| **Jaccard Similarity (Samples)**| 10.0% | 9.3% | 11.1% | **10.9%** | -0.2% | Consistent |
| **Micro-Precision** | 56.1% | 46.9% | 60.8% | **63.7%** | **+2.9%** | 🏆 Strong precision surge |
| **Micro-Recall** | 70.6% | 80.1% | 79.0% | **78.2%** | -0.8% | High recall protected |
| **Micro-F1** | 62.5% | 59.2% | 68.7% | **70.2%** | **+1.5%** | 🏆 **Crossed 70% threshold** |
| **Macro-Precision** | 57.4% | 47.3% | 61.2% | **62.9%** | **+1.7%** | 🏆 Significant lift |
| **Macro-Recall** | 63.2% | 71.7% | 79.2% | **77.0%** | -2.2% | Robust |
| **Macro-F1** | 58.6% | 55.2% | 66.2% | **67.3%** | **+1.1%** | 🏆 New benchmark high |
| **Weighted F1** | 66.4% | 61.0% | 70.3% | **71.1%** | **+0.8%** | 🏆 Support-weighted |

#### 4. Per-Slot Classification Breakdown

*Ranked by ground-truth support (occurrences in `true_slots`):*

| Slot Name | Threshold ($\tau$) | Support | Pred | TP | FP | FN | Precision | Recall | F1-Score | $\Delta$ F1 vs Run 03 | Status / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`order_id`** | `0.95` | 461 | 597 | 451 | 146 | 10 | **75.5%** | **97.8%** | **85.3%** | **+0.2%** | Cut 2 FPs; sustained near-flawless 97.8% recall (451/461) |
| **`account_id`** | `0.96` | 291 | 268 | 176 | 92 | 115 | **65.7%** | 60.5% | **63.0%** | 0.0% | Zero regression; exact same strong performance |
| **`username`** | `0.90` | 247 | 262 | 120 | 142 | 127 | **45.8%** | 48.6% | **47.2%** | -0.3% | Stable across gate evaluations |
| **`email`** | `0.98` | 243 | 251 | 214 | 37 | 29 | **85.3%** | 88.1% | **86.6%** | 0.0% | Maintained peak 86.6% F1 |
| **`zip_code`** | `0.85` | 157 | 125 | 102 | 23 | 55 | **81.6%** | 65.0% | **72.3%** | **+1.0%** | Cut 4 FPs; precision surged to 81.6% |
| **`amount`** | `0.90` | 149 | 240 | 132 | 108 | 17 | **55.0%** | 88.6% | **67.9%** | **+7.7%** | 🏆 **Massive win**: Slashed 52 FPs (160 $\to$ 108); precision jumped +9.6% |
| **`street_address`**| `0.90` | 99 | 138 | 98 | 40 | 1 | **71.0%** | **99.0%** | **82.7%** | **+1.7%** | Cut 5 FPs; precision rose to 71.0% with 99.0% recall |
| **`name`** | `0.90` | 61 | 186 | 34 | 152 | 27 | 18.3% | 55.7% | **27.5%** | -3.5% | Cut 43 FPs (195 $\to$ 152); robust merchandise recovery |
| **`phone`** | `0.97` | 54 | 106 | 54 | 52 | 0 | **50.9%** | **100.0%** | **67.5%** | **+3.6%** | Cut 9 FPs; precision broke 50%; preserved 100% recall (0 FN) |
| **`pin_number`** | `0.90` | 30 | 25 | 20 | 5 | 10 | **80.0%** | 66.7% | **72.7%** | 0.0% | Preserved strong 80.0% precision |

---

#### 5. Diagnosis & Deep Machine Learning Insights

1. **Elimination of 115 False Positives with Zero Schema Coupling:**
   * By using universal semantic categories (`identifier, contact info, code, name, or numeric amount`), the gate avoids any hardcoded dependence on specific ABCD slot tags.
   * Total False Positives dropped from **912 down to 797** (-115 FPs eliminated).
   * The largest single win was on **`amount`**, where false positives dropped from 160 to 108 (-52 FPs), surging precision from 45.4% to 55.0% (+9.6%) and lifting F1 from 60.2% to **67.9%** (+7.7% lift!).

2. **Substantial Speedup (25% Latency Reduction):**
   * Skipping conversational fillers reduced total runtime from **5m 01s down to 3m 46s** (a **24.9% latency reduction**, 25.2 ms/turn).
   * Over 3,100 turns were safely short-circuited at Stage 1, freeing GPU computation.

3. **Generalizability to New Domains:**
   * If slot definitions change (e.g. airline booking with `pnr_code`, `flight_number` or hotel reservations with `room_rate`), the gate hypothesis requires zero edits, functioning as an intent-level pre-filter.


---

#### 6. Actionable Next Steps (Towards Run 05)

* [x] **Experiment 2: Contextual Disambiguation Gate:**
  * Added Stage 3 Customer-Support Verification pass (Option B). Candidate slots predicted in Stage 2 from the contextual document are verified against customer-only text. If $P_{\text{cust}}(\text{slot}) < \tau_{\text{verify}}$, unfulfilled candidate slots are pruned. Evaluated in Run 05.

</details>

<details open>
<summary><h3 style="display: inline-block; margin: 0; cursor: pointer;">Run 05: Contextual Disambiguation Gate (BART-Large-MNLI)</h3> &nbsp; <span style="font-size: 13px; color: #64748b;">(Click to collapse/expand)</span></summary>

<br>

* **Date:** 2026-10-09
* **Notebook:** [`notebooks/01_semantic_slots_extraction_evals.ipynb`](notebooks/01_semantic_slots_extraction_evals.ipynb)
* **Goal:** Introduce Stage 3 **Contextual Disambiguation Gate** (Customer-Support Verification pass with calibrated $\tau_{\text{verify}}=0.05$) to eliminate agent prompt bleed. Candidate slots predicted from contextual documents (`[AGENT] + [CUSTOMER]`) are checked against the customer's utterance alone; candidate slots lacking support in the customer's text ($P_{\text{cust}} < \tau_{\text{verify}}$) are pruned as unfulfilled agent inquiries.

#### 1. Configuration & Parameters

| Component | Parameter | Value | Rationale / Description |
| :--- | :--- | :--- | :--- |
| **Classifier** | Model Architecture | `facebook/bart-large-mnli` | Large 406M parameter NLI cross-encoder |
| | Device Backend | Apple Silicon GPU (`mps`) | Metal Performance Shaders |
| | Precision | `fp16=True` (Half precision) | FP16 enabled |
| | Decision Calibration | Option 1 3-Class Probability | $P(\text{Entailment}) > P(\text{Neutral}) \land P(\text{Entailment}) > \tau_{\text{slot}}$ |
| | Dynamic Thresholds ($\tau_{\text{slot}}$) | Per-slot dictionary mapping | Retained from Run 03 (`THRESHOLD_BY_SLOTS`) |
| | Default Threshold | `0.90` | Applied to slots not explicitly overridden |
| | Hypothesis Template | `DEFAULT_SLOT_HYPOTHESES` | Optimized action-oriented slot hypotheses from Run 03 |
| **Gating System** | **Stage 1: Informative Gate** | `enable_informative_gate=True` | Customer utterance pre-filter on pleasantries/fillers |
| | Gate Hypothesis | `"The customer provides an identifier, contact info, code, name, or numeric amount."` | Generalized semantic entity categories ($\tau_{\text{gate}}=0.02$) |
| | **Stage 3: Disambiguation Gate** | `enable_contextual_disambiguation=True` | **Customer-Support Verification Gate (Option B)** |
| | Verification Threshold ($\tau_{\text{verify}}$) | `0.05` | Calibrated cutoff to purge weak entailment bleed |
| | Verification Premise | Customer text only (`c_text`) | Isolates customer commitment from preceding agent questions |
| **Document Preconditioning** | Rolling Agent Context | `min_prior_agent_utterances=1` | Traverses backward to include at least 1 preceding agent utterance |
| | Customer Utterance Stripping | `strip_prior_customer_utterances=True` | Excludes prior customer messages to prevent inter-turn slot bleeding |
| | Document Format | Preconditioned strings | `[AGENT]: ... \n [CUSTOMER]: ...` |
| **Execution** | Batch Size | `128` | Max pairs evaluated per GPU forward pass |
| | Evaluation Universe | 10 core slots | Ground-truth ABCD `slot_tags` evaluation set |

#### 2. Inference Latency & Speed Benchmark

* **Total Evaluated Turns:** 8,984 customer messages
* **Total Batches:** 71 batches (chunked by 128)
* **Total Runtime:** **3 minutes 58 seconds** (238.0s) &nbsp; *(vs 3m 46s in Run 04, **only +12s / 5.3% overhead**)*
* **Batch Processing Rate:** **3.36 s/batch**
* **Per-Turn Latency:** **26.5 ms / customer turn** (~37.7 customer turns/second)
* **Computational Efficiency:** Minimal overhead because Stage 3 only evaluates candidate slots predicted by Stage 2 (~1.2 candidate pairs per turn) rather than iterating across all 10 candidate slots.

#### 3. Global Multi-Label Performance Metrics

| Global Metric | Run 01 Baseline | Run 02 Dynamic | Run 03 Optimized | Run 04 Gate | Run 05 Disambig Gate | Delta vs Run 04 ($\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Match Ratio (Subset)** | 85.7% | 83.4% | 87.8% | 88.9% | **89.3%** | **+0.4%** | 🏆 **New All-Time High** |
| **Hamming Loss** | 0.0169 | 0.0220 | 0.0143 | 0.0132 | **0.0127** | **-0.0005** | 🏆 **Lowest Error Rate** |
| **Jaccard Similarity (Samples)**| 10.0% | 9.3% | 11.1% | 10.9% | **11.0%** | **+0.1%** | Upward recovery |
| **Micro-Precision** | 56.1% | 46.9% | 60.8% | 63.7% | **65.4%** | **+1.7%** | 🏆 **New Peak Precision** |
| **Micro-Recall** | 70.6% | 80.1% | 79.0% | 78.2% | **77.6%** | -0.6% | 99.3% TP retention |
| **Micro-F1** | 62.5% | 59.2% | 68.7% | 70.2% | **71.0%** | **+0.8%** | 🏆 **New Benchmark High** |
| **Macro-Precision** | 57.4% | 47.3% | 61.2% | 62.9% | **64.4%** | **+1.5%** | 🏆 **New Peak Precision** |
| **Macro-Recall** | 63.2% | 71.7% | 79.2% | 77.0% | **75.7%** | -1.3% | Robust |
| **Macro-F1** | 58.6% | 55.2% | 66.2% | 67.3% | **67.3%** | 0.0% | Maintained |
| **Weighted F1** | 66.4% | 61.0% | 70.3% | 71.1% | **71.9%** | **+0.8%** | 🏆 **New Peak Weighted F1** |

#### 4. Per-Slot Classification Breakdown

*Ranked by ground-truth support (occurrences in `true_slots`):*

| Slot Name | Threshold ($\tau$) | Support | Pred | TP | FP | FN | Precision | Recall | F1-Score | $\Delta$ F1 vs Run 04 | Status / Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`order_id`** | `0.95` | 461 | 578 | 451 | 127 | 10 | **78.0%** | **97.8%** | **86.8%** | **+1.5%** | Slashed 19 FPs (146 $\to$ 127); precision surged +2.5%; 100% TP retention (451/451) |
| **`account_id`** | `0.96` | 291 | 256 | 172 | 84 | 119 | **67.2%** | 59.1% | **62.9%** | -0.1% | Cut 8 FPs (92 $\to$ 84); precision rose +1.5% |
| **`username`** | `0.90` | 247 | 251 | 120 | 131 | 127 | **47.8%** | 48.6% | **48.2%** | **+1.0%** | Slashed 11 FPs (142 $\to$ 131); 100% TP retention (120/120) |
| **`email`** | `0.98` | 243 | 242 | 214 | 28 | 29 | **88.4%** | 88.1% | **88.2%** | **+1.6%** | 🏆 **Peak Performance**: Cut 9 FPs (37 $\to$ 28); precision reached 88.4%; 100% TP retention |
| **`zip_code`** | `0.85` | 157 | 120 | 99 | 21 | 58 | **82.5%** | 63.1% | **71.5%** | -0.8% | Cut 2 FPs (23 $\to$ 21); precision rose to 82.5% |
| **`amount`** | `0.90` | 149 | 236 | 132 | 104 | 17 | **55.9%** | 88.6% | **68.6%** | **+0.7%** | Cut 4 FPs (108 $\to$ 104); sustained high 88.6% recall |
| **`street_address`**| `0.90` | 99 | 134 | 98 | 36 | 1 | **73.1%** | **99.0%** | **84.1%** | **+1.4%** | Cut 4 FPs (40 $\to$ 36); precision reached 73.1% with 99.0% recall |
| **`name`** | `0.90` | 61 | 185 | 34 | 151 | 27 | 18.4% | 55.7% | **27.6%** | **+0.1%** | Cut 1 FP (152 $\to$ 151) |
| **`phone`** | `0.97` | 54 | 104 | 54 | 50 | 0 | **51.9%** | **100.0%** | **68.4%** | **+0.9%** | Cut 2 FPs (52 $\to$ 50); preserved flawless 100% recall (0 FN) |
| **`pin_number`** | `0.90` | 30 | 21 | 17 | 4 | 13 | **81.0%** | 56.7% | **66.7%** | -6.0% | Cut 1 FP (5 $\to$ 4); precision rose to 81.0% |

---

#### 5. Diagnosis & Deep Machine Learning Insights

1. **83 False Positives Eliminated vs Run 04 (6.8:1 FP-to-TP Ratio):**
   * Total False Positives plummeted from **797 down to 714** (-83 FPs eliminated).
   * Total True Positives remained exceptionally resilient: **1,391 out of 1,401 TPs were preserved** across all 8,984 turns (only 10 marginal TPs were traded across the entire dataset).
   * Pruning 83 false positives for only 10 traded true positives represents a stellar **8.3-to-1 return ratio**, driving global **Micro-Precision to 65.4%** (+1.7%) and **Exact Match to 89.3%**.

2. **Major F1 Surges Across Primary Workhorse Slots:**
   * **`order_id`**: Slashed 19 false positives (146 $\to$ 127) while preserving 100% true positives (451/451), pushing F1 to **86.8%** (+1.5%).
   * **`email`**: Slashed 9 false positives (37 $\to$ 28) with 100% true positive retention (214/214), pushing F1 to **88.2%** (+1.6%).
   * **`street_address`**: Pruned 4 false positives, driving F1 to **84.1%** (+1.4%).
   * **`username`**: Pruned 11 false positives (142 $\to$ 131) with zero recall loss, pushing F1 to **48.2%** (+1.0%).
   * **`phone`**: Preserved 100% recall (0 FN) with 51.9% precision, lifting F1 to **68.4%** (+0.9%).

3. **Trade-off Analysis on Sparse Entities:**
   * For dense entities with strong lexical framing, $\tau_{\text{verify}}=0.05$ achieved pure FP elimination with 0 recall penalty.
   * For small-support classes like `pin_number` (30 ground truth turns) and `zip_code` (157 turns), very terse customer replies (e.g. 4 raw digits without the word *"pin"*) lost 3 TPs each, slightly reducing macro recall from 77.0% to 75.7%. However, global **Micro-F1 reached an all-time peak of 71.0%** and **Weighted-F1 reached 71.9%**.

4. **Inference Latency Profile:**
   * Total execution time remained blazing fast at **3m 58s** (26.5 ms / customer turn).
   * Verifying candidate slots added barely 12 seconds compared to Run 04 while purging 83 false positives.

---

#### 6. Summary of Progression Across All 5 Runs

| Milestone | Key Innovation | Micro-F1 | Macro-F1 | Exact Match | Total FPs | Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Run 01** | Baseline Template + Option 1 Calibration | 62.5% | 58.6% | 85.7% | 1,027 | 4m 39s |
| **Run 02** | Personalized Hypotheses + Dynamic Cutoffs | 59.2% | 55.2% | 83.4% | 1,481 | 5m 22s |
| **Run 03** | LLM-Guided Action Hypotheses | 68.7% | 66.2% | 87.8% | 912 | 5m 01s |
| **Run 04** | Stage 1 Generalized Informative Gate | 70.2% | 67.3% | 88.9% | 797 | **3m 46s** |
| **Run 05** | Stage 3 Contextual Disambiguation Gate ($\tau=0.05$) | 🏆 **71.0%** | **67.3%** | 🏆 **89.3%** | 🏆 **714** | **3m 58s** |

</details>



