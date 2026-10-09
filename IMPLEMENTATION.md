# Implementation Guide: Conversational Analytics on ABCD

**Project:** Post-Hoc Conversation State Tracking (CST/DST) & Action State Tracking (AST) Analytics  
**Reference Document:** [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md)  
**Package:** `abcd_helper` (installed in editable mode via `pyproject.toml`)  
**Status:** **Phase 1 Complete** (Data Ingestion, Target Labeling, Queryable Data Structures, EDA Environment)

---

## 1. Executive Summary

This project establishes a post-mortem tabular modeling pipeline on the **Action-Based Conversations Dataset (ABCD)** ([Chen et al., 2021](https://arxiv.org/abs/2104.00783)) to discover what conversational friction points, slot-filling bottlenecks, and agent guideline deviations predict whether a customer conversation resolves successfully or fails/escalates.

Rather than training a real-time conversational agent, this environment is architected for **post-hoc exploratory data analysis, feature engineering, and explainable tabular modeling (XGBoost/LightGBM + SHAP)**.

---

## 2. Directory Structure

```text
nlp-dialogue-state-tracking/
├── ANALYSIS_PLAN.md              # Project analytical blueprint and methodology
├── EVALS_SLOT_EXTRACTION.md      # Experiment log and leaderboard for DST slot extraction
├── IMPLEMENTATION.md             # This implementation and architecture guide
├── pyproject.toml                # Project metadata, dependencies, and setuptools package config
├── uv.lock                       # Deterministic dependency lockfile
├── .gitignore                    # Ignores large raw/parquet data, cache, and checkpoints
├── data/
│   ├── raw/                      # Downloaded source files from ASAPP research repository
│   │   ├── abcd_v1.1.json.gz     # Complete corpus (10,042 multi-turn conversations)
│   │   ├── ontology.json         # 55 intents, 30 discrete actions, slot definitions
│   │   ├── guidelines.json       # Human-readable SOP flowcharts
│   │   ├── kb.json               # Canonical action sequences for all 55 intents
│   │   ├── utterances.json       # Agent response candidate pool
│   │   └── abcd_sample.json      # 3-conversation sample
│   └── processed/                # Normalized, high-performance queryable files
│       ├── conversations.parquet # 1 row per conversation (10,042 rows)
│       ├── turns.parquet         # 1 row per turn/action (220,983 rows)
│       ├── subflows.parquet      # Canonical SOP action sequence per intent (55 rows)
│       ├── actions.parquet       # Discrete system actions and slot schemas (30 rows)
│       ├── intent_schema.parquet # Normative ground-truth slot requirements in CNF (55 rows)
│       └── abcd.db               # Indexed SQLite database for SQL queries
├── notebooks/
│   ├── 00_conversation_data_exploration.ipynb        # Starter EDA and interactive viewer notebook
│   └── 01_semantic_slots_extraction_evals.ipynb      # Phase 2 DST extraction benchmark notebook
├── scripts/
│   ├── run_abcd_ingestion.py     # Standalone CLI runner for ABCD ETL data ingestion
│   └── run_slot_extraction.py    # Batched GPU runner for two-stage slot extraction
├── src/
│   ├── abcd_helper/              # Core dataset loading and queryable schema library
│   │   ├── __init__.py           # Exports public loading API and Conversation viewer
│   │   ├── config.py             # File paths and remote URLs
│   │   ├── ingestion.py          # Pure ETL transformation functions
│   │   ├── intent_schema.py      # 55 SOP intent slot specifications (CNF mandatory & action slots)
│   │   ├── loader.py             # High-level caching query API
│   │   ├── resolution.py         # Ground-truth target variable Y logic
│   │   └── viewer.py             # Jupyter HTML chat bubble & action renderer
│   ├── nlp/                      # Dialogue State Tracking NLP engine
│   │   ├── __init__.py           # Package exports
│   │   ├── context.py            # Rolling dialogue turn context extraction (DialogueTurnContext)
│   │   └── slots.py              # TwoStageSlotExtractor, RegexMatcher, Calibrated SemanticMatcher
│   ├── data_science/             # Data science and ML evaluation utilities
│   │   ├── __init__.py           # Package exports
│   │   └── evaluation.py         # Generic evaluate_multilabel() and MultiLabelEvaluationResult
│   └── utils/                    # Shared presentation utilities
│       ├── __init__.py           # Package exports
│       └── display.py            # Dataset-agnostic ConversationView with dark/light themes
└── tests/
    ├── test_data_loader.py       # Ingestion, schema, CNF, and resolution logic tests (10 tests)
    ├── test_display.py           # Interactive conversation viewer & theme tests (3 tests)
    ├── test_evaluation.py        # Generic multi-label evaluation metric tests (3 tests)
    ├── test_nlp_context.py       # Rolling dialogue context window extractor tests (9 tests)
    └── test_slots.py             # Regex, 3-class NLI calibration, and two-stage routing tests (3 tests)
```

---

## 3. Data Tables & Schema Specification

The raw nested JSON was ingested, validated, and normalized into 5 relational tables in `data/processed/`:

### 3.1. `conversations.parquet` (Conversation Level)
* **Granularity:** 1 row per conversation
* **Total Rows:** 10,042 (Train: 8,034 | Dev: 1,004 | Test: 1,004)
* **File Size:** ~609 KB (Parquet snappy compressed)

| Column | Type | Description |
| :--- | :--- | :--- |
| `convo_id` | `int64` | Unique global conversation identifier |
| `split` | `string` | Data split: `'train'`, `'dev'`, or `'test'` |
| `flow` | `string` | Top-level customer domain (10 distinct flows, e.g. `product_defect`, `order_issue`) |
| `intent` | `string` | Canonical customer intent (55 distinct subflows, e.g. `return_size`, `refund_initiate`) |
| `scenario_subflow` | `string` | Granular scenario prompt template identifier |
| `num_turns` | `int64` | Total turn count in conversation |
| `num_customer_turns` | `int64` | Count of customer utterances |
| `num_agent_turns` | `int64` | Count of agent utterances |
| `num_action_turns` | `int64` | Count of discrete system action executions |
| `customer_name` | `string` | Fictional customer name from scenario |
| `username` | `string` | Customer username |
| `email` | `string` | Customer email |
| `phone` | `string` | Customer phone number |
| `member_level` | `string` | Loyalty tier: `bronze`, `silver`, `gold` |
| `account_id` | `string` | Customer account identifier |
| `order_id` | `string` | Associated order identifier |
| `order_amount` | `float64`| Total order dollar value (derived from product amounts) |
| `order_products` | `string` | Names of items involved in conversation |
| `order_city` / `state` / `zip` | `string` | Shipping address attributes |
| `actions_taken` | `list[str]`| Observed sequence of system action buttons executed by agent |
| `num_actions` | `int64` | Total count of system actions fired |
| `canonical_actions` | `list[str]`| Expected guideline SOP sequence from `kb.json` |
| `canonical_terminal` | `string` | Designated canonical resolving action for this intent |
| `actual_terminal` | `string` | Last action executed in the observed conversation |
| `has_escalation` | `bool` | `True` if `notify-team` was executed at any turn |
| `reached_canonical_terminal` | `bool` | `True` if `actual_terminal == canonical_terminal` |
| **`resolved`** | **`int64`** | **Binary Target ($Y \in \{0, 1\}$)**: 1 if positive resolution, 0 otherwise |

### 3.2. `turns.parquet` (Turn & Utterance Level)
* **Granularity:** 1 row per turn (conversation utterance or system action execution)
* **Total Rows:** 220,983 (Agent: 95,129 | Customer: 89,372 | Action: 36,482)
* **File Size:** ~10 MB (Parquet snappy compressed)

| Column | Type | Description |
| :--- | :--- | :--- |
| `convo_id` | `int64` | Foreign key matching `conversations.convo_id` |
| `turn_id` | `int64` | 0-indexed sequential turn order within conversation |
| `turn_count` | `int64` | 1-indexed turn counter matching ABCD format |
| `split` | `string` | `'train'`, `'dev'`, or `'test'` |
| `speaker` | `string` | `'customer'`, `'agent'`, or `'action'` |
| `text` | `string` | Raw natural language utterance as spoken |
| `text_delexed` | `string` | Delexicalized text with slot tokens (`<order_id>`, `<email>`, `<username>`) |
| `flow` | `string` | Domain flow |
| `intent` | `string` | Canonical subflow intent |
| `next_step` | `string` | `'retrieve_utterance'`, `'take_action'`, or `None` |
| `action_name` | `string` | Action name if `speaker == 'action'`, else `None` (30 discrete actions) |
| `action_values` | `string (JSON)`| Slot values passed as arguments into the action |
| `slot_tags` | `string` | Comma-separated list of slot entities present in delexed text |
| `char_len` | `int64` | Character length of utterance |
| `word_len` | `int64` | Word count of utterance |

### 3.3. `subflows.parquet` (SOP Metadata)
* **Granularity:** 1 row per canonical intent (55 rows)
* Contains `intent`, `flow`, `canonical_actions` (list), `num_canonical_actions`, and `canonical_terminal`.

### 3.4. `actions.parquet` (Action Ontology)
* **Granularity:** 1 row per discrete action (30 rows)
* Contains `action_name`, `action_category` (`kb_query`, `interaction`, `faq_policy`), and `required_slots`.

### 3.5. `intent_schema.parquet` (Normative Intent Slot Specification)
* **Granularity:** 1 row per canonical intent (55 rows across 10 flows)
* **Key Columns:** `flow`, `intent`, `canonical_actions`, `mandatory_slots` (`list[list[str]]`), `num_mandatory_slots`, `agent_action_slots` (`list[str]`), `all_slots` (`list[str]`), `sop_summary`.

### 3.6. More about `intent_schema.parquet`

#### 3.6.1. Methodological Design: Normative SOP Inference vs. Alternatives

In Conversation State Tracking (CST / DST), determining the ground-truth slot requirements for an intent is a foundational methodological decision. In the raw ABCD dataset, slot requirements are defined only at the atomic *action level* in `ontology.json` (e.g. `validate-purchase` requires `username`, `email`, `order_id`), but the benchmark provides no explicit relational table mapping high-level customer *intents* to required customer slots. 

For this reason, an LLM-powered inferential approach was chosen to define what customer information *ought* to be collected for each intent. **Gemini 3.8 Flash - medium** was used as the inference agent to synthesize the Standard Operating Procedures (SOPs) governing each intent across three authoritative dataset files:
1. `guidelines.json`: Human-written natural language SOP flowcharts and instructions describing prerequisite steps, authentication rules, and branching policies per subflow.
2. `ontology.json`: Formal schema defining the 30 discrete system actions and their required input parameters.
3. `kb.json`: Canonical action trajectories specifying the intended sequence of steps for all 55 intents.

Inference agent is asked to directly make changes to the source code `intent_schema.py`. The variable `INTENT_SCHEMA_DATA` is the literal parsing outputs from the LLM, which is then transformed and saved as `intent_schema.parquet`.

**Alternative Approach Considered:** An alternative approach would be empirical frequency aggregation from actual conversation logs to extract slot occurrence distributions per intent and apply a threshold (e.g. marking any slot present in $\ge 50\%$ of conversations for an intent as mandatory). However, given the availability of gold-standard SOP metadata, potential agent non-compliance in historical logs, and sample sparsity across rare intents, normative inference was chosen.

---

#### 3.6.2. Conjunctive Normal Form (CNF) for Customer Slots

Customer-provided requirements are represented in **Conjunctive Normal Form (CNF)** as `list[list[str]]`:
* **Outer List (Logical AND):** Every clause (inner list) represents an essential requirement that must be satisfied.
* **Inner List (Logical OR):** Represents acceptable alternatives or disjunctions (e.g. `["customer_name", "account_id"]` for account pull-up, or `["street_address", "full_address"]` for shipping labels).

```python
# Example: product_defect -> refund_initiate
mandatory_slots = [
    ["customer_name", "account_id"],   # Clause 1: Account identification (Name OR Account ID)
    ["username"],                       # Clause 2: Purchase validation credential
    ["email"],                          # Clause 3: Purchase validation contact
    ["order_id"],                       # Clause 4: Target transaction identifier
    ["full_address", "street_address"], # Clause 5: Return shipping destination
]
```

**Advantages of CNF Representation for DST Metrics:**
* **Exact Denominator:** $|\text{mandatory\_slots}|$ is the exact count of independent requirements, avoiding denominator inflation that would occur if alternative slots were flattened into a simple list.
* **Boolean Clause Satisfaction:**
  $$\text{clause\_satisfied}_i = \bigvee_{s \in C_i} \mathbb{I}(s \in \text{extracted\_slots})$$
* **Upfront Slot Density & Completeness:**
  $$\text{upfront\_slot\_density} = \frac{\sum_{i=1}^{M} \text{clause\_satisfied}_i}{M}, \quad \text{where } M = |\text{mandatory\_slots}|$$
* **Evaluation Code Snippet:**
  ```python
  satisfied = [any(s in extracted for s in group) for group in mandatory_slots]
  is_complete = all(satisfied)
  density = sum(satisfied) / len(mandatory_slots) if mandatory_slots else 1.0
  ```

---

#### 3.6.3. Slot Taxonomy & Role Separation

To avoid confounding customer information collection with agent decision-making, the schema strictly categorizes slot types:

| Slot Type | Data Structure | Role in Conversation | Examples |
| :--- | :--- | :--- | :--- |
| **`mandatory_slots`** | `list[list[str]]` (CNF) | Informational constraints customer *must* provide to proceed with SOP | `[["customer_name", "account_id"], ["order_id"]]` |
| **`agent_action_slots`** | `list[str]` (flat list) | Discrete operational parameters configured by agent when triggering system actions | `["order_slotval", "shipping_option", "membership_level"]` |
| **`all_slots`** | `list[str]` (flat list) | Complete ordered union of all parameters accepted by the intent's canonical actions | $\bigcup_{a \in A_{\text{canonical}}} \text{required\_slots}(a)$ |

### 3.7. `abcd.db` (SQLite Database)
* Replicates all 5 tables (`conversations`, `turns`, `subflows`, `actions`, `intent_schema`) with relational B-tree indexes on `convo_id`, `split`, `intent`, `(flow, intent)`, `speaker`, and `resolved` for high-performance ad-hoc SQL queries.

---

## 4. Target Variable ($Y$) Grounding

Per Section 2 of `ANALYSIS_PLAN.md`, resolution is determined objectively against ABCD's guideline ontology:

$$Y = \begin{cases} 1 & \text{if agent executes the canonical terminal action for intent and finishes cleanly} \\ 0 & \text{if conversation escalates, routes to fallback, or terminates prematurely} \end{cases}$$

### Implementation Logic (`src/abcd_helper/resolution.py`):
1. For each conversation, retrieve the expected canonical SOP sequence $A_{\text{guideline}} = (a_1, a_2, \dots, a_k)$ from `kb.json`. The canonical terminal action is $a_k$.
2. Identify the observed action sequence $A_{\text{observed}} = (o_1, o_2, \dots, o_m)$. The actual terminal action is $o_m$.
3. Check for escalation actions: `has_escalation = any(act == 'notify-team' for act in A_observed)`.
4. Check terminal compliance: `reached_canonical_terminal = (o_m == a_k)`.
5. Target label:
   $$\text{resolved} = 1 \iff (\text{reached\_canonical\_terminal} \land \neg\text{has\_escalation})$$

### Corpus Distribution:
* **Positive Resolution ($Y = 1$):** 5,432 conversations (**54.1%**)
* **Negative Resolution ($Y = 0$):** 4,610 conversations (**45.9%**)
* **Escalations (`notify-team`):** 567 conversations (**5.6%**)
* This produces an evenly balanced target suitable for standard binary classification without requiring aggressive synthetic oversampling.

---

## 5. Python API Usage Guide (`abcd_helper`)

The package is installed in editable mode. Import as `import abcd_helper as abcd`:

```python
import abcd_helper as abcd

# 1. Load conversation-level DataFrame (with in-memory cache)
df_convos = abcd.load_conversations()
df_train = abcd.load_conversations(split="train")
df_failed = abcd.load_conversations(resolved=0)

# 2. Load turn-level DataFrame
df_turns = abcd.load_turns()  # full 220k turns
turns_3592 = abcd.load_turns(convo_id=3592)  # fast slice for one conversation

# 3. Load metadata, guidelines, and intent slot schemas
df_subflows = abcd.load_subflows()
df_actions = abcd.load_actions()
guidelines = abcd.load_guidelines()
kb = abcd.load_kb()

# 4. Load Normative Intent Slot Schemas (CNF mandatory slots & action parameters)
df_all_intents = abcd.load_intent_slots()  # full 55 rows
refund_schema = abcd.load_intent_slots(intent="refund_initiate")  # returns dict
# refund_schema['mandatory_slots'] -> [['customer_name', 'account_id'], ['username'], ['email'], ['order_id'], ['full_address', 'street_address']]
# refund_schema['agent_action_slots'] -> ['reason_slotval', 'refund_target', 'details_slotval', 'amount']
# refund_schema['all_slots'] -> union of canonical action parameters

# Filter by flow (returns 4 subflows)
faq_schema = abcd.load_intent_slots(flow="single_item_query")

# 5. Interactive Conversation Inspection in Jupyter Notebook
convo = abcd.get_conversation(3592)
convo  # Automatically renders color-coded HTML chat bubbles & action pills in cell output!

# 6. CLI transcript printing
convo.print_transcript()
```

---

## 6. Dialogue State Tracking (DST) & Two-Stage Slot Extraction (Phase 2)

Phase 2 engineers conversation-level tabular features ($X_{\text{DST}}$) by tracking when entity slots are disclosed across multi-turn customer dialogues.

### 6.1. Ground-Truth Principles & Zero-Cheating Constraint
In real-world post-mortem conversational intelligence, models never have access to scenario metadata (`convo.meta`) or internal simulator states during runtime. Slot extraction must operate **strictly on the observed conversational transcript**:
* **Input Documents:** Contextualized dialogue documents (`DialogueTurnContext`).
* **Rolling Agent Context Window:** Customer responses often contain short answers, isolated IDs, or disambiguations (e.g., `"cz303929"`, `"90210"`, `"yes, overnight"`). The context window rolls backwards until capturing up to **2 prior agent utterances**, bridging sub-dialogues without cross-conversation bleeding.
* **Pure Transcript Inference:** No access to customer scenario ground truth fields during inference.

### 6.2. Semantic Extraction Architecture (`nlp.slots`)
The slot classification pipeline focuses on zero-shot / one-shot Multi-Genre Natural Language Inference (MNLI):
1. **Calibrated Semantic Matcher (`SemanticMatcher`):**
   * Operates as a pure NLP sequence classifier on raw text strings (`str` and `list[str]`), decoupled from domain data structures (`DialogueTurnContext`).
   * Evaluates slots via NLI hypothesis testing using Option 1 3-class probability calibration:
     $$p_{\text{entailment}} > p_{\text{neutral}} \quad \land \quad p_{\text{entailment}} > \tau$$
   * Returns active slot names directly (`list[str]` for single utterances, `list[list[str]]` for batches), or continuous probability scores when `return_scores=True`.
2. **External Data Preconditioning (Separation of Concerns):**
   * Text preparation (such as rolling context formatting, stripping prior customer turns, or custom prompt wrapping) is executed explicitly in the caller/pipeline before passing strings to the model:
     ```python
     # External preconditioning in data pipeline:
     doc_texts = [doc.get_full_document(strip_prior_customer_utterances=True) for doc in documents]
     inferred = matcher.extract_batch(doc_texts, available_slots)
     ```
   * Eliminates tight coupling between the model and dialogue data models, allowing seamless reproducibility, interactive testing, and prompt experimentation.

### 6.3. Option 1: 3-Class NLI Probability Calibration
Standard Hugging Face zero-shot pipelines (`multi_label=True`) calculate scores via $\text{softmax}([\text{logit}_{\text{contradiction}}, \text{logit}_{\text{entailment}}])$, **completely discarding the neutral logit**. For irrelevant utterances where the model is 86% neutral, 11% entailment, and 3% contradiction, dropping neutral causes entailment to artificially explode to $\frac{11}{11+3} = 78\%$, yielding ~34,800 false positives (6.1% precision).

We resolved this by implementing **Option 1 3-Class Probability Calibration**:
1. Compute full 3-class softmax:
   $$[p_{\text{contradiction}}, p_{\text{neutral}}, p_{\text{entailment}}] = \text{Softmax}(\mathbf{z})$$
2. Slot activation decision rule:
   $$\text{Slot is Active} \iff p_{\text{entailment}} > p_{\text{neutral}} \quad \land \quad p_{\text{entailment}} > \tau$$
   where $\tau$ is the confidence threshold (default $0.5$).
3. Hypothesis template framing: `"The customer specifies their {}."` to prevent the model from misinterpreting arbitrary conversational statements as entity disclosures.

### 6.4. Generic Multi-Label Evaluation Engine (`data_science.evaluation`)
Provides comprehensive multi-label evaluation utilities:
* `evaluate_multilabel(data, true_col, pred_col)`: Computes micro/macro/weighted/samples Precision, Recall, F1-Score, Exact Match Ratio (Subset Accuracy), Hamming Loss, and Jaccard similarity.
* Produces per-label breakdown ranking classes by support, with visual F1 progress bars.
* `MultiLabelEvaluationResult`: Implements interactive `_repr_html_()` rendering with responsive light/dark theme adaptation for JupyterLab and VS Code.

### 6.5. Empirical Benchmarking (Before vs. After Calibration)
Evaluated across customer turns against ABCD human-annotated `slot_tags`:

| Slot | Old Uncalibrated Precision | Two-Stage Routed Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **`order_id`** | ~6.0% | **100.0%** | **100.0%** | **1.000** |
| **`zip_code`** | ~8.0% | **100.0%** | **100.0%** | **1.000** |
| **`street_address`**| ~5.0% | **100.0%** | **100.0%** | **1.000** |
| **`email`** | 6.1% | **66.7% – 100.0%\*** | **100.0%** | **0.800 – 1.000** |
| **`account_id`** | 16.7% | **44.4%** | **100.0%** | **0.615** |
| **Global Micro Recall** | 99.8% | **90.5% – 97.7%** | — | — |

*\*Note: The single false positive on email was Turn 15 of Convo 10295 (`"Asanders asanders@gmail.com"`), which is an actual email that human annotators omitted from `slot_tags` in ABCD.*

---

## 7. Verification & Automated Tests

All functionality across data ingestion, NLP context framing, slot extraction, display, and evaluation is validated by **32 automated pytest unit tests**:

* [`tests/test_data_loader.py`](tests/test_data_loader.py) (10 tests):
  * Validates 10,042 conversation rows, split distribution, turn sequences, 55 intents, 30 actions, CNF boolean satisfaction, and target $Y$ resolution.
* [`tests/test_nlp_context.py`](tests/test_nlp_context.py) (9 tests):
  * Validates rolling context backward traversal, agent utterance counting, turn ordering, and conversation boundary isolation.
* [`tests/test_slots.py`](tests/test_slots.py) (5 tests):
  * Validates `SemanticMatcher` with raw `str` and `list[str]`, Option 1 3-class probability calibration, `return_scores` behavior, and empty utterance handling.
* [`tests/test_evaluation.py`](tests/test_evaluation.py) (5 tests):
  * Validates multi-label classification metrics, edge cases, `drop_zero_support=True` filtering, and HTML dashboard formatting.
* [`tests/test_display.py`](tests/test_display.py) (3 tests):
  * Validates `ConversationView` HTML rendering, turn badges, speaker bubbles, and ABCD conversation viewer integration.

Run all tests anytime with:
```bash
uv run pytest
```
Code formatting and linting:
```bash
uv run ruff check
uv run ruff format
```

---

## 8. Next Steps Roadmap (Phase 2 & Phase 3)

The environment and queryable data layer are fully ready for feature engineering:

1. **Phase 2: Feature Extraction Pipeline**
   * **DST / CST Features ($X_{\text{DST}}$):**
     * `slot_filling_latency`: Turns until required slots are mentioned (`turn_id` difference).
     * `slot_revision_count`: Frequency of conflicting slot values across turns.
     * `upfront_slot_density`: Entities supplied in turns 1–2 vs total required slots.
   * **AST Features ($X_{\text{AST}}$):**
     * `sop_edit_distance`: Levenshtein distance between `actions_taken` and `canonical_actions`.
     * `action_thrashing_count`: Consecutive duplicate actions ($\sum \mathbb{I}(a_t = a_{t-1})$).
     * `premature_action_attempts`: Actions fired before prerequisite slots are populated.
   * **NLP / Semantic Trajectory ($X_{\text{NLP}}$):**
     * `semantic_alignment_score`: Query-reply similarity via sentence transformers.
     * `frustration_velocity`: Emotional trajectory slope across conversation turns.

2. **Phase 3: Tabular Matrix Assembly**
   * Join engineered features with `conversations[['convo_id', 'split', 'intent', 'resolved']]`.
   * Form $X \in \mathbb{R}^{N \times D}$ and $Y \in \{0, 1\}^N$.

3. **Phase 4: Interpretable Modeling & SHAP Discovery**
   * Train LightGBM / XGBoost with Stratified 5-Fold Cross-Validation.
   * Run SHAP global importance, dependence plots, and cluster failure modes.
