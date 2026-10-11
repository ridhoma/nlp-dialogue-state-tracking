# Conversational Analytics Blueprint: Predicting Conversation Resolution on ABCD

**Project Objective:** Formulate a post-mortem tabular modeling pipeline to answer: *“What conversational friction points, conversation state tracking (CST/DST) dynamics, and action state tracking (AST) bottlenecks predict positive vs. negative conversation resolution?”*

---

## 1. Scope & Operating Principles

* **Post-Mortem Analytics (Not Real-Time Automation):** Analyze completed transcripts rather than training models for live response generation or next-action recommendations.
* **Predefined Schema as Ground Truth:** Eliminate unsupervised schema induction. Leverage ABCD's existing ontology:
  * **Intents:** 55 predefined customer intents.
  * **Slots:** Intent-specific mandatory and optional entity fields.
  * **System Actions:** 30 discrete system actions executed by the agent.
  * **Standard Operating Procedures (SOPs):** Defined flowcharts and execution trees in `guidelines.json`.

---

## 2. Target Variable ($Y$) Definition

Resolution is framed as an objective, rule-grounded binary label extracted from the terminal action sequence of each conversation:

$$Y =  \begin{cases}  1 & \text{if the agent successfully executes the canonical resolving action for the given intent and ends cleanly} \\ 0 & \text{if the conversation escalates, routes to fallback, or terminates prematurely}  \end{cases}$$

* **Positive Resolution ($Y = 1$):**
  * The agent executes the intent's designated terminal action (e.g., `confirm_refund`, `send_replacement_label`, `update_shipping_address`).
  * The interaction concludes with `end_conversation` without raising unhandled error states.
* **Negative Resolution ($Y = 0$):**
  * Fallback or escalation action is executed (`escalate_to_supervisor`, `transfer_to_tier_2`).
  * The conversation terminates before reaching the required terminal action (abandonment or agent error).
  * An invalid or unauthorized system action is triggered.

---

## 3. Feature Space ($X$) Engineering

Each conversation transcript is compressed into a single feature vector representing conversation-level friction, mechanical compliance, and trajectory dynamics.

### 3.1. Conversation State Tracking (CST / DST) Features
*Focus: Customer communication efficiency, information density, entity updates, and normative schema compliance.*

In post-mortem dialogue analysis, Conversation State Tracking measures how smoothly and completely the customer provides the prerequisite information required by standard operating procedures (SOPs). Mandatory slot requirements are defined in **Conjunctive Normal Form (CNF)** within `intent_schema`:

$$\text{Mandatory Requirement} = \bigwedge_{k=1}^{M} \left( \bigvee_{s \in \text{Clause}_k} s \right)$$

where the outer conjunction represents distinct required pieces of information (AND), and each inner clause represents alternative acceptable entity forms (OR, e.g. `['customer_name', 'account_id']`, `['product', 'name']`, or `['street_address', 'full_address']`). A clause $k$ is satisfied at turn $t$ if at least one alternative is present in the customer's cumulative disclosed slot set $\mathcal{S}_t = \bigcup_{i=1}^t \text{inferred\_slots}_i$.

---

#### 3.1.1. Summary Specification of Extracted DST Features

| Feature Name | Type | Value Range | Formula / Implementation | Core Operational Meaning |
| :--- | :---: | :---: | :--- | :--- |
| **`all_mandatory_slots_filled`** | `int` | $\{0, 1\}$ | $\mathbb{I}\left(\sum_{k=1}^M \text{satisfied}_k = M\right)$ | Binary flag indicating if 100% of the intent's mandatory CNF clauses were satisfied. |
| **`mandatory_slots_count`** | `int` | $[0, 5]$ | $M = |\text{mandatory\_clauses}|$ | Normative complexity: total number of distinct CNF clauses required by the intent. |
| **`mandatory_slots_satisfied`** | `int` | $[0, M]$ | $\sum_{k=1}^M \mathbb{I}(\exists s \in \text{Clause}_k : s \in \mathcal{S}_{\text{final}})$ | Count of distinct mandatory CNF clauses satisfied by the end of the dialogue. |
| **`mandatory_slot_coverage`** | `float` | $[0.0, 1.0]$ | $\frac{\text{mandatory\_slots\_satisfied}}{\max(M, 1)}$ | Completeness ratio: fraction of required information successfully collected. |
| **`slot_filling_latency`** | `float` | $[0.0, N_{\text{turns}}]$ | $t_{\text{all\_filled}} - t_{\text{start}}$ *(or $N_{\text{turns}}$ if incomplete)* | Dialogue turn distance elapsed before all mandatory slots were collected. |
| **`slot_filling_turn`** | `float` | $[0, N_{\text{turns}}] \cup \{\text{NaN}\}$ | Turn index $t$ where $\text{coverage} = 1.0$ | Absolute turn ID at which mandatory requirements were completed (NaN if never completed). |
| **`upfront_slot_density`** | `float` | $[0.0, 1.0]$ | $\frac{\sum_{k=1}^M \mathbb{I}(\exists s \in \text{Clause}_k : s \in \mathcal{S}_{\text{cust} \le 2})}{\max(M, 1)}$ | Voluntary upfront disclosure: fraction of mandatory clauses provided in customer turns 1 & 2. |
| **`unused_slot_ratio`** | `float` | $[0.0, 1.0]$ | $\frac{\|\mathcal{S}_{\text{final}} \setminus \mathcal{S}_{\text{schema}}\|}{\|\mathcal{S}_{\text{final}}\|}$ *(0.0 if empty)* | Information irrelevance: proportion of customer-disclosed entities not used by intent. |
| **`slot_repetition_count`** | `int` | $[0, \infty)$ | $\sum_{t} \sum_{s \in \text{inferred}_t} \mathbb{I}(s \in \mathcal{S}_{<t})$ | Friction / Nagging: frequency with which customer had to re-state previously seen slots. |
| **`total_slots_disclosed`** | `int` | $[0, 11]$ | $\|\mathcal{S}_{\text{final}}\|$ | Customer information volume: total distinct unique slots extracted across the entire chat. |

---

#### 3.1.2. Detailed Feature Definitions, Mechanics & Predictive Hypotheses

##### 1. `all_mandatory_slots_filled`
* **Definition:** A binary indicator $\{0, 1\}$ denoting whether the conversation ever satisfied all $M$ mandatory CNF clauses specified in the normative schema for the intent.
* **Calculation:** If $M = 0$ (e.g. general FAQ intents like `storewide`), defaults to $1$. For $M > 0$, evaluates whether the cumulative set of customer slots satisfies every clause $k \in \{1, \dots, M\}$.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Primary prerequisite for resolution.** If $0$, the agent cannot authorize or execute the terminal system action (e.g., executing a refund without an order ID or account identity). Hypothesized to be one of the strongest positive predictors of $Y = 1$.

##### 2. `mandatory_slots_count`
* **Definition:** Integer count $M \ge 0$ of mandatory CNF clauses specified by the intent schema.
* **Calculation:** Direct length of the intent's `mandatory_slots` CNF list.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Baseline operational cognitive load.** Higher values represent high-friction flows (e.g., identity verification + payment change + address verification) that introduce multiple failure points for customer drop-off.

##### 3. `mandatory_slots_satisfied`
* **Definition:** Integer count of mandatory CNF clauses successfully fulfilled by the customer at any point in the transcript.
* **Calculation:** $\sum_{k=1}^M \mathbb{I}(\text{Clause}_k \cap \mathcal{S}_{\text{final}} \neq \emptyset)$.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** Measures partial progress toward resolution. Provides a granular count metric alongside the binary completion flag.

##### 4. `mandatory_slot_coverage`
* **Definition:** Normalized completeness ratio $[0.0, 1.0]$ of satisfied mandatory clauses over total mandatory clauses.
* **Calculation:** $\frac{\text{mandatory\_slots\_satisfied}}{M}$ (defaults to $1.0$ when $M = 0$).
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** Normalizes information collection across intents with varying schema complexities. Strongly correlated with resolution; conversations stalling below $0.5$ coverage are hypothesized to fail primarily due to identity verification bottlenecks.

##### 5. `slot_filling_latency`
* **Definition:** The number of dialogue turns elapsed from the conversation's first customer utterance until the exact turn where all mandatory clauses were satisfied.
* **Calculation:**
  $$\Delta t_{\text{latency}} = t_{\text{all\_mandatory\_filled}} - t_{\text{start}}$$
  * *Incomplete Imputation:* If the customer never supplies all mandatory slots, latency is imputed to the full conversation duration ($N_{\text{turns}}$), penalizing stalled dialogues without introducing null values into tree models.
  * *Zero Mandatory Imputation:* If $M = 0$, latency is defined as $0.0$.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Direct proxy for communication friction.** High latency indicates extensive agent probing, customer confusion, or reluctance, increasing conversation fatigue and resolution failure probability.

##### 6. `slot_filling_turn`
* **Definition:** The absolute integer turn ID at which the final mandatory CNF clause was satisfied ($t_{\text{all\_mandatory\_filled}}$).
* **Calculation:** Extracted during chronological turn traversal; remains `NaN` if incomplete.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** Captures the temporal milestone of information readiness. Complements relative latency by indicating where in the timeline resolution became technically feasible.

##### 7. `upfront_slot_density`
* **Definition:** The proportion of mandatory CNF clauses satisfied voluntarily by the customer during their first two speaking turns ($\text{CustomerTurns} \le 2$).
* **Calculation:** Evaluates CNF clause satisfaction strictly on the slot subset disclosed during customer turns 1 and 2:
  $$\text{upfront\_slot\_density} = \frac{\sum_{k=1}^M \mathbb{I}(\text{Clause}_k \cap \mathcal{S}_{\text{cust} \le 2} \neq \emptyset)}{\max(M, 1)}$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Proactive customer behavior indicator.** Customers who provide their account number, order ID, or issue details immediately upfront minimize agent inquiry turns, hypothesized to correlate strongly with shorter, successful resolutions.

##### 8. `unused_slot_ratio`
* **Definition:** The fraction of unique customer-provided entities that are completely unreferenced by the intent's schema ($\mathcal{S}_{\text{disclosed}} \setminus \mathcal{S}_{\text{schema}}$).
* **Calculation:**
  $$\text{unused\_slot\_ratio} = \begin{cases} \frac{|\mathcal{S}_{\text{disclosed}} \setminus \mathcal{S}_{\text{schema}}|}{|\mathcal{S}_{\text{disclosed}}|} & \text{if } |\mathcal{S}_{\text{disclosed}}| > 0 \\ 0.0 & \text{if } |\mathcal{S}_{\text{disclosed}}| = 0 \end{cases}$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Customer over-specification / misunderstanding.** Customers volunteering irrelevant personal details (e.g., offering credit cards or PIN numbers for simple delivery status checks) indicate confusion regarding the service model, leading to clarification loops and lower resolution rates.

##### 9. `slot_repetition_count`
* **Definition:** Total frequency with which the customer discloses a slot entity that was already previously provided in an earlier turn.
* **Calculation:** Sum of redundant slot extractions:
  $$\text{slot\_repetition\_count} = \sum_{t=1}^{N_{\text{cust}}} \sum_{s \in \text{inferred}_t} \mathbb{I}(s \in \mathcal{S}_{<t})$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Agent memory failure / Customer nagging.** Occurs when an agent repeatedly asks for an order ID or email address that the customer already supplied, or when an impatient customer re-states their ID multiple times. A prominent operational marker of agent incompetence and customer escalation.

##### 10. `total_slots_disclosed`
* **Definition:** Total count of distinct unique slot entities provided by the customer across the entire dialogue ($|\mathcal{S}_{\text{final}}|$).
* **Calculation:** Cardinality of the cumulative set of extracted slots.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** Baseline entity volume control metric. Separates taciturn conversations (minimal disclosures) from highly communicative dialogues.

---

#### 3.1.3. Post-Hoc Dialogue State Tracking (DST) Extraction Architecture
In post-mortem conversational analytics, calculating metrics like `slot_filling_latency` or `upfront_slot_density` requires determining which entity slots are populated at each turn *without cheating* (i.e. strictly from dialogue text, never peeking at scenario metadata `convo.meta`).

The extraction pipeline supports two verified production artifacts:
1. **Production Champion LLM Agent (`customer_slots_extracted_run_02.parquet`):** Generative multi-turn extraction via Claude Haiku 5.5 CLI Agent adhering to [`SLOTS_EXTRACTION_PROMPT`](src/nlp/prompts.py). Delivers **84.7% Micro-F1 (93.2% excluding `name`)** and **98.8% Global Recall** across all 89,372 customer turns, resolving historical alphanumeric ambiguities.
2. **Local GPU NLI Architecture (`customer_slots_extracted_run_01.parquet`):** 3-stage calibrated cross-encoder pipeline (`facebook/bart-large-mnli`) with Stage 1 Informative Gating ($\tau_{\text{gate}}=0.02$) and Stage 3 Disambiguation Gating ($\tau_{\text{verify}}=0.05$). Produces **70.1% Micro-F1** with 24.2 ms/turn local inference.

### 3.2. Action State Tracking (AST) Features
*Focus: Agent workflow compliance, system latency, and execution bottlenecks.*

| Feature Name | Definition | Metric / Formula |
| :--- | :--- | :--- |
| **`sop_edit_distance`** | Levenshtein distance between observed action sequence and optimal guideline path | $\text{Levenshtein}(A_{\text{observed}}, A_{\text{guideline}})$ |
| **`action_thrashing_count`** | Frequency of repeated consecutive identical actions | $\sum \mathbb{I}(a_t = a_{t-1})$ |
| **`action_execution_latency`** | Turns between prerequisite slot satisfaction and corresponding action execution | $t_{\text{action\_executed}} - t_{\text{slots\_satisfied}}$ |
| **`premature_action_attempts`** | Count of actions triggered before prerequisite slots are populated | $\sum \mathbb{I}(a_t \text{ fired} \land \text{Prerequisites Missing})$ |

### 3.3. Semantic & Trajectory Features
*Focus: Comprehension friction, mutual understanding, and emotional momentum.*

| Feature Name | Definition | Metric / Formula |
| :--- | :--- | :--- |
| **`semantic_alignment_score`** | Mean cross-encoder similarity between customer queries and agent replies | $\frac{1}{N} \sum_{i=1}^{N} \text{CrossEncoder}(u_{\text{cust}, i}, u_{\text{agent}, i})$ |
| **`clarification_loop_count`** | Turns where customer repeats meaning while agent divergence remains high | Count of turns where $\cos(u_t, u_{t-2}) > 0.85$ (user) and $\cos(a_t, a_{t-1}) < 0.4$ |
| **`frustration_velocity`** | Linear regression slope of customer negative emotion scores over turn index | $\beta_1 \text{ from } \text{FrustrationScore}_t = \beta_0 + \beta_1 t$ |
| **`turn_count_total`** | Total conversation length (volume control metric) | $N_{\text{turns}}$ |

---

## 4. End-to-End Pipeline Architecture


```text
┌────────────────────────────────┐
│   Raw ABCD JSON Transcripts    │
│ (10,000+ Multi-Turn Conversations) │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│   Turn-Level Extraction Core   │
│  - Slot presence (DST)         │
│  - Action sequence logs (AST)  │
│  - Emotion & Sentence Encoders │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│  Tabular Feature Matrix (X, Y) │
│  - 1 row per conversation      │
│  - 12–15 engineered features   │
│  - Binary target Y ∈ {0, 1}    │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│  Interpretable Tree Modeling   │
│  - XGBoost / LightGBM          │
│  - Stratified 5-Fold CV        │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ Explainable AI (XAI) Discovery │
│  - Global SHAP Importances     │
│  - Feature Interaction Plots   │
│  - Decision Boundary Auditing  │
└────────────────────────────────┘
```


---

## 5. Modeling, Validation & Explainability

### 5.1. Classifier Configuration
* **Algorithms:** Gradient Boosted Trees (XGBoost, LightGBM).
* **Validation Strategy:** Stratified 5-Fold Cross-Validation grouped or stratified across the 55 distinct intents to evaluate generalizability across varying subdomains.
* **Evaluation Metrics:** ROC-AUC, PR-AUC (preferred if class imbalance exists between positive and negative resolutions), and Brier Score for probability calibration.

### 5.2. Post-Mortem Interpretability via SHAP
* **Global Importance:** Rank the relative influence of agent workflow deviations (AST) versus customer communication friction (DST) in driving failed resolutions.
* **Non-Linear Thresholding:** Use SHAP dependence plots to identify actionable operational tipping points (e.g., determining the exact threshold where `slot_filling_latency` or `sop_edit_distance` sharply degrades resolution probability).
* **Failure Clustering:** Apply $k$-means to local SHAP attribution vectors to identify distinct failure modes (e.g., *SOP Non-Compliance Failures* vs. *Information Asymmetry Loops*).

---

## 6. Project Milestones

1. **Phase 1: Ingestion & Target Labeling**
   * Parse ABCD conversation JSON files.
   * Write determinator functions to map terminal actions to binary target $Y$.
2. **Phase 2: Feature Extraction Pipeline**
   * Build rule-based parser for AST metrics against `guidelines.json`.
   * Implement DST tracker using slot presence across turns.
   * Run embedding/emotion models (`sentence-transformers`, `GoEmotions`) to compute trajectory slopes.
3. **Phase 3: Tabular Matrix Assembly**
   * Validate matrix schema, check missingness, and assess feature collinearity.
4. **Phase 4: Predictive Modeling & Interpretation**
   * Train baseline Logistic Regression vs. XGBoost/LightGBM.
   * Generate SHAP summary, dependence, and waterfall diagnostic plots.