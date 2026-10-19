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
| **`mandatory_slots_count`** | `int` | $[0, 5]$ | $M = \|\text{mandatory\_clauses}\|$ | Normative complexity: total number of distinct CNF clauses required by the intent. |
| **`mandatory_slots_satisfied`** | `int` | $[0, M]$ | $\sum_{k=1}^M \mathbb{I}(\exists s \in \text{Clause}_k : s \in \mathcal{S}_{\text{final}})$ | Count of distinct mandatory CNF clauses satisfied by the end of the dialogue. |
| **`mandatory_slot_coverage`** | `float` | $[0.0, 1.0]$ | $\frac{\text{mandatory\_slots\_satisfied}}{\max(M, 1)}$ | Completeness ratio: fraction of required information successfully collected. |
| **`slot_filling_latency`** | `float` | $[0.0, N_{\text{turns}}]$ | $t_{\text{all\_filled}} - t_{\text{start}}$ *(or $N_{\text{turns}}$ if incomplete)* | Number of dialogue turns elapsed before all mandatory slots were collected (measured in turn counts, not wall-clock time). |
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
* **Definition:** The **number of dialogue turns** (sequential turn count, as ABCD contains no wall-clock timestamps) elapsed from the conversation's first customer utterance ($t_{\text{start}}$) until the turn where all mandatory clauses were satisfied ($t_{\text{all\_mandatory\_filled}}$).
* **Calculation:**
  $$\Delta t_{\text{latency}} = t_{\text{all\_mandatory\_filled}} - t_{\text{start}}$$
  * *Incomplete Imputation:* If the customer never supplies all mandatory slots, latency is imputed to the full conversation turn count ($N_{\text{turns}}$), penalizing stalled dialogues without introducing `NaN` values into linear (OLS/Logit) or tree models.
  * *Zero Mandatory Imputation:* If $M = 0$, latency is defined as $0.0$.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Direct proxy for communication friction.** High turn latency indicates extensive agent probing, customer confusion, or reluctance, increasing conversation fatigue and resolution failure probability.

##### 6. `upfront_slot_density`
* **Definition:** The proportion of mandatory CNF clauses satisfied voluntarily by the customer during their first two speaking turns ($\text{CustomerTurns} \le 2$).
* **Calculation:** Evaluates CNF clause satisfaction strictly on the slot subset disclosed during customer turns 1 and 2:
  $$\text{upfront\_slot\_density} = \frac{\sum_{k=1}^M \mathbb{I}(\text{Clause}_k \cap \mathcal{S}_{\text{cust} \le 2} \neq \emptyset)}{\max(M, 1)}$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Proactive customer behavior indicator.** Customers who provide their account number, order ID, or issue details immediately upfront minimize agent inquiry turns, hypothesized to correlate strongly with shorter, successful resolutions.

##### 7. `unused_slot_ratio`
* **Definition:** The fraction of unique customer-provided entities that are completely unreferenced by the intent's schema ($\mathcal{S}_{\text{disclosed}} \setminus \mathcal{S}_{\text{schema}}$).
* **Calculation:**
  $$\text{unused\_slot\_ratio} = \begin{cases} \frac{|\mathcal{S}_{\text{disclosed}} \setminus \mathcal{S}_{\text{schema}}|}{|\mathcal{S}_{\text{disclosed}}|} & \text{if } |\mathcal{S}_{\text{disclosed}}| > 0 \\ 0.0 & \text{if } |\mathcal{S}_{\text{disclosed}}| = 0 \end{cases}$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Customer over-specification / misunderstanding.** Customers volunteering irrelevant personal details (e.g., offering credit cards or PIN numbers for simple delivery status checks) indicate confusion regarding the service model, leading to clarification loops and lower resolution rates.

##### 8. `slot_repetition_count`
* **Definition:** Total frequency with which the customer discloses a slot entity that was already previously provided in an earlier turn.
* **Calculation:** Sum of redundant slot extractions:
  $$\text{slot\_repetition\_count} = \sum_{t=1}^{N_{\text{cust}}} \sum_{s \in \text{inferred}_t} \mathbb{I}(s \in \mathcal{S}_{<t})$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Agent memory failure / Customer nagging.** Occurs when an agent repeatedly asks for an order ID or email address that the customer already supplied, or when an impatient customer re-states their ID multiple times. A prominent operational marker of agent incompetence and customer escalation.

##### 9. `total_slots_disclosed`
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
*Focus: Agent workflow compliance, system latency, execution bottlenecks, and prerequisite synchronization.*

In post-mortem dialogue analysis, Action State Tracking evaluates how faithfully, cleanly, and promptly the human agent adheres to the designated Standard Operating Procedure (SOP) leading up to resolution.

**Zero Target-Leakage Pre-Terminal Conditioning:** Because the binary target variable $Y = \text{resolved}$ is defined directly by whether the agent executed the canonical terminal action ($a_M^* = A_{\text{canonical}}[-1]$) without triggering an escalation (`notify-team`), including `reached_canonical_terminal`, `has_escalation`, or full-sequence metrics containing $a_M^*$ would leak $Y$ into $X$. Therefore, all AST features are computed strictly on the **pre-terminal prerequisite action sequence**:
* **Canonical Prerequisite Sequence ($A_{\text{canonical}}^{\text{pre}}$):** $A_{\text{canonical}}[:-1] \setminus \{\text{\texttt{notify-team}}\}$ (all intermediate SOP steps prior to the final resolving action).
* **Observed Prerequisite Sequence ($A_{\text{observed}}^{\text{pre}}$):** $A_{\text{observed}} \setminus \{a_M^*, \text{\texttt{notify-team}}\}$ (all intermediate actions executed by the agent excluding the terminal resolving action and escalation action).

---

#### 3.2.1. Summary Specification of Extracted AST Features

| Feature Name | Type | Value Range | Formula / Implementation | Core Operational Meaning |
| :--- | :---: | :---: | :--- | :--- |
| **`num_actions_taken`** | `int` | $[0, \infty)$ | $N_{\text{act}} = \|A_{\text{observed}}^{\text{pre}}\|$ | Total count of pre-terminal intermediate system actions executed by the agent. |
| **`num_canonical_actions`** | `int` | $[0, \infty)$ | $M_{\text{act}} = \|A_{\text{canonical}}^{\text{pre}}\|$ | Prerequisite workflow depth: expected number of intermediate actions prior to resolution. |
| **`action_count_ratio`** | `float` | $[0.0, \infty)$ | $\frac{\|A_{\text{observed}}^{\text{pre}}\|}{\max(\|A_{\text{canonical}}^{\text{pre}}\|, 1)}$ | Prerequisite workflow inflation / truncation ratio ($>1.0$ extra steps, $<1.0$ skipped steps). |
| **`action_count_delta`** | `int` | $(-\infty, \infty)$ | $\|A_{\text{observed}}^{\text{pre}}\| - \|A_{\text{canonical}}^{\text{pre}}\|$ | Absolute step difference between observed and canonical pre-terminal actions. |
| **`sop_edit_distance_raw`** | `int` | $[0, \infty)$ | $\text{Levenshtein}(A_{\text{observed}}^{\text{pre}}, A_{\text{canonical}}^{\text{pre}})$ | Minimum token-level insertions, deletions, and substitutions on prerequisite SOP steps. |
| **`sop_edit_distance_norm`** | `float` | $[0.0, 1.0]$ | $\frac{\text{sop\_edit\_distance\_raw}}{\max(\|A_{\text{obs}}^{\text{pre}}\|, \|A_{\text{canon}}^{\text{pre}}\|, 1)}$ | Normalized pre-terminal SOP divergence: bounded sequence misalignment score. |
| **`action_thrashing_count`** | `int` | $[0, \infty)$ | $\sum_{i=1}^{N-1} \mathbb{I}(a_i = a_{i+1})$ | Execution loops: frequency of consecutive identical pre-terminal actions fired consecutively. |
| **`premature_action_attempts`** | `int` | $[0, \infty)$ | $\sum_{a_t \in \mathcal{A}_{\text{prereq}}^{\text{pre}}} \mathbb{I}(\neg \text{PrereqsMet}(a_t, \mathcal{S}_{< t}))$ | Jumping the gun: pre-terminal actions triggered before customer provided required entity inputs. |
| **`action_execution_latency_mean`** | `float` | $[0.0, \infty)$ | $\frac{1}{\|\mathcal{A}_{\text{prereq}}^{\text{pre}}\|} \sum_{a_t \in \mathcal{A}_{\text{prereq}}^{\text{pre}}} (t - t_{\text{prereqs\_satisfied}})$ | Agent / system delay: average turn lag between prerequisite readiness and action execution. |

---

#### 3.2.2. Detailed Feature Definitions, Mechanics & Predictive Hypotheses

##### 1. `num_actions_taken` & `num_canonical_actions`
* **Definition:** Absolute count of executed pre-terminal intermediate system actions ($\|A_{\text{observed}}^{\text{pre}}\|$) versus expected prerequisite guideline actions ($\|A_{\text{canonical}}^{\text{pre}}\|$).
* **Calculation:** Direct length of pre-terminal action token sequences.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** Conversations with excessively high `num_actions_taken` relative to canonical prerequisite depth indicate trial-and-error troubleshooting or agent confusion ($r = -0.14$), correlating with failed resolution.

##### 2. `action_count_ratio` & `action_count_delta`
* **Definition:** Relative and absolute measures of prerequisite workflow elongation or truncation:
  $$\text{action\_count\_ratio} = \frac{\|A_{\text{observed}}^{\text{pre}}\|}{\max(\|A_{\text{canonical}}^{\text{pre}}\|, 1)}, \quad \text{action\_count\_delta} = \|A_{\text{observed}}^{\text{pre}}\| - \|A_{\text{canonical}}^{\text{pre}}\|$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):**
  * $\text{ratio} \ll 1.0$: Agent skipped prerequisite verification/lookup steps before attempting or abandoning resolution.
  * $\text{ratio} \gg 1.0$: Agent thrashing or executing redundant verification attempts.
  * $\text{ratio} \approx 1.0$: Optimal procedural adherence on intermediate SOP steps, associated with $Y = 1$.

##### 3. `sop_edit_distance_raw` & `sop_edit_distance_norm`
* **Definition:** Token-level Levenshtein distance computed via dynamic programming between the ordered sequence of observed pre-terminal agent actions ($A_{\text{observed}}^{\text{pre}}$) and the normative pre-terminal SOP sequence ($A_{\text{canonical}}^{\text{pre}}$).
* **Calculation:** Bounded normalization:
  $$\text{sop\_edit\_distance\_norm} = \frac{\text{Levenshtein}(A_{\text{observed}}^{\text{pre}}, A_{\text{canonical}}^{\text{pre}})}{\max(\|A_{\text{observed}}^{\text{pre}}\|, \|A_{\text{canonical}}^{\text{pre}}\|, 1)}$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Primary operational compliance feature.** Directly quantifies procedural deviations during the troubleshooting and verification phase without leaking whether the terminal resolving button was clicked. High normalized edit distance reflects out-of-order execution, skipped prerequisite steps, or unauthorized actions ($r = -0.39$, ROC-AUC = `0.71`).

##### 4. `action_thrashing_count`
* **Definition:** The number of times the agent executed the exact same pre-terminal system action consecutively without an intervening distinct action ($a_t = a_{t+1}$).
* **Calculation:** $\sum_{i=1}^{N_{\text{act}} - 1} \mathbb{I}(a_i = a_{i+1})$.
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **System friction and retry marker.** In ABCD, thrashing frequently occurs when backend queries fail (e.g. repeated `pull-up-account` or `validate-purchase` attempts due to invalid customer credentials). High thrashing indicates acute operational deadlock and predicts conversation failure.

##### 5. `premature_action_attempts`
* **Definition:** Count of pre-terminal actions ($a_t \in \mathcal{A}_{\text{prereq}}^{\text{pre}}$) triggered by the agent before the required customer entity inputs were present in the cumulative conversation state.
* **Prerequisite Rules Applied:**
  * `validate-purchase`: Requires purchase credentials (`username` AND `email` AND `order_id`).
  * `verify-identity`: Requires at least one tracked verification credential (`order_id`, `zip_code`, `account_id`, `username`, `email`, `phone`, OR `pin_number`).
  * `offer-refund`: Requires financial context (`amount`).
  * `enter-details`: Requires contact/address/credential context (`full_address`, `street_address`, `phone`, `email`, `pin_number`, `account_id`, OR `username`).
  * *(Note: `pull-up-account` is excluded from prerequisite gating as 97.1% of calls use untagged dummy `customer_name` entities.)*
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Agent incompetence / System error.** Premature actions trigger unhandled API error responses, forcing the agent to backtrack, ask clarifying questions, and disrupt dialogue flow.

##### 6. `action_execution_latency_mean`
* **Definition:** The mean number of dialogue turns elapsed between the exact turn when a customer supplied the required prerequisite slots and the turn when the agent finally triggered the corresponding pre-terminal system action.
* **Calculation:**
  $$\text{latency} = \frac{1}{\|\mathcal{A}_{\text{prereq}}^{\text{pre}}\|} \sum_{a_t \in \mathcal{A}_{\text{prereq}}^{\text{pre}}} (t_{\text{action\_executed}} - t_{\text{prerequisites\_satisfied}})$$
* **Predictive Hypothesis ($Y \in \{0, 1\}$):** **Agent / System Sluggishness.** High latency reflects agent hesitation, unnecessary conversational chit-chat after the customer already provided all needed info, or slow internal tooling, increasing customer drop-off risk.

---

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