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
*Focus: Customer communication efficiency, information density, and entity updates.*

| Feature Name | Definition | Metric / Formula |
| :--- | :--- | :--- |
| **`slot_filling_latency`** | Conversation turns elapsed before all mandatory slots are provided | $\Delta t = t_{\text{all\_slots\_filled}} - t_{\text{start}}$ |
| **`slot_revision_count`** | Number of times previously captured slot values are corrected | $\sum \mathbb{I}(\text{slot\_value}_{t} \neq \text{slot\_value}_{t-1})$ |
| **`upfront_slot_density`** | Ratio of required slots supplied voluntarily in the first 2 customer turns | $\frac{\text{Slots Provided in Turns 1–2}}{\text{Total Required Slots for Intent}}$ |
| **`unused_slot_ratio`** | Ratio of extracted user entities irrelevant to the current intent schema | $\frac{\text{Extracted Entities} \setminus \text{Schema Slots}}{\text{Total Extracted Entities}}$ |

#### 3.1.1. Post-Hoc Dialogue State Tracking (DST) Extraction Methodology
In post-mortem conversational analytics, calculating metrics like `slot_filling_latency` or `upfront_slot_density` requires determining which entity slots are populated at each turn *without cheating* (i.e. strictly from dialogue text, never peeking at scenario metadata `convo.meta`).

The DST tracking pipeline uses **Calibrated Semantic Natural Language Inference (MNLI)**:
1. **Contextualized Turn Documents (Preconditioning):** Each customer utterance is framed within a rolling dialogue context window capturing preceding agent prompts (default: 2 prior agent turns) to bridge clarification loops and short customer responses (e.g., bare IDs or confirmations). This preprocessing happens upstream before entering the neural classifier.
2. **Calibrated Semantic NLI Classification:** Candidate slots are evaluated via zero-shot / one-shot Natural Language Inference (BART/DistilBART-MNLI) on raw text strings (`extract_slots(str)` and `extract_batch(list[str])`). To eliminate false-positive bias where standard zero-shot pipelines discard the neutral logit, predictions are calibrated using **Option 1 3-Class Probability Calibration**:
   $$\text{Slot is Active} \iff \text{Prob}(\text{Entailment}) > \text{Prob}(\text{Neutral}) \quad \land \quad \text{Prob}(\text{Entailment}) > \tau$$
   where $\tau$ is the confidence threshold (default $0.5$) with hypothesis framing *"The customer specifies their {slot}."*

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