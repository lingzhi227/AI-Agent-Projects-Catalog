**practice-oriented checklist of the most relevant papers from **NeurIPS / ICML / ICLR 2025** on “**RL → LLM/agentic multi-agent**” (grouped by venue, with one-sentence takeaways + actionable inspirations). I also marked each item with the official page or arXiv link so you can read further or cite them by name in emails.

---

# NeurIPS 2025: Representative Works of RL Empowering LLM/Agents

* **Act Only When It Pays: Efficient RL for LLM Reasoning via Selective Rollouts** — uses **selective rollouts** to skip low-value samples, achieving **2× training speed-up** without accuracy loss; applicable to high-cost reasoning tasks. ([arXiv][1])
  *Inspiration: in multi-agent collaboration or bio tool-chain calls, pre-filter “low-value samples/low-confidence trajectories” to save expensive evaluation or simulation.*

* **Accelerating RL for LLM Reasoning with Optimal Advantage Regression (A*-PO)** — a two-stage paradigm: **offline estimate of optimal value** + simple **least-squares on-policy**, reducing over-sampling and critic dependence, making **training faster and more memory-efficient**. ([arXiv][2])
  *Inspiration: when using RL to optimize an agent’s tool selection and planning policy, use lightweight policy optimization like A*-PO instead of PPO/GRPO.*

* **Absolute Zero: Reinforced Self-play Reasoning with Zero Data** — **zero external data** self-play with self-generated tasks and verification (code-execution-verifiable rewards), reaching SOTA on math/code. ([arXiv][3])
  *Inspiration: in bioinformatics, use **verifiable programs/rules** (e.g., GO annotation rules, Pfam/PDB alignment scores, CAFA evaluation scripts) to build **self-play/self-evolving** training pipelines.*

* **S-GRPO: Early Exit via RL in Reasoning Models** — proposes **sequence-group decayed rewards** to encourage **early stopping in CoT**, significantly shortening reasoning length and improving accuracy on GSM8K/AIME. ([NeurIPS][4])
  *Inspiration: in multi-agent collaboration, introduce **early-stop/hand-off** mechanisms to reduce verbose thinking and tool overuse.*

* **SPC: Evolving Self-Play Critic for LLM Reasoning** — extends **self-play** to **adversarial critic evolution**, stabilizing self-play alignment. ([NeurIPS][5])

* **SPACE: Noise-Contrastive Estimation Stabilizes Self-Play Fine-Tuning for LLMs** (listed) — reports **contrastive estimation** to stabilize self-play fine-tuning. ([NeurIPS][6])

* **Token-Level Self-Play with Importance-Aware Guidance** — performs self-play at the **token level** with **importance-aware guidance**. ([NeurIPS][6])

* **Triplets Better Than Pairs: Towards Stable and Effective Self-Play Fine-Tuning for LLMs** — uses triplets instead of pairs to improve self-play stability and effectiveness. ([NeurIPS][6])

* **SQL-R1: Training NL→SQL Reasoning Model by RL** — for “**verifiable tasks**” (SQL execution correctness), uses RL to improve reasoning and tool use. ([NeurIPS][7])

* **ToolRL: Reward is All Tool Learning Needs** — treats **tool calling** itself as an RL, reward-driven learning problem. ([NeurIPS][8])

> Note: the above items all come from the official entries or poster pages of the NeurIPS 2025 virtual library/download page. ([NeurIPS][6])

---

# ICML 2025: Highlights for “Agentic Reasoning + RL”

* **Satori: RL with Chain-of-Action-Thought (COAT) Enhances LLM Reasoning via Autoregressive Search** — two stages (**format fine-tuning + large-scale RL self-improvement**), **internalizes search/self-reflection** into a single model; ICML 2025. ([arXiv][9])
  *Inspiration: encode “plan-act-reflect” as **learnable meta-actions**, enabling the model to learn “when to retrieve/when to reflect/when to proceed” during training.*

> ICML 2025 also has multiple workshops on **“programmatic representations and agent learning,”** focusing on “planning/code/verifiable RL.” This suits converting your **bioinformatics pipelines/database APIs** into programmable rewards and state machines. ([ICML][10])

---

# ICLR 2025: From Long-Horizon Interaction to Self-Evolving Curricula

* **Reinforcement Learning for Long-Horizon Interactive LLM Agents** — formalizes interactive digital agents (API-driven) as POMDPs; proposes **LOOP (critic-free, single-model memory)**, surpassing o1 on AppWorld; emphasizes **long-horizon context management** and **task completion rewards**. ([arXiv][11])

* **WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum RL** — ICLR 2025 (main conference/paper), introduces **self-evolving online curricula** + confidence-filtered replay to stabilize training of open-source LLM web agents. ([OpenReview][12])

* **ICLR 2025 Reasoning & Planning for LLMs** (workshop) — aggregates frontiers in RL/post-training optimization/multi-task interaction; several papers on **multi-agent training (e.g., MALT)** and **game-theoretic alignment** were accepted. ([ICLR 2025 Workshop][13])

---

## Shared Technical Themes (to map onto “agentic multi-agent + bio tool-chain”)

1. **Low-cost/high-efficiency RL training paradigms:**

   * **Selective rollouts**, **critic-free value approximation**, **two-stage lightweight policy optimization** → making RL for LLM-Agents more **compute/memory efficient**. (Act-Only, A*-PO) ([arXiv][1])

2. **Self-play and self-evolving curricula:**

   * **Zero/few human labels** relying on **verifiable rewards** (executors/unit tests/symbolic checks) for self-improvement; **triplet/contrastive** tricks to enhance stability. (Absolute-Zero, SPACE/Triplets/Token-level self-play, WebRL) ([arXiv][3])

3. **Internalizing learnable meta-actions for “plan-act-reflect”:**

   * Turn “search/reflect/terminate/continue” into parts of the action space, supporting **long-horizon interaction**. (Satori, LOOP, S-GRPO early exit) ([arXiv][9])

4. **Tool use as RL:**

   * Integrate tool call success, execution correctness, and cost (API/latency) into rewards, forming **unified tool-RL training**. (ToolRL, SQL-R1) ([NeurIPS][8])

---

## Porting These Methods to Your **CAFA5 Protein Function Prediction Agentic System**

**Scenario** (you already have): multiple specialist agents (sequence similarity, structure prediction, annotation transfer, knowledge-base retrieval, etc.) coordinating via JSON protocols and multi-level memory.

**Actionable plan:**

1. **Reward design (verifiable/extensible):**

   * **Primary reward:** use CAFA official metrics (e.g., **Fmax/AvgPr**) or proxy metrics (cross-validated **F1/precision**) as episodic reward.
   * **Intermediate rewards:** tool-verifiable goals (e.g., Pfam/InterPro hits, structure alignment TM-score, GO term consistency) → dense rewards; **cost penalties:** API call count/latency/failure rate.
   * Corresponding methods: **executable verification + rewards** in **ToolRL/SQL-R1**, **self-generated tasks + programmatic verification** in **Absolute-Zero**. ([NeurIPS][8])

2. **Training paradigm and efficiency:**

   * Use **A*-PO** or **Act-Only** to reduce rollout cost; prioritize **failed samples** for “retry-reflect” before adding to replay. ([arXiv][2])
   * Combine **S-GRPO early exit**: when retrieval/alignment is confident enough, **early-stop/hand-off** to the summarizing agent, reducing verbose thinking and API waste. ([NeurIPS][4])

3. **Self-play/curriculum learning:**

   * Start with **synthetic tasks** (randomly mask/perturb GO annotations) to learn the loop “find evidence → verify → fix,” then progressively add real hard cases (**WebRL’s self-evolving curriculum**). ([OpenReview][12])

4. **Meta-actions and multi-agent orchestration:**

   * Borrow **Satori (COAT)** to define **meta-actions**: `{retrieve, align, structure_predict, evidence_aggregate, reflect, submit}`;
   * Use the **LOOP** framework to formalize the whole orchestration as a POMDP, achieving stable training with a **single model/single replica**. ([arXiv][9])

---

## Two–Three “alignment sentences” you can place directly in your email to Prof. Li

> I plan to adapt **efficient RL post-training** (e.g., **A*-PO**, **selective rollouts**) and **self-play curricula** to train **agentic multi-agent LLMs** that orchestrate bioinformatics tools for **CAFA5** protein function prediction, with **verifiable rewards** from annotation metrics and executable checks. Building on **COAT/early-exit** mechanisms, I will internalize *plan-act-reflect* meta-actions and introduce **early-stop/hand-off** policies to reduce compute while improving reliability. These directions closely align with your group’s **RL theory for constrained/non-stationary settings** and **distributed learning**, and I hope to develop **provable and scalable** training schemes for scientific agents. ([arXiv][2])

---

### References (partial)

(click to view official page/abstract)

* NeurIPS 2025 virtual library & downloads (includes multiple self-play / RL-for-Reasoning / ToolRL entries). ([NeurIPS][6])
* **Act Only When It Pays** (NeurIPS 2025). ([arXiv][1])
* **A*-PO** (NeurIPS 2025). ([arXiv][2])
* **Absolute Zero** (NeurIPS 2025). ([arXiv][3])
* **S-GRPO** (NeurIPS 2025). ([NeurIPS][4])
* **SPC** (NeurIPS 2025). ([NeurIPS][5])
* **SQL-R1** (NeurIPS 2025). ([NeurIPS][7])
* **ToolRL** (NeurIPS 2025). ([NeurIPS][8])
* **Satori (ICML 2025)**: arXiv, ICML page, institution page. ([arXiv][9])
* **LOOP (long-horizon interactive agent, ICLR-related)**: arXiv/Apple research page. ([arXiv][11])
* **WebRL (ICLR 2025)**: OpenReview paper and PDF. ([OpenReview][12])

---

If you’d like, I can turn the **deployment plan** above into a one-page **“Research Plan Overview (1–2 page PDF)”** including: problem definition → POMDP formalization → reward design → training paradigms → experiments & metrics (CAFA5/Fmax/tool success rate) → alignment with Li group. Just tell me which main thread you want to emphasize (efficiency, verifiability, or personalization/federation), and I’ll generate it directly.

[1]: https://arxiv.org/abs/2506.02177?utm_source=chatgpt.com "Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts"
[2]: https://arxiv.org/abs/2505.20686?utm_source=chatgpt.com "Accelerating RL for LLM Reasoning with Optimal Advantage Regression"
[3]: https://arxiv.org/abs/2505.03335?utm_source=chatgpt.com "Absolute Zero: Reinforced Self-play Reasoning with Zero Data"
[4]: https://neurips.cc/virtual/2025/poster/115333 "NeurIPS Poster S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models"
[5]: https://neurips.cc/virtual/2025/poster/118706 "NeurIPS Poster SPC: Evolving Self-Play Critic via Adversarial Games for LLM Reasoning"
[6]: https://neurips.cc/Downloads/2025 " Downloads"
[7]: https://neurips.cc/virtual/2025/poster/116624 "NeurIPS Poster SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning"
[8]: https://neurips.cc/virtual/2025/poster/116923 "NeurIPS Poster ToolRL: Reward is All Tool Learning Needs"
[9]: https://arxiv.org/abs/2502.02508?utm_source=chatgpt.com "Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search"
[10]: https://icml.cc/virtual/2025/workshop/39950?utm_source=chatgpt.com "Programmatic Representations for Agent Learning - icml.cc"
[11]: https://arxiv.org/abs/2502.01600?utm_source=chatgpt.com "Reinforcement Learning for Long-Horizon Interactive LLM Agents"
[12]: https://openreview.net/forum?id=oVKEAFjEqv&utm_source=chatgpt.com "WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum ..."
[13]: https://workshop-llm-reasoning-planning.github.io/index.html?utm_source=chatgpt.com "ICLR 2025 Workshop on Reasoning and Planning for Large Language Models"
