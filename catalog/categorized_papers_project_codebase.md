# Agent Projects Notes

## dataset

### NLP & QA Datasets

- 2WikiMultiHopQA · [GitHub](https://github.com/Alab-NII/2WikiMultiHopQA)
  - description: Multi-hop QA dataset used in IoA’s RAG experiments to assess reasoning over multiple pieces of evidence.
  - description: Multi-hop QA dataset; used for information-exchange tasks with split evidence between agents.

- 2WikiMultihopQA · [GitHub](https://github.com/Alab-NII/2wikimultihop)
  - description: Multi-hop QA dataset used to evaluate the proposed method’s generalization and performance.
  - description: Multi-hop QA dataset evaluated through LongBench tasks in the paper.

- AmbigQA · [GitHub](https://github.com/shmsw25/ambigqa) · [Website](https://nlp.cs.washington.edu/ambigqa/)
  - description: Ambiguous open-domain QA dataset used for single-hop evaluation and out-of-domain tests.

- AutoForm · [GitHub](https://github.com/thunlp/AutoForm)
  - description: Prior THUNLP baseline that uses concise, non–natural-language formats for agent communication; OPTIMA compares against it and adopts its insight for format-diverse initialization.
  - description: Official code release for this paper; contains prompts and scripts to reproduce the single-LLM reasoning and multi-agent communication experiments.

- CAIL2018 (China AI and Law Challenge 2018) · [GitHub](https://github.com/thunlp/CAIL2018) · [Website](http://cail.cipsc.org.cn)
  - description: Large-scale Chinese legal judgment dataset used in this paper (400 sampled cases) for confusing charge prediction; legal rules and charges are matched from this dataset.

- Chart-to-text· [GitHub](https://github.com/vis-nlp/Chart-to-text)
  - description: Benchmark used to obtain human-annotated chart–summary pairs and to form the Pew test split; also used to assess GPT-4-turbo paragraph–table linking accuracy.

- ChartQA · [GitHub](https://github.com/vis-nlp/ChartQA)
  - description: Chart question answering dataset used as an image source for building the image pool in data synthesis.
  - description: Benchmark dataset used to validate the chart-to-table extraction step; the authors evaluated Gemini-1.0-pro-vision on 100 ChartQA chart images with gold tables.

- CommaQA · [GitHub](https://github.com/allenai/CommaQA)
  - description: Long-context synthetic multi-hop QA dataset (CommaQA‑E variant) used to test decomposition vs. CoT in reading‑comprehension settings.

- CraigslistBargain (CB) · [GitHub](https://github.com/stanfordnlp/craigslistbargains) · [Website](https://stanfordnlp.github.io/craigslistbargains/)
  - description: Price negotiation benchmark used for evaluation; the paper uses its test split to assess agents’ negotiation success, efficiency, and sale-to-list ratio.

- DyLAN · [GitHub](https://github.com/SALT-NLP/DyLAN)
  - description: Dynamic LLM-agent network optimizing temporal communication; used as a temporal-communication baseline in experiments.

- EXAMS-V · [GitHub](https://github.com/mbzuai-nlp/EXAMS)
  - description: Multilingual, multi-discipline multimodal exam benchmark; EMMA filters its physics/chemistry problems for inclusion.

- FreshQA (FreshLLMs) · [GitHub](https://github.com/google-research/google-research/tree/master/freshqa)
  - description: Provides fast-changing vs. slow-changing factual questions; used to build the Time domain of SMART-ER and evaluate temporal tool use.

- GAIR-NLP/Abel · [GitHub](https://github.com/GAIR-NLP/abel)
  - description: Generative AI for Math project explicitly linked in the references; included as a related open-source implementation practitioners can inspect.

- GAOKAO-MM · [GitHub](https://github.com/FudanNLP/GAOKAO-MM) · [Website](https://gaokao-mm.github.io)
  - description: Chinese human-level multimodal benchmark used for cross-domain evaluation and to build a domain-specific retrieval base.

- GeoQA · [GitHub](https://github.com/chenjiaqi/GeoQA)
  - description: Multimodal geometric QA dataset used for training and evaluating geometry reasoning models.

- GrailQA · [GitHub](https://github.com/dki-lab/GrailQA) · [Website](https://dki-lab.github.io/GrailQA/)
  - description: One of the KBQA sources used to construct KBQA-AGENT; contributes complex multi-hop Freebase questions.
  - description: In-domain KGQA dataset emphasizing generalization (i.i.d., compositional, zero-shot) over Freebase; used for instruction synthesis and evaluation.

- HellaSwag · [GitHub](https://github.com/rowanz/hellaswag)
  - description: Commonsense inference benchmark used in automatic evaluation.

- HoneyComb · [GitHub](https://github.com/BangLab-UdeM-Mila/NLP4MatSci-HoneyComb)
  - description: Official code release for the paper; contains the LLM-based agent system with MatSciKB (knowledge base), ToolHub (general and materials-science tools), retriever components, and evaluation setups for reproducing results.

- KGQAn· [GitHub](https://github.com/qcri/KGQAn)
  - description: Dataset of questions with annotated SPARQL over YAGO introduced by Omar et al.; used as a benchmark, and the KGQAn repo hosts resources associated with the dataset and baseline system.
  - description: Universal QA platform for knowledge graphs used as a comparison baseline and source for the YAGO-QA benchmark referenced in the paper.

- Lawformer · [GitHub](https://github.com/thunlp/LAWformer)
  - description: Chinese legal-domain pre-trained model used in the paper’s additional comparisons on general charge prediction versus confusing charge prediction.

- LC-QuAD 1.0 · [GitHub](https://github.com/AskNowQA/LC-QuAD) · [Website](http://lc-quad.sda.tech/)
  - description: Benchmark dataset of natural-language questions with SPARQL over DBpedia; used to evaluate Triad.

- MEAN / dyMEAN · [GitHub](https://github.com/THUNLP-MT/MEAN)
  - description: Antibody design baselines; retrained with unified data/settings, and the repo hosts the referenced rabd_summary.jsonl.

- MetaMath · [GitHub](https://github.com/hkust-nlp/MetaMath)
  - description: Tool-free math data augmentation and finetuning baseline; also used as an alternative Stage-1 checkpoint in two-stage ablations.

- Middleware for LLMs (this paper) · [GitHub](https://github.com/OSU-NLP/Middleware)
  - description: Official code release implementing the middleware tool framework, error-feedback, and decoupled-generation schemes used throughout the paper.

- Mind2Web · [GitHub](https://github.com/OSU-NLP/Mind2Web) · [Dataset](https://huggingface.co/datasets/osunlp/Mind2Web)
  - description: Web task dataset/environment used by the paper to source trajectories for IDMBench and CriticBench.
  - description: Web agent dataset/benchmark cited and compared in related work; UI-Vision targets desktop instead of web.
  - description: Dataset/benchmark used for evaluation (and cited for statistics); AgentTrek reports results on the multimodal extension and compares against HTML and HTML+Image settings.
  - description: Generalist web agent dataset; used as a held-in web task with step success metrics.

- MiniWob++ · [GitHub](https://github.com/stanfordnlp/miniwob-plusplus)
  - description: Classic web interaction benchmark used in the paper’s zero-shot evaluation to test low-level GUI skills.
  - description: Parameterizable web UI benchmark whose tasks are integrated into AndroidWorld as MobileMiniWoB++; serves as the web task suite used in the paper’s experiments.
  - description: Classic suite of synthetic web UI tasks for training/evaluating web agents (incl. MiniWoB/FormWoB); used throughout the literature summarized by the survey for step/task-level evaluation.

- Mol-Instructions · [GitHub](https://github.com/zjunlp/Mol-Instructions)
  - description: Biomolecular instruction dataset/model used as a baseline; paper notes its limited template diversity.

- MuSiQue · [GitHub](https://github.com/StonyBrookNLP/musique) · [GitHub](https://github.com/stanfordnlp/musique)
  - description: Multi-hop QA dataset used as part of LongBench evaluation in the paper.

- Musique (dataset) · [GitHub](https://github.com/stonybrooknlp/musique) · [Website](https://musique-data.github.io/)
  - description: Multi-hop QA via single-hop question composition; used as an evaluation dataset in the open-domain experiments.
  - description: Multi-hop QA dataset used to assess MindSearch on closed-set tasks.

- NarrativeQA · [GitHub](https://github.com/deepmind/narrativeqa)
  - description: LongBench task used in evaluation; measures comprehension over long narratives.

- OpenDelta · [GitHub](https://github.com/thunlp/OpenDelta)
  - description: Library the authors used to implement LoRA adapters for fine-tuning the base LM.

- OpenEQA · [GitHub](https://github.com/allenai/OpenEQA) · [Website](https://openeqa.github.io)
  - description: Embodied QA benchmark used to evaluate ThinkAct’s embodied reasoning performance.

- PathVQA · [GitHub](https://github.com/UCSD-AI4H/PathVQA)
  - description: Pathology visual question answering dataset used for image-based evaluation.

- Qasper · [GitHub](https://github.com/allenai/qasper)
  - description: Scientific paper QA dataset included via LongBench evaluation in the paper.

- ScienceQA · [GitHub](https://github.com/lupantech/ScienceQA) · [Website](https://scienceqa.github.io)
  - description: Multimodal multiple‑choice benchmark used for MLLM evaluation (QCM format) in the paper.

- SeeAct · [GitHub](https://github.com/OSU-NLP-Group/SeeAct) · [Website](https://osu-nlp-group.github.io/SeeAct/)
  - description: Web agent baseline compared in the paper; uses planning plus grounding with GPT-4V and serves as a strong reference implementation.
  - description: Baseline web agent adapted by the authors to the Android setting for comparison; their implementation follows the SeeActchoice variant with Android-specific actions.
  - description: Prompting-based multimodal web agent (GPT-4V/4o) baseline; AdaptAgent augments its prompt with 1-shot multimodal in-context demonstrations to obtain the proprietary-model results.
  - description: Web task-completion agent that grounds actions on webpage screenshots; INFOGENT augments SeeAct (adding GO BACK and AGGREGATE actions) to build its Navigator in the Interactive Visual Access setting.
  - description: Web agent used for web tasks; the paper evaluates task-specific and systemic risks against web agents by referencing SeeAct.
  - description: Generalist web agent used by the paper as a target agent; GuardAgent moderates SeeAct’s predicted actions and reasoning traces on Mind2Web-SC.

- SeeAct · [GitHub](https://github.com/OSU-NLP/SeeAct)
  - description: Baseline method referenced for multimodal planning on Mind2Web; used for comparison in the evaluation tables.

- SimCSE · [GitHub](https://github.com/princeton-nlp/SimCSE) · [Doc](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large)
  - description: Alternative retriever baseline evaluated in analysis to study retriever quality on tool retrieval.
  - description: Sentence embedding method used to embed tool descriptions and perform clustering into toolkits.
  - description: Contrastive sentence-embedding method used as a technical discriminator (SimCSE-RoBERTa-large) to compute semantic similarity in the Rumor Chain experiments.

- SOCIALIQA · [GitHub](https://github.com/rowanz/social-iqa)
  - description: Commonsense reasoning benchmark about social interactions; used in the paper to evaluate LLM social knowledge/reasoning limits.

- Solo-Performance-Prompting (SPP) · [GitHub](https://github.com/GAIR-NLP/solo-performance-prompting)
  - description: Flat-structure coding setup via multi-persona prompting of a single model; used as a Flat baseline on code generation.

- Tree of Thoughts (ToT) · [GitHub](https://github.com/princeton-nlp/tree-of-thought-llm) · [Website](https://arxiv.org/abs/2305.10601)
  - description: Reasoning strategy baseline; also influences module designs within the Reasoning module space.
  - description: Decision-making paradigm that inspires D2A’s multi-candidate Activity Proposal/Evaluation/Selection process.

- TruthfulQA · [GitHub](https://github.com/sylinrl/TruthfulQA)
  - description: Benchmark measuring propensity to produce falsehoods; used in automatic evaluation.

- UGround · [GitHub](https://github.com/OSU-NLP/UGround)
  - description: Universal GUI grounding model used as a competitive baseline and as a grounding component in planner+grounder ablations reported by the paper.

- UniKGQA · [GitHub](https://github.com/RUCAIBox/UniKGQA)
  - description: Unified retrieval-and-reasoning framework for multi-hop KGQA; used as a baseline on Freebase-based datasets.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop) · [Website](https://princeton-nlp.github.io/WebShop/)
  - description: Prior web interaction environment benchmark referenced as a domain-specific baseline; contrasts with AgentStudio’s broader action/observation spaces.
  - description: Simulated e-commerce environment and agent baseline; used in MMInA as a heuristic web-agent baseline (queries adapted via GPT-3.5).
  - description: E-commerce web interaction benchmark used as one of the six evaluation tasks.
  - description: Simulated e-commerce website; used with exploration and answer-forcing to create shopping trajectories and evaluate average reward.
  - description: Referenced web shopping environment used to evaluate grounded web agents; included for practitioners looking at adjacent agent-eval settings.
  - description: Open-domain web shopping environment and benchmark for grounded language agents; listed in the survey’s browser benchmarks.
  - description: Realistic online shopping website environment with 1.18M products and 12k instructions; used to test ReAct’s web navigation and compared against IL/IL+RL baselines.
  - description: Web-shopping simulation environment and benchmark; used as one of the main evaluation environments.
  - description: Web navigation/shopping environment used as a held-in benchmark in the experiments.
  - description: Large-scale simulated online shopping environment; the paper evaluates on its 500 test instructions and uses 12K human instructions to collect trajectories and fine-tune the Critic.
  - description: E-commerce web navigation environment and dataset used as one of the main benchmarks; the paper trains/evaluates agents on WebShop and uses its scoring rules.
  - description: Interactive web shopping environment used to evaluate agent navigation and task completion (Sections 5.1, 5.2.1).

- WebThinker · [GitHub](https://github.com/RUC-NLPIR/WebThinker)
  - description: Open-source baseline and concurrent agentic framework (SFT+RL) reproduced and compared against in experiments.

- WikiExtractor · [GitHub](https://github.com/attardi/wikiextractor)
  - description: Used to extract and clean text from Wikipedia dumps when building the retrieval corpus.

- WikiPlots · [GitHub](https://github.com/markriedl/WikiPlots)
  - description: Public dataset of plot summaries referenced for dataset scale/characteristics comparison against TELL ME A STORY.

- WikiSQL · [GitHub](https://github.com/salesforce/WikiSQL)
  - description: Table semantic parsing/QA dataset widely used to pre-train or evaluate table LLMs.

- WikiTableQuestions · [GitHub](https://github.com/ppasupat/WikiTableQuestions)
  - description: Training dataset used for planner curriculum on table reasoning and operations.

### Benchmark

- AdvBench: Harmful Behaviors · [GitHub](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench)
  - description: Dataset of 520 malicious requests introduced with GCG and used by this paper for evaluation (ASR and Recheck metrics).

- Alibaba DAMO-ConvAI (FlowBench mirror/hub) · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI)
  - description: Alibaba Research organization hub that hosts conversational AI resources; the paper points here as an official location for accessing FlowBench.

- AlpacaEval· [GitHub](https://github.com/tatsu-lab/alpaca_eval) · [Website](https://tatsu-lab.github.io/alpaca_eval/)
  - description: Primary benchmark used (length-controlled win rate) to evaluate MoA and baselines.
  - description: General capability benchmark used to measure catastrophic forgetting when fine-tuning.
  - description: GPT-4-based evaluator used to compare instruction-following quality.

- Arquero · [GitHub](https://github.com/uwdata/arquero) · [Website](https://idl.uw.edu/arquero/)
  - description: JavaScript data-wrangling library whose verb taxonomy informed BLADE’s transform-verb set and graph-based matching for data transformations.

- ASDiv · [GitHub](https://github.com/chaochun/nlu-asdiv-dataset)
  - description: Arithmetic reasoning dataset used in further math transfer experiments.
  - description: Diverse arithmetic word problems; used for out-of-domain evaluation.

- Bamboogle (dataset) · [GitHub](https://github.com/ofirpress/self-ask/tree/main/bamboogle)
  - description: Closed-set QA dataset used for evaluation; introduced with the self-ask work.

- BenchForm · [GitHub](https://github.com/Zhiyuan-Weng/BenchForm)
  - description: Official repository for the paper’s BENCHFORM benchmark and code, used to reproduce all protocols, prompts, metrics, and experiments studying conformity in multi‑agent LLMs.

- BIG-bench · [GitHub](https://github.com/google/BIG-bench)
  - description: The original large language model evaluation benchmark from which BBH is drawn; relevant background and task sources referenced in the paper.
  - description: Source for three reasoning datasets used in single-LLM experiments (Logic Grid Puzzle, Information Essentiality, Minute Mysteries QA); the paper downloads these tasks from the official repo.

- BIG-bench Hard (BBH) · [GitHub](https://github.com/suzgunmirac/BIG-bench-hard) · [GitHub](https://github.com/suzgunmirac/BIG-bench-Hard)
  - description: Challenging subset of BIG-bench used to assess reasoning improvements with MoA.
  - description: Logical reasoning benchmark used for training/validation and analysis (e.g., router ablations, heterogeneity, scalability) in the paper.
  - description: Curated subset of challenging BIG-bench tasks; BENCHFORM derives its reasoning-intensive multiple‑choice problems from BBH.

- BIRD (Text-to-SQL) · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/BIRD) · [Website](https://bird-bench.github.io/)
  - description: Primary database benchmark used without oracle knowledge; evaluates agents’ ability to navigate real DB content via the middleware tools.
  - description: Large-scale, realistic text-to-SQL benchmark; integrated to assess OpenHands’ database-grounded code generation.

- BLADE · [GitHub](https://github.com/behavioral-data/BLADE)
  - description: Official repository for the paper’s benchmark, including the 12 datasets, expert-annotated ground-truth decision space, prompts, automatic evaluation modules, and baseline agent code to reproduce and extend results.

- BIRD: Big Bench for Large-Scale Database-Grounded Text-to-SQL · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/BIRD) · [Website](https://bird-bench.github.io/)
  - description: Large-scale, realistic text-to-SQL benchmark; integrated to assess OpenHands’ database-grounded code generation.

- CATH 4.2 processed dataset (NeurIPS’19 Graph Protein Design) · [GitHub](https://github.com/jingraham/neurips19-graph-protein-design)
  - description: Repository providing the curated CATH 4.2 inverse folding dataset and splits originally used by Ingraham et al.; SurfPro trains/evaluates on these splits (following Jing et al. 2020).

- COCOA Datasets: MutualFriends and CraigslistBargain · [GitHub](https://github.com/stanfordnlp/cocoa) · [Website](https://stanfordnlp.github.io/cocoa)
  - description: Original datasets for the cooperative (MutualFriends) and competitive (Craigslist Bargain) tasks used as representative scenarios in the paper’s evaluations and analyses.

- CRAB Benchmark-v0 (dataset and tasks) · [GitHub](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0) · [Doc](https://github.com/camel-ai/crab/blob/main/crab-benchmark-v0/README.md)
  - description: The released suite of 120 cross-environment tasks (Ubuntu + Android) with subtask templates and graph evaluators; used in the paper’s evaluations.

- CURIE Benchmark · [GitHub](https://github.com/google/curie)
  - description: Official release of the CURIE benchmark with data, prompts, evaluation code (LLMSim/LMScore), and model outputs; primary resource to reproduce and extend all experiments.

- DeepMind Mathematics Dataset · [GitHub](https://github.com/deepmind/mathematics_dataset)
  - description: Source of 1,000 math problems across 56 subjects for the Math domain; also used to create the Comprehend+ subset and the planner-/solver-shift tasks.

- Dense Passage Retrieval (DPR)· [GitHub](https://github.com/facebookresearch/DPR)
  - description: Retrieval framework the authors follow to train their dense retriever (RoBERTa-base backbone) for function/tool retrieval.
  - description: Off‑the‑shelf dense retriever employed in CAVE’s text-retrieval module to fetch Wikipedia passages for verification (cited as the retriever used).

- DSBench · [GitHub](https://github.com/LiqiangJing/DSBench)
  - description: Official repository released by the paper; contains the data and code for DSBench, enabling reproduction of the benchmark and experiments.

- EleutherAI LM Evaluation Harness · [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
  - description: Framework the authors used to assess general reasoning/knowledge (MMLU, ARC, GSM8K, HellaSwag) for catastrophic-forgetting analysis.
  - description: Evaluation codebase referenced (Gao et al., 2021) for running standard LLM benchmarks.

- EmbodiedBench· [Website](https://embodiedbench.github.io) · [GitHub](https://github.com/EmbodiedBench)
  - description: Official code, benchmark tasks, and datasets released by the paper; includes the unified agent, evaluation scripts, EB-ALFRED/EB-Habitat/EB-Navigation/EB-Manipulation environments, and the auto task-generation script for EB-Navigation.

- Error Correction Zoo (EC Zoo) · [GitHub](https://github.com/errorcorrectionzoo/eczoo_data) · [Website](https://errorcorrectionzoo.org/)
  - description: Open-source repository and site for EC code entries; the QECC task asks models to produce YAML entries following this template and schema.

- ETDataset (ETT) · [GitHub](https://github.com/zhouhaoyi/ETDataset)
  - description: Research time-series forecasting dataset family (e.g., ETTm2) used in development/deployment tasks.

- EvalPlus· [GitHub](https://github.com/evalplus/evalplus) · [Website](https://evalplus.github.io/)
  - description: Enhanced HumanEval with additional test cases; used for coding evaluation.
  - description: Extended unit tests for HumanEval/MBPP used in the paper’s evaluation (EvalPlus results reported and used to augment testing).

- FlowBench · [GitHub](https://github.com/Justherozen/FlowBench)
  - description: Official repository for the FlowBench benchmark, including benchmark data, workflow knowledge in multiple formats (text/code/flowchart), evaluation scripts, and prompts used in the paper.

- Google Landmarks Dataset v2 (Web‑Landmark) · [GitHub](https://github.com/cvdfoundation/google-landmark)
  - description: Landmark recognition/retrieval dataset used as an image source in the image pool.

- GSM-Hard · [GitHub](https://github.com/FranxYao/GSM8K-Hard)
  - description: Hard subset of GSM8K used as another held-out math domain to evaluate transfer performance.

- HaluEval · [GitHub](https://github.com/RUCAIBox/HaluEval)
  - description: Hallucination evaluation benchmark cited for assessing and mitigating hallucinations in medical LLM agents.

- HarmBench · [GitHub](https://github.com/centerforaisafety/HarmBench)
  - description: A standardized evaluation framework for automated red teaming and robust refusal referenced in the paper’s discussion/limitations; useful for broader benchmarking of jailbreak methods.

- HotpotQA (dataset) · [GitHub](https://github.com/hotpotqa/hotpot) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used for real and simulated human–agent collaboration experiments and evaluation in the paper.
  - description: Multi-hop QA dataset used as a primary benchmark for training, evaluation, and ablations.
  - description: Multi-hop QA dataset used in Case Study 2; the paper evaluates in the fullwiki setting using the official Wikipedia 2017 abstracts index.
  - description: Multi-hop QA dataset used to build 60 needle pairs for the paper’s multi-needle (two-needle) evaluation setting.
  - description: Multi-hop QA dataset; used in the information-exchange setting where contexts are split across two agents.
  - description: Multi-hop QA dataset; the paper reports main closed-set results and ablations on it.

- HotpotQA (via Reflexion data) · [GitHub](https://github.com/noahshinn/reflexion/tree/main/hotpotqa_runs/data) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used for the multi-agent communication setting; the paper obtains the evaluation split from the Reflexion repository.

- Hugging Face Datasets · [GitHub](https://github.com/huggingface/datasets) · [Codewiki](https://codewiki.google/github.com/huggingface/datasets) · [Doc](https://huggingface.co/docs/datasets)
  - description: Dataset handling library used during model fine-tuning (per Appendix F).

- HumanEval · [GitHub](https://github.com/openai/human-eval) · [Website](https://huggingface.co/datasets/openai_humaneval)
  - description: Code generation benchmark; used to measure pass@1 and to visualize sampled workflows.
  - description: Code-generation benchmark used by the paper for cross-benchmark comparison with BLADE to analyze correlations between coding ability and analysis-generation performance.
  - description: Code generation benchmark used for evaluation; the paper reports pass@1 and compares against multi-agent baselines.
  - description: Code generation benchmark used to measure program synthesis performance.
  - description: Code-generation benchmark with unit tests; used to evaluate pass@k for MACNET.
  - description: Code generation benchmark used to evaluate Pass@1 and study error-type and error-rate effects under AUTOINJECT/AUTOTRANSFORM.
  - description: Function-level code generation benchmark used for Pass@1 evaluation and cost comparisons.
  - description: Code generation benchmark where the paper evaluates targeted injection (adding a safety_check function) and DoS attacks.
  - description: Functional code synthesis benchmark; used to build programming trajectories.
  - description: Code generation benchmark used for baseline comparison and MegaAgent evaluation.
  - description: Code-generation benchmark used for performance and cost comparisons.
  - description: Code generation benchmark; correctness evaluated via test cases.
  - description: Program synthesis benchmark on which CODESIM is evaluated (pass@1), including base tasks and sample I/O used for simulation and testing.
  - description: Code generation benchmark (pass@1) used to evaluate programming ability.

- HUSKY (and Husky-QA dataset) · [GitHub](https://github.com/allenai/husky)
  - description: Open-source language agent framework used as a strong multi-agent baseline; its Husky-QA dataset (train/test splits) is the primary benchmark used to train/evaluate AOP.

- INFOSEEK · [GitHub](https://github.com/google-research-datasets/infoseek) · [Website](https://github.com/google-research-datasets/infoseek)
  - description: Visual information-seeking VQA dataset used as the image source for constructing MORE (images/entities linked to Wikipedia).

- IRCoT (Interleaving Retrieval with Chain-of-Thought) · [GitHub](https://github.com/allenai/ir-cot)
  - description: Strong multi-round retrieval baseline implemented for comparison with LONGAGENT on long-document QA.

- LightEval · [GitHub](https://github.com/huggingface/lighteval)
  - description: Evaluation toolkit used to compute pass@1 and run AIME, MATH500, and GPQA benchmarks in the paper.

- LiveBench · [GitHub](https://github.com/LiveBench/LiveBench) · [Website](https://livebench.ai)
  - description: Uncontaminated LLM benchmark used to obtain the “Reasoning Average” scores that the paper correlates with catastrophic-behavior and deception rates.

- LongBench · [GitHub](https://github.com/THUDM/LongBench)
  - description: Public long-context benchmark; the paper evaluates all long-document QA tasks from LongBench (NarrativeQA, Qasper, MuSiQue, HotpotQA, 2WikiMQA).

- MaSIF-Site (PPI site prediction dataset) · [GitHub](https://github.com/LPDI-EPFL/masif)
  - description: Foundational molecular surface fingerprinting method; cited as prior work on mesh-based surface features that motivates the paper’s INR-based surface modeling.
  - description: Surface-based geometric deep learning framework; used by the authors for a head-to-head binder design comparison by ranking ProteinMPNN-generated candidates.
  - description: Dataset and code from Gainza et al. for PPI site prediction; the paper adopts this dataset (with a proximity-based relabeling) for node‑level evaluation.

- Megablocks (Mixture-of-Experts) · [GitHub](https://github.com/stanford-futuredata/megablocks)
  - description: Library leveraged to implement efficient sparse MoE layers used in ProGen3.

- Microsoft COCO (MS-COCO) · [GitHub](https://github.com/cocodataset/cocoapi) · [Website](https://cocodataset.org)
  - description: Generic image dataset used to provide diverse visual inputs for each character; required if re-generating or extending the dataset pipeline.

- MINT · [GitHub](https://github.com/xingyaoww/MINT-benchmark)
  - description: Multi-turn interaction benchmark with tool use and simulated language feedback; OpenHands evaluates math and code subsets.

- MMBench · [GitHub](https://github.com/open-compass/MMBench) · [Website](https://mmbench.opencompass.org.cn)
  - description: Primary image understanding benchmark (TEST_V11/TEST_EN variants) used for main results, ablations, and hyperparameter analyses.

- MMLU (Massive Multitask Language Understanding)· [GitHub](https://github.com/hendrycks/test) · [Website](https://people.eecs.berkeley.edu/~hendrycks/) · [Doc](https://arxiv.org/abs/2009.03300)
  - description: General reasoning benchmark used for evaluation; the paper reports accuracy on MMLU subsets and the full benchmark.
  - description: Multi-task QA benchmark used to assess general reasoning and as a transfer target for agents discovered on math.
  - description: Benchmark used to evaluate general reasoning.
  - description: Evaluation dataset for multitask language understanding; used to measure MACNET accuracy on multiple-choice reasoning.
  - description: General reasoning benchmark used for evaluation and cost/performance plots.
  - description: Knowledge and reasoning benchmark used in the paper’s reasoning evaluation suite.
  - description: General knowledge and reasoning benchmark used for additional evaluation in the appendix.
  - description: Benchmark used for multiple-choice tasks (biology and physics subsets) in both targeted-behavior and DoS attack evaluations.
  - description: General knowledge benchmark used to assess whether mixture training preserves general abilities.
  - description: Source dataset used to construct the College Physics and College Chemistry benchmarks evaluated in this paper.
  - description: General-purpose knowledge benchmark with a medical subset; cited for measuring cross-domain factual knowledge of LLM agents.
  - description: Knowledge benchmark used for evaluation; authors sample subsets for experiments.
  - description: Benchmark used for evaluation; the paper reports accuracy and cost on MMLU.
  - description: General knowledge benchmark used both in training (subset) and testing.
  - description: Massive multitask language understanding benchmark; used in debate evaluations.
  - description: Used in an expanded experiment (abstract algebra subset) to test DMAD on more challenging multi‑hop reasoning.
  - description: Multi-task benchmark; the Math categories are used to test transferability to multi-choice math.
  - description: Multiple-choice academic benchmark used in automatic evaluation.

- MMLU‑Pro · [GitHub](https://github.com/TIGER-Lab/MMLU-Pro) · [Website](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
  - description: Harder professional-level extension of MMLU included in the knowledge-task evaluations.

- MMLU‑Pro · [GitHub](https://github.com/TIGER-AI-Lab/MMLU-Pro) · [Website](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
  - description: Robust multi-task language understanding benchmark used as a closed-domain evaluation dataset in the paper.
  - description: A harder, more robust multi‑task benchmark the authors plan to incorporate in future extensions of BENCHFORM.

- MMMU· [GitHub](https://github.com/MMMU-Benchmark/MMMU) · [Website](https://mmmu-benchmark.github.io)
  - description: Multimodal benchmark used to test EVOAGENT with GPT‑4V and Gemini‑Pro on multiple‑choice validation questions (§4.1).
  - description: Multi-discipline multimodal benchmark from which EMMA sources physics/chemistry items after applying stricter filtering.
  - description: Massive multi-discipline multimodal benchmark (with a math track) used to assess broad MLLM reasoning in expert AGI settings.
  - description: One of the core evaluation benchmarks for complex multimodal reasoning; the paper reports accuracy improvements of GAM-Agent on this dataset.
  - description: Massive multi-discipline multimodal understanding benchmark; SCIVERSE sources problems from MMMU as part of its curated pool.

- MT-Bench (LMSYS) · [GitHub](https://github.com/lm-sys/FastChat#mt-bench-and-chatbot-arena) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat#mt-bench-and-chatbot-arena)
  - description: GPT-4 judging benchmark used to evaluate multi-turn conversational ability.

- MuMath (Multi-perspective Data Augmentation) · [GitHub](https://github.com/youweihao-tal/MuMath)
  - description: Prior dataset and augmentation framework used for Stage-1 training (Dµ, ~750K CoT samples) and as the source of the augmented question set Q referenced throughout the paper.

- Musique (dataset) · [GitHub](https://github.com/stonybrooknlp/musique) · [Website](https://musique-data.github.io/)
  - description: Multi-hop QA via single-hop question composition; used as an evaluation dataset in the open-domain experiments.
  - description: Multi-hop QA dataset used to assess MindSearch on closed-set tasks.

- MVBench · [GitHub](https://github.com/OpenGVLab/MVBench) · [Website](https://mvbench.github.io)
  - description: Video temporal reasoning benchmark; the paper evaluates GAM-Agent’s debate framework on multi-frame/video inputs here.

- Natural Questions (NQ) · [GitHub](https://github.com/google-research-datasets/natural-questions)
  - description: Open-domain QA dataset used in RAG experiments to evaluate IoA’s multi-agent retrieval and synthesis.

- NLG-Eval · [GitHub](https://github.com/Maluuba/nlg-eval)
  - description: Toolkit referenced as “NLPEval” in the appendix; used to compute linguistic captioning metrics (BLEU, ROUGE, CIDEr, BERTScore) for motion-to-text evaluation.

- NormBank · [GitHub](https://github.com/behavioral-data/NormBank)
  - description: Knowledge bank of situational social norms; referenced when discussing how norms shape interactions and as a resource for modeling social context.

- OpenAI Simple Evals · [GitHub](https://github.com/openai/simple-evals)
  - description: Evaluation utilities referenced for prompting/evaluation practice (e.g., DROP one-shot style), relevant for reproducing the paper’s evaluation setup.

- Overcooked-AI Human Trajectories (BC dataset) · [GitHub](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/overcooked_ai/data/human_data)
  - description: Human gameplay dataset bundled with Overcooked-AI; the paper uses BC agents trained on this data as human proxies and as partners/opponents in zero-shot coordination.

- PEER Benchmark · [GitHub](https://github.com/DeepGraphLearning/PEER_Benchmark)
  - description: Provides datasets/splits for subcellular localization and protein–protein interaction tasks used in evaluation.

- PlanBench (BlocksWorld) · [GitHub](https://github.com/karthikvalmeekam/planbench)
  - description: Planning benchmark suite providing the BlocksWorld tasks, domain description, and solver-generated ground-truth plans/feedback; used for training (100), validation (500), and attribution analyses of action/constraint tokens.

- RE-Bench Environments (ai-rd-tasks) · [GitHub](https://github.com/METR/ai-rd-tasks)
  - description: Official release of the seven RE‑Bench AI R&D environments, including starting solutions, scoring functions, and reference solutions used for all experiments in the paper.

- RedPajama-Data · [GitHub](https://github.com/togethercomputer/RedPajama-Data)
  - description: Open corpus used as the starting source to mine GUI-like tutorials; AgentTrek’s tutorial harvesting and filtering are performed on this dataset.

- RLBench · [GitHub](https://github.com/stepjam/RLBench) · [Website](https://sites.google.com/view/rlbench)
  - description: Human-designed robotics benchmark used as a comparison baseline for task and scene diversity.

- RoCo / RoCoBench · [GitHub](https://github.com/roco-llm/roco) · [Website](https://roco-llm.github.io/)
  - description: Multi-robot collaboration system and benchmark that the paper extends into DV-RoCoBench and also uses as a strong baseline (RoCo) and simulator for tabletop tasks.

- Schema-Guided Dialogue (SGD) · [GitHub](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
  - description: Multi-turn dialogue corpus used to generate the paper’s CRA (Conversational ReAct API) data with GPT-4o, forming a key part of CoALM-IT.

- [SciBench](https://github.com/SciBench/SciBench)

- SciBench · [GitHub](https://github.com/JetRunner/SciBench)
  - description: College-level scientific problem-solving benchmark used for out-of-domain testing.

- SciRepEval · [GitHub](https://github.com/allenai/SciRepEval)
  - description: Multi-format benchmark for scientific document representations, cited as a key evaluation resource for graph-aware scientific LLMs.

- SNIPS NLU (repurposed for DST) · [GitHub](https://github.com/snipsco/nlu-benchmark)
  - description: Public intent/slot dataset (Coucke et al., 2018) used by the paper as an additional seed source for generating realistic executable instructions during Android Instruct data construction.
  - description: Single-turn dataset transformed by the authors into dialogue state tracking format for CoALM-IT training.

- SQuAD· [GitHub](https://github.com/rajpurkar/SQuAD-explorer) · [Website](https://rajpurkar.github.io/SQuAD-explorer/) · [Doc](https://huggingface.co/datasets/squad_v2)
  - description: Reading comprehension dataset with answerable/unanswerable questions used to analyze decision protocols on edge cases.
  - description: Dataset used to construct the 25k member-agent training set and to source 100 “needles” for single-needle evaluation in Needle-in-a-Haystack PLUS.

- SRBench · [GitHub](https://github.com/cavalab/srbench)
  - description: Benchmark suite for symbolic regression; used to evaluate and compare against symbolic regression baselines in the constitutive-law task reformulation.

- StableToolBench · [GitHub](https://github.com/THUDM/StableToolBench)
  - description: Benchmark and evaluation framework used for end-to-end tool-use assessment (SoPR/SoWR) with stabilized tasks and GPT-4-based simulation for failed tools.

- ToolBench / ToolLLM · [GitHub](https://github.com/THUDM/ToolBench)
  - description: Benchmark and evaluation suite for multi-turn tool-use the authors use to test pass rate and generalization (Sections 5.1, A.2).

- ToolHallucination (Relign + RelyToolBench) · [GitHub](https://github.com/X-LANCE/ToolHallucination)
  - description: Official release for this paper. Contains the Relign reliability-alignment implementation, the RelyToolBench benchmark and prompts, and scripts to run evaluation metrics (RePR, Benefit-Cost Utility) and hallucination detection.

- TOOLMAKER (includes TM-BENCH) · [GitHub](https://github.com/KatherLab/ToolMaker)
  - description: Official repository released with the paper; contains the TOOLMAKER agent framework, the TM-BENCH benchmark tasks, unit tests, and scripts to reproduce all experiments.

- TriviaQA · [GitHub](https://github.com/mandarjoshi90/triviaqa) · [Website](https://nlp.cs.washington.edu/triviaqa/)
  - description: Compositional QA dataset integrated into AgentBank with a search interface.
  - description: Reading comprehension/QA dataset; used as an information-exchange benchmark in OPTIMA evaluations.

- TS50 and TS500 benchmark sets· [GitHub](https://github.com/drorlab/gvp-pytorch)
  - description: Structure encoder backbone used in ProSST’s quantization module to embed residue-level local structures before k-means clustering.
  - description: Core geometric module underlying the GVP-Transformer encoder used as the structural expert in DPLM’s adapter-tuned inverse folding.
  - description: GVP-based encoder components used to extract invariant backbone geometric features for the structure tokenizer.
  - description: Standard inverse folding test sets employed for evaluation; commonly provided via the GVP PyTorch repository referenced by prior work.

- VisualWebArena (VWA) · [GitHub](https://github.com/web-arena-x/visualwebarena) · [Website](https://visualwebarena.github.io)
  - description: Realistic visual web-browsing benchmark; MMInA follows VWA’s condensed action space and multimodal web interaction setup.
  - description: Provides the multimodal web environments and Set-of-Marks observation interface that VideoWebArena builds upon; also supplies evaluators reused by this paper.
  - description: Realistic multimodal web-agent environment the paper extends to create VWA-Adv; provides the web environments, evaluation primitives, and baseline agents used throughout the experiments.

- VLMbench · [GitHub](https://github.com/UM-ARM-Lab/VLMbench)
  - description: Compositional vision-language manipulation benchmark that EB-Manipulation extends; provides categories such as pick-place, stacking, shape sorting, and wiping.

- WikiHow Dataset · [GitHub](https://github.com/mahnazkoupaee/WikiHow-Dataset) · [Website](https://www.wikihow.com)
  - description: Professional knowledge corpus cited as a source for constructing and organizing workflow-related knowledge in FlowBench.

### Corpora

- Abstraction and Reasoning Corpus (ARC) · [GitHub](https://github.com/fchollet/ARC) · [Website](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)
  - description: Visual reasoning dataset used as the core case study; agents must learn programmatic transformations from examples and predict test outputs.

- Android in the Wild (AITW) · [GitHub](https://github.com/google-research/google-research/tree/master/android_in_the_wild)
  - description: Large-scale dataset of Android device control episodes with a shared action space. Used as a benchmark to train and evaluate agents and to mix with AMEX for improved performance.
  - description: Large-scale Android device-control dataset from which Auto-UI is derived; referenced as a key prior dataset in the paper.

- AndroidControl· [GitHub](https://github.com/microsoft/AndroidControl)
  - description: Large-scale Android GUI-control dataset and benchmark. Used in the paper to train and evaluate SphAgent and to measure gains from adding AMEX Level I/II data.

- AndroidEnv · [GitHub](https://github.com/google-research/android_env)
  - description: RL platform for Android; cited among mobile/online environments to contrast with UI-Vision’s desktop offline benchmark.
  - description: Interactive Android RL platform whose action space the paper draws from; relevant dependency/background for the defined Tap/Swipe/Type/Long-Press/Home/Back action space used in AndroidLab.

- AndroidLab · [GitHub](https://github.com/THUDM/AndroidLab) · [Website](https://arxiv.org/abs/2410.24024)
  - description: Android benchmark with 138 tasks and fine-grained metrics; used to assess MobileUse with the reported success rates.

- ANLI (Adversarial NLI) · [GitHub](https://github.com/facebookresearch/anli) · [Website](https://huggingface.co/datasets/anli)
  - description: Adversarial natural language inference dataset; used to train a verifier and evaluate MAPoRL on logical inference.

- APPS · [GitHub](https://github.com/hendrycks/apps)
  - description: Python coding benchmark; instances were reformatted into trajectories for programming skill training.
  - description: Competitive programming dataset used for contest-level evaluation of CODESIM.

- AQuA-RAT· [GitHub](https://github.com/deepmind/AQuA)
  - description: Multiple-choice math word problems dataset (AQuA-RAT) used for evaluation.
  - description: Multiple-choice math reasoning dataset used in evaluation.
  - description: Multi-choice math reasoning dataset; used in experiments.
  - description: Algebraic word-problem dataset with rationales used as part of the training query pool.
  - description: Algebra question answering with rationales; used to test transferability to multi-choice math reasoning.

- ARC (AI2 Reasoning Challenge) · [GitHub](https://github.com/allenai/ai2-arc) · [Website](https://allenai.org/data/arc)
  - description: Multiple-choice science questions; the ARC-Challenge subset is used in debate experiments.
  - description: Grade-school science QA benchmark used in automatic evaluation.

- Ask-before-Plan · [GitHub](https://github.com/magicgh/Ask-before-Plan)
  - description: Official repository for the paper; releases the Ask-before-Plan dataset, CEP multi-agent code, prompts, and environment modifications used to reproduce results.

- Auto-UI (You Only Look at Screens) · [GitHub](https://github.com/amazon-science/auto-ui)
  - description: Benchmark and data derived from AITW; the authors fine-tune/evaluate ReachAgent on its official split for cross-dataset testing.

- BabyAI · [GitHub](https://github.com/mila-iqia/babyai)
  - description: Grid-world instruction-following environment used as a held-in benchmark; also used in ablations for value-function and critical-step verification.

- Bamboogle · [GitHub](https://github.com/ofirpress/Bamboogle)
  - description: Held-out reasoning benchmark requiring compositional web search; used to test generalization.

- BUTTON (BUTTONInstruct) · [GitHub](https://github.com/PKU-Baichuan-MLSystemLab/BUTTON)
  - description: Official release for the paper, providing the BUTTON pipeline, prompts, and the BUTTONInstruct dataset (8k multi-turn function-calling trajectories) used to train/evaluate models.

- ChatDev· [GitHub](https://github.com/OpenBMB/ChatDev)
  - description: SOP-style multi-agent framework cited in the method section as a contrast to SimClass’s dynamic session controller (referenced when discussing systems with standardized operating procedures).
  - description: Real-world multi-agent software development framework evaluated under AiTM by intercepting agents (CEO/CPO/CTO/Programmer) across phases.
  - description: LLM-powered agent collaborative software development framework used as a single-team baseline and underlying execution environment; the Croto implementation is released as a branch within this repository.
  - description: Dataset used for software generation experiments (15 tasks); provided with the ChatDev project and cited as the task source in the paper.
  - description: Multi-agent software development system used as a task-specific baseline in comparisons.
  - description: Software-development MAS framework; its memory variant (ChatDev-M) provides inside- and cross-trial memory baselines for comparison.

- CIAR (Counter-Intuitive Arithmetic Reasoning) · [GitHub](https://github.com/Skytliang/CIAR)
  - description: Math reasoning dataset with hidden traps; used to measure system resilience on math problem solving.

- CMU-MOSEI · [GitHub](https://github.com/A2Zadeh/CMU-MultimodalSDK) · [Website](https://multicomp.cs.cmu.edu/resources/cmu-mosei/)
  - description: Large-scale multimodal sentiment/emotion dataset and SDK; referenced for modeling emotion/sentiment in conversations and broader multimodal social understanding.

- CogMir · [GitHub](https://github.com/XuanL17/CogMir)
  - description: Official repository for the paper’s open-ended Multi-LLM Agents framework, including code, prompts, and the constructed datasets (Known/Unknown MCQ, Inform, CogScene, CogAction, CogIdentity) used to mirror cognitive biases and reproduce experiments.

- COIG (Chinese Open Instruction Generalist) · [GitHub](https://github.com/BAAI-Open/COIG) · [Website](https://huggingface.co/datasets/BAAI/COIG)
  - description: Included as part of the general reasoning knowledge base for retrieval.

- CommonGen · [GitHub](https://github.com/INK-USC/CommonGen) · [Website](https://inklab.usc.edu/CommonGen/)
  - description: Commonsense composition benchmark; the paper evaluates on the harder CommonGen-Hard subset to measure open-ended generation quality.

- compfiles · [GitHub](https://github.com/dwrensha/compfiles)
  - description: Catalog of Olympiad-style math problems formalized in Lean; used in the initial curriculum ordering and dataset construction.

- Croto (Cross-Team Orchestration) · [GitHub](https://github.com/OpenBMB/ChatDev/tree/macnet)
  - description: Official implementation released by this paper; contains MACNET code to build DAG-based multi-agent topologies and run experiments, along with scripts and resources used in evaluation (including the SRDD benchmark/metrics from the ChatDev project).
  - description: Official code and data release for the paper; implements the cross-team orchestration framework (greedy aggregation, hierarchical partitioning, pruning) on top of ChatDev for software and story-generation experiments.

- Decomposed Prompting (DecomP) · [GitHub](https://github.com/allenai/DecomP)
  - description: Official repository released by the paper; provides datasets, prompts, and code to run the decomposer, sub-task handlers, and all experiments.

- DialogStudio · [GitHub](https://github.com/salesforce/DialogStudio)
  - description: Unified multi-domain instruction/conversation datasets used as part of the general instruction-tuning mixture for xLAM (Section 3.5).

- DiffAb · [GitHub](https://github.com/luost26/diffab)
  - description: Diffusion-based antigen-specific antibody design; used as a baseline (with multiple samples) in the antibody benchmark.

- dockur/windows · [GitHub](https://github.com/dockur/windows) · [Codewiki](https://codewiki.google/github.com/dockur/windows)
  - description: Windows-in-Docker image the authors adapt to deploy a Windows 11 VM within their Docker-based benchmark infrastructure.

- e3nn (Tensor Field Networks) · [GitHub](https://github.com/e3nn/e3nn)
  - description: Library for SO(3)/SE(3) equivariant neural networks; used for TFN and spherical tensor operations in the benchmark.

- EasyJailbreak · [GitHub](https://github.com/EasyJailbreak/EasyJailbreak)
  - description: An open-source framework for jailbreaking LLMs cited as a recent benchmark; relevant for extending evaluations beyond the paper’s main setup.

- EGNN · [GitHub](https://github.com/vgsatorras/egnn)
  - description: E(n)-equivariant GNN baseline evaluated across tasks in the benchmark.

- EnvDistraction · [GitHub](https://github.com/xbmxb/EnvDistraction)
  - description: Official code release for this paper. Contains scripts to simulate environmental distractions in GUIs, implement the three working patterns, run evaluations, and reproduce results on the proposed dataset.

- Flickr30k Entities · [GitHub](https://github.com/BryanPlummer/flickr30k_entities) · [Website](https://bryanplummer.com/Flickr30kEntities/)
  - description: Phrase-to-region grounding dataset used alongside COCO when fine-tuning Grounding DINO to retain open-set grounding skills.

- Graphein · [GitHub](https://github.com/a-r-j/graphein)
  - description: Library for geometric deep learning on biomolecular structures; listed as a dependency for dataset/graph construction.

- HumanAct12 · [GitHub](https://github.com/EricGuo5513/action-to-motion)
  - description: Action-conditioned motion dataset contributing sequences to HumanML3D as noted in the experimental setup.

- HumanML3D · [GitHub](https://github.com/EricGuo5513/HumanML3D)
  - description: Dataset of 3D human motions with text descriptions; main training and evaluation set for MotionLLM and Motion-Agent.

- idpGAN · [GitHub](https://github.com/feiglab/idpGAN)
  - description: GAN-based baseline for sequence-conditioned conformation ensembles; authors ran the official code for comparisons.
  - description: GAN-based generator for intrinsically disordered protein ensembles; used as an open-source baseline on the IDP benchmark.

- IIRC · [GitHub](https://github.com/allenai/iirc) · [Website](https://allenai.org/data/iirc)
  - description: Incomplete Information Reading Comprehension dataset; used as another evaluation benchmark (decontextualized subset) for AOP.

- Lean Math Workshop · [GitHub](https://github.com/yuma-mizuno/lean-math-workshop)
  - description: Detailed Lean tutorial and exercises; part of the initial curriculum and dataset generation.

- LexicalRichness · [GitHub](https://github.com/LSYS/lexicalrichness)
  - description: Python package used in dataset quality analysis to compute lexical diversity metrics for generated questions.

- LIBERO · [GitHub](https://github.com/UT-Austin-RPL/LIBERO) · [Website](https://libero-project.github.io/)
  - description: Long-horizon, compositional manipulation benchmark used to evaluate ThinkAct, including few-shot adaptation experiments.

- mABC · [GitHub](https://github.com/knediny/mABC)
  - description: Official repository for the paper’s framework and resources; contains the MABC multi-agent system implementation and the newly released Train-Ticket-based dataset used for experiments and ablations.

- MACE · [GitHub](https://github.com/ACEsuit/mace)
  - description: Higher‑order equivariant GNN; included as an evaluated architecture class in the benchmark.

- MasRouter · [GitHub](https://github.com/yanweiyue/masrouter)
  - description: Official code release of the paper; implements the cascaded controller (collaboration determiner, role allocator, LLM router) and experiments across benchmarks.

- mathlib4 · [GitHub](https://github.com/leanprover-community/mathlib4)
  - description: The community mathematics library for Lean 4; ReProver’s retriever is pre-trained/fine-tuned on mathlib4 and its premises are part of the retrieval corpus used by LeanAgent.

- MathVista · [GitHub](https://github.com/lupantech/MathVista) · [Website](https://mathvista.github.io/)
  - description: Visual mathematical reasoning benchmark; used to test multimodal MAD (with GPT-4o) and measure accuracy/cost trade-offs under different topologies.
  - description: Visual math benchmark used both as an evaluation benchmark and as part of the multimodal retrieval corpus.
  - description: Visual math benchmark cited for multimodal evaluation; used in method and benchmark discussions to test diagram/figure-based reasoning.
  - description: Visual mathematics benchmark for geometry and multimodal reasoning; used to evaluate geometry-capable LLMs.

- MBPP· [GitHub](https://github.com/google-research/google-research/tree/master/mbpp) · [Website](https://huggingface.co/datasets/mbpp)
  - description: Python program synthesis benchmark; used to evaluate code generation performance.
  - description: Python coding benchmark used to measure AiTM’s targeted behavior injection and DoS attack success rates.
  - description: Beginner-level Python tasks; used to construct programming trajectories and evaluation.
  - description: Programming benchmark used to evaluate MegaAgent’s foundational performance.
  - description: Python coding problems benchmark; used to compare against MAS and routing baselines.
  - description: Python program synthesis dataset used in training and evaluation.
  - description: Basic Python programming benchmark used to evaluate CODESIM (including MBPP and MBPP-ET variants reported).

- MGSM (Multilingual Grade School Math) · [GitHub](https://github.com/google-research/google-research/tree/master/mgsm)
  - description: Multilingual math word-problem benchmark used as a primary search domain and source of agents for cross-domain transfer.

- miniF2F-Lean4 · [GitHub](https://github.com/yangky11/miniF2F-lean4)
  - description: Lean4 port of MiniF2F benchmark repository; used as a new repo appended after the initial curriculum to demonstrate LeanAgent’s ability to extend to fresh repositories and for a Lean4 test-set Pass@1 comparison.

- MiniWoB++ · [GitHub](https://github.com/miniwob/miniwob-plusplus)
  - description: Classic web GUI benchmark referenced for context; contrasts with UI-Vision’s desktop focus.
  - description: Diverse web interaction tasks; used as a held-out web benchmark for generalization.
  - description: Classic GUI/web task environment cited among web-agent benchmarks; provides context for agent evaluation beyond function calling.

- MiniWoB++ · [GitHub](https://github.com/google-deepmind/miniwob-plusplus)
  - description: Interactive synthetic web tasks benchmark; used to evaluate OpenHands’ browsing agent.

- MMInA · [GitHub](https://github.com/shulin16/MMInA)
  - description: Official code and data release for the MMInA benchmark; contains tasks, evaluation protocol, and scripts to run and assess multihop multimodal Internet agents.

- MM‑Vet · [GitHub](https://github.com/yuweihao/MM-Vet) · [Website](https://mm-vet.github.io)
  - description: Open‑ended multimodal reasoning benchmark; the paper evaluates all MLLM methods here and uses the official GPT‑4–based evaluator provided by MM‑Vet.

- MORE (Quantifying and Mitigating Unimodal Biases) · [GitHub](https://github.com/OpenCausaLab/MORE)
  - description: Official project page/repository for this paper; releases the MORE dataset (12k MCQ VQA with causal rationales) and CAVE framework for bias mitigation and evaluation.

- MultiWOZ 2.4 · [GitHub](https://github.com/smartyfh/MultiWOZ2.4)
  - description: TOD benchmark used to evaluate Success and JGA; the paper runs zero-shot evaluations on its 2.4 test set.

- Needle-in-a-Haystack PLUS · [GitHub](https://github.com/zuucan/NeedleInAHaystack-PLUS)
  - description: Official benchmark released by the paper for long-document QA with single-needle and multi-needle tasks; used throughout to evaluate LONGAGENT up to 128k tokens and to test reasoning while preventing data leakage.

- OpenCompass · [GitHub](https://github.com/open-compass/opencompass) · [Website](https://opencompass.org.cn)
  - description: Evaluation framework referenced for generative/discriminative scoring (e.g., CircularEval) in benchmarking LLM/MLLM mathematical reasoning.

- Papyrus Scripts · [GitHub](https://github.com/OlivierBeq/Papyrus-scripts)
  - description: Scripts around the Papyrus dataset; used to construct chemistry tasks and analyses.

- ProFSA · [GitHub](https://github.com/bowen-gao/ProFSA)
  - description: Official code and data release for the paper; contains the pocket encoder, training pipeline, and the large pseudo ligand–pocket dataset used to pretrain/evaluate ProFSA.

- ProtT3 · [GitHub](https://github.com/acharkq/ProtT3)
  - description: Official code release for the paper, including training/evaluation scripts, dataset processing, and pretrained checkpoints for the proposed protein-to-text generation framework.

- PyBullet · [GitHub](https://github.com/bulletphysics/bullet3) · [Website](http://pybullet.org/)
  - description: Physics engine integrated with Habitat; the paper disables PyBullet-based physics during benchmark runs but notes it can be re-enabled for extended tasks.

- QMSum · [GitHub](https://github.com/Yale-LILY/QMSum)
  - description: Meeting summarization dataset used to source non-policy negative samples in the LLM-as-a-Judge validation of policy reasonableness.

- RAVEN · [GitHub](https://github.com/WellyZhang/RAVEN)
  - description: Visual reasoning dataset used to supplement EMMA’s math Pattern Inference category with inherently multi-hop visual reasoning tasks.

- ReHAC · [GitHub](https://github.com/XueyangFeng/ReHAC)
  - description: Official code and released datasets for the paper’s Reinforcement Learning-based Human-Agent Collaboration method; primary resource to reproduce experiments and train the collaboration policy.

- ReProver (LeanDojo)· [GitHub](https://github.com/lean-dojo/LeanDojo) · [Website](https://leandojo.org)
  - description: Open-source framework used to parse Lean repos, trace proofs, build datasets (LeanDojo Benchmark 4), and run retrieval-augmented proving; LeanAgent relies on it for data extraction, dataset generation, and the proving pipeline.
  - description: Retrieval-augmented baseline prover from LeanDojo; LeanAgent initializes from ReProver’s ByT5 retriever and uses its tactic generator, and compares against ReProver as the main baseline.

- S2ORC: Semantic Scholar Open Research Corpus · [GitHub](https://github.com/allenai/s2orc) · [Website](https://allenai.org/data/s2orc)
  - description: Paper sources for multiple CURIE tasks (e.g., DFT, MPV, HFD/HFE, GEO) were selected from S2ORC; useful to replicate paper selection and extend the benchmark.
  - description: Large corpus of scholarly literature cited as the data backbone for paper/meta-data access; relevant for reproducing the paper’s literature analysis.

- SaProt · [GitHub](https://github.com/DeepGraphLearning/SaProt)
  - description: Provides datasets and benchmarks for downstream predictive tasks where DPLM-2 representations are evaluated against sequence-only and structure-aware baselines.

- SciDocs · [GitHub](https://github.com/allenai/scidocs)
  - description: Benchmark suite for scientific document representations; used in the survey to evaluate link prediction, recommendation, and retrieval for graph-aware LLMs.

- ScreenSpot · [GitHub](https://github.com/Sea-Snell/ScreenSpot)
  - description: GUI grounding dataset cited as prior work focusing on grounding rather than full desktop interaction; contrasted with UI-Vision’s broader tasks.

- ScreenSpot-Pro · [GitHub](https://github.com/Sea-Snell/ScreenSpot-Pro)
  - description: Professional high-resolution GUI grounding dataset referenced as prior grounding-focused work; UI-Vision expands to layout and actions.

- self-ask · [GitHub](https://github.com/ofirpress/self-ask)
  - description: Baseline method compared in the appendix; also hosts the Bamboogle dataset.

- ShowUI · [GitHub](https://github.com/showlab/ShowUI)
  - description: Vision-language-action GUI agent included as an open-source baseline, especially for action prediction.
  - description: GUI grounding dataset/model used in the paper’s Stage-1 grounding mix to initialize perception/grounding for the CUA models.

- ShowUI · [GitHub](https://github.com/ShowLab/ShowUI)
  - description: Vision-language-action GUI agent included as an open-source baseline, especially for action prediction.
  - description: GUI grounding dataset/model used in the paper’s Stage-1 grounding mix to initialize perception/grounding for the CUA models.

- SimplerEnv · [GitHub](https://github.com/google-research/simpler-env)
  - description: Simulation benchmark (Google-VM, Google-VA, Bridge-VM) where ThinkAct is evaluated for robot manipulation robustness.

- Social-AI Community Resource · [GitHub](https://github.com/l-mathur/social-ai)
  - description: Official repository released with the paper; a continually updated collection of papers, datasets, benchmarks, simulators, and courses to support research on the technical challenges outlined in the position paper.

- SOTOPIA· [GitHub](https://github.com/sotopia-lab/sotopia) · [Website](https://sotopia.world) · [Doc](https://pypi.org/project/sotopia/)
  - description: Related interactive social‑intelligence benchmark cited and compared in the paper’s discussion/table; useful for practitioners to inspect complementary evaluation settings.
  - description: The simulation/evaluation framework the paper builds on; used to run AGENTS and MINDREADERS modes via its state-space agent library and to compute goal completion metrics.
  - description: Interactive environment for evaluating social intelligence in language agents; cited as a dynamic setting to study dyadic and multi-party interactions.

- Spider (Text-to-SQL) · [GitHub](https://github.com/taoyds/spider) · [Website](https://yale-lily.github.io/spider)
  - description: Large-scale text-to-SQL dataset from which the InterCodeSQL database is constructed; provides schemas and queries underlying the interactive SQL evaluations.

- Stanford Alpaca· [GitHub](https://github.com/tatsu-lab/stanford_alpaca) · [Codewiki](https://codewiki.google/github.com/tatsu-lab/stanford_alpaca)
  - description: The instruction-following (Instruction-Input-Output) format adopted for supervised fine-tuning of SMARTAgent.
  - description: Instruction dataset used to synthesize a pseudo-preference dataset for DPO training to bias models toward following the user goal.
  - description: Dataset/model reference used to construct the comparison baseline “gpt‑2 (small) finetuned on Stanford Alpaca” in the QA finetuning environment.
  - description: Provides the Alpaca-style data generation recipe the authors adapt to expand user queries and plans.
  - description: Instruction-tuned LLaMA baseline included in comparisons.
  - description: Seed instruction data and training code; authors expanded Alpaca to 70k and retrained Alpaca-13B as a baseline and initial seed for evolution.

- Super-NaturalInstructions (SNI) · [GitHub](https://github.com/allenai/natural-instructions)
  - description: Alternative instruction dataset used in ablations (random 70k sampled to train an LLaMA-13B baseline).

- SVAMP · [GitHub](https://github.com/arkilpatel/SVAMP)
  - description: Math reasoning dataset used for evaluation; the paper compares accuracy and token cost on SVAMP.
  - description: Math word-problem dataset used in additional transfer evaluations from MGSM-discovered agents.
  - description: Math word-problem dataset assessing robustness to variations; used in evaluation.
  - description: Math reasoning dataset; part of the paper’s benchmark suite.
  - description: Out-of-domain math word problem benchmark used for generalization evaluation.

- TAPE · [GitHub](https://github.com/songlab-cal/tape) · [Doc](https://tape.readthedocs.io)
  - description: Source of the Metal Ion Binding dataset and splits used in supervised fine-tuning comparisons.
  - description: Source for secondary-structure prediction (SSP) data and evaluation protocol used to train the classifier providing discrete guidance for controllable generation.
  - description: Provides the contact map prediction task used to evaluate structural understanding of model representations.
  - description: The paper follows TAPE splits for four tasks (secondary structure, contact prediction, remote homology, stability).
  - description: Benchmark and evaluation protocol followed for secondary structure prediction and contact prediction to assess protein understanding.
  - description: Source of the secondary structure dataset and splits used for residue-level classification probes.

- Time-Series-Library (TSLib) · [GitHub](https://github.com/thuml/Time-Series-Library)
  - description: Repository providing the Weather and Electricity forecasting datasets and evaluation setup used in the experiments.

- TimeGAN · [GitHub](https://github.com/jsyoon0823/TimeGAN)
  - description: Baseline time-series generative model used for comparison in fidelity metrics across datasets.

- Train-Ticket · [GitHub](https://github.com/FudanSELab/train-ticket)
  - description: Open-source microservices benchmark from Fudan University; used by the paper as the application environment to construct the Train-Ticket dataset and simulate RCA scenarios.

- Uni-Mol · [GitHub](https://github.com/dptech-corp/Uni-Mol) · [Website](https://openreview.net/forum?id=6K2RM6wVqKu)
  - description: Pretrained 3D molecular representation model used to compute quantum mechanical property values; the paper uses a UniMol model fine-tuned on QM9.
  - description: 3D molecular representation learning framework; its released dataset (19M molecules, 209M conformations) is used for ESM-AA pretraining and it serves as a molecule-encoder baseline.
  - description: Universal 3D molecular representation framework; ProFSA uses the official pretrained molecular encoder and architecture from Uni-Mol and adopts Uni-Mol's pocket druggability dataset and pocket construction settings.

## Database
- LevelDB · [GitHub](https://github.com/google/leveldb)
- FAISS· [GitHub](https://github.com/facebookresearch/faiss) · [CodeWiki](https://codewiki.google/github.com/facebookresearch/faiss)
- Chroma (Vector Database) · [GitHub](https://github.com/chroma-core/chroma) · [CodeWiki](https://codewiki.google/github.com/chroma-core/chroma)
- Milvus · [GitHub](https://github.com/milvus-io/milvus) · [CodeWiki](https://codewiki.google/github.com/milvus-io/milvus)

## LLM Models

### Huggingface
- Hugging Face Transformers· [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers/agents)
  - description: Library used via the “Hugging Face interface” to run and fine-tune Mistral-7B-Instruct; required to reproduce local fine-tuning and inference.
  - description: External agent toolbox used in ablation (7-tool setup) to compare against the proposed three-agent design.
  - description: Library used to load and run open-source MLLMs during EMMA’s experiments.
  - description: Library used in generated training code (e.g., BERT/RoBERTa/DeBERTa models) within DS-Agent’s experiments.
  - description: Core modeling library used for hosting/fine-tuning LMs in DSPy experiments (listed in Appendix F).
  - description: Transformer implementations; used to implement S-T5 and S-GPT structure language models (encoder-decoder and decoder-only variants).
  - description: Library and model hub used to load and fine-tune the THUDM/cogagent-chat-hf checkpoint during meta-learning and adaptation.
  - description: Training framework used for full-parameter supervised fine-tuning of Llama3 and Qwen2 models.
  - description: Core training/inference library the authors base their SFT/DPO code on (Section 4.1).

### Meta
- Ollama · [GitHub](https://github.com/ollama/ollama) · [Codewiki](https://codewiki.google/github.com/ollama/ollama) · [Website](https://ollama.com/) · [Doc](https://ollama.com/docs)
  - description: Local model runner used to host open-weight LLaMA 3.1 models in the experiments.
  - description: Runtime used by the authors to obtain and run nine open‑source LLM checkpoints; enables local inference for models like Llama, Gemma, and Qwen.
  - description: Local LLM runner used by the authors to deploy Qwen-2.5 models for reproducing MAS experiments.
  - description: Local serving/runtime the authors note for integrating Qwen 2.5 into their simulator setup.

- Llama 2 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open-weight LLM family from Meta; used as a baseline/backbone across multiple agent frameworks and fine-tuning experiments.

### Deepseek
- DeepSeek-V2 · [GitHub](https://github.com/deepseek-ai/DeepSeek-V2)
  - description: Additional LLM backend tested in the appendix to demonstrate MindSearch generalization.
- DeepSeek-R1 · [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) · [Website](https://www.deepseek.com)
- Open R1 · [GitHub](https://github.com/huggingface/open-r1)
- DeepSeek-Coder · [GitHub](https://github.com/deepseek-ai/deepseek-coder) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/deepseek-coder)
  - description: Open-source code LLM baselines (6.7B/33B instruct) evaluated as code agents in the experiments.
- DeepSeek-Coder-V2 · [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-Coder) · [Model](https://huggingface.co/deepseek-ai/deepseek-coder-v2-instruct)
  - description: Open-source code-focused LLM corresponding to the DeepSeek-V2.5 model used in the experiments; repository provides model access and usage guidance for reproduction.
  - description: Open-source code LLM baselines (6.7B/33B instruct) evaluated as code agents in the experiments.
  - description: Code-focused LLM used in Appendix experiments as an expert Code Agent within AOP.


### Qwen
- Qwen (Qwen2/2.5)· [GitHub](https://github.com/QwenLM/Qwen) · [Website](https://qwenlm.github.io)
- Qwen2.5-0.5B Model · [Model](https://huggingface.co/Qwen/Qwen2.5-0.5B)
- Qwen2.5-Coder-7B-Instruct · [Model](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- Qwen2.5-Math-PRM-7B Model · [Model](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B)
- Qwen3 Blog · [Blog](https://qwenlm.github.io/blog/qwen3/)
- Qwen2-VL· [GitHub](https://github.com/QwenLM/Qwen2-VL)
- Qwen-Coder · [GitHub](https://github.com/QwenLM/Qwen-Coder)
- Qwen-Agent · [GitHub](https://github.com/QwenLM/Qwen-Agent)
- Qwen (Qwen2/2.5)· [GitHub](https://github.com/QwenLM/Qwen) · [Codewiki](https://codewiki.google/github.com/QwenLM/Qwen) · [Website](https://qwenlm.ai/)
  - description: Open-source LLM baseline controller (Qwen1.5‑72B‑chat) used for comparison.
  - description: Open LLM family tested as another backbone to analyze ResearchAgent’s robustness across models.
  - description: Major open-source LLMs used as proposers and as the final aggregator (e.g., Qwen1.5-110B-Chat, Qwen1.5-72B-Chat).
  - description: Alternative backbone models used for scaling and backbone studies (0.5B–7B Chat variants).
  - description: Base AR model; the authors fine-tune Qwen2.5‑32B‑Instruct to obtain Multiverse‑32B.

- Qwen-Audio · [GitHub](https://github.com/QwenLM/Qwen-Audio)
  - description: Audio-language model used as a baseline and leveraged for audio tasks within the study.

- Qwen-VL· [GitHub](https://github.com/QwenLM/Qwen-VL) · [Website](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  - description: Open-source vision-language baseline evaluated on GroundUI, IDMBench, and CriticBench; useful for extending experiments with open models.
  - description: Open-source vision–language chat model used as a strong baseline in experiments.
  - description: Open-source multimodal Qwen model evaluated as a generalist baseline.
  - description: Alternative open-source LVLM evaluated as a backbone in the appendix to verify the generality of the method across models.
  - description: Earlier Qwen multimodal model used as an open baseline for VAB fine-tuning and evaluation.
  - description: Open-source vision-language model fine-tuned to build MMRole-Agent and also used to train the specialized reward model; serves as a baseline MRPA (Qwen-VL-Chat).
  - description: Open-source VLM baseline; also fine-tuned on the paper’s page navigation split for comparison.
  - description: Open-source vision-language model baseline tested on MORE (7B).

- Qwen2· [GitHub](https://github.com/QwenLM/Qwen2) · [Model](https://huggingface.co/Qwen/Qwen2-Math-7B-Instruct)
  - description: Open-source LLM family from Alibaba; Qwen2‑7B‑Instruct is used as a baseline and fine-tuned to create the “Qwen2‑7B‑Proactive” model.
  - description: Open-source LLMs (Qwen2-7B-Instruct, Qwen2-72B-Instruct) used in additional experiments to demonstrate FlowBench’s applicability to open-source models.
  - description: Text LLM backbones for PRM and open-source MLLMs used in experiments.
  - description: Task-specific math LLM used in Appendix experiments as an expert Math Agent to extend AOP.
  - description: External model used to generate intentionally incorrect/irrelevant answers in challenge experiments.
  - description: Alibaba’s open LLM series evaluated in the paper, with detailed conformity behaviors reported.
  - description: Base and instruction models (Qwen2-7B/72B) fine-tuned and compared in experiments.
  - description: Open-source models used to evaluate MALR across sizes (1.5B–72B) and analyze scaling behavior and effectiveness on smaller LLMs.
  - description: Open-source LLM family (7B/72B) used as baselines and for episodic/parametric memory-updating experiments and attribution analyses.

- Qwen2-VL· [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://qwenlm.github.io) · [Doc](https://github.com/QwenLM/Qwen2-VL#readme)
  - description: The other VLM backbone the authors fine-tune with MM‑Traj to build T3‑Agent (Qwen2‑VL‑7B).
  - description: Vision-language model backbone; the 7B variant is fine-tuned to produce Explorer-7B and used in several evaluations.
  - description: Open-source VLM baseline evaluated for element and layout grounding.
  - description: Open-source MLLM baseline evaluated on EMMA; link to model family repo and model card for replication.
  - description: Vision-language model fine-tuned with AgentTrek multimodal trajectories to build the vision-based web agent.
  - description: Open LMM baseline fine-tuned on VAB trajectories; used for comparison across all environments.
  - description: Vision-language model used as a comparison baseline on the ScreenSpot evaluation.
  - description: Open-source MLLM backbone used in experiments with AR-MCTS.
  - description: Vision–language model baseline used for image and video comparisons.
  - description: Open-source vision-language model evaluated on SCIVERSE; results reported for knowledge and CoT metrics.
  - description: Base multimodal model used as the Visual Element Locator; the paper fine-tunes Qwen2‑VL‑Instruct with LoRA on GUI data to predict on-screen coordinates.
  - description: Open-source VLM family used as base models (e.g., Qwen2-VL-7B) for supervised fine-tuning into OpenCUA variants.
  - description: Open-source multimodal LLM family used both as baselines (7B/72B) and, in ablations, as the autonomous task proposer and outcome evaluator.

- Qwen2.5· [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io) · [Doc](https://huggingface.co/Qwen)
  - description: Open-source LLM backbone (Qwen-2.5-72B-Instruct) accessed via API; used for evaluations and transfer experiments.
  - description: Open‑source models used both as agent backbones (7b/14b/72b) and as judge models.
  - description: Qwen model used as an alternative/heterogeneous agent in experiments assessing multi-model collaboration with MAPoRL.
  - description: Another base LLM used (Qwen2.5-72B-Instruct) in the experiments.
  - description: Alternative base models (1.5B–14B) used to study ToolGen’s scaling across model sizes.
  - description: Open‑source baseline models used in the paper to test generality of the proposed attribution methods across model families.
  - description: LLM family fine-tuned on AgentTrek text trajectories (AXTree + Playwright actions) to build the text-based web agent.
  - description: Open-source LLM used as one of the main models in experiments; enables reproducing the open-weight model results reported.
  - description: Open-source models used as Mauto baselines in all scenarios; specific Instruct checkpoints (7B/32B/72B) are evaluated.
  - description: Base models fine-tuned by the authors (full-parameter SFT and LoRA/mDPO) to produce MAGNET‑7B/14B variants.
  - description: Another base model fine-tuned and evaluated in the experiments (Qwen2.5-7B-Instruct).
  - description: Open-source LLM backbone (7B/14B) used for most experiments; models are locally served in the paper’s setup.
  - description: Open-source LLM family with function-calling and parallel tool invocation; Qwen2.5-7B-Instruct is evaluated as a strong open-source baseline.
  - description: Alternative backbone model used to test framework adaptability; integrated via Ollama in the paper’s experiments.
  - description: Additional LLM backend (Qwen-2.5-7B) evaluated in the appendix for generalization.

- Qwen2.5-Math · [GitHub](https://github.com/QwenLM/Qwen2.5-Math) · [Website](https://qwenlm.github.io)
  - description: Math-specialized models cited in the Math-LLM progress timeline; commonly used as strong baselines and for self-improvement instruction tuning.

- Qwen2.5-VL · [GitHub](https://github.com/QwenLM/Qwen2.5-VL) · [Website](https://qwenlm.github.io/)
  - description: Newer Qwen VLM variant also evaluated as an open-source baseline for grounding.
  - description: Open-source multimodal LLM family; the paper uses Qwen2.5-VL-72B/32B/7B Instruct as the backbone model for MobileUse.
  - description: Used as a primary VLM backbone (7B and 72B variants) for all experiments; GAM-Agent is layered on top of Qwen2.5-VL to evaluate improvements on MMMU, MMBench, MVBench, and V*Bench.
  - description: Newer Qwen VL model cited and evaluated in SCIVERSE for comparison with other open-source LMMs.
  - description: The enhanced high-resolution VLMs (e.g., 32B/72B) that serve as the primary backbones for OpenCUA-32B/72B.
  - description: Multimodal LLM backbone for ThinkAct’s reasoning module; cold-started via SFT and then optimized with GRPO.

- QwQ‑32B‑Preview (Qwen Team) · [GitHub](https://github.com/QwenLM/QwQ) · [Website](https://huggingface.co/qwen/QwQ-32B-Preview)
  - description: Open-source “reasoning” model variant evaluated (marked as o1‑like in the paper) to study catastrophic and deceptive behaviors.

### General & Task-specific Models

- AutoGPT · [GitHub](https://github.com/Significant-Gravitas/AutoGPT) · [Website](https://auto-gpt.ai/)
  - description: Third-party autonomous agent integrated into IoA and also used as a baseline; IoA orchestrates AutoGPT within teams on open-ended tasks.
  - description: Baseline system; an autonomous LLM agent framework with planning and tool use against which MACNET is compared.
  - description: Referenced agent-based modeling framework; cited as foundational to the LLM-agent paradigm that AgentReview builds upon and useful for extensions to multi-agent setups.
  - description: Referenced as a representative autonomous agent framework with memory and tool use; cited in related work to contrast with KG-Agent’s open-source 7B setup.
  - description: Autonomous agent scaffold; INFOGENT’s Direct API-Driven Navigator is built upon an AutoGPT-style tool-using agent.
  - description: Popular autonomous agent framework; adopted as a baseline system for comparison.

- BERTScore · [GitHub](https://github.com/Tiiiger/bert_score) · [Doc](https://bert-score.readthedocs.io/)
  - description: Metric used to quantify similarity between reviews and meta-reviews when evaluating AC involvement strategies.
  - description: Reference-based evaluation metric used to automatically compare generated stories with human references.
  - description: Semantic similarity metric referenced for evaluating generated clinical text (e.g., reports, summaries) produced by LLM agents.
  - description: Similarity metric used in the self-instruct filtering pipeline to curate and deduplicate generated demonstrations for AgentToken training.

- ChemBERTa · [GitHub](https://github.com/seyonechithrananda/chemberta)
  - description: RoBERTa-style molecular language model on SMILES data included among encoder-only molecular LLMs.

- ClinicalBERT · [GitHub](https://github.com/EmilyAlsentzer/clinicalBERT)
  - description: EHR-adapted BERT referenced for clinical tasks (NER, NLI, prediction) in the survey.

- CodeAct · [GitHub](https://github.com/xingyaoww/CodeAct)
  - description: Referenced agent approach incorporating executable code actions; included as a closely related open-source implementation for practitioners.

- CodeContests · [GitHub](https://github.com/deepmind/code_contests)
  - description: DeepMind CodeContests dataset used as a competitive programming benchmark in the paper.
  - description: Competitive programming dataset used to build the Contest-level Coding tasks, including E2E standard, planner-shift, solver-shift, problem parsing, and self-correction.

- CodeScientist · [GitHub](https://github.com/allenai/codescientist)
  - description: Official open-source implementation of the CodeScientist system introduced in the paper; includes the end-to-end pipeline, codeblock library, instrumented execution sandbox, prompts, and example experiments used for the reported results.

- CODESIM · [GitHub](https://github.com/KagNLP/codesim.github.io) · [Website](https://kagnlp.github.io/codesim.github.io/)
  - description: Official release of the paper’s multi‑agent, simulation-driven planning/coding/debugging framework; the site hosts the open-source code and project page referenced in the paper.

- ColBERT / ColBERTv2 · [GitHub](https://github.com/stanford-futuredata/ColBERT) · [Website](https://colbert.ai)
  - description: Late-interaction neural retriever used as the default search retriever in DSPy (built-in dspy.Retrieve) and for HotPotQA experiments; the paper mentions using a ColBERTv2 retrieval server.

- ConceptNet 5 · [GitHub](https://github.com/commonsense/conceptnet5) · [Website](https://conceptnet.io) · [Doc](https://github.com/commonsense/conceptnet5/wiki)
  - description: Commonsense knowledge graph referenced and used in codeblocks (e.g., knowledge-graph memory/lookup) to augment agents and analyses.

- Direct Preference Optimization (DPO) · [GitHub](https://github.com/eric-mitchell/direct-preference-optimization)
  - description: Preference-based RL algorithm adopted by the paper in the second stage to optimize ReachAgent with constructed preference pairs.
  - description: Reference implementation of DPO; the paper optimizes agents with outcome-level and step-level DPO losses as part of its mixed objective.

- Flan‑T5 (Instruction‑tuned T5) · [GitHub](https://github.com/google-research/FLAN) · [Doc](https://huggingface.co/google/flan-t5-xxl)
  - description: Smaller open models (Large/XL/XXL) used to test model-scale effects for the decomposition framework in open-domain QA.

- FrugalGPT · [GitHub](https://github.com/stanford-futuredata/FrugalGPT) · [Doc](https://arxiv.org/abs/2305.05176)
  - description: Cost-aware LLM routing baseline included in comparisons.

- Gemma 2 · [GitHub](https://github.com/google-deepmind/gemma) · [Website](https://ai.google.dev/gemma) · [Doc](https://ai.google.dev/gemma/docs)
  - description: Google’s open models tested on BENCHFORM to analyze conformity and independence rates.

- GPT-Engineer · [GitHub](https://github.com/AntonOsika/gpt-engineer) · [Codewiki](https://codewiki.google/github.com/AntonOsika/gpt-engineer)
  - description: Single-agent software generation system used as a comparison baseline in the experiments.

- GPT-J 6B · [GitHub](https://github.com/kingoflolz/mesh-transformer-jax) · [Doc](https://huggingface.co/EleutherAI/gpt-j-6B)
  - description: Open-source model used as a Critic backbone; fine-tuned with LoRA on trajectory–reward data.

- GPTSwarm · [GitHub](https://github.com/SchmidhuberAI/GPTSwarm)
  - description: Multi-agent optimization framework used as an automated baseline in experiments.

- GPTSwarm · [GitHub](https://github.com/microsoft/gptswarm)
  - description: Graph-based multi-agent framework referenced for appendix experiments (star vs. complete topologies) to validate that hierarchical oversight improves robustness.

- GPTSwarm · [GitHub](https://github.com/metauto-ai/GPTSwarm)
  - description: Optimizable-graph multi-agent framework used as a strong baseline and as a backbone combined with AgentPrune; authors note minor code modifications for fair comparison.

- GPTSwarm · [GitHub](https://github.com/VITA-Group/GPTSwarm)
  - description: Agents-as-graphs framework with optimizable collaboration patterns; used as a multi-agent comparison baseline.

- Hugging Face TRL (Transformer Reinforcement Learning) · [GitHub](https://github.com/huggingface/trl) · [Codewiki](https://codewiki.google/github.com/huggingface/trl) · [Doc](https://huggingface.co/docs/trl/index)
  - description: Training utilities referenced among the fine-tuning stack used in experiments (Appendix F).

- LLM-Blender · [GitHub](https://github.com/yuchenlin/LLM-Blender) 970
  - description: Ensemble framework with PairRanker and GenFuser; treated as a spatial message-passing baseline for comparing response fusion strategies.

- SPECTER · [GitHub](https://github.com/allenai/specter) 563
  - description: Citation-informed Transformer producing paper embeddings; used as a representative contrastive/document-representation model in the survey.

- Tulu / Open-Instruct · [GitHub](https://github.com/allenai/open-instruct) 3.3K
  - description: Open-source instruction-tuned model suite cited as a comparison baseline for instruction-following agents.
  - description: Training toolkit used by the authors for SFT and as the base framework for their mDPO stage implementation.

- HuggingGPT (JARVIS) · [GitHub](https://github.com/microsoft/JARVIS) 24.5K · [Codewiki](https://codewiki.google/github.com/microsoft/JARVIS) · [Website](https://hugginggpt.github.io/)
  - description: The broader project repo used in the paper (cited as HFmodels) for tool/model integration; the EASYTOOL code lives under this repo.
  - description: Tool-use agent baseline coordinating models/tools via Hugging Face; included among hand-crafted baselines.

- InterCode / InterCodeSQL · [GitHub](https://github.com/princeton-nlp/InterCode) 230 · [Website](https://intercode-bench.github.io/)
  - description: Interactive coding benchmark (SQL split used) providing the environment and “Try Again” prompting setup for coding experiments.
  - description: Interactive coding benchmark; IC-SQL is a held-in programming task and IC-Bash is used as a held-out evaluation.
  - description: Interactive coding benchmark framework used to build the InterCodeSQL environment evaluated in the paper; authors adapt and evaluate their agents on this SQL task setup.

- iTransformer · [GitHub](https://github.com/thuml/iTransformer) 1.9K
  - description: Strong time-series forecasting model used as a human-designed baseline for the Weather/Electricity tasks.

- Language Models as Zero-Shot Planners (baseline) · [GitHub](https://github.com/huangwl18/language-planner) 276
  - description: Baseline LLM planning approach reproduced in the paper for comparisons on Minecraft, ALFWorld, and Tabletop tasks.

- LaWGPT · [GitHub](https://github.com/pengxiao-song/LaWGPT) 6K
  - description: Chinese legal large language model baseline evaluated against AgentsCourt; trained/fine-tuned on legal corpora and instructions.

- LLaMA· [GitHub](https://github.com/facebookresearch/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open foundation/chat models evaluated on the CoordinationQA suite.
  - description: General LLM baseline used for comparison on sequence understanding tasks.

- Llama 3 · [GitHub](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)
  - description: Base model whose 8B variant was fine-tuned to act as IoA’s communication layer LLM in additional experiments.

- Meta Llama 3 (Meta-Llama-3-70B-Instruct) · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
  - description: One of the base LLMs used to build MAS for evaluation (Meta-Llama3-8B-Instruct).
  - description: Base foundation model (Llama‑3‑8B) used for ToolGen training with the Llama‑3 chat template.
  - description: Open-source Meta models used as proposers/aggregators in MoA and in small-model experiments.
  - description: Base model family; the authors fine-tune LLaMA‑3.1‑8B‑Instruct as their main backbone.
  - description: Open LLM family referenced as a backbone in related baselines (LLaMA3-LLaVA-NeXT-8B).
  - description: Open-weight LLMs (8B and 70B) used as agent backbones in several experimental settings.
  - description: Open LLM backbone used as part of the routing pool.
  - description: Open-weight LLM used in ablations to replace GPT-4/3.5 and evaluate model sensitivity to the base LLM.
  - description: The paper uses Llama3-8B as the text encoder inside the diffusion model for conditioning with refined textual descriptions.
  - description: Base foundation model (Llama 3 8B / 3.2 3B) used to instantiate agents for all OPTIMA experiments.
  - description: Base and instruction models (Llama3-8B/70B) fine-tuned and compared in experiments.
  - description: Open‑source LLMs (70B and 8B Instruct) used as base models in the paper’s LLM experiments.
  - description: Evaluated LLM (Llama‑3 Instruct 70B) used for extraction and normalization in all settings.

- Llama 3.1· [GitHub](https://github.com/meta-llama/llama-models) · [Website](https://ai.meta.com/llama/)
  - description: Open-source LLM baseline; authors fine-tune an 8B version to create SCIAGENT-LLAMA3.
  - description: Open-source LLM backbone (llama-3.1-70B) used in cross-model evaluations and transferability tests.

- Llama Guard 3 · [GitHub](https://github.com/meta-llama/llama-guard) · [Doc](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
  - description: Safety classifier model used as a guardrail baseline (LLaMA-Guard 3) in the paper’s comparisons.

- LlamaGuard (Llama Guard 3) · [GitHub](https://github.com/meta-llama/PurpleLlama) · [Doc](https://ai.meta.com/resources/models-and-libraries/llama-guard/)
  - description: Input–output safety classifier from Meta used in the paper as a model-guarding baseline to compare against GuardAgent.

- LlamaIndex · [GitHub](https://github.com/run-llama/llama_index) · [Codewiki](https://codewiki.google/github.com/run-llama/llama_index) · [Website](https://www.llamaindex.ai/)
  - description: Used in Appendix A.5 to extract and match major comments between human and LLM-generated reviews during validation analyses.

- LlamaIndex · [GitHub](https://github.com/jerryjliu/llama_index) · [Website](https://www.llamaindex.ai/)
  - description: Retrieval/agent library discussed as a popular framework relying on prompt engineering; serves as a comparison point to DSPy’s abstractions.

- LMDeploy · [GitHub](https://github.com/InternLM/lmdeploy) · [Doc](https://lmdeploy.readthedocs.io/)
  - description: Toolkit used by the authors to deploy and serve open-source MLLMs during evaluation.

- LMQL (Language Model Query Language) · [GitHub](https://github.com/eth-sri/lmql) · [Website](https://lmql.ai/)
  - description: Related query language for constrained decoding that the paper notes could be used to implement specific advanced DSPy modules.

- LongLLaMA · [GitHub](https://github.com/CStanKonrad/long_llama)
  - description: Open-source long-context model included as a baseline; used to test long-context comprehension on CURIE.

- Magicoder · [GitHub](https://github.com/ise-uiuc/Magicoder)
  - description: Open-source code LLM baseline (Magicoder-S-DS-6.7B) evaluated within MatPlotAgent and in direct decoding comparisons.

- MapCoder · [GitHub](https://github.com/KagNLP/MapCoder)
  - description: Prior multi-agent baseline used for comparison; the authors state they collected all datasets/evaluation setup from this repository for fair comparison.

- MAS-GPT · [GitHub](https://github.com/rui-ye/MAS-GPT)
  - description: Official code release of the paper, providing the MAS generation model, training scripts, and executable MAS examples used in all experiments.

- MDM (Motion Diffusion Model) · [GitHub](https://github.com/GuyTevet/motion-diffusion-model)
  - description: Diffusion-based text-to-motion baseline included in quantitative comparisons.

- Melting Pot Contest 2023 Starter Code · [GitHub](https://github.com/rstrivedi/Melting-Pot-Contest-2023)
  - description: PPO starter pipeline for Melting Pot that the authors used to implement and train their PPO baseline.

- MetaGPT · [GitHub](https://github.com/metagpt-dev/MetaGPT)
  - description: Meta-programming multi-agent framework evaluated as a baseline; authors report results with MetaGPT v0.8.1.

- MindSearch · [GitHub](https://github.com/InternLM/MindSearch)
  - description: Official code release of the paper; contains the multi-agent WebPlanner/WebSearcher implementation used for all experiments.

- Mixtral‑8x7B‑Instruct · [GitHub](https://github.com/mistralai) · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) · [Doc](https://mistral.ai/)
  - description: Open LLM used to instantiate agents for alignment labeling tasks; the paper compares performance/cost of MAD topologies using this model.
  - description: Open-source LLM baseline the authors ran as an agent in their evaluations.
  - description: Evaluated LLM from Mistral AI used for TDMR extraction and normalization experiments.

- ModernBERT · [GitHub](https://github.com/AnswerDotAI/ModernBERT)
  - description: Modern bidirectional encoder architecture; used for masked sentence prediction in the modernbert_predict_masked task.

- MotionGPT · [GitHub](https://github.com/OpenMotionLab/MotionGPT)
  - description: Bidirectional motion–text model used as a comparison baseline; also substituted into Motion-Agent for an ablation of the agent component.

- nanoGPT · [GitHub](https://github.com/karpathy/nanoGPT) · [Codewiki](https://codewiki.google/github.com/karpathy/nanoGPT)
  - description: Lightweight GPT training codebase used as the foundation for the “Fix Embedding” and “Scaling Law Experiment” environments.
  - description: Lightweight autoregressive baseline trained from scratch for unconditional generation comparisons.

- OAG-BERT · [GitHub](https://github.com/THUDM/OAG-BERT)
  - description: Graph-aware academic LLM pre-trained with text and metadata, highlighted as a Type 1.B example.

- Octo · [GitHub](https://github.com/octo-model/Octo)
  - description: Generalist robot policy baseline included in SimplerEnv comparisons.

- OPRO (LLMs as Optimizers) · [GitHub](https://github.com/google-deepmind/opro)
  - description: Prompt-search baseline method used for comparison; LLM-based optimizer for instruction/prompt refinement.

- OPTIMA · [GitHub](https://github.com/thunlp/Optima)
  - description: Official code release of the paper’s framework for training LLM-based multi-agent systems via iterative SFT/DPO and MCTS-inspired data generation; used to reproduce the methods and experiments.

- OWL (Optimized Workforce Learning / WORKFORCE) · [GitHub](https://github.com/camel-ai/owl)
  - description: Official code release for the paper, including the multi-agent WORKFORCE framework, OWL training pipelines, tools, prompts, and assets to reproduce results and extend the system.

- Project CodeNet · [GitHub](https://github.com/IBM/Project_CodeNet) · [Website](https://developer.ibm.com/exchanges/data/all/project-codenet/)
  - description: Large-scale code dataset used for the code readability task (the paper follows prior work to sample a Python subset); official dataset repository and information page.

- PyDirectInput · [GitHub](https://github.com/learncodebygaming/pydirectinput)
  - description: Input library used by CRADLE to send low-level keyboard events compatible with DirectX-based games.

- RoBERTa-base · [GitHub](https://github.com/facebookresearch/fairseq) · [Codewiki](https://codewiki.google/github.com/facebookresearch/fairseq) · [Doc](https://huggingface.co/roberta-base)
  - description: Encoder backbone used to train the authors’ dense retriever for function retrieval.
  - description: Backbone referenced for the supervised SimCSE variant used to compute tool embeddings.

- RT-1 / RT-1-X · [GitHub](https://github.com/google-research/robotics_transformer) · [Website](https://robotics-transformer.github.io)
  - description: Robotics Transformer baselines; RT‑1‑X/RT‑X variants are included as comparison methods in SimplerEnv evaluations.

- SelfCheckGPT · [GitHub](https://github.com/potsawee/selfcheckgpt)
  - description: Zero-resource hallucination detection toolkit cited as one of the “state-of-the-art technical discriminators” within CogMir’s evaluator set.

- statsmodels · [GitHub](https://github.com/statsmodels/statsmodels) · [Doc](https://www.statsmodels.org/)
  - description: Statistical modeling library (e.g., OLS, GLM/logit) imported in the sandbox environment and used in the example/modeling code for agents.

- StructGPT · [GitHub](https://github.com/RUCAIBox/StructGPT)
  - description: A synergy-augmented LLM framework to reason over structured data; used as a strong comparison baseline and speed reference.

- Tongyi DeepResearch (WebDancer code and demo) · [GitHub](https://github.com/Alibaba-NLP/DeepResearch) [Star_count: 17.2K] · [Codewiki](https://codewiki.google/github.com/Alibaba-NLP/DeepResearch)
  - description: Official release accompanying the paper; contains code and demos for WebDancer, including data synthesis (CRAWLQA/E2HQA), ReAct-style agent, SFT cold-start, and DAPO-based RL training.

- Vicuna / FastChat· [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat) · [Website](https://lmsys.org/blog/2023-03-30-vicuna/) · [Doc](https://chat.lmsys.org)
  - description: Serving framework used to host the agent, tools, and web UI for interactive demos in the paper.
  - description: Open-source chat model used as a backbone (Vicuna-13B/30B) in several baselines and comparisons.
  - description: Multi-turn evaluation benchmark (and evaluation scripts via FastChat) used to score models; authors report turn-based scores.
  - description: Open-source chat model family evaluated on the CoordinationQA benchmark.
  - description: Training/evaluation framework used by the authors (with MT-Bench); leveraged for efficient fine-tuning and benchmarking.
  - description: Referenced for its evaluation methodology; MMRole-Eval’s reward-model scoring is inspired by Vicuna/LLaVA style comparative judgments.
  - description: Open-source aligned chat model used as a white-box target model (Vicuna-7B) and also shown in examples (Vicuna-33B demo).
  - description: Baseline model and evaluation framework; the paper compares WizardLM to Vicuna-13B (v1.1) and uses FastChat for MT-Bench.

- WizardCoder / Evol-CodeAlpaca (code data) · [GitHub](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder)
  - description: Code-focused Evol-Instruct dataset used as 10% of the training mixture to boost code/agent planning skills.

- WizardLM· [GitHub](https://github.com/nlpxucan/WizardLM)
  - description: Open-source LLM family (e.g., WizardLM-2-8x22B) used as proposers and analyzed for proposer/aggregator specialization.
  - description: Method used to automatically generate ~2.3K instruction–response pairs for tuning the Prompt Agent’s JSON parsing.
  - description: Open-source code LLM baseline (WizardCoder-Python-33B-V1.1) used for comparisons with and without MatPlotAgent.
  - description: Official code/checkpoints for the paper’s instruction-evolved models; used to reproduce WizardLM training and inference.

- World Models (VAE + MDN-RNN controller) · [GitHub](https://github.com/hardmaru/WorldModels)
  - description: Baseline implementation framework the authors adapt for a recurrent world model comparator.

- xCodeEval · [GitHub](https://github.com/microsoft/xCodeEval)
  - description: Multilingual code benchmark used in the paper’s ablation to assess performance across programming languages.

- xLAM (Large Action Models) · [GitHub](https://github.com/SalesforceAIResearch/xLAM) · [Website](https://huggingface.co/Salesforce)
  - description: Official release of the xLAM model family introduced in the paper (weights/checkpoints and usage). The models are used throughout the experiments and are the main contribution.

- XTuner · [GitHub](https://github.com/InternLM/xtuner)
  - description: Fine-tuning toolkit used for InternVL-based models referenced in the experiments.



### Vision & Multimodal Models

- DDIM (Denoising Diffusion Implicit Models) · [GitHub](https://github.com/ermongroup/ddim)
  - description: Sampling method used across experiments to generate videos from the compositional diffusion world model.

- DeBERTa · [GitHub](https://github.com/microsoft/DeBERTa)
  - description: Pretrained model used (as in prior work) to generate candidate elements for Multimodal-Mind2Web evaluation inputs.

- HuatuoGPT-Vision · [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-Vision)
  - description: Medical LVLM baseline used for image-based evaluation and as one of the specialized models leveraged in the study.

- InternLM-XComposer 2.5 (InternVL2.5-XComposer) · [GitHub](https://github.com/InternLM/InternLM-XComposer)
  - description: Open-source vision-language model for free-form composition/comprehension; evaluated on SCIVERSE.
  - description: Open-source VLM baseline (InternVL‑2.5‑XComposer‑7B/8B) evaluated under the same web-navigation setup.

- MiniGPT-v2 · [GitHub](https://github.com/Vision-CAIR/MiniGPT-4) · [Codewiki](https://codewiki.google/github.com/Vision-CAIR/MiniGPT-4) · [Website](https://minigpt-4.github.io)
  - description: Open-source multimodal model baseline; assessed on SCIVERSE for open-source comparisons.

- OFA· [GitHub](https://github.com/OFA-Sys/OFA)
  - description: Vision-language model used as an alternative captioner in baselines (ImageRef + OFA-Caption).

- Phi-3.5 Vision· [GitHub](https://github.com/microsoft/Phi-3CookBook) · [Website](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
  - description: Microsoft’s 3.4B instruction-tuned model used as the main base LLM and verifier backbone for MAPoRL training and evaluation.
  - description: One of the base open-source LLMs the paper finetunes end-to-end for multiagent generation and critic roles.
  - description: Open multimodal model evaluated as an additional baseline (frame-based setting) in the benchmark.

- ShareGPT4V · [GitHub](https://github.com/InternLM/ShareGPT4V) · [Website](https://sharegpt4v.github.io)
  - description: Used to produce captions for the collected source images that seed multi-modal file generation.
  - description: Open-source large multimodal model; included among SCIVERSE evaluation baselines.

- T5 (Text-to-Text Transfer Transformer)· [GitHub](https://github.com/google-research/text-to-text-transfer-transformer) · [Doc](https://github.com/google-research/text-to-text-transfer-transformer#readme)
  - description: The paper uses a T5-XXL encoder to preprocess text action prompts for the video diffusion model.
  - description: Used as a held-out encoder (T5-3B) to compute embedding-based diversity metrics for the analysis.
  - description: Open-source model used as a Critic backbone; the authors fine-tune T5 (3B) for trajectory reward prediction.

- VideoLLaMA2 · [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
  - description: Video-LLM baseline compared to MAM on MedVidQA and used as a specialized video model.

- VILA (Visual Language Model) · [GitHub](https://github.com/Efficient-Large-Model/VILA) · [Website](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b)
  - description: Backbone used to build the reward model; the paper fine-tunes a small VILA model (e.g., VILA1.5-3B) to score trajectories.

- Vision Transformer (ViT) · [GitHub](https://github.com/google-research/vision_transformer)
  - description: ImageNet-pretrained ViT embeddings are used to assess scene-image diversity.

- Visual Studio Code · [GitHub](https://github.com/microsoft/vscode) · [Codewiki](https://codewiki.google/github.com/microsoft/vscode) · [Website](https://code.visualstudio.com)
  - description: IDE targeted by several tasks (extensions, settings, editing) within the Windows environment.

- ActivityWatch · [GitHub](https://github.com/ActivityWatch/activitywatch) · [Website](https://activitywatch.net/) · [Doc](https://docs.activitywatch.net/en/latest/)
  - description: Open-source activity tracker used by the authors to build their monitoring software for collecting real-world keyboard/mouse, browser, and app-usage events that seed scenarios and example events in the Environment Gym.

- AVDC (Learning to Act from Actionless Videos through Dense Correspondences) · [GitHub](https://github.com/apple/ml-avdc)
  - description: Video-diffusion codebase the authors build upon (with architectural modifications) to implement their compositional world model.

- BEHAVIOR / BEHAVIOR-1K · [GitHub](https://github.com/StanfordVL/Behavior) · [Website](https://behavior.stanford.edu/)
  - description: Embodied household-activity benchmark used as a baseline for the diversity evaluation.

- BLIP· [GitHub](https://github.com/salesforce/BLIP)
  - description: Image captioning model used in the Image-to-Text Bridge (I2T-B) to verbalize generated images; also used as a captioning baseline (ImageRef + BLIP-Caption).

- BLIP-2 / LAVIS (Q-Former)· [GitHub](https://github.com/salesforce/LAVIS) · [Doc](https://huggingface.co/docs/transformers/model_doc/blip-2)
  - description: Vision-language captioning used by MMInA to generate image captions for the “caption-augmented” text-only baselines.
  - description: Source of the Q-Former cross-modal projector architecture and training objectives (contrast/match/caption) that ProtT3 adapts to bridge PLM and LM.
  - description: Vision-language model used to visually validate retrieved 3D assets and assist asset selection.
  - description: Open-source MLLM baseline evaluated on MORE; authors use the InstructBLIP-Vicuna-13B variant.
  - description: Q-Former module (from BLIP-2/LAVIS) is used as the latent projector to inject ThinkAct’s visual plan latent into the action model.

- CLIP (Contrastive Language-Image Pre-training) · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Contrastive vision-language model used for cross-modal retrieval over the hybrid-modal corpus.
  - description: Vision-language model used to compute alignment and to support CLIPScore-based evaluation of image plans.
  - description: Vision-language model used by prior work and as a comparator/option for the selector and scene description in DEPS ablations.
  - description: Contrastive text–image pre-training framework repeatedly used/extended in the biomedical and geoscience vision–language models.
  - description: Vision-language model mentioned in the survey’s perception subsystem for processing medical images as part of multimodal agent pipelines.
  - description: Used to compute embeddings and filter icons by cosine similarity when constructing the paper’s OOD variant of ScreenSpot (ScreenSpot‑P&P‑OOD).
  - description: Used to compute embedding similarity of scene images for evaluating visual diversity.
  - description: Text encoder used in ThinkAct’s state encoder to process language instructions.
  - description: Vision-language model family used as black-box surrogates for the targeted image attack (CLIP attack); the paper ensembles several CLIP variants to generate transferable adversarial perturbations.

- CLIPort · [GitHub](https://github.com/cliport/cliport) · [Website](https://cliport.github.io)
  - description: Robot manipulation policy used as the controller in the Tabletop Manipulation experiments; DEPS provides plans that CLIPort executes.

- clipscore· [GitHub](https://github.com/jmhessel/clipscore)
  - description: Reference-free image captioning/vision-language metric; used to automatically evaluate visual plan quality and alignment.

- CogVLM · [GitHub](https://github.com/THUDM/CogVLM)
  - description: Open visual-language model used as a baseline; fine-tuned with VAB data in experiments.
  - description: Visual-language backbone family behind CogAgent; cited as the underlying pretrained model family for GUI understanding.

- CogVLM2 · [GitHub](https://github.com/THUDM/CogVLM2)
  - description: Open-source VLM (including Llama3-chat-19B variant) evaluated on GroundUI; a baseline implementation for UI grounding comparisons.
  - description: Newer CogVLM variant evaluated as an open-source baseline on VAB after multitask fine-tuning.

- DINOv2 · [GitHub](https://github.com/facebookresearch/dinov2)
  - description: Visual encoder used to extract image features for SphAgent training/evaluation.
  - description: Visual encoder used in ThinkAct’s state encoder for the action policy.

- Grounding DINO · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
  - description: Open-set detector that the authors fine-tune to create a medical grounding tool for object detection/localization across modalities.
  - description: Open-set detector used for icon/image detection to augment Set-of-Marks for visual grounding.
  - description: Open-set object detector used to locate interactive icons (icon/logo prompts) as part of the visual prompt generation.
  - description: Suggested alternative for identifying action-relevant regions when setting agent-dependent loss scaling without reachability info (Appendix B).
  - description: Open-set object detector used by CRADLE’s Information Gathering module to provide bounding boxes for GUI/object grounding.
  - description: Open-set detector used by the OmniParser baseline referenced in the paper.

- GUI Odyssey · [GitHub](https://github.com/OpenGVLab/GUI-Odyssey)
  - description: Cross-app GUI navigation dataset and tasks; SPA-Bench references its task taxonomy and draws most English cross-app tasks from it.

- InternVideo2 / InternVideo2.5 · [GitHub](https://github.com/OpenGVLab/InternVideo2)
  - description: Referenced in the video understanding setup for MVBench as a video-specialized model option when token logprobs are available for uncertainty estimation.

- InternVL· [GitHub](https://github.com/OpenGVLab/InternVL) · [Doc](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B)
  - description: Open-source VLM baseline controller evaluated by the paper (InternVL2‑8B).
  - description: Open-source VLMs (InternVL2/2.5) used as VLM baselines for UI grounding and layout tasks.
  - description: Open-source MLLM baseline; EMMA evaluates the Llama3‑76B variant.
  - description: Updated open-source MLLM baseline evaluated on EMMA.
  - description: Open-source LMMs fine-tuned/evaluated as baselines in VAB; practitioners can reproduce the open-model experiments and training.
  - description: Open-source MLLM backbone (InternVL2-8B) evaluated with AR-MCTS.
  - description: Employed as another main VLM backbone (14B and 78B variants); the paper implements GAM-Agent over InternVL3 and reports gains across all benchmarks.
  - description: Open-source multimodal suite; InternVL-1.5 and InternVL-2 are evaluated across SCIVERSE’s versions.
  - description: Open-source InternVL-Chat-V1.5 is evaluated as an MRPA baseline.
  - description: Open-source MLLM suite; InternVL2-8B is the base model for the MetaAgent used throughout the paper’s experiments.
  - description: Vision-language model used by the authors to generate image captions of GUI pages during dataset construction.

- Jarvis-VLA · [GitHub](https://github.com/CraftJarvis/JARVIS-VLA)
  - description: Open-source vision-language-action model used in MCU’s AutoEval experiments as another open-access evaluator variant.

- Jina-CLIP-v1 · [GitHub](https://github.com/jina-ai/jina-clip) · [Website](https://huggingface.co/jinaai/jina-clip-v1)
  - description: Alternative multimodal retriever used in ablations for cross-modal retrieval.

- LLaVA (LLaVA-1.6 / LLaVA-Next) · [GitHub](https://github.com/haotian-liu/LLaVA) 24K · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io) · [Doc](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
  - description: Open-source VLM baseline controller evaluated by the paper (LLaVA‑NeXT‑8B).
  - description: General MLLM baseline compared against MMedAgent; also informs the instruction format for tool use via prior LLaVA work.
  - description: Open-source vision-language model evaluated as a generalist baseline.
  - description: Another open-source LVLM baseline/backbone the paper experiments with to compare/improve visual grounding for web agents.
  - description: Vision-language model fine-tuned and used to implement COMBO’s planning sub-modules (Action Proposer, Intent Tracker, Outcome Evaluator).
  - description: Open LMM baselines the authors fine-tune on VAB’s multi-environment trajectories for agent evaluation.
  - description: Open-source LVLM evaluated as a model-only baseline on DSBench data analysis tasks.
  - description: Vision–language assistant paradigm referenced for adapting general LLMs to medical and math images (e.g., LLaVA‑Med, G‑LLaVA).
  - description: Vision–language model baseline compared against MAM on image and video tasks.
  - description: Open-source LLaVA baseline version evaluated across SCIVERSE’s five problem versions.
  - description: Open-source LMMs evaluated as baselines (LLaVA-NeXT-34B and LLaVA-NeXT-Mistral-7B) in the MMRole-Eval benchmark.
  - description: Open‑source MLLM used as one of the base models (LLaVA‑1.6‑13B) for ScienceQA and discussed on MM‑Vet in the paper.
  - description: Alternative VLM baseline used in ablations for reward modeling (compared against VILA-13B).
  - description: Open-source MLLM baseline evaluated on MORE (v1.5, 13B).
  - description: Open-weight VLM used as the white-box captioner component; the paper performs gradient-based attacks against this captioner and uses it to preprocess images for policy models.
  - description: Open-source VLM used as the base agent policy (LLaVA‑1.6‑7B and LLaVA‑1.6‑34B) that PAE fine-tunes via RL/Filtered-BC.

- LLaVA-OneVision · [GitHub](https://github.com/LLaVA-VL/LLaVA-OneVision)
  - description: Open-source MLLM backbone used in experiments and for generating pseudo-answers for corpus construction.

- LLaVA-NeXT (Video) · [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) · [Website](https://llava-vl.github.io/blog/2024-llava-next-video/)
  - description: Open-source MLLM baseline (LLaMA3-LLaVA-NeXT-8B) used in additional evaluations.
  - description: Video-capable LLaVA baseline used for video-based evaluation.
  - description: Recent LLaVA model emphasizing easy visual task transfer; evaluated on SCIVERSE including vision-rich/vision-only settings.

- LLaVA‑OneVision · [GitHub](https://github.com/haotian-liu/LLaVA-OneVision) · [Doc](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf)
  - description: Open-source MLLM baseline used in EMMA evaluations; repo and evaluated checkpoint.

- Lookup-Free Quantizer (LFQ, MAGVIT-v2) · [GitHub](https://github.com/google-research/magvit-v2)
  - description: Discrete tokenizer used by DPLM-2 to convert encoded structural features into structure tokens; the paper reports LFQ substantially outperforms VQ-VAE for structure tokenization.

- MathVerse · [GitHub](https://github.com/OpenGVLab/MathVerse) · [Website](https://mathverse-cuhk.github.io)
  - description: Multimodal math dataset used to construct the multimodal retrieval knowledge base.
  - description: Multimodal visual math benchmark the survey uses to illustrate step-wise generative evaluation and diagram understanding.

- MathVision · [GitHub](https://github.com/OpenGVLab/MathVision)
  - description: Multimodal math dataset included in the hybrid-modal retrieval corpus.

- MindSearch · [GitHub](https://github.com/OpenGVLab/MindSearch) · [Doc](https://arxiv.org/abs/2407.20183)
  - description: Multi-agent search framework modeling information seeking via iterative graph construction; used as a baseline for Direct API-Driven Access comparisons.

- mPLUG-Owl · [GitHub](https://github.com/X-PLUG/mPLUG-Owl)
  - description: Open-source multimodal model baseline assessed on MORE (Llama-7B variant).

- OpenVLA · [GitHub](https://github.com/openvla/openvla) · [Website](https://openvla.github.io)
  - description: Vision-language-action baseline compared against ThinkAct on LIBERO and SimplerEnv; also used for inference speed comparison.

- OWL‑ViT · [GitHub](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
  - description: Open-vocabulary object localization model used as a real executable tool in T3‑Agent.

- Predicting-Activity-by-Machine-Learning · [GitHub](https://github.com/psa-lab/predicting-activity-by-machine-learning)
  - description: QSAR/ML workflows; adapted for activity prediction and feature interpretation tasks.

- pyperclip · [GitHub](https://github.com/asweigart/pyperclip) · [Doc](https://pyperclip.readthedocs.io)
  - description: Used to read and write clipboard contents (text/image descriptions) for agent observations/actions.

- RoomR (Visual Room Rearrangement) · [GitHub](https://github.com/allenai/roomr)
  - description: Embodied rearrangement tasks; gold action sequences constructed via heuristic DFS and used in AgentBank.

- SAMtools/BCFtools · [GitHub](https://github.com/samtools/samtools) · [Website](http://www.htslib.org/)
  - description: Used to generate consensus sequences from read alignments during sequence verification.

- SeeClick · [GitHub](https://github.com/OpenGVLab/SeeClick)
  - description: GUI grounding model evaluated as an open-source baseline for element and layout grounding.

- Segment Anything (SAM) · [GitHub](https://github.com/facebookresearch/segment-anything) · [Codewiki](https://codewiki.google/github.com/facebookresearch/segment-anything) · [Website](https://segment-anything.com/)
  - description: Dataset/model cited as an image source to diversify the pool of images for MM‑Traj synthesis.
  - description: Segmentation model used to augment screenshots (SAM2SOM-style overlays) for robust GUI element grounding in CRADLE.

- Segment Anything 2 (SAM 2) · [GitHub](https://github.com/facebookresearch/segment-anything-2)
  - description: Suggested tool to segment relevant regions to adjust loss scaling for the video diffusion training when reachability masks are unavailable (Appendix B).

- TraceVLA · [GitHub](https://github.com/microsoft/TraceVLA)
  - description: Baseline that uses visual traces prompting for spatial-temporal awareness; compared to ThinkAct in manipulation benchmarks.

- UI-Vision (project site repo) · [GitHub](https://github.com/uivision/uivision.github.io)
  - description: GitHub repository for the project website; referenced as the official site and the paper states code and processing scripts will be released alongside the benchmark for reproducibility.

- VERL (vLLM Efficient Reinforcement Learning) · [GitHub](https://github.com/vllm-project/verl)
  - description: RL framework used to run on-policy rollouts and optimize with the DAPO algorithm; the authors state they implement VERL to support RL algorithm and rollouts.

- VideoGUI · [GitHub](https://github.com/ShowLab/VideoGUI)
  - description: Benchmark from instructional videos for GUI automation; cited as a smaller-scale desktop dataset compared to UI-Vision.

- VideoWebArena · [GitHub](https://github.com/ljang0/videowebarena) · [Website](https://www.youtube.com/@webarenawarrior) · [Doc](https://drive.google.com/file/d/17DwmsM7KzBWyz1BN1aq7NHDvgcTIrCgx/view?usp=drive_link)
  - description: Official codebase and video dataset for the benchmark introduced in the paper; includes environment, tasks, evaluators, and 74 tutorial videos (YouTube/Drive) needed to reproduce experiments.

- Visual Sketchpad · [GitHub](https://github.com/allenai/visual-sketchpad)
  - description: Sketch-based visual chain-of-thought system referenced as a planner-style MLLM approach enabling intermediate sketches for math reasoning.

- VisualWebArena · [GitHub](https://github.com/jykoh/visualwebarena)
  - description: Multimodal extension of WebArena for visual web tasks; included in the survey’s browser platform benchmarks.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai/) · [Doc](https://docs.vllm.ai/en/latest/)
  - description: LLM serving system the paper used to deploy all open‑source models for experiments.
  - description: High-throughput LLM inference engine used by the authors to run Llama3 and Qwen2.5 experiments.
  - description: High-throughput LLM serving engine (PagedAttention) underpinning VERL; enables efficient rollouts and training in the RL stage referenced by the paper.
  - description: Inference/serving library used to speed up multi-sample generations during EMMA’s filtering and evaluation.
  - description: High-throughput LLM serving used by the authors to run open-source vision-language models locally for benchmarking.
  - description: Suggested in limitations as an efficient serving/decoding stack; relevant for reproducing inference at scale.
  - description: High-throughput LLM inference engine used to run the open-source code LLM baselines in the experiments.
  - description: Serving framework used to run the open-source LLM (Mixtral-8x7B-Instruct) efficiently in the deployment stage.
  - description: Inference engine used for running the fine-tuned PLANNER and EXECUTOR during evaluation.
  - description: High-throughput LLM serving engine used to deploy the multimodal model for MobileUse experiments.
  - description: Inference serving stack used to deploy the Qwen2‑VL‑7B Locator for latency measurements.
  - description: High-throughput LLM inference engine used to host and evaluate open-source models reproducibly in the paper’s experiments.
  - description: LLM serving library used to host open-source models locally (e.g., Llama, Mistral, Phi) for both data synthesis and agent policies.
  - description: High-throughput LLM serving/inference engine used for all generations in the experiments (e.g., to sample trajectories and run MC step-reward estimation).

- VPT (Video Pre-Training) · [GitHub](https://github.com/openai/Video-Pre-Training)
  - description: Baseline agent evaluated in MCU (both behavior-cloned and RL-tuned-to-diamond variants); MineStudio provides wrappers to run VPT in the MCU tasks.

- Yi / Yi-VL · [GitHub](https://github.com/01-ai/Yi) · [Website](https://01.ai)
  - description: Open foundation model family providing the Yi‑VL‑34B baseline evaluated against MMedAgent.
  - description: Open foundation models from 01.AI; Yi-VL-34B and Yi-VL-6B are included as open-source MRPA baselines.


## Fine-tuning
- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory)
- PEFT (Parameter-Efficient Fine-Tuning) · [GitHub](https://github.com/huggingface/peft) · [CodeWiki](https://codewiki.google/github.com/huggingface/peft)
- NVIDIA NeMo-Aligner · [GitHub](https://github.com/NVIDIA/NeMo-Aligner)
- bitsandbytes · [GitHub](https://github.com/TimDettmers/bitsandbytes)
  - description: Library used for QLoRA (nf4) during CoALM 405B fine-tuning, as described in the training details.

- DeepSeekMath· [GitHub](https://github.com/deepseek-ai/DeepSeek-Math)
  - description: Math-pretrained LLM baseline; authors fine-tune a 7B version to create SCIAGENT-DEEPMATH.
  - description: Open math-specialized LLM used in the survey’s landscape of Math-LLMs and as a strong open-source baseline for math reasoning.

- DeepSpeed (ZeRO Stage 3) · [GitHub](https://github.com/microsoft/DeepSpeed) · [Website](https://www.deepspeed.ai/)
  - description: Distributed training library; the paper uses ZeRO Stage 3 to speed up fine-tuning of the backbones.
  - description: Training system used with ZeRO‑3 to fine-tune ToolGen efficiently across GPUs.
  - description: Dependency used to facilitate efficient LLM training for instruction tuning.
  - description: Training system used to scale PRM fine-tuning (ZeRO-3) efficiently.
  - description: Distributed training framework used to train all models except the 70B variants.
  - description: Distributed training system used for fine-tuning (8×V100 with ZeRO-3).
  - description: Training system used in ThinkAct’s SFT cold-start (ZeRO-3) to scale MLLM optimization.

- DeepSpeed-Chat · [GitHub](https://github.com/microsoft/DeepSpeedExamples) · [Doc](https://www.deepspeed.ai/tutorials/deepspeed-chat/)
  - description: Training framework used for efficient SFT/DPO fine-tuning of LLMs in this work.

- FreeSASA · [GitHub](https://github.com/mittinatten/freesasa) · [Website](https://freesasa.github.io)
  - description: Solvent accessible surface area library; used to compute relative buried surface area (rBSA) for distributional alignment and analysis of pseudo vs. real complexes.

- GLoRIA · [GitHub](https://github.com/marshuang80/gloria)
  - description: Medical global‑local contrastive pre-training approach used for image–text alignment tasks referenced in the survey.

- Guanaco-7B (QLoRA) · [GitHub](https://github.com/artidoro/qlora) · [Doc](https://huggingface.co/timdettmers/guanaco-7b)
  - description: Efficient low-rank finetuning for quantized LLMs; used to fine-tune base models and verifiers in MAPoRL due to compute constraints.
  - description: Guanaco-7B model and finetuning framework (QLoRA) used as one of the open-source white-box target models in experiments.

- HH-suite3 (HHblits, hhalign)· [GitHub](https://github.com/soedinglab/hh-suite) · [Website](https://wwwuser.gwdg.de/~compbiol/data/hhsuite/)
  - description: Used to search for homologs of generated sequences to assess novelty against sequence databases.
  - description: Profile HMM-based MSA tool used in the AF2 baseline pipeline for sequence alignment construction.
  - description: Toolkit used to build MSAs and compute E-values; provides HHblits and hhalign commands used for MSA baselines and analyses.

- HMMER/JackHMMER· [GitHub](https://github.com/EddyRivasLab/hmmer) · [Website](http://hmmer.org/)
  - description: MSA search tool used in the AF2 baseline pipeline to build alignments (UniRef90, MGnify, BFD).
  - description: JackHMMER is used in the paper’s Accelerated MSA pipeline to align top retrieved sequences.

- LoRA· [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA) · [Doc](https://arxiv.org/abs/2106.09685)
  - description: Parameter-efficient tuning method employed to train the three specialized agents during SPAN distillation.
  - description: Parameter-efficient fine-tuning method; used for the Multi-LLMone-stage (LoRA) and α-UMi (LoRA) baselines compared against full fine-tuning.
  - description: Parameter-efficient fine-tuning method used to finetune LLaVA for COMBO’s planning sub-modules.
  - description: Parameter‑efficient fine‑tuning method employed to train ToolACE models.
  - description: Parameter-efficient fine-tuning method used to train the end-to-end baselines and specialized agents on limited data.
  - description: Parameter-efficient finetuning method used for dynamic replanning experiments due to compute constraints.
  - description: Parameter-efficient fine-tuning method applied to the base language model in ProtT3’s stage-2 training.
  - description: Parameter-efficient fine-tuning method used to adapt the ESM model in VFN-IFE.
  - description: Parameter-efficient fine-tuning method applied to both the visual encoder and language model layers when adapting Qwen2‑VL to GUI grounding.
  - description: Parameter-efficient fine-tuning method employed to fine-tune the relatively small LLM Critics on trajectory–reward data.
  - description: Low-Rank Adaptation used for alignment and to preserve pretrained abilities, especially for xLAM-8x22B and during DPO (Section 4.1).

- OpenWebVoyager · [GitHub](https://github.com/MinorJerry/OpenWebVoyager)
  - description: Official release of the paper including code and data for training the multimodal web agent, running the imitation-learning and exploration–feedback–optimization cycles, and reproducing results.

- Oumi · [GitHub](https://github.com/oumi-ai/oumi) · [Website](https://oumi.ai)
  - description: Open, end-to-end training platform used by the authors to fine-tune and scale CoALM models; enables reproducible training pipelines.

- seaborn· [GitHub](https://github.com/mwaskom/seaborn) · [Codewiki](https://codewiki.google/github.com/mwaskom/seaborn) · [Doc](https://seaborn.pydata.org/)
  - description: Statistical visualization library imported in the environment for data exploration during agent analysis.
  - description: Statistical data visualization used in plotting distributions and comparative figures in code-generated experiment reports.

- UIAutomator2 · [GitHub](https://github.com/openatx/uiautomator2)
  - description: Android UI automation library; used by the authors to execute end-to-end actions for fine-tuned agents that lack direct Android interaction capabilities.

## Prompt Engineering
- AutoDAN · [GitHub](https://github.com/SheltonLiu-N/AutoDAN)
  - description: Official implementation of the paper’s hierarchical genetic algorithm for generating stealthy jailbreak prompts; primary resource for reproducing all methods and experiments.

- Automatic Prompt Engineer (APE) · [GitHub](https://github.com/keirp/automatic_prompt_engineer)
  - description: Baseline method compared in the experiments; code for generating and selecting prompts based on LLM feedback.

- ComplexCoT (Complexity-Based Prompting) · [GitHub](https://github.com/FranxYao/Complexity-Based-Prompting)
  - description: Baseline for prompting; the paper follows its official implementation for ComplexCoT.

- G-Memory · [GitHub](https://github.com/bingreeky/GMemory)
  - description: Official code release for the paper’s hierarchical memory system for MAS, including prompts and integration hooks for AutoGen, DyLAN, and MacNet to reproduce experiments and extend the method.

- GraphRouter · [GitHub](https://github.com/snap-stanford/GraphRouter) · [Website](https://snap.stanford.edu) · [Doc](https://arxiv.org/abs/2410.03834)
  - description: Routing framework referenced; the paper follows GraphRouter to build LLM profiles and compares with its PromptLLM baseline.

- Hypothetical Minds · [GitHub](https://github.com/locross93/Hypothetical-Minds/)
  - description: Official code release for the paper; contains the LLM-agent implementation, prompts, ToM module, subgoal/action planner, and experiment scripts to reproduce results.

- LLMLingua · [GitHub](https://github.com/microsoft/LLMLingua)
  - description: Prompt compression toolkit used as a comparison in the appendix; the paper shows naive compression can harm tool-use versus EASYTOOL.

- MC-Planner (DEPS) · [GitHub](https://github.com/CraftJarvis/MC-Planner)
  - description: Official implementation of the paper’s “Describe, Explain, Plan and Select” interactive planner, including prompts, selector, and Minecraft integrations used in all experiments.

- Program-of-Thoughts (PoT) Prompting· [GitHub](https://github.com/wenhuchen/Program-of-Thoughts)
  - description: Numerical reasoning prompting used as one of the three diverse methods in DMAD for LLMs and as a standalone baseline.
  - description: Prompting approach that separates computation from reasoning via code; cited as an antecedent to code-nested solutions.

- Reflexion · [GitHub](https://github.com/noahshinn/reflexion)
  - description: Agent prompting framework with verbal reinforcement learning; used to test ReHAC’s generalization across prompt frameworks.

- SABM_ShallWeTeamUp · [GitHub](https://github.com/wuzengqing001225/SABM_ShallWeTeamUp)
  - description: Official code release for this paper; implements the three simulations (Keynesian Beauty Contest, Bertrand Competition, Emergency Evacuation), prompts, and SABM-based multi-agent workflow used in the experiments.

- Self-Instruct · [GitHub](https://github.com/yizhongw/self-instruct) · [Website](https://arxiv.org/abs/2212.10560)
  - description: Procedure used to synthesize new web task queries in each exploration–feedback–optimization cycle.
  - description: Instruction-generation approach that inspires the paper’s Alpaca-style synthetic query generation used in their data pipeline.
  - description: Instruction-generation procedure the authors follow to create multi-granularity prompts (statement/category/tool/API) for MGToolBench.
  - description: Used to generate diverse task instructions from documentation during data synthesis.

- Semantic Kernel · [GitHub](https://github.com/microsoft/semantic-kernel) · [Codewiki](https://codewiki.google/github.com/microsoft/semantic-kernel) · [Doc](https://learn.microsoft.com/semantic-kernel/)
  - description: Microsoft’s orchestration framework mentioned as a related toolkit that connects LMs and tools via prompt templates; relevant for practitioners comparing SDKs.

- Set‑of‑Mark (SoM) · [GitHub](https://github.com/CASIA-IVA-Lab/SoM)
  - description: Visual grounding prompting method used for OSWorld baselines (GPT‑4o+SoM) to compare against CRADLE’s image-only augmentation.

- SimClass · [GitHub](https://github.com/THU-MAIC/SimClass)
  - description: Official code and service released by the paper for their LLM-based multi-agent classroom simulator, including role agents, the session controller, prompts, and the online system used in experiments.

- Step‑Back Prompting · [GitHub](https://github.com/google-research/google-research/tree/master/step-back-prompting)
  - description: Abstraction‑first prompting method employed as a DMAD agent strategy and as a baseline in the LLM experiments.

- textgrad· [GitHub](https://github.com/stanfordmlgroup/textgrad)
  - description: Gradient-based natural-language optimization toolkit used as a multi-agent prompt-optimization baseline in the experiments.

- Triad · [GitHub](https://github.com/ZJU-DCDLab/Triad)
  - description: Official code and data release for this paper; implements the multi‑role LLM agent framework (G-/D-/A-Agent), prompts, and scripts used in all experiments.

## Structured Output
- lm-format-enforcer · [GitHub](https://github.com/noamgat/lm-format-enforcer)
- Cytopus · [GitHub](https://github.com/wallet-maker/cytopus)
  - description: Knowledge base for cell-type gene programs; used to generate Spectra-ready JSON in the cytopus_db task.

- INFOGENT · [GitHub](https://github.com/gangiswag/infogent)
  - description: Official code release for the paper’s modular web information aggregation framework (Navigator, Extractor, Aggregator) covering both Direct API-Driven and Interactive Visual Access setups.

- NexusRaven · [GitHub](https://github.com/NexusflowAI/NexusRaven)
  - description: Function-calling model cited in the data unification section as a prior strong approach, motivating a universal function-calling format (Section 3.1).

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui) · [Doc](https://pyautogui.readthedocs.io/)
  - description: Forms part of the action space and initial state setup scripts for mouse/keyboard automation within the VM.
  - description: Python library for mouse/keyboard automation on the Ubuntu VM; used to implement desktop interaction actions.
  - description: Cross‑platform GUI automation library used for keyboard/mouse control (especially in OSWorld and standard desktop apps).
  - description: Used to implement pixel-level actions for the vision-based agent by mapping higher-level Playwright actions to screen-coordinate interactions.
  - description: Automation library used to format and execute action triplets into runnable code in the OSWorld environment.
  - description: Cross-platform GUI automation library; the paper defines its action space as a subset of PyAutoGUI actions for agent execution.
  - description: GUI automation library; actions in OSWorld/Spider2-V examples are expressed with PyAutoGUI commands.

- Scanpy · [GitHub](https://github.com/scverse/scanpy)
  - description: Single-cell analysis toolkit; used for bioinformatics tasks involving scRNA-seq processing and visualization.

- Scirpy · [GitHub](https://github.com/scverse/scirpy)
  - description: TCR single-cell repertoire analysis; used for immunology-related bioinformatics tasks.

- SimpleDeepSearcher · [GitHub](https://github.com/RUCAIBox/SimpleDeepSearcher)
  - description: Closely related open-source implementation for deep information seeking via trajectory synthesis; cited as a comparison/alternative to RL-based training.

- Str2Str · [GitHub](https://github.com/lujiarui/Str2Str)
  - description: Official implementation of the paper’s score-based structure-to-structure framework for zero-shot protein conformation sampling; used for all experiments and released by the authors.

- Str2Str · [GitHub](https://github.com/DeepGraphLearning/Str2Str)
  - description: Score-based structure-perturbation framework for conformation sampling; used for multi-state and distribution prediction comparisons.

- Vega-Lite · [GitHub](https://github.com/vega/vega-lite) · [Website](https://vega.github.io/vega-lite/)
  - description: Grammar of interactive graphics cited as a basis for transform abstractions; BLADE’s transformation representation references verbs/concepts aligned with Vega/Vega-Lite.

## Planning
- MC-Planner: Describe, Explain, Plan and Select · [GitHub](https://github.com/CraftJarvis/MC-Planner)
- The AIPlan4EU Unified Planning Library · [GitHub](https://github.com/aiplan4eu/unified-planning)
- Workflow Induction Toolkit · [GitHub](https://github.com/zorazrw/workflow-induction-toolkit)
- ADAPT (As-needed Decomposition and Planning) · [GitHub](https://github.com/allenai/adapt) · [Website](https://adapt-agent.github.io)
  - description: Baseline hierarchical decomposition/planning approach; compared with SELFGOAL on the same environments.

- BabyAGI · [GitHub](https://github.com/yoheinakajima/babyagi)
  - description: Minimal agentic workflow example cited as related engineering resource for building LLM agent systems.
  - description: Baseline maintaining a prioritized task list; used for comparison with D2A in activity generation quality.

- Chameleon· [GitHub](https://github.com/lupantech/Chameleon)
  - description: Baseline system that plans multi-step tool use; included for comparison in the experiments.
  - description: Plug-and-play compositional reasoning framework cited as a planner-style system that assembles tools for complex (including multimodal) math tasks.

- Concordia · [GitHub](https://github.com/google-deepmind/concordia) · [Website](https://arxiv.org/abs/2312.03664)
  - description: Text-based simulator framework the authors build upon to implement their daily-activity simulator, memory, planning, and environment control; also used for an ablation with its multi-step planning component.

- Flow (official) · [GitHub](https://github.com/tmllab/2025_ICLR_FLOW)
  - description: Official implementation of Flow, the modularized agentic workflow automation framework proposed in the paper; used to reproduce experiments and extend the AOV-based dynamic workflow system.

- Open Motion Planning Library (OMPL) · [GitHub](https://github.com/ompl/ompl) · [Website](https://ompl.kavrakilab.org)
  - description: Motion-planning library used for action primitives; the paper specifically employs the BIT* planner through OMPL.

- Recast Navigation · [GitHub](https://github.com/recastnavigation/recastnavigation)
  - description: Used to build the L4 navmesh for path planning in Habitat-MAS scenes.

- stable-diffusion· [GitHub](https://github.com/CompVis/stable-diffusion)
  - description: Image generation model used as a tool within T3‑Agent.
  - description: Text-to-image generation model referenced in open-ended tasks (T2I) for image synthesis.
  - description: Text-to-image diffusion model used to generate image plans; serves as the backbone T2I model in TIP and in baseline comparisons.


## Formal Verification
- Stormpy: Python bindings for Storm · [GitHub](https://github.com/moves-rwth/stormpy) · [Website](https://www.stormchecker.org/)
- FormalBook (“Proofs from THE BOOK”) · [GitHub](https://github.com/Mo271/FormalBook)
  - description: Lean formalizations inspired by Proofs from THE BOOK; part of the sub-curriculum where LeanAgent proved results including Wedderburn’s Little Theorem.

- Hairy Ball Theorem (Lean) · [GitHub](https://github.com/corent1234/hairy-ball-theorem-lean)
  - description: Lean repository for the Hairy Ball Theorem; included in the sub-curriculum where LeanAgent proved a key step (“HairyBallDiff”).

- Lean (Lean 4) · [GitHub](https://github.com/leanprover/lean4) · [Website](https://leanprover.github.io)
  - description: Interactive theorem prover used to type-check and verify all automatically generated proofs; LeanAgent switches Lean versions to match each target repository.

- Llemma · [GitHub](https://github.com/EleutherAI/llemma)
  - description: Open mathematics-focused language models cited among Math-LLMs and baselines for symbolic and theorem-oriented reasoning tasks.

- Minimap2 · [GitHub](https://github.com/lh3/minimap2)
  - description: Used in the sequencing validation pipeline to map nanopore reads to reference amplicons for clone verification.

- MUMmer (DNADiff) · [GitHub](https://github.com/mummer4/mummer) · [Website](http://mummer.sourceforge.net/)
  - description: DNADiff from MUMmer used to call variants between consensus and reference sequences in the verification pipeline.

- PathFinderCRC · [GitHub](https://github.com/LiangJunhao-THU/PathFinderCRC)
  - description: Repository used to verify biomarkers from WSI probability maps; serves as the target project for the pathfinder_verify_biomarker task.

- Saturn · [GitHub](https://github.com/siddhartha-gadgil/Saturn)
  - description: Experiments with SAT solvers with proofs in Lean 4; part of the initial curriculum.

- Zeta 3 Irrational · [GitHub](https://github.com/ahhwuhu/zeta_3_irrational)
  - description: Proof of ζ(3) irrationality; used in the sub-curriculum.

## RAG (Retrieval-Augmented Generation)
- ColBERT (v2) · [GitHub](https://github.com/stanford-futuredata/ColBERT)
- Pyserini · [GitHub](https://github.com/castorini/pyserini)
- Hyperopt: Distributed Hyperparameter Optimization · [GitHub](https://github.com/hyperopt/hyperopt)
- RAGFlow · [GitHub](https://github.com/infiniflow/ragflow) · [Website](https://ragflow.io)
- AnythingLLM: Product Overview · [GitHub](https://github.com/Mintplex-Labs/anything-llm) · [Website](https://anythingllm.com)
- ChaosBlade · [GitHub](https://github.com/chaosblade-io/chaosblade) · [Website](https://chaosblade.io)
  - description: Chaos engineering toolkit used to inject network, CPU, memory, storage, and code faults in the Train-Ticket system to create realistic failure cases.

- DBpedia · [GitHub](https://github.com/dbpedia) · [Website](https://www.dbpedia.org)
  - description: Open knowledge base extracted from Wikipedia used as a primary KB in experiments (DBpedia-04/DBpedia-10); Triad indexes DBpedia and queries it via SPARQL.

- DNA Chisel · [GitHub](https://github.com/Edinburgh-Genome-Foundry/DnaChisel)
  - description: Used to codon-optimize gene fragments for E. coli in constructing GFP11 fusion libraries for split-GFP assays.

- E5 · [GitHub](https://github.com/microsoft/unilm/tree/master/e5) · [Doc](https://huggingface.co/intfloat/e5-large-v2)
  - description: Text-embedding retriever family used as another alternative retriever in additional experiments.

- MMOA-RAG · [GitHub](https://github.com/chenyiqun/MMOA-RAG)
  - description: Official code release of the paper’s multi-agent reinforcement learning framework for joint optimization of RAG modules (Query Rewriter, Selector, Generator) using MAPPO; used for all experiments.

- OpenAdapt · [GitHub](https://github.com/OpenAdaptAI/OpenAdapt)
  - description: Process-automation toolkit leveraged by AGENTNET TOOL for input tracking and trajectory processing during annotation.

- python-docx · [GitHub](https://github.com/python-openxml/python-docx) · [Doc](https://python-docx.readthedocs.io/)
  - description: Library used by WriterAgent to edit Word/Writer documents programmatically as part of the CLI-based agents.

- SELF-RAG · [GitHub](https://github.com/AkariAsai/self-rag)
  - description: Public implementation used as a comparison baseline in the experiments.

- YAGO 4 · [GitHub](https://github.com/yago-naga/yago-4) · [Website](https://yago-knowledge.org)
  - description: Reason-able knowledge base used as a second KB (YAGO-4) in experiments; Triad indexes and queries it as part of the KBQA pipeline.


## Tools
- BUTTON: Facilitating Multi-turn Function Calling for LLMs
  - [GitHub](https://github.com/PKU-Baichuan-MLSystemLab/BUTTON)
  - paper: BUTTON: Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning

- Gorilla: Large Language Model Connected with Massive APIs
  - Github: https://github.com/ShishirPatil/gorilla

- ToolUniverse
  - [GitHub](https://github.com/mims-harvard/ToolUniverse)

- universal-tool-calling-protocol
  - [github](https://github.com/universal-tool-calling-protocol)
  - description: 缩小工具调用长度、简化工具调用

### Dataset
- ToolLLM · [GitHub](https://github.com/beijixiong1/ToolLLM) · [Website](https://toolllm.github.io)
- ToolBench · [GitHub](https://github.com/THUDM/ToolBench)
- ToolAlpaca · [GitHub](https://github.com/thunlp/ToolAlpaca)
  - description: Simulated tool-use benchmark with 3k cases; used as an additional evaluation set in the paper.
- APIBench (from Gorilla) · [GitHub](https://github.com/gorilla-llm/gorilla) · [Website](https://gorilla.cs.berkeley.edu)
  - description: Dataset and evaluation harness for API/tool-use (covering TorchHub, HuggingFace, TensorFlow); used to test Tool-Planner and report pass/win/hallucination rates.

- GAIA: A Benchmark for General AI Assistants· [GitHub](https://github.com/GAIA-benchmark/GAIA) · [Website](https://huggingface.co/datasets/gaia-benchmark/GAIA)
  - description: Multi-domain tool-use benchmark (web, files, multimodal); main benchmark for tool-use experiments and baselines.
  - description: General AI assistant benchmark (Levels 1–3) used to evaluate agentic reasoning, browsing, and tool-use.
  - description: General AI assistant benchmark spanning tool use, browsing, and multimodality; OpenHands evaluates GPTSwarm and agents on GAIA.
  - description: Benchmark used for real-world question answering; the paper uses its public subset for training/testing.

- GAIA (General AI Assistant Benchmark) · [GitHub](https://github.com/gaia-benchmark/GAIA) · [Website](https://huggingface.co/datasets/gaia-benchmark/GAIA)
  - description: Multi-domain tool-use benchmark (web, files, multimodal); main benchmark for tool-use experiments and baselines.
  - description: General AI assistant benchmark (Levels 1–3) used to evaluate agentic reasoning, browsing, and tool-use.
  - description: General AI assistant benchmark spanning tool use, browsing, and multimodality; OpenHands evaluates GPTSwarm and agents on GAIA.
  - description: Benchmark used for real-world question answering; the paper uses its public subset for training/testing.

- Gorilla (APIBench) · [GitHub](https://github.com/ShishirPatil/gorilla) · [Website](https://gorilla.cs.berkeley.edu/)
  - description: LLM connected to massive APIs; its dataset/documentation statistics are referenced in the paper’s analysis of tool documentation.
  - description: Tool‑use baseline family and related ecosystem; BFCL originates from the Gorilla team and Gorilla‑OpenFunctions‑v2 is a comparison baseline.
  - description: Tool-API calling LLM referenced as an open-source baseline; paper notes limitations consuming tool responses in conversational settings.
  - description: API calling benchmark used to test OpenHands’ ability to select and call software APIs.

- GSM8K· [GitHub](https://github.com/openai/grade-school-math) · [Website](https://openai.com/research/gsm8k) · [Dataset](https://huggingface.co/datasets/gsm8k)
  - description: Grade-school math reasoning benchmark; used for evaluation (train/test split 1:4).
  - description: Grade school math word problems; used as a primary text reasoning benchmark for comparing sparse vs. fully-connected MAD and baselines.
  - description: Grade school math word problems; used to collect training trajectories and evaluate α-UMi’s program-aided math reasoning.
  - description: Used for preliminary tool-overuse analysis and as an out-of-distribution evaluation benchmark for SMARTAgent.
  - description: Math word problem benchmark used for evaluation; the paper reports substantial gains on GSM8K.
  - description: Math word problem dataset and verifier training resources; used to train the verifier and evaluate MAPoRL on mathematical reasoning.
  - description: Math word-problem benchmark used as a held-out domain to test transfer of math agents discovered via MGSM.
  - description: Math word-problem dataset for evaluating mathematical reasoning.
  - description: Math word problem dataset used to evaluate mathematical reasoning and token costs.
  - description: Math word-problem dataset used for finetuning and evaluation; the paper trains on 500 examples and tests on held-out sets.
  - description: Math word problems; official solution paths were reformatted into interaction trajectories.
  - description: Text-only math dataset used to build the mathematics-specific retrieval corpus.
  - description: Grade School Math dataset used throughout the survey as a core benchmark for text-only mathematical reasoning and robustness analyses.
  - description: Dataset of math word problems used in the paper’s Case Study 1 to evaluate CoT, reflection, and DSPy compilation strategies.
  - description: Math word problem dataset highlighted as a dominant benchmark for training and evaluating math LLMs.
  - description: Grade-school math word problem benchmark used to evaluate reasoning capability.
  - description: Math word-problem dataset used for evaluation and ablations.
  - description: Grade-school math dataset used for training and evaluation.
  - description: Grade-school math word problem dataset from which the widely used GSM-Hard split is derived; used for the paper’s closed-domain math reasoning evaluation.
  - description: Grade-school math word problems; used in the debate setting (solver–critic agents) and for transfer/scaling studies.
  - description: Core math word-problem dataset; its training split seeds augmentation and its test split is a main in-domain evaluation.
  - description: Math reasoning benchmark on which the paper reports test accuracy; dataset repo and dataset card for downloading and evaluation.
  - description: Grade school math benchmark used (4-shot, pass@1) for math reasoning evaluation.
  - description: Grade-school math word problems; used to evaluate a simple decomposition that post-processes CoT to fix answer-extraction errors.

- HotpotQA · [GitHub](https://github.com/hotpotqa/hotpotqa) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset adapted into a multi-turn search-tool environment for AgentBank.
  - description: Open-domain multi-hop QA dataset; used in the paper’s retrieval-augmented decomposition experiments (fullwiki setting).

- ToolPlanner / MGToolBench · [GitHub](https://github.com/XiaoMi/toolplanner)
  - description: Official code and data release for the paper; includes the ToolPlanner two-stage RL framework, prompts, training scripts, evaluation, and the MGToolBench multi-granularity instruction dataset constructed from ToolBench.

- TravelPlanner Benchmark · [GitHub](https://github.com/OSU-NLP-Group/TravelPlanner) · [Website](https://travelplanner-bench.github.io)
  - description: Real-world travel planning benchmark and simulator on which Ask-before-Plan is built; the authors adapt its tools/environment and evaluation for their new dataset and tasks.
  - description: Real-world travel planning benchmark used as a primary evaluation bed; the paper uses its “sole-planning” mode, training split (45) and validation split (180), and its evaluation protocol and hard/commonsense constraints.

- GPQA · [GitHub](https://github.com/Idavidrein/gpqa) · [Website](https://huggingface.co/datasets/Idavidrein/gpqa)
  - description: Graduate-level science QA benchmark used for evaluation (Physics, Chemistry, Biology).
  - description: Graduate-level, Google-proof QA benchmark; OpenHands evaluates tool-augmented reasoning on GPQA subsets.
  - description: Graduate‑level, Google‑proof multiple-choice benchmark used as a knowledge task.
  - description: Graduate‑level, Google‑proof multiple‑choice benchmark used in the paper to evaluate LLM reasoning under various prompting/debate strategies.

- GPQA · [GitHub](https://github.com/Idavidrein/gpqa) · [Website](https://huggingface.co/datasets/Idavidrein/gpqa)
  - description: Graduate-level science QA benchmark used for evaluation (Physics, Chemistry, Biology).
  - description: Graduate-level, Google-proof QA benchmark; OpenHands evaluates tool-augmented reasoning on GPQA subsets.
  - description: Graduate‑level, Google‑proof multiple-choice benchmark used as a knowledge task.
  - description: Graduate‑level, Google‑proof multiple‑choice benchmark used in the paper to evaluate LLM reasoning under various prompting/debate strategies.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Real-world web navigation dataset; the paper uses 37 of its websites to create training queries and evaluates on its cross-task and cross-website test splits.
  - description: Real-world web agent dataset/benchmark used for offline evaluation and fine-tuning; the paper also evaluates on its online variant Mind2Web-Live.
  - description: Dataset and benchmark for web navigation generalization; used for both offline (train-induced workflows) and online AWM evaluation. The repo also provides the MindAct baseline components (e.g., element filtering) compared against AWM.
  - description: Generalist web-agent benchmark referenced in related work; useful for broader comparisons with TOOLSANDBOX’s tool-use evaluation.
  - description: Popular web-agent dataset cited as related work; relevant for practitioners seeking additional training/evaluation data aligned with the paper’s domain.
  - description: Large-scale web-agent dataset used for training/meta-learning and evaluation (cross-task, cross-website, cross-domain); the paper also amends the cross-task split for proper evaluation.
  - description: Web agent benchmark used for offline evaluation (element accuracy and step success rate) under Cross-Task/Website/Domain splits.
  - description: Large-scale dataset and evaluation suite for generalist web agents; cited in the survey among core browser benchmarks (static and live variants).
  - description: Realistic web interaction dataset behind Mind2Web-SC (the safety-control split used in the paper via GuardAgent); supports reproducing the web-agent safety evaluations.
  - description: Large benchmark of real web tasks across many sites; the paper filters Mind2Web tasks and augments them with user profiles to create Mind2Web-SC and evaluate GuardAgent with SeeAct.

- StrategyQA · [GitHub](https://github.com/allenai/strategyqa) · [Website](https://allenai.org/data/strategyqa)
  - description: Implicit reasoning QA dataset used for training/evaluation under the ReAct framework and GPT-4-simulated human setting.
  - description: Implicit reasoning QA used as a reasoning task with a search tool in AgentBank.

- TheoremQA · [GitHub](https://github.com/wenhuchen/TheoremQA)
  - description: Held-out theorem-driven QA in math/science; used to evaluate math/generalization with Python and Wikipedia tools.
  - description: Theorem-driven QA dataset included in the composition of the College Physics benchmark used for evaluation.

- BERT· [GitHub](https://github.com/google-research/bert) · [Codewiki](https://codewiki.google/github.com/google-research/bert)
  - description: Used as the dense retriever baseline in ToolBench for tool retrieval comparisons.
  - description: Used to encode textual actions in the Recurrent World Models baseline (VAE + MDN-RNN).
  - description: Pretrained language encoder used in the critic network to extract textual features for value/advantage estimation.
  - description: Foundational masked language model architecture that underpins many encoder-style scientific LLMs surveyed (e.g., SciBERT, BioBERT).
  - description: Used to obtain instruction embeddings for t-SNE/k-means analysis of topic breadth in the evolved datasets.

- Llama 3 8B Instruct · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com/) · [Doc](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
  - description: Open-source LLM baseline controller (LLaMA‑3‑70B‑Instruct) used for comparison.
  - description: Open LLM used in auxiliary experiments comparing different backbones within the ResearchAgent framework.
  - description: The time‑efficiency study deploys Llama‑3.1‑70B‑Instruct with SGLang to compare single‑ vs multi‑agent runtimes (App. B.1).
  - description: Open‑source baseline models evaluated (Llama‑2‑7b/13b/70b‑Chat; Llama‑3‑8b/70b‑Instruct).
  - description: Meta’s Llama 3 8B instruction-tuned model used as another heterogeneous agent to study co-training across models under MAPoRL.
  - description: Open-source LLM evaluated in an extended experiment to show ReAd’s applicability beyond closed-source models.
  - description: Open-weight model used in ablations comparing idea diversity across different base LLMs.
  - description: Base LLM used for finetuning and evaluation, including multi-iteration studies and broader benchmarks.
  - description: Open-source chat models (7B/13B) evaluated as alternative agent backbones within the middleware framework.
  - description: Primary base LLM (LLaMA2-7B) fine-tuned to build KG-Agent; also used in scaling/ablation experiments.
  - description: Open-source LLM baselines evaluated with and without HoneyComb to demonstrate framework improvements.
  - description: Open-weight Meta model evaluated by the paper; provides instructions and weights access for replicating the Llama-based runs.
  - description: Base open-source LLM family; the authors fine-tune LLaMA‑3.1‑8B‑Instruct both as the proactive agent and as the reward model, and also evaluate baseline performance.
  - description: Open-source baseline model evaluated as Mauto in all scenarios; used to compare safety behavior with other open/closed models.
  - description: Base and teacher models used throughout—Llama‑3.1‑8B‑Instruct as the executor/challenger and Llama‑3.1‑70B‑Instruct for distillation demonstrations.
  - description: Backbone model used for ToolPlanner (LLaMA-7B); repository contains inference/fine-tuning code and model access instructions.
  - description: One of the MAS-driving LLMs used during dataset construction and testing for baseline and MAS execution.
  - description: Base open-source LLM (LLaMA2-7B) that the paper fine-tunes to instantiate the leader and member agents in LONGAGENT.
  - description: Open LLM family from Meta evaluated across BENCHFORM protocols; models fetched via Ollama.
  - description: Updated Meta LLMs (including 70B and 405B via API) assessed in the benchmark; highlighted for differing conformity characteristics.
  - description: Open-source baseline LLMs used as Generator/Evaluator agents to compare the agentic framework against direct prompting.
  - description: Open-source base VLM used in experiments; the paper runs four agent instances locally on GPUs.
  - description: Backbone models fine-tuned to obtain DTA-Llama; the paper reports results for Llama2-7B, Llama2-13B, and Llama3-8B.
  - description: Open-source LLM family used at 8B/70B/405B scales for baseline and analysis; the 405B variant is run via Vertex AI and others are locally fine-tuned/evaluated.
  - description: Open-source LLM baseline evaluated by the paper as an agent; used to test alignment under the proposed prompting conditions.
  - description: Default backbone LLM (Llama 3.1-70B) for both agents and the environment controller in the experiments.
  - description: One of the evaluated LLMs (Llama‑2 Chat 70B) used for zero-shot prompting in TDMR extraction and normalization.

- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory) · [Codewiki](https://codewiki.google/github.com/hiyouga/LLaMA-Factory) · [Doc](https://llamafactory.readthedocs.io/)
  - description: Training framework used for faithfulness-improvement experiments (LoRA + DPO) on Llama-based models.
  - description: Training framework the authors used to run SFT and DPO; includes scripts and docs for efficient post-training of LLMs.
  - description: Training framework the paper uses for additional backbone experiments (e.g., finetuning Qwen2.5-VL).
  - description: Training toolkit whose PPO codebase the authors build upon to implement MAPPO-based optimization and SFT for MMOA-RAG.
  - description: Training framework used to fine-tune the base LLM; the paper cites using standard SFT via this tool.
  - description: Training framework the authors used to fine-tune the probed model on datasets for confidence integration.

- RestGPT / RestBench · [GitHub](https://github.com/Yifan-Song793/RestGPT)
  - description: Benchmark provider for two subsets (TMDB and Spotify) used in experiments; also a multi-agent baseline compared by the authors.
  - description: Framework and benchmark for connecting LLMs to real-world RESTful APIs; the paper evaluates EASYTOOL on the TMDB subset of RestBench and compares against RestGPT baselines.
  - description: Repository containing RestBench (TMDB and Spotify scenarios) used as evaluation datasets; the paper reports CP%/Win% on RestBench-TMDB and RestBench-Spotify.

- Sentence-Transformers· [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [Doc](https://www.sbert.net/)
  - description: Sentence embedding toolkit mentioned as an alternative lightweight encoder for the controller’s embedding function.
  - description: Lightweight text-embedding model used as the NodeEncoder to embed agent profiles and task text.
  - description: Embedding model/library used to compute query similarity and filter near-duplicate queries when generating task sets.
  - description: Used to compute sentence embedding similarity between reviews and meta-reviews as part of content-level analysis.
  - description: Library and model used for idea de-duplication; the study encodes ideas with all-MiniLM-L6-v2 and filters pairs by cosine similarity.
  - description: Used in analysis to retrieve top-K KB triples by semantic similarity when testing “direct prompting with sampled triples.”
  - description: Sentence embedding toolkit used to compute Text-S, Cap-S, and ALL-S scores for semantic similarity of text and image plans to references.
  - description: Used by the paper’s goal parser to map free-form LLM plans to predefined controller skills via semantic similarity.
  - description: Embedding model used as the backbone of AOP’s reward model to encode sub-tasks and agent descriptions.
  - description: Library for SBERT embeddings; the paper uses a paraphrase-tuned model to compute cosine similarity and merge semantically similar criteria.
  - description: Used to compute cosine similarity between agents’ final answers for measuring answer diversity.
  - description: Encoder used in the collaboration determiner to extract query semantics.
  - description: Dense retriever baseline used in the ablation comparing tag extraction vs. retrieval for selecting candidate tools/APIs.
  - description: Text-embedding toolkit used for retrieving textually similar Objaverse assets and for computing task-description diversity (Sentence-BERT similarity).
  - description: Library used to compute semantic similarity between legal rules (via Sentence-BERT) for retrieving similar rules and transferring rule-insights across datasets.
  - description: Embedding library used for dense retrieval and cosine-similarity baseline; includes the embedding models used in the paper.

- ToolBench / ToolLLM (ToolLLaMA) · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolbench.github.io/) · [Doc](https://huggingface.co/ToolBench/ToolLLaMA-2-7b-v2)
  - description: Primary dataset and baseline framework used in the paper. The authors sample tasks for SPAN distillation and evaluation, and compare against ToolLLM’s DFSDT decision module.
  - description: Benchmark, datasets, baselines (ToolLLaMA), and evaluation toolkit (including ReAct and DFSDT/ToolEval) for API-tool use; the paper trains and evaluates α-UMi on ToolBench and uses its real-time and static evaluation protocols.
  - description: Dataset and toolkit of real-world REST APIs used extensively for evaluation (I1/I2/I3 subsets, DFSDT baseline, retriever); the paper converts its tool docs with EASYTOOL and benchmarks performance.
  - description: Tool-use dataset/tooling referenced in the paper’s ToolUse module design (e.g., retrieval-based tool selection in derived modules).
  - description: Real-world tool/dataset repository with ~47k APIs and trajectories; used to build ToolGen’s tool set, retrieval pairs, and agent-tuning data, and provides the ToolRetriever baseline.
  - description: Widely used tool‑learning dataset and system; used as a comparison baseline and in additional data training comparisons in the paper.
  - description: Referenced benchmark/dataset and evaluation suite for tool-use LLMs; used in the paper’s comparisons (ToolEval) and to contextualize differences in task design and evaluation.
  - description: Public tool-use dataset and framework the paper builds upon; the authors use the G3 split as seed data, the official 100-task test set, DFSDT tree decoding, and ToolEval for Win Rate evaluation; also provides ToolLLaMA baseline code.
  - description: Benchmark dataset and agent framework for tool-augmented LLMs; the paper evaluates retrieval on ToolBench subsets and uses ToolLLaMA with DFSDT as the agent for end-to-end tests.
  - description: Large-scale tool-use dataset and training/evaluation framework used to construct training data (10k samples for SFT/DPO), provide baselines (ToolLLaMA-7B), and supply the APIBench OOD test used in this paper’s generalization study.
  - description: Benchmark and tooling with 16,464 real-world APIs sourced from RapidAPI; used as a primary dataset and evaluation suite (ToolEval) and provides the DFSDT baseline implementation used in comparisons.
  - description: Deep-first search over decision trees for tool use; Tool-Planner improves upon DFSDT by searching at the toolkit level; implementation available in the ToolBench repo.
  - description: Large-scale repository of real-world APIs and tool-use data (ToolLLM/ToolBench); used as seed data to extract/expand scenarios during bottom-up instruction construction.
  - description: Benchmark and tooling suite of real-world APIs and the DFSDT baseline; the paper evaluates on the I3-Instruction subset and uses DFSDT as a comparison method.
  - description: Comprehensive multi-tool learning dataset and toolkit used to construct DTA-Tool (training data transformed from ToolBench) and as the source of APIs; also provides DFSDT search and baseline implementations referenced in the method.
  - description: Toolkit and dataset for tool-use LLMs; the paper fine-tunes ToolLLaMA (ToolLLM) and uses it as a baseline for the static execution/tool-learning subtask.

- AgentBench · [GitHub](https://github.com/THUDM/AgentBench) · [Website](https://agentbench.dev)
  - description: External agent evaluation suite; the paper reports additional results on this benchmark for reference.
  - description: Generalist agent benchmark suite cited to situate tool-use evaluation within broader multi-task agent testing.
  - description: Multi-domain agent evaluation suite; OpenHands uses the OS (bash) subset for code-grounded system interaction.
  - description: Benchmark and Dockerized environments; used to launch WebShop and ALFWorld setups and to follow evaluation protocols.
  - description: Benchmark for evaluating LLMs as agents; referenced as an existing agent evaluation dataset that informed CogMir’s dataset construction.
  - description: Benchmark and agent suite used as the OS agent base; the Safe-OS benchmark in the paper is constructed on top of AgentBench’s OS agent and data format.

- APIGen · [GitHub](https://github.com/SalesforceAIResearch/APIGen) · [Website](https://arxiv.org/abs/2409.10019)
  - description: Public function-calling dataset pipeline used for comparison in ablations (training Qwen2.5‑Coder with APIGen+ToolAce vs. MAGNET data).
  - description: Automated pipeline the authors use to synthesize 50k verifiable function-calling datapoints from 3,673 executable APIs; central to the paper’s data generation in Section 3.4.

- ToolAlpaca · [GitHub](https://github.com/tangqiaoyu/ToolAlpaca)
  - description: Simulated tool-learning dataset referenced for documentation statistics and context in the paper.
  - description: Simulated tool‑learning dataset/baseline cited in the paper’s comparisons of API coverage and capabilities.

- Tools Fail · [GitHub](https://github.com/jiminsun/tools-fail)
  - description: Official code and data release for this paper; includes the calculator experiments, ALFRED-based tool-error detection datasets, prompts, and scripts to reproduce the detection results.

- ESMFold (ESM)· [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com) · [Doc](https://esmatlas.com/docs)
  - description: Alternative structure prediction model suggested by the paper for sequence-only cases; also used in the sequence-only analysis.
  - description: Protein language model and folding pipeline used as the second backbone (ESMFLOW) that the authors adapt to flow matching and fine-tune.
  - description: Used throughout for baselines and evaluation—ESM-2 as a masked-LM baseline and initialization checkpoint, ESMFold to predict structures/pLDDT for generated sequences, and the ESM inverse-folding (GVP-Transformer) encoder as the structure expert for DPLM’s adapter-based inverse folding.
  - description: Provides pretrained sequence representations concatenated into the conditional score model; also subjected to noise injection during training to enhance diversity.
  - description: Protein LM used as a feature extractor to embed generated sequences for clustering/visualization in design evaluation.
  - description: Used to predict structures from ProteinMPNN-designed sequences to compute scRMSD and determine designability.
  - description: Baseline large protein language model capable of structure prediction; evaluated under the paper’s unconditional benchmarks.
  - description: Single-sequence structure predictor used to initialize the starting conformation for STR2STR sampling.
  - description: Structure predictor used extensively for evaluation (pLDDT, refolding for scTM/scRMSD) and baselines; DPLM-2 compares against ESMFold and uses its outputs.
  - description: Protein language models and tools; the esm_fold_predict task uses the esm2_t33_650M_UR50D model for sequence representations and contact maps.
  - description: Pre-trained protein language model used as the sequence encoder in the series-fusion setup for pre-training and downstream tasks.
  - description: Protein language model used as the frozen protein encoder in ProtT3; the paper uses ESM-2 (150M) and its smaller variants in ablations.
  - description: Used as the pretrained encoder for dense retrieval embeddings (ESM‑1b), as a baseline (ESM‑1b/ESM‑2, MSA Transformer), and as a folding model augmented by RSA (ESMFold).
  - description: Pretrained protein language model used as the encoder/decoder backbone to obtain CLS embeddings and fine-tuned initial transformer layers for sequence reconstruction in the VED.
  - description: Protein language model family used as initialization (ESM-2 35M checkpoint) for ESM-AA and as a baseline protein encoder in comparisons.
  - description: Base protein language model whose residual stream activations the authors train SAEs on and probe against.
  - description: Fast structure predictor used to predict designed binder structures before computing AlphaFold2 pAE interaction scores.
  - description: Protein language model family (including ESMFold) used to fold designed sequences and, via LoRA, as the external knowledge model in VFN-IFE to refine VFN-IF predictions.
  - description: ESMFold code and the ESM Atlas dataset of predicted structures; the benchmark provides loaders for ESM Atlas and evaluates ESM-2-650M as a baseline.
  - description: Protein language models used as a baseline; paper evaluates ESM‑2‑650M augmented with structural features.
  - description: FAIR’s protein modeling suite; used for structure prediction (ESMFold), inverse folding (ESM-IF1), and evaluation/generation with ESM3 open models.
  - description: Structure prediction used for self-consistency evaluation; the best ESMFold prediction per designed sequence/backbone is used to compute scTM/scRMSD.
  - description: Protein language models and ESMFold structure predictor used as encoders (ESM-2 family), for pseudo‑perplexity scoring, and to predict 3D structures (pLDDT) throughout evaluation.
  - description: Protein language models and folding model used in FOLDFLOW-2; ESM2-650M provides sequence embeddings (frozen) and ESMFold is used for refolding in designability/self-consistency evaluation.
  - description: Language model-based protein structure prediction; proposed by the authors as another source of predicted structures to expand ProFSA’s pretraining corpus.
  - description: Structure prediction model used to fold designed sequences for designability evaluation (pLDDT, RMSD) and for benchmarking.
  - description: Structure prediction model used to predict structures/secondary structure and to score/generated sequences in several analyses (e.g., Figs. 2f, 3a, 4f).

- RDKit · [GitHub](https://github.com/rdkit/rdkit) · [Website](https://www.rdkit.org) · [Doc](https://www.rdkit.org/docs/)
  - description: Open-source cheminformatics toolkit used to analyze molecular properties from SMiCRM and generate new chemistry questions (structure recognition/bond counting).
  - description: Cheminformatics toolkit used for molecule 3D conformation generation (ETKDG) and MMFF94 optimization in the molecular design tasks.
  - description: Cheminformatics toolkit used to generate 3D conformations from SMILES for fine-tuning; also provides ETKDG and MMFF94 used in the molecule dataset’s conformation generation.

- WORD (Abdominal Organ Segmentation from CT) · [GitHub](https://github.com/HiLab-git/WORD)
  - description: Large-scale CT organ segmentation dataset used to generate grounding labels and to evaluate segmentation-related tools.

- ADB Keyboard · [GitHub](https://github.com/senzhk/ADBKeyBoard)
  - description: Android input method for ADB text entry; the paper installs it to enable reliable Chinese text input for several agents (e.g., AppAgent, SeeAct, M3A, T3A).
  - description: Open-source Android IME used in the paper’s annotation tool to programmatically input text (ADB keyboard ON) for recording/type actions consistently during dataset creation.

- ALFRED· [GitHub](https://github.com/askforalfred/alfred) · [Website](https://askforalfred.com)
  - description: Household instruction-following dataset and tasks on AI2-THOR; EB-ALFRED is developed from ALFRED with simulator fixes and multi-instance support.
  - description: Embodied instruction-following benchmark used to create the multimodal tool-error detection datasets (object detector and action planner) and collect agent trajectories for evaluation.
  - description: Vision-language instruction dataset/environment extended by ALFWorld; referenced as part of the ALFWorld stack used in evaluation.
  - description: Instruction-following dataset underlying ALFWorld; authors use its 3,553 training tasks and 134 unseen test tasks, and sample 3K training tasks to fine-tune the LLM Critic.

- AndroidLab · [GitHub](https://github.com/THUDM/Android-Lab)
  - description: Official release from the paper. Provides the unified Android agent environment (XML and SoM modes), the 138-task benchmark with prebuilt AVD images, evaluation scripts/metrics, and the Android Instruct data and annotation tool needed to reproduce results and fine-tune models.

- BIRD (Text-to-SQL) · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) · [Website](https://bird-bench.github.io/)
  - description: Primary database benchmark used without oracle knowledge; evaluates agents’ ability to navigate real DB content via the middleware tools.
  - description: Large-scale, realistic text-to-SQL benchmark; integrated to assess OpenHands’ database-grounded code generation.

- Contriever · [GitHub](https://github.com/facebookresearch/contriever) · [Website](https://huggingface.co/facebook/mcontriever-msmarco)
  - description: Another alternative dense retriever baseline used in analysis to compare against the authors’ fine-tuned retriever.
  - description: Dense retrieval model used in HoneyComb’s hybrid retriever (combined with BM25) for semantically re-ranking candidate knowledge entries and tool outputs.
  - description: Dense retriever used as the fixed first-stage retrieval model in the main experiments.
  - description: Dense text retriever used to retrieve top-K documents from the text-only corpus.
  - description: Text embedding model used in ablations for alternative clustering similarity comparisons.
  - description: Dense retrieval model used in the paper’s tool-retrieval analysis (NDCG@1/@10) to test whether refined docs improve retrieval.

- CREATOR Challenge (CREATION Challenge) · [GitHub](https://github.com/amazon-science/creator)
  - description: External tool-use benchmark re-purposed by the authors to form a global toolset and report additional accuracy results.

- DSSP · [GitHub](https://github.com/cmbi/dssp) · [Website](https://swift.cmbi.nl/structure/dssp/)
  - description: Physics-based secondary structure assignment used as an alternative discretization baseline in ProSST’s ablations.
  - description: Secondary structure assignment tool used to annotate training structures for coil-ratio filtering in ablation studies.
  - description: Secondary-structure assignment tool used on predicted structures to compare secondary-structure content with natural proteins.
  - description: Used to compute secondary structure and solvent-accessible surface area (SASA) for dataset filtering and analysis.

- Fpocket · [GitHub](https://github.com/Discngine/fpocket)
  - description: Open-source pocket detection and scoring tool; its scores (Fpocket score, druggability, SASA, hydrophobicity) define the Uni-Mol druggability dataset on which ProFSA is evaluated.

- GluonTS · [GitHub](https://github.com/awslabs/gluonts) · [Doc](https://ts.gluon.ai/)
  - description: Probabilistic/neural time-series toolkit the authors use as a dataset provider; many in-domain benchmarks (e.g., Electricity, Solar, Traffic, etc.) are obtained from GluonTS.

- MATH· [GitHub](https://github.com/hendrycks/math) · [Website](https://people.eecs.berkeley.edu/~hendrycks/MATH.html)
  - description: Base math corpus the authors build upon to synthesize MATHFUNC samples; provides training problems used to teach math skills and tool-use.
  - description: Mathematical problem-solving benchmark; used for evaluation and cost analysis.
  - description: Challenging competition-style math problems; the paper evaluates MAD variants on the algebra linear 1d composed sub-task.
  - description: Math problem benchmark; the paper collects trajectories (via GPT-3.5/4) for training and evaluates α-UMi with a program-aided agent on MATH.
  - description: Source dataset for the Math domain in SMART-ER; provides problems and solutions used to compose tool-free and tool-needed reasoning steps.
  - description: Math reasoning benchmark used in the paper’s reasoning evaluations and MoA layer-depth ablation.
  - description: Competition-level math dataset used extensively for training and evaluation (including multiple finetuning iterations and zero-shot tests).
  - description: Challenging math competition problems; turned into interactive tasks using Python and Wikipedia tools.
  - description: Benchmark for mathematical reasoning; the paper evaluates level-5 problems in several categories and compares against MathChat/AutoGen.
  - description: Competition math dataset providing step-by-step solutions; used in the text-only math retrieval corpus.
  - description: Competition-level math benchmark repeatedly referenced for evaluating LLM and MLLM mathematical reasoning capabilities.
  - description: Dataset used for mathematical reasoning experiments; the paper samples training/testing problems from MATH.
  - description: High-difficulty mathematics problem set used extensively in the survey for evaluating mathematical reasoning.
  - description: Mathematical problem-solving dataset; used in Appendix experiments to evaluate AOP with multiple specialized math/code agents.
  - description: Benchmark of 12,500 competition math problems; the paper evaluates on 120 Level-5 problems to assess AgentEval and baseline solvers.
  - description: Math word problem dataset used to evaluate AgentNet and baselines on mathematics tasks; the paper constructs specific train/test splits from MATH.
  - description: Mathematical reasoning dataset used to assess MegaAgent on standard benchmarks.
  - description: Mathematical reasoning benchmark; the paper samples problems and reports performance/cost.
  - description: Training and testing benchmark for mathematical reasoning used to build queries and evaluate MAS-GPT.
  - description: High-school competition math problems; used in the debate setting and as a source model for transfer to GSM8K.
  - description: High‑school competition math benchmark used for LLM evaluation; the paper samples problems per subject to test reasoning methods and debate settings.
  - description: Source of the MATH500 evaluation subset used to assess reasoning performance.
  - description: High-school math competition dataset; training split seeds augmentation and test split is a main in-domain evaluation.

- MMseqs2· [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Sequence search/cluster engine underlying the ColabFold MSA pipeline and used for PDB clustering in dataset prep.
  - description: Tool used to compute sequence identity and cluster/filter proteins (e.g., 70% identity) and to define sequence similarity for debiased sampling.
  - description: Used for sequence-based clustering to balance datasets (e.g., D21M clustering and AFDB preprocessing).
  - description: Fast MSA/search toolkit referenced in relation to AF2/ColabFold pipelines used in experiments.
  - description: Sequence clustering/search toolkit used in extended evaluations for sequence diversity and novelty analyses.
  - description: Tool used to cluster Swiss-Prot sequences at 30% identity for evaluation and to reduce redundancy.
  - description: Fast sequence search/clustering; used for antibody dataset clustering (CDR-H3) and involved in ColabFold MSA generation.
  - description: Sequence search and clustering toolkit; used to cluster/filter sequences when curating the IDP test set and to avoid data leakage.
  - description: Used for clustering sequences (e.g., computing CD0.5/CD0.95, co‑clustering with datasets) to assess diversity and distribution coverage.
  - description: Tool used to recluster the PDB dataset at 50% sequence identity during data preparation.
  - description: Fast sequence clustering used to derive non-redundant PDB chains at 70% identity for ProFSA’s large-scale pseudo-pair construction.
  - description: Used to cluster sequences and quantify sequence diversity (DIV-seq) of designable samples.

- OpenHands· [GitHub](https://github.com/All-Hands-AI/OpenHands)
  - description: Generalist code-generation/SE agent framework used as a baseline (CodeAct v1.9) in the paper’s evaluations.
  - description: Generalist software/agent platform compared as a baseline for ML and open-ended tasks; noted for event streaming and sandboxing.
  - description: Official code release of the paper (MIT licensed) including agent implementations, sandboxed runtime, skills library, multi-agent delegation, UI, and integrated benchmarks for reproduction.
  - description: Software-engineering agent used as the main comparison baseline; adapted by the authors to generate install scripts and tool functions for TM-BENCH tasks.

- OpenInterpreter · [GitHub](https://github.com/OpenInterpreter/open-interpreter) · [Website](https://openinterpreter.com)
  - description: Third-party agent integrated into IoA and used as a baseline; IoA coordinates it for tool-use and coding in the open-ended instruction benchmark.
  - description: CLI-based agent baseline compared/re-implemented in Table 1; serves as a generalist coding/CLI agent reference.

- OS-Kairos · [GitHub](https://github.com/Wuzheng02/OS-Kairos)
  - description: Official codebase and dataset release for the paper’s adaptive GUI agent with confidence scoring; includes training, evaluation, and the collaborative probing toolkit.

- Prodigal · [GitHub](https://github.com/hyattpd/Prodigal)
  - description: Prokaryotic gene prediction tool cited/used alongside Prodigal-GV during dataset construction.

- ProofWriter · [GitHub](https://github.com/allenai/proofwriter) · [Website](https://allenai.org/data/proofwriter)
  - description: Deductive reasoning dataset; OpenHands evaluates long-hop logical reasoning with tool assistance.

- ReAct· [GitHub](https://github.com/ysymyth/react) · [Website](https://react-lm.github.io/)
  - description: Single-agent baseline prompting method (reasoning + acting) against which CONAGENTS is compared; also run with multiple attempts (ReAct@N).
  - description: Reasoning-and-acting operator referenced as a building block in the MaAS operator set.
  - description: Agent framework used to build the paper’s iterative notebook-interacting baseline agent for analysis generation and exploration.
  - description: Reasoning‑and‑acting prompting method used in ScienceWorld experiments and as a baseline in TravelPlanner (§4.2, §4.3).
  - description: Reasoning-acting agent prompting framework; the paper builds agents with ReAct for ToolBench real-time evaluations.
  - description: Baseline prompting algorithm combining reasoning and acting; used in comparisons on ToolBench and RestBench.
  - description: Closed-loop LLM reasoning-and-acting baseline compared against ReAd-J in the experiments.
  - description: Reason+Act agent framework the paper adopts for QA tasks; provides action space (Search/Lookup/Finish) and Wikipedia-API tooling used in their setup.
  - description: Reasoning-and-acting prompting framework; cited as one of the advanced reasoning strategies explored in the paper’s LLM agent scaffolding.
  - description: The paper uses ReAct as the backbone prompting framework for reasoning-acting loops, on top of which their error-feedback and decoupled-generation are built.
  - description: Baseline framework re-implemented (“ReAct-like”) for comparison; provides the reasoning+acting prompting pattern.
  - description: Reasoning-and-acting agent framework cited by the paper as compatible with HiAgent’s hierarchical memory management; useful for practitioners looking to integrate the proposed memory mechanism into existing agent paradigms.
  - description: Reasoning-and-acting framework the paper adopts for agent tasking, where the model emits rationale before each action.
  - description: Reasoning+acting prompting framework; used as the inference template for agent planning (thought, action, observation) in FlowBench experiments.
  - description: Reasoning-and-acting agent baseline; the authors train a ReAct agent with their AgentOptimizer.
  - description: Prompting style used for baseline executors without an explicit planner; forms the “No Planner / ReAct-style” training baseline.
  - description: Baseline method combining reasoning and acting; compared against AOP in the experiments.
  - description: Original ReAct prompting approach for reasoning-and-acting; serves as a comparison baseline (via LangChain ReAct) in the experiments.
  - description: Reasoning-and-acting prompting framework; AgentNet’s agents adopt ReAct-style reasoning/acting and ReAct is also used as a single-agent baseline.
  - description: Chain-based reasoning-and-acting method used to implement the chain decoding baseline (CoT@N) for comparisons in the paper.
  - description: Reasoning-and-acting agent framework used as a baseline; prompts/logic adapted to the paper’s embodied multi-agent settings.
  - description: Baseline/agent template repeatedly used as a codeblock and comparison point (e.g., modified ReAct agents, goal tracking, spatial memory) within the system’s experiments.
  - description: Prompting framework the authors adopt to build a single-agent crawler/reasoner for collecting candidate documents and extracting general-purpose text templates in their multi-agent data-prep stage.
  - description: Reason+Act prompting framework for tool-using LLM agents; used to implement the tool-based Navigator for Direct API-Driven Access.
  - description: Baseline method combining reasoning and acting; used for comparison across datasets.
  - description: Baseline method for reasoning-and-acting LLM agents; the paper evaluates against ReAct (implemented via LangChain).
  - description: Reasoning-and-acting prompting framework used as a baseline in the paper’s comparisons.
  - description: Widely used tool-using/agent prompting framework; serves as one of the baseline paradigms compared against DTA-Llama.
  - description: Reasoning-and-acting baseline framework; used as a comparison baseline in the experiments.
  - description: Reasoning-and-acting prompting framework used by the paper as a baseline for dynamic tool interaction and dynamic planning.
  - description: Reasoning-and-acting prompting framework adopted for the agent policy format during training and evaluation.
  - description: Baseline ICL agent framework combining reasoning and acting; used as the base prompting approach that Prospector extends with AskAct and compares against.
  - description: Reason+Act prompting framework used to collect expert trajectories and to structure agent interactions; the paper fine-tunes agents on ReAct-style trajectories.
  - description: Baseline agent that reasons before acting; the paper compares D2A against ReAct on human-likeness and dissatisfaction metrics.
  - description: Reasoning-and-acting baseline framework; used as a comparison and ablation baseline for WebPlanner/WebSearcher.
  - description: Baseline agent framework combining reasoning and acting; used for comparison against SELFGOAL across tasks.

- ReAct· [GitHub](https://github.com/ysymyth/ReAct) · [Website](https://react-lm.github.io/)
  - description: Single-agent baseline prompting method (reasoning + acting) against which CONAGENTS is compared; also run with multiple attempts (ReAct@N).
  - description: Reasoning-and-acting operator referenced as a building block in the MaAS operator set.
  - description: Agent framework used to build the paper’s iterative notebook-interacting baseline agent for analysis generation and exploration.
  - description: Reasoning‑and‑acting prompting method used in ScienceWorld experiments and as a baseline in TravelPlanner (§4.2, §4.3).
  - description: Reasoning-acting agent prompting framework; the paper builds agents with ReAct for ToolBench real-time evaluations.
  - description: Baseline prompting algorithm combining reasoning and acting; used in comparisons on ToolBench and RestBench.
  - description: Closed-loop LLM reasoning-and-acting baseline compared against ReAd-J in the experiments.
  - description: Reason+Act agent framework the paper adopts for QA tasks; provides action space (Search/Lookup/Finish) and Wikipedia-API tooling used in their setup.
  - description: Reasoning-and-acting prompting framework; cited as one of the advanced reasoning strategies explored in the paper’s LLM agent scaffolding.
  - description: The paper uses ReAct as the backbone prompting framework for reasoning-acting loops, on top of which their error-feedback and decoupled-generation are built.
  - description: Baseline framework re-implemented (“ReAct-like”) for comparison; provides the reasoning+acting prompting pattern.
  - description: Reasoning-and-acting agent framework cited by the paper as compatible with HiAgent’s hierarchical memory management; useful for practitioners looking to integrate the proposed memory mechanism into existing agent paradigms.
  - description: Reasoning-and-acting framework the paper adopts for agent tasking, where the model emits rationale before each action.
  - description: Reasoning+acting prompting framework; used as the inference template for agent planning (thought, action, observation) in FlowBench experiments.
  - description: Reasoning-and-acting agent baseline; the authors train a ReAct agent with their AgentOptimizer.
  - description: Prompting style used for baseline executors without an explicit planner; forms the “No Planner / ReAct-style” training baseline.
  - description: Baseline method combining reasoning and acting; compared against AOP in the experiments.
  - description: Original ReAct prompting approach for reasoning-and-acting; serves as a comparison baseline (via LangChain ReAct) in the experiments.
  - description: Reasoning-and-acting prompting framework; AgentNet’s agents adopt ReAct-style reasoning/acting and ReAct is also used as a single-agent baseline.
  - description: Chain-based reasoning-and-acting method used to implement the chain decoding baseline (CoT@N) for comparisons in the paper.
  - description: Reasoning-and-acting agent framework used as a baseline; prompts/logic adapted to the paper’s embodied multi-agent settings.
  - description: Baseline/agent template repeatedly used as a codeblock and comparison point (e.g., modified ReAct agents, goal tracking, spatial memory) within the system’s experiments.
  - description: Prompting framework the authors adopt to build a single-agent crawler/reasoner for collecting candidate documents and extracting general-purpose text templates in their multi-agent data-prep stage.
  - description: Reason+Act prompting framework for tool-using LLM agents; used to implement the tool-based Navigator for Direct API-Driven Access.
  - description: Baseline method combining reasoning and acting; used for comparison across datasets.
  - description: Baseline method for reasoning-and-acting LLM agents; the paper evaluates against ReAct (implemented via LangChain).
  - description: Reasoning-and-acting prompting framework used as a baseline in the paper’s comparisons.
  - description: Widely used tool-using/agent prompting framework; serves as one of the baseline paradigms compared against DTA-Llama.
  - description: Reasoning-and-acting baseline framework; used as a comparison baseline in the experiments.
  - description: Reasoning-and-acting prompting framework used by the paper as a baseline for dynamic tool interaction and dynamic planning.
  - description: Reasoning-and-acting prompting framework adopted for the agent policy format during training and evaluation.
  - description: Baseline ICL agent framework combining reasoning and acting; used as the base prompting approach that Prospector extends with AskAct and compares against.
  - description: Reason+Act prompting framework used to collect expert trajectories and to structure agent interactions; the paper fine-tunes agents on ReAct-style trajectories.
  - description: Baseline agent that reasons before acting; the paper compares D2A against ReAct on human-likeness and dissatisfaction metrics.
  - description: Reasoning-and-acting baseline framework; used as a comparison and ablation baseline for WebPlanner/WebSearcher.
  - description: Baseline agent framework combining reasoning and acting; used for comparison against SELFGOAL across tasks.

- TravelPlanner · [GitHub](https://github.com/TIGER-AI-Lab/TravelPlanner)
  - description: Real-world tool-use planning benchmark used as one of the tool-oriented evaluation tasks.

- WebArena / WebArena-Lite · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Realistic web environment referenced for comparison; AgentStudio extends beyond web-only action spaces.
  - description: Suite of realistic web environments; MMInA adopts WebArena-style browser/display settings and evaluation practices for single-hop tasks.
  - description: Realistic web environment referenced in related work; useful for practitioners exploring alternative evaluation/simulation setups for web agents.
  - description: Realistic web environment for autonomous agents; referenced as a web benchmark contrasting with desktop UI-Vision.
  - description: Realistic web agent benchmark and environment used for evaluation; the paper follows its accessibility-tree observation format and reports both overall and per-site results, including a cross-template subset.
  - description: Benchmark used to evaluate the text-based web agent trained with AgentTrek trajectories.
  - description: Related web-interaction benchmark cited in related work to contrast web-browsing agents with TOOLSANDBOX’s stateful tool-use focus.
  - description: Interactive web environment the paper adapts and cleans to form VAB-WebArena-Lite; used with SoM annotations and automated via Playwright.
  - description: Realistic, self-hostable web-agent benchmark; integrated for OpenHands web evaluation and baseline comparison.
  - description: The paper hosts and reuses these realistic web environments and evaluators; VideoWebArena maps its skill-retention tasks to WebArena templates and uses its evaluation utilities.
  - description: Primary web-navigation benchmark and environment used for training, ablations, and main evaluation; WebArena-Lite provides the 165 human-verified tasks and executor prompt format used by the paper.
  - description: Realistic online web environment and benchmark for autonomous web agents; referenced as a primary browser-based interactive benchmark.
  - description: Realistic web environment and benchmark used for all main evaluations; provides sandbox sites and program-based evaluators the paper relies on.
  - description: The broader suite of realistic web environments that VWA builds upon; relevant for setting up and extending the web-based evaluation used in the paper.
  - description: Realistic, self-hosted multi-website environment with ground-truth functional verifiers; used as a major training/evaluation environment (OpenStreetMap, PostMill, OneStopMarket) and for success/failure detection.

### Eval: function-calling capability
- Berkeley Function Call Leaderboard · [GitHub](https://github.com/gorilla-llm/berkeley-function-call-leaderboard) · [Website](https://bfcl.cs.berkeley.edu) · [Docs](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)

### Libraries & Frameworks

- Flask · [GitHub](https://github.com/pallets/flask) · [Codewiki](https://codewiki.google/github.com/pallets/flask) · [Website](https://flask.palletsprojects.com)
  - description: Python web server used as the bridge inside the VM to receive commands, execute them, and return observations/files.

- GeoPandas · [GitHub](https://github.com/geopandas/geopandas)
  - description: Core geospatial dataframe library; widely used across the GIS tasks in the benchmark.

- Gradio · [GitHub](https://github.com/gradio-app/gradio) · [Codewiki](https://codewiki.google/github.com/gradio-app/gradio) · [Website](https://www.gradio.app/) · [Doc](https://www.gradio.app/docs)
  - description: Used to deploy and run open-source MLLMs locally during evaluations.
  - description: Deployment toolkit used by the Operation Agent to produce web endpoints for trained models.
  - description: Web UI framework used by the MobileUse Toolkit to provide a visual interface for issuing commands and monitoring agent execution.
  - description: UI toolkit used by the authors to build the human annotation interface for evaluator alignment and error analysis.

- Hugging Face Accelerate · [GitHub](https://github.com/huggingface/accelerate)
  - description: Training launcher/runtime the paper uses to scale experiments (Section 4.1).

- JAX · [GitHub](https://github.com/google/jax) · [Codewiki](https://codewiki.google/github.com/google/jax) · [Website](https://jax.readthedocs.io)
  - description: Second backend for GPUDrive’s Gymnasium environments; also used by comparison simulators like Waymax.
  - description: The implementation leverages JAX (noted in runtime analysis) for efficient sampling via JIT-compilation.

- LLM Foundry · [GitHub](https://github.com/mosaicml/llm-foundry) · [Doc](https://docs.mosaicml.com/projects/llm-foundry)
  - description: Training/finetuning framework used to build the “Optimize LLM Foundry” environment where agents speed up a finetuning script without changing behavior.

- Matplotlib · [GitHub](https://github.com/matplotlib/matplotlib) · [Website](https://matplotlib.org/stable/)
  - description: Plotting library imported in the environment for exploratory analysis steps during agent runs.
  - description: Plotting library referenced in codeblocks and used for visualizations (e.g., ROC curves, scatter plots, metric distributions) throughout the experiments.
  - description: Visualization library targeted by the chart-to-code generation; all charts are reproduced via Matplotlib code.

- MosaicML Composer · [GitHub](https://github.com/mosaicml/composer)
  - description: Training orchestration framework used by the authors to run distributed pretraining.

- MosaicML Streaming · [GitHub](https://github.com/mosaicml/streaming)
  - description: Data streaming/dataloading library employed for efficient large-scale pretraining.

- NumPy · [GitHub](https://github.com/numpy/numpy) · [Doc](https://numpy.org/doc/)
  - description: Numerical computing dependency used in transformation/modeling code within the sandbox.
  - description: Numerical computing dependency used for vectorization and array operations in the verifier implementation.
  - description: Numerical computing library used within code-nested solutions.

- OpenCV · [GitHub](https://github.com/opencv/opencv) · [Codewiki](https://codewiki.google/github.com/opencv/opencv) · [Doc](https://docs.opencv.org/)
  - description: Image processing library used in the verifier for HSV color masking, histogram computation, and image handling.

- pandas · [GitHub](https://github.com/pandas-dev/pandas) · [Codewiki](https://codewiki.google/github.com/pandas-dev/pandas) · [Doc](https://pandas.pydata.org/docs/)
  - description: Core data-frame and transformation library used throughout agent transform code and the evaluation environment.
  - description: Python data analysis library used in preprocessing; authors convert spreadsheets to text for LLM baselines and handle data files during experiments.
  - description: Data handling/analysis in experiments (e.g., summarizing generation methods, computing summary statistics and tables).

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Doc](https://playwright.dev/python/docs/intro)
  - description: Browser automation framework; MMInA uses Playwright to render pages, extract accessibility trees, and execute the 12 summarized actions.
  - description: Browser automation framework the authors use to execute actions and collect screenshots, HTML, and accessibility trees during trajectory synthesis.
  - description: Browser automation toolkit used for precise web actions and for logging reproducible traces (DOM snapshots, network, action sequences) during guided replay.
  - description: Browser automation toolkit used to write program-based solvers and to execute web actions when collecting training trajectories and during evaluation in Web tasks.
  - description: Headless Chromium driver used by OpenHands’ runtime browser to execute BrowserGym actions.
  - description: Browser automation framework used by the benchmark to execute agent actions (each action maps to Playwright Python code in the environment).
  - description: Browser automation framework used to execute predicted actions on webpages in the agent loop (mentioned as an execution tool).
  - description: Browser automation framework; used (via the SEEACT tooling) to simulate real web browsing for INFOGENT’s Interactive Visual Access.

- PyTorch · [GitHub](https://github.com/pytorch/pytorch) · [Website](https://pytorch.org) · [Doc](https://pytorch.org/docs/stable/)
  - description: One of the two backends for GPUDrive’s Gymnasium environments and training loops.
  - description: Deep learning library providing autograd and optimization; the inner-level differentiable optimization and physics-code templates are implemented in PyTorch.
  - description: Framework used to implement the residue-level linear classifier for secondary structure prediction.
  - description: Core deep learning framework used throughout the benchmark.
  - description: Deep learning framework used to implement and train all ProGen3 models.

- PyTorch Geometric · [GitHub](https://github.com/pyg-team/pytorch_geometric) · [Codewiki](https://codewiki.google/github.com/pyg-team/pytorch_geometric) · [Doc](https://pytorch-geometric.readthedocs.io)
  - description: Library used to load Planetoid graph datasets (Cora, Citeseer) and implement node classification baselines per the paper’s setup.
  - description: Graph learning framework used to implement GNN architectures in the benchmark.

- PyTorch Lightning · [GitHub](https://github.com/Lightning-AI/lightning)
  - description: Training framework used to organize experiments and runs.

- pytorch-fid· [GitHub](https://github.com/mseitzer/pytorch-fid)
  - description: Standard generative image quality metric used to evaluate the realism of generated image plans.

- Ray RLlib · [GitHub](https://github.com/ray-project/ray) · [Codewiki](https://codewiki.google/github.com/ray-project/ray) · [Doc](https://docs.ray.io/en/latest/rllib/index.html)
  - description: RL library used to train the PPO baseline agents reported in the paper.

- scikit-image · [GitHub](https://github.com/scikit-image/scikit-image) · [Doc](https://scikit-image.org/)
  - description: Used to compute SSIM in the verifier’s overall similarity metric between reference and generated charts.

- scikit-learn· [GitHub](https://github.com/scikit-learn/scikit-learn) · [Codewiki](https://codewiki.google/github.com/scikit-learn/scikit-learn) · [Website](https://scikit-learn.org/stable/)
  - description: ML utilities imported in the sandbox environment available to agents during analysis.
  - description: Classic ML toolkit used for tabular baselines, metrics, and utilities (e.g., KMeans, train/test split) in generated pipelines.
  - description: ML toolkit commonly invoked by DS-Agent and baselines for tabular tasks (e.g., logistic regression, preprocessing).
  - description: Library used for logistic regression and ridge regression probes (protein-level tasks) with hyperparameter search.
  - description: Used for evaluation utilities such as ROC curve and AUC in the state-prediction confidence analysis and related metrics.
  - description: Provides cosine_similarity for comparing color histograms in the verifier.

- SciPy · [GitHub](https://github.com/scipy/scipy) · [Codewiki](https://codewiki.google/github.com/scipy/scipy) · [Doc](https://docs.scipy.org/doc/scipy/reference/stats.html)
  - description: Scientific computing routines imported in the notebook environment supporting analysis.
  - description: Used to compute Pearson correlation and p-values to validate that GPT-4V automatic scores correlate with human evaluations.
  - description: Statistical tools (e.g., Pearson correlation) used to analyze confidence–accuracy relationships and other experiment metrics.
  - description: Scientific computing library referenced as part of the tool stack employed in synthesized/executed code.

- Selenium · [GitHub](https://github.com/SeleniumHQ/selenium) · [Codewiki](https://codewiki.google/github.com/SeleniumHQ/selenium) · [Website](https://www.selenium.dev/) · [Doc](https://www.selenium.dev/documentation/)
  - description: Browser automation toolkit used to implement the online web environment and execute actions (click, type, scroll, restart) during trajectory collection.
  - description: Standard browser automation toolkit underpinning ChromeDriver-based control; applicable to reproducing the simulated browsing environment used for data collection and rollouts.

- Soft Actor-Critic (SAC, PyTorch) · [GitHub](https://github.com/denisyarats/pytorch_sac)
  - description: Reinforcement learning algorithm used to train manipulation and locomotion subtasks in AnomalyGen.

- Text Generation Inference (TGI) · [GitHub](https://github.com/huggingface/text-generation-inference) · [Doc](https://huggingface.co/docs/text-generation-inference)
  - description: High-performance inference server used to host LMs for DSPy (listed as a requirement in Appendix F).

- The Alignment Handbook · [GitHub](https://github.com/huggingface/alignment-handbook)
  - description: Training recipes used to fine-tune Llama 3 8B as IoA’s communication LLM (following the SFT config referenced in the paper).

- tiktoken · [GitHub](https://github.com/openai/tiktoken) · [Codewiki](https://codewiki.google/github.com/openai/tiktoken)
  - description: Tokenizer used for analysis (cl100k_base) to measure documentation/instruction token lengths.

- torchtune · [GitHub](https://github.com/pytorch/torchtune)
  - description: Training framework specified in the paper’s hyperparameters; used to fine-tune the PLANNER and EXECUTOR.

- XFeat (Accelerated Features) · [GitHub](https://github.com/verlab/accelerated_features)
  - description: Lightweight feature matching toolkit used in evaluator functions for image matching on Ubuntu tasks.

### Utilities & Execution

- UTCP · [GitHub](https://github.com/universal-tool-calling-protocol)
  - description: Standardizes and shortens tool-call schemas so agents can invoke functions with minimal arguments.

- Functionary (Functionary‑Small‑v3.1) · [GitHub](https://github.com/MeetKai/functionary)
  - description: Open-source function-calling LLM used as a comparison baseline on BFCL‑v3 and a drop-in tool executor for agent experiments.

- PyMuPDF (fitz) · [GitHub](https://github.com/pymupdf/PyMuPDF) · [Doc](https://pymupdf.readthedocs.io/)
  - description: PDF text extraction dependency used by learned functions in GAIA experiments to parse papers and slides.

- pytesseract · [GitHub](https://github.com/madmaze/pytesseract)
  - description: Python wrapper around Tesseract used in the pipeline to obtain OCR text from screenshots during environment interactions.

- API-Bank · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
  - description: Function-calling/tool-use benchmark used to test API/tool-augmented planning; the paper samples training and test tasks from API-Bank and annotates categories/difficulties.
  - description: Function-calling benchmark used for Level-1/Level-2 evaluations; authors follow official evaluation setups (via baseline repos) when reporting results.

- BMTools · [GitHub](https://github.com/OpenBMB/BMTools)
  - description: Tool aggregation framework used by ToolBench; referenced as another source of APIs leveraged in the evaluation.

- CLIN · [GitHub](https://github.com/allenai/clin)
  - description: Continually learning language agent baseline that stores causal abstractions; used as a comparison to SELFGOAL.

- DLATK (Differential Language Analysis ToolKit) · [GitHub](https://github.com/wwbp/dlatk) · [Website](https://dlatk.wwbp.org)
  - description: Python toolkit the authors used for feature extraction, correlation analysis, and word‑cloud visualization of n‑grams and lexicon features in their belief-generation analyses.

- EASYTOOL (in JARVIS) · [GitHub](https://github.com/microsoft/JARVIS/tree/main/easytool)
  - description: Official code release from the paper; provides scripts to transform tool documentation into concise tool instructions and reproduce the reported experiments.

- IPython/Jupyter · [GitHub](https://github.com/ipython/ipython) · [Codewiki](https://codewiki.google/github.com/ipython/ipython) · [Website](https://ipython.org)
  - description: Provides the interactive Python server OpenHands uses for IPythonRunCellAction to execute agent-generated code.

- Iris (SciTools) · [GitHub](https://github.com/SciTools/iris)
  - description: Python library for meteorological and climate data; used to support GIS/climate-related analysis tasks.

- Label Studio · [GitHub](https://github.com/HumanSignal/label-studio) · [Website](https://labelstud.io/)
  - description: Annotation platform used to collect expert human evaluations and induce human-aligned criteria for reviewing agents.

- LeanEuclid · [GitHub](https://github.com/loganrjmurphy/LeanEuclid)
  - description: Euclidean geometry in Lean; included in the initial curriculum.

- Mermaid · [GitHub](https://github.com/mermaid-js/mermaid) · [Codewiki](https://codewiki.google/github.com/mermaid-js/mermaid) · [Website](https://mermaid.js.org) · [Doc](https://mermaid.js.org/intro/)
  - description: Diagramming and visual programming syntax used to encode flowchart-format workflow knowledge (Markdown Mermaid) throughout FlowBench.

- Moatless Tools· [GitHub](https://github.com/aorwall/moatless-tools)
  - description: Base agent framework extended by the authors (“Moatless-Adapted”) to support tree-structured states, backtracking, and test running; used as the primary baseline and foundation for SWE-Search.
  - description: Open-source SWE toolkit used as a comparison baseline against OpenHands on SWE-Bench Lite.

- OBS Studio · [GitHub](https://github.com/obsproject/obs-studio)
  - description: Open-source screen recorder used in AGENTNET TOOL to capture high-quality screen videos of demonstrations.

- PartNet-Mobility (SAPIEN) · [GitHub](https://github.com/haosulab/SAPIEN) · [Website](https://sapien.ucsd.edu) · [Doc](https://sapien.ucsd.edu/downloads/partnet-mobility-dataset)
  - description: Interactive articulated-object dataset/platform; the paper curates its anomalous household asset list from a subset of PartNet-Mobility.

- pip-tools · [GitHub](https://github.com/jazzband/pip-tools)
  - description: Used to resolve and install Python dependencies for each generated program during evaluation.

- pipreqs · [GitHub](https://github.com/bndr/pipreqs)
  - description: Tool the authors use to infer a program’s Python dependencies when setting up execution environments for evaluation.

- RapidFuzz python-Levenshtein · [GitHub](https://github.com/rapidfuzz/python-Levenshtein)
  - description: Library used to compute Levenshtein similarity in the paper’s correlation analyses (Appendix B).

- scvi-tools · [GitHub](https://github.com/scverse/scvi-tools)
  - description: Probabilistic deep generative models for single-cell omics; adapted in single-cell analysis tasks.

- SeeClick · [GitHub](https://github.com/TencentARC/SeeClick)
  - description: Specialized GUI grounding model evaluated on GroundUI; a strong open-source baseline practitioners can inspect for improving UI localization.

- Tool-Planner · [GitHub](https://github.com/OceannTwT/Tool-Planner)
  - description: Official code release of this paper’s framework for toolkit-based task planning and evaluation.

- Toolformer · [GitHub](https://github.com/TimoSchick/Toolformer)
  - description: Tool-use method referenced for multi-candidate tool invocation and voting strategies used in evolved ToolUse modules.

- ToolGen · [GitHub](https://github.com/Reason-Wang/ToolGen)
  - description: Official code and data release for the paper; provides training scripts for tool virtualization, memorization, retrieval, and end-to-end agent tuning, plus evaluation assets to reproduce results.

- TOOLSANDBOX · [GitHub](https://github.com/apple/ToolSandbox)
  - description: Official release of the paper’s stateful, conversational, interactive tool-use evaluation framework with 1032 scenarios, Python tool implementations, user-simulator prompts, and milestone/minefield evaluation.

- voyageai Python client· [GitHub](https://github.com/voyage-ai/voyageai-python) · [Website](https://www.voyageai.com) · [Doc](https://docs.voyageai.com)
  - description: Alternative embedding model used for ablations and reviewer ranking; accessed via the voyageai Python client.
  - description: Client library the paper used to compute voyage-3 embeddings for similarity metrics and reviewer retrieval.

- AutoGluon · [GitHub](https://github.com/autogluon/autogluon) · [Website](https://auto.gluon.ai)
  - description: State-of-the-art AutoML toolkit used as a baseline (Tabular, TimeSeries, Multimodal variants) in the experiments.
  - description: AutoML system used for empirical comparison in Appendix C.1 on tabular tasks.

- DSFD (Dual Shot Face Detector) · [GitHub](https://github.com/sfzhang15/DSFD)
  - description: Face detection model used as a tool within T3‑Agent.

- DuckTrack · [GitHub](https://github.com/TheDuckAI/DuckTrack)
  - description: Open-source mouse/keyboard event capture library the paper builds upon in AGENTNET TOOL for recording human interactions.

- Firecrawl · [GitHub](https://github.com/mendableai/firecrawl)
  - description: Web crawling/extraction tool mentioned as part of the paper’s document/web processing toolkit; aids in robust content extraction.

- FlashAttention-2· [GitHub](https://github.com/Dao-AILab/flash-attention) · [Codewiki](https://codewiki.google/github.com/Dao-AILab/flash-attention)
  - description: Fast attention kernels employed during training/inference to reduce memory and speed up ToolGen’s agent experiments.
  - description: Dependency used during training for memory-efficient exact attention.
  - description: Accelerated attention kernel employed during PRM training for efficiency.
  - description: Kernel used by the authors to speed up ESM-2 and Galactica during training/inference.
  - description: Optimized attention kernels used during ProGen3 model training for efficiency.

- InstructPix2Pix · [GitHub](https://github.com/timothybrooks/instruct-pix2pix)
  - description: Image editing model used as a tool within T3‑Agent.

- Lizard · [GitHub](https://github.com/terryyin/lizard)
  - description: Static analysis tool used by the paper to compute cyclomatic complexity of generated functions.

- LLMCompiler · [GitHub](https://github.com/mit-han-lab/llm-compiler)
  - description: Non-training-based system for parallel function calling; used as an open-source baseline for comparison with DTA-Llama.

- MAST-ML · [GitHub](https://github.com/uw-cmg/MAST-ML)
  - description: Materials Simulation Toolkit for ML; adapted to construct materials data-driven tasks.

- MODNet · [GitHub](https://github.com/ppdebreuck/modnet)
  - description: Materials property prediction toolkit; adapted for materials modeling/feature selection tasks.

- MUSK · [GitHub](https://github.com/lilab-stanford/MUSK)
  - description: Vision-language oncology foundation model; the vision component is wrapped by TOOLMAKER in the musk_extract_features task.

- nanobind · [GitHub](https://github.com/wjakob/nanobind)
  - description: C++/Python binding tool used to expose GPUDrive’s C++ engine through a Pythonic interface.

- OPENAGI · [GitHub](https://github.com/agiresearch/OpenAGI)
  - description: Tool-use/agent framework baseline for comparison.

- PaddleOCR · [GitHub](https://github.com/PaddlePaddle/PaddleOCR) · [Codewiki](https://codewiki.google/github.com/PaddlePaddle/PaddleOCR) · [Doc](https://paddleocr.readthedocs.io/en/latest/)
  - description: OCR toolkit used to augment the accessibility tree with screenshot text for improved GUI grounding in the ACI.
  - description: OCR toolkit used in SPA-Bench’s single-app success detection (coarse stage) to extract screen text for key-component matching.
  - description: OCR toolkit explicitly required in the open-ended OCR tasks; used to extract fields and amounts from images.

- pNeRF · [GitHub](https://github.com/aqlaboratory/pnerf)
  - description: Tool for reconstructing Cartesian coordinates from torsions; used in torsional denoising tasks to rebuild structures before feature computation.

- Prodigal-GV · [GitHub](https://github.com/apcamargo/prodigal-gv)
  - description: Gene-calling tool (via geNomad) used for protein-coding prediction from genomic/metagenomic inputs when constructing PPA-1.

- ProDy · [GitHub](https://github.com/prody/ProDy) · [Website](http://prody.csb.pitt.edu)
  - description: Toolkit used to run normal mode analysis (GNM/ANM) baselines for comparison against AlphaFLOW-MD.

- pynput · [GitHub](https://github.com/moses-palmer/pynput) · [Doc](https://pynput.readthedocs.io/en/latest/)
  - description: Python library to control mouse/keyboard for GUI automation; used as an alternative tool to execute actions.

- PyTube · [GitHub](https://github.com/pytube/pytube)
  - description: Used in IoA’s GAIA configuration to download YouTube video transcripts as part of the YouTube agent’s toolset.

- rembg · [GitHub](https://github.com/danielgatis/rembg) · [Codewiki](https://codewiki.google/github.com/danielgatis/rembg)
  - description: Image background removal tool used in the open-ended “Image Background Removal” tasks.

- Whisper (OpenAI) · [GitHub](https://github.com/openai/whisper) · [Codewiki](https://codewiki.google/github.com/openai/whisper)
  - description: Employed when YouTube videos lack transcripts; IoA’s YouTube tool uses Whisper to transcribe audio to text.
  - description: Speech-to-text system used to transcribe video audio for the “video frames in-context” and “video summary” agent baselines.

## Model Context Protocol (MCP)
- Model Context Protocol (MCP): Specification · [GitHub](https://github.com/modelcontextprotocol/specification) · [Website](https://modelcontextprotocol.io) · [Docs](https://www.anthropic.com/news/model-context-protocol)
- Model Context Protocol: Servers · [GitHub](https://github.com/modelcontextprotocol/servers)
- Model Context Protocol: Organization · [GitHub](https://github.com/modelcontextprotocol)
- ShareDrop (now LimeWire): File sharing · [GitHub](https://github.com/ShareDropio/sharedrop)

## Context Engineering
- MemGPT· [GitHub](https://github.com/cpacker/MemGPT) · [arxiv](https://arxiv.org/pdf/2310.08560)
  - description: Long-term memory framework used as a comparative memory strategy in the appendix ablation.
- 2025-01 Letta (formerly MemGPT) · [GitHub](https://github.com/letta-ai/letta)
- 2025-04 Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory · [GitHub](https://github.com/suzgunmirac/dynamic-cheatsheet)
- 2025-07 Context Engineering · [GitHub](https://github.com/davidkimai/Context-Engineering)
- 2025-07 [A Survey of Context Engineering for Large Language Models](https://arxiv.org/pdf/2507.13334)
- 2025-10 Accumulating Context Changes the Beliefs of Language Models · [GitHub](https://github.com/JiayiGeng/lm-belief-change) · [Website](https://lm-belief-change.github.io/)

## Agent SDK providers
### Openai
- github repo  [GitHub](https://github.com/orgs/openai/repositories)
- OpenAI Agents JS · [Github](https://github.com/openai/openai-agents-js)
- OpenAI Python API (Chat Completions)· [GitHub](https://github.com/openai/openai-python) · [Codewiki](https://codewiki.google/github.com/openai/openai-python) · 
- OpenAI API (GPT-4, GPT-3.5) · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: LLM API used to run agents and generate profiles; the paper evaluates systems primarily with gpt-4-1106-preview and gpt-3.5-turbo-0125.
  - description: Commercial LLMs used as the backbone in most experiments (model names given in the paper), accessed via the OpenAI API/SDK.
  - description: The authors run GPT-4-0314 via openai==0.28.0 and ChatCompletion.create for all core simulations; these docs and client library are required to reproduce the LLM calls.

#### LLM Dev Tools
- CLIP: Contrastive Language-Image Pre-Training · [GitHub](https://github.com/openai/CLIP) · [Website](https://openai.com/research/clip)

### Google ADK
- Google Agent Development Kit(ADK) Documentation · [Docs](https://google.github.io/adk-docs/)
- Google ADK Sessions & Memory · [Docs](https://google.github.io/adk-docs/sessions/memory/)
- Google ADK Safety Documentation · [Docs](https://google.github.io/adk-docs/safety/)
- Google ADK Guardrails Best Practices · [Docs](https://google.github.io/adk-docs/callbacks/design-patterns-and-best-practices/#guardrails-policy-enforcement)

## Agent SDK projects
- LangGraph · [GitHub](https://github.com/langchain-ai/langgraph) · [Docs](https://langchain-ai.github.io/langgraph/)
- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [CodeWiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Website](https://langchain.com)
- Langroid · [GitHub](https://github.com/langroid/langroid)
- n8n: Secure Workflow Automation for Technical Teams · [GitHub](https://github.com/n8n-io/n8n) · [Website](https://n8n.io)
- AutoGen · [GitHub](https://github.com/microsoft/autogen)
- CrewAI: Fast and Flexible Multi-Agent Automation Framework · [GitHub](https://github.com/crewAIInc/crewAI) · [Website](https://www.crewai.com)
- dspy· [GitHub](https://github.com/stanfordnlp/dspy) · [Website](https://dspy.ai/api/optimizers/GEPA/overview/) · [Docs](https://dspy.ai/api/optimizers/MIPROv2/)
- CAMEL: Framework Design Principles · [GitHub](https://github.com/camel-ai/camel)
- MetaGPT: The Multi-Agent Framework · [GitHub](https://github.com/geekan/MetaGPT)
- LlamaIndex · [GitHub](https://github.com/jerryjliu/llama_index) · [Docs](https://docs.llamaindex.ai/en/stable/module_guides/querying/)
- AutoGPT: Build, Deploy, and Run AI Agents · [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
- Agent Starter Pack · [GitHub](https://github.com/GoogleCloudPlatform/agent-starter-pack)
- A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence · [GitHub](https://github.com/CharlesQ9/Self-Evolving-Agents)
- DAMO ConvAI · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
- dspy· [GitHub](https://github.com/stanfordnlp/dspy) · [Codewiki](https://codewiki.google/github.com/stanfordnlp/dspy) · [Website](https://dspy.ai)
  - description: Official code and docs for the programming model introduced in the paper; provides signatures, modules (Predict, ChainOfThought, ReAct, etc.), teleprompters, and the compiler used in all experiments.
  - description: Prompt/program optimizer baseline (MIPROv2) used for comparison against SIRIUS.

### Frameworks & Platforms

- AG2 (AgentGym/AG2)· [GitHub](https://github.com/ag2ai/ag2) · [Doc](https://docs.ag2.ai/)
  - description: Open-source agent framework referenced in related work; a useful baseline/toolkit for practitioners comparing orchestration approaches.
  - description: Agent framework used in the paper to auto‑generate multi‑agent systems via the CaptainAgent algorithm; provides team formation, tools, and orchestration used to build the algorithm‑generated portion of Who&When.

- Agent S · [GitHub](https://github.com/simular-ai/Agent-S)
  - description: Official code release for the paper’s agentic framework, including the Experience-Augmented Hierarchical Planning, Agent-Computer Interface (ACI), memory modules, and evaluation scripts for OSWorld and WindowsAgentArena.

- Agent Skill Induction (ASI) · [GitHub](https://github.com/zorazrw/agent-skill-induction)
  - description: Official code release for this paper; implements programmatic skill induction, verification, and agent execution over BrowserGym/WebArena to reproduce results.

- Agent Workflow Memory (AWM) · [GitHub](https://github.com/zorazrw/agent-workflow-memory)
  - description: Official code release for this paper; contains the offline/online workflow induction pipeline, prompts, and evaluation scripts for WebArena and Mind2Web.

- agent-attack (VWA-Adv, attacks, defenses) · [GitHub](https://github.com/ChenWu98/agent-attack)
  - description: Official release from the paper containing the curated VWA-Adv adversarial tasks (200 tasks), evaluation scripts, attack/defense implementations, and code to reproduce the ARE analysis.

- Agent-Oriented-Planning (AOP) · [GitHub](https://github.com/lalaliat/Agent-Oriented-Planning)
  - description: Official code release of the paper; implements the AOP framework (meta-agent, reward model, detector, representative works, prompts) used in all experiments.

- AgentBoard (PDDL tasks) · [GitHub](https://github.com/hkust-nlp/AgentBoard)
  - description: Provides the PDDL game tasks used by the paper as the strategic game benchmark for evaluating MAS memory.

- AgentDropout · [GitHub](https://github.com/wangzx1219/AgentDropout)
  - description: Official code release for the paper; contains the implementation of Node Dropout and Edge Dropout, training/evaluation scripts, and configs to reproduce the reported results.

- AgentEval · [GitHub](https://github.com/Narabzad/AgentEval/)
  - description: Official code, data, prompts, and logs released by this paper for reproducing the AgentEval framework and experiments on MATH and ALFWorld.

- AgentGPT · [GitHub](https://github.com/reworkd/AgentGPT) · [Website](https://agentgpt.reworkd.ai)
  - description: LM-driven agent baseline used in preliminary tool-overuse experiments to show unnecessary tool invocation behavior.
  - description: Cited agent framework; example of LLM-based agents discussed in the introduction.

- AgentLego · [GitHub](https://github.com/InternLM/agentlego)
  - description: Open-source tool API library used as a comparison baseline agent framework in experiments.

- AgentLite · [GitHub](https://github.com/salesforce/AgentLite)
  - description: Referenced lightweight library for task-oriented LLM agent systems; a related open-source option for reproducing or scaling agent-based experiments.

- AgentOhana · [GitHub](https://github.com/SalesforceAIResearch/agentohana)
  - description: Prior open-source agent training pipeline cited for unified data and used as a comparison baseline in experiments (Sections 2.1, 5.2.1).

- AgentOhana / xLAM · [GitHub](https://github.com/Agent-Ohana/xLAM)
  - description: Baseline agent family (xLAM-7B-r) used for comparison; useful for practitioners examining alternative agent-tuning pipelines.

- AgentReview · [GitHub](https://github.com/Ahren09/AgentReview) · [Website](https://agentreview.github.io/)
  - description: Official codebase and project page for the paper’s LLM-agent simulation framework of peer review; primary resource to reproduce experiments and access assets described in the paper.

- AgentSense · [GitHub](https://github.com/ljcleo/agent_sense)
  - description: Official code and data release for the paper’s benchmark; includes scenario templates, prompts, simulation/evaluation scripts to reproduce AgentSense.

- AgentSquare · [GitHub](https://github.com/tsinghua-fib-lab/agentsquare)
  - description: Baseline for agent search; the paper adapts code snippets from this repo for textual gradient and includes AgentSquare as a comparison baseline.
  - description: Official implementation of the paper’s modular agent search framework (AgentSquare) and standardized module interfaces for Planning, Reasoning, Tool Use, and Memory.

- AgentTuning / AgentLM · [GitHub](https://github.com/THUDM/AgentTuning)
  - description: Baseline fine-tuned agent models and datasets compared against ATLAS; relevant for reproducing baseline results.

- AiTM (Agent-in-the-Middle) · [GitHub](https://github.com/PengfeiHePower/AiTM)
  - description: Official code release from this paper implementing the AiTM communication attack, including prompts and experiment scripts for AutoGen/CAMEL structures and real-world tests.

- AppAgent · [GitHub](https://github.com/RUC-GSAI/AppAgent)
  - description: Smartphone agent baseline whose action space the paper adopts alongside AndroidEnv; helpful for practitioners to compare agent formulations and prompts.

- Automated Design of Agentic Systems (ADAS)· [GitHub](https://github.com/ShengranHu/ADAS)
  - description: Baseline/related toolkit; the paper adapts part of ADAS’s textual-gradient implementation for operator updates and compares against ADAS in experiments.
  - description: Official code release for the paper’s Meta Agent Search algorithm and framework; used to generate, evaluate, and archive discovered agents across all experiments.

- BioDiscoveryAgent · [GitHub](https://github.com/snap-stanford/BioDiscoveryAgent)
  - description: Official code release from the paper; includes prompts, agent implementation, tool integrations, and scripts to reproduce the closed-loop experiment design experiments.

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent) · [Doc](https://huggingface.co/THUDM/cogagent-chat-hf)
  - description: Open-source VLM agent baseline evaluated on GroundUI; relevant for reproducing baseline results and comparisons.
  - description: Visual language model for GUI/web agents; evaluated as a multimodal agent baseline on MMInA.
  - description: Specialist GUI agent evaluated for coordinate-based action prediction in the paper’s tasks.
  - description: Visual-language model for GUI agents; included as an “agent-as-a-model” baseline integrated and evaluated in SPA-Bench.
  - description: Vision-language GUI agent evaluated as an open-source baseline on UI-Vision tasks.
  - description: GUI-focused visual-language agent used as an open-source baseline and fine-tuned on VAB trajectories.
  - description: Open-source GUI-focused LVLM used as a comparison baseline on ScreenSpot in the paper.
  - description: Open-weights GUI-focused MLLM used as the open-source backbone; the paper meta-trains (FOMAML) and few-shot adapts CogAgent, and loads the THUDM/cogagent-chat-hf checkpoint.
  - description: GUI-specialized multimodal model used as a grounding/agent baseline in comparisons.

- CRAB (Cross-environment Agent Benchmark) · [GitHub](https://github.com/camel-ai/crab) · [Doc](https://github.com/camel-ai/crab/blob/main/crab-benchmark-v0/README.md)
  - description: Official repository for the paper’s framework and benchmark; includes code, evaluator graphs, environment interface, and reproducibility instructions used to build and run CRAB Benchmark-v0.

- DMAD (Diverse Multi‑Agent Debate) · [GitHub](https://github.com/MraDonkey/DMAD)
  - description: Official code release of the paper; provides prompts, debate pipelines, and scripts to reproduce DMAD and all baselines across LLM and MLLM benchmarks.

- DS-Agent · [GitHub](https://github.com/guosyjlu/DS-Agent)
  - description: Official code release for the paper; includes the DS-Agent framework, datasets/splits, prompts, case banks, and scripts to reproduce both development and deployment experiments.

- Generative Agents· [GitHub](https://github.com/joonspk-research/generative_agents) · [Codewiki](https://codewiki.google/github.com/joonspk-research/generative_agents) · [Website](https://generativeagents.dev)
  - description: Single-agent simulation framework baseline; its memory module was reused in discovered high-performing combinations.
  - description: Prior cognitive-agent architecture that Hypothetical Minds builds upon (modules for perception, memory, and planning), extended here with a new Theory-of-Mind module.
  - description: Agent framework featuring observational and reflective memory; used as a single-agent memory baseline adapted to MAS for comparison.
  - description: Referenced codebase the authors inspected (Appendix C) to determine simulation modes; earlier versions used a SCRIPT-like mode, informing their analysis of omniscient simulations.
  - description: Proof-of-concept single-LLM agent system cited when describing individual agent architecture; practitioners can compare memory and behavior modules with CogMir’s design.

- Hammer 2.0 · [GitHub](https://github.com/MadeAgents/Hammer) · [Website](https://huggingface.co/MadeAgents/Hammer2.0-7b)
  - description: Baseline function-calling system; the paper incorporates Hammer-style data for CoALM-IT training and evaluates against Hammer on API-Bank and BFCL.

- HiAgent · [GitHub](https://github.com/HiAgent2024/HiAgent)
  - description: Official code release for the paper; contains the hierarchical working-memory agent implementation, prompts, and scripts needed to reproduce the results reported in the paper.

- Internet of Agents (IoA) · [GitHub](https://github.com/OpenBMB/IoA)
  - description: Official code release of the paper’s framework, including the server/client implementations, message protocol, team formation, and conversation flow control used in all experiments.

- KnowAgent · [GitHub](https://github.com/zjunlp/KnowAgent)
  - description: Knowledge-augmented agent baseline referenced and compared in the paper’s discussion of workflow-guided planning.

- LeanAgent · [GitHub](https://github.com/lean-dojo/LeanAgent)
  - description: Official code release for the paper; implements the lifelong learning framework (curriculum construction, dynamic database, progressive retriever training, best-first proof search) and scripts to reproduce experiments and generate PRs.

- MatPlotAgent / MatPlotBench · [GitHub](https://github.com/thunlp/MatPlotAgent)
  - description: Official release from the paper containing the MatPlotAgent framework, MatPlotBench benchmark (100 cases), prompts, and evaluation scripts needed to reproduce results.

- Mixture-of-Agents (MoA) · [GitHub](https://github.com/togethercomputer/moa)
  - description: Official code release for the paper; includes prompts and evaluation scripts to reproduce the Mixture-of-Agents results reported (footnote link in the paper).

- MMedAgent · [GitHub](https://github.com/Wangyixinxin/MMedAgent)
  - description: Official code and web UI for the paper’s multi-modal medical agent, including the instruction-tuning data and tool integration needed to reproduce the system.

- MMRole (MMRole-Agent, MMRole-Data, MMRole-Eval) · [GitHub](https://github.com/YanqiDai/MMRole)
  - description: Official release of this paper containing code, dataset, prompts, trained models, and the reward-model based evaluation needed to reproduce MMRole-Agent, MMRole-Data, and MMRole-Eval.

- MobileReach (dataset) · [GitHub](https://github.com/XiaoMi/reachagent)
  - description: Official code release of the paper, including the MobileReach dataset, two-stage SFT+RL implementation with action alignment, and evaluation scripts for reproducing results.
  - description: The paper’s new dataset for page navigation, page reaching, and page operation; used to train and evaluate ReachAgent and released together with the code.

- MobileUse Toolkit · [GitHub](https://github.com/MadeAgents/mobile-use)
  - description: Official code release from the paper; provides the hierarchical-reflection mobile GUI agent and an out-of-the-box toolkit with WebUI for operating physical Android devices via ADB.

- ModelScope-Agent · [GitHub](https://github.com/modelscope/modelscope-agent) · [Website](https://modelscope.cn/)
  - description: Open-source agent framework; the paper adopts its static evaluation idea to compare model outputs with annotated references for ToolBench.

- Multi-LLM-Agent (α-UMi) · [GitHub](https://github.com/X-PLUG/Multi-LLM-Agent)
  - description: Official code release for the paper’s multi-LLM agent with GLPFT training, prompts for planner/caller/summarizer, and evaluation scripts.

- Open-SMARTAgent · [GitHub](https://github.com/qiancheng0/Open-SMARTAgent)
  - description: Official code release for SMART, including SMART-ER dataset, training/inference scripts, and SMARTAgent checkpoints used throughout the paper.

- OS Agents Survey · [GitHub](https://github.com/os-agent-survey/os-agent-survey.github.io) · [Website](https://os-agent-survey.github.io/)
  - description: Official project page and continuously-updated repository maintained by the authors with curated papers, benchmarks, products, and resources on OS Agents; cited in the paper as the open-source resource accompanying the survey.

- ProAgent · [GitHub](https://github.com/THUDM/ProAgent)
  - description: Agentic process automation system using code/control flows; listed as a related workflow-guided agent baseline in the benchmark comparison.

- Qwen-Agent · [GitHub](https://github.com/QwenLM/Qwen-Agent)
  - description: Agent toolkit used to implement the ReAct-based system and tool-calling; the paper builds WebDancer on top of this framework and trains with chatml formatting.

- Qwen-Agent Code Interpreter Benchmark · [GitHub](https://github.com/QwenLM/Qwen-Agent/tree/main/benchmark)
  - description: External benchmark whose visualization subsets were used to further evaluate MatPlotAgent and analyze the effect of visual feedback.

- ResearchAgent · [GitHub](https://github.com/JinheonBaek/ResearchAgent)
  - description: Official code release for the paper; implements the ResearchAgent pipeline, prompts, and evaluation setup for iterative research idea generation with literature, entity knowledge store, and reviewing agents.

- ResearchAgent · [GitHub](https://github.com/snap-stanford/ResearchAgent)
  - description: Baseline agent compared against DS-Agent in the development stage experiments.

- ScreenAgent (dataset) · [GitHub](https://github.com/niuzaisheng/ScreenAgent/tree/main/data/ScreenAgent/train)
  - description: Dataset referenced in the dataset comparison and statistics; used as a point of comparison for scale/steps in AgentTrek’s analysis.

- smolagents (Hugging Face Agents) · [GitHub](https://github.com/huggingface/smolagents) · [Codewiki](https://codewiki.google/github.com/huggingface/smolagents) · [Website](https://huggingface.co/blog/open-deep-research)
  - description: Open-source library and “Open Deep Research” baseline referenced in experiments; provides agents and search/browsing baselines compared against WORKFORCE.

- SWE-Agent· [GitHub](https://github.com/princeton-nlp/SWE-agent) · [Website](https://swe-agent.com)
  - description: Prior ACI design for software engineering agents that inspired Agent S’s ACI abstraction for GUI control, relevant for practitioners extending ACI concepts.
  - description: Software-engineering agent whose Agent-Computer Interface and file-editing utilities are adapted in OpenHands’ AgentSkills and used as a SWE baseline.

- TapeAgents · [GitHub](https://github.com/ServiceNow/TapeAgents)
  - description: Agent development and optimization framework; used as a baseline on GAIA.

- TELL ME A STORY (Agents’ Room) · [GitHub](https://github.com/google-deepmind/tell_me_a_story)
  - description: Official release from the paper including the high-quality prompts and human-written stories dataset, splits, and evaluation metrics/scripts used to assess long-form narratives.

- TrustAgent · [GitHub](https://github.com/agiresearch/TrustAgent)
  - description: Official code and data release for the paper’s Agent-Constitution framework implementing pre-, in-, and post-planning safety strategies, plus the synthetic datasets and evaluation scripts used in the experiments.

- VisualAgentBench (VAB)· [GitHub](https://github.com/THUDM/VisualAgentBench)
  - description: Benchmark suite from the authors where parts of AndroidLab’s SoM modes are included as the VAB-Mobile component; useful for inspecting the SoM grounding and comparing across visual agent settings referenced by the paper.
  - description: Official release of the benchmark, environments, training trajectories, prompts, and evaluation code for all five settings (VAB-OmniGibson, VAB-Minecraft, VAB-AndroidLab, VAB-WebArena-Lite, VAB-CSS).

- WindowsAgentArena · [GitHub](https://github.com/microsoft/WindowsAgentArena)
  - description: Windows OS benchmark used to evaluate cross-OS generalization; includes the NAVI baseline referenced in the paper.
  - description: Microsoft’s Windows-centric benchmark used by the paper to evaluate online agent performance on Windows.

- XAgent · [GitHub](https://github.com/OpenBMB/XAgent)
  - description: Agent framework evaluated in the preliminary study; used to demonstrate tool overuse patterns on GSM8K-style tasks.
  - description: Autonomous agent framework included as a baseline in the ML benchmark.

- H2O AutoML · [GitHub](https://github.com/h2oai/h2o-3) · [Doc](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
  - description: Related AutoML system discussed in the paper’s comparison of DS-Agent vs. AutoML approaches.

- Self-Refine · [GitHub](https://github.com/madaan/self-refine)
  - description: Hand-crafted agent baseline that iteratively refines outputs using self-feedback.
  - description: Iterative self-feedback method re-implemented as one of the baseline MAS designs in the MAS pool.
  - description: Iterative self-correction baseline compared against in the paper’s experiments.


### Multi-agent Systems

- AgentPrune · [GitHub](https://github.com/yanweiyue/AgentPrune)
  - description: Official code release for the paper’s economical multi-agent communication pruning framework; used to reproduce AgentPrune and its integrations.

- AgentScope · [GitHub](https://github.com/modelscope/agentscope) · [Website](https://modelscope.cn/agentscope)
  - description: Multi-agent framework used to implement VIRSCI; the paper specifically uses AgentScope’s KnowledgeBank module to store and retrieve scientist profiles during collaboration.
  - description: Modular multi-agent platform cited in related work; included as a relevant toolkit practitioners may inspect alongside the paper’s decentralized approach.

- AgentsCourt / SimuCourt · [GitHub](https://github.com/Zhitao-He/SimuCourt)
  - description: Official release of the paper, containing the multi-agent AgentsCourt framework and the SimuCourt benchmark; includes code for court debate simulation, retrieval, and evaluation, plus the constructed Legal-KB resources used in experiments.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse) · [Doc](https://openbmb.github.io/agentverse/)
  - description: Multi-agent collaboration framework; included as a hand-crafted multi-agent baseline.
  - description: Baseline multi‑agent framework compared against EVOAGENT in Table 1; discussed as a human‑designed pipeline that EVOAGENT seeks to automate.
  - description: Hierarchical multi-agent framework with dynamic recruitment; used as a Hierarchical-structure baseline in all tasks.
  - description: Multi-agent framework for assembling expert agents in structured topologies; used as a comparison baseline.
  - description: Multi-agent collaboration toolkit used as a baseline for Gobang and national policy generation.
  - description: Multi-agent collaboration framework used as a dynamic MAS baseline in experiments.
  - description: Referenced agent-oriented framework for multi-agent collaboration; an alternative platform to build and extend the social simulations described.
  - description: Multi-agent collaboration framework referenced in related work; useful for practitioners extending CogMir’s multi-agent interactions.
  - description: Platform for multi-agent collaboration and emergent behavior analysis; cited as an interactive environment for Social-AI research.

- AgentVerse · [GitHub](https://github.com/THUDM/AgentVerse) · [Website](https://agentverse.ai)
  - description: Multi-agent collaboration framework compared as a baseline in the experiments.
  - description: Baseline multi-agent framework that assembles expert agents in chained/hierarchical structures; used for comparison in experiments.

- AutoAgents · [GitHub](https://github.com/Link-AGI/AutoAgents)
  - description: Automatic multi-agent generation framework; used as a comparison baseline.
  - description: Baseline automatic agent‑generation framework compared in Table 1; the paper contrasts its fixed human‑designed architecture with EVOAGENT’s EA‑driven generation.

- AutoML-Agent · [GitHub](https://github.com/deepauto-ai/automl-agent)
  - description: Official code release for the paper’s multi-agent full-pipeline AutoML framework; used to reproduce all methods, planning, execution, and verification components.

- CAMEL (Communicative Agents for “Mind” Exploration) · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel) · [Website](https://www.camel-ai.org/)
  - description: Communicative role‑playing agents framework; the paper demonstrates EVOAGENT can automatically generate roles to use within CAMEL instead of manual role design (Appendix D).
  - description: Two-agent “User–Assistant” flat collaboration framework; used as a Flat-structure baseline across tasks and as a target for AUTOTRANSFORM/AUTOINJECT case studies.
  - description: Role-playing multi-agent framework used as a second implementation platform for all communication structures and experiments.
  - description: Multi-agent framework emphasizing role-playing agents; used as a comparison baseline in the experiments.
  - description: Communicative agents framework used as a baseline across tasks (benchmarks, Gobang, policy simulation).
  - description: Open-source instruction-tuned baseline model compared in automatic benchmarks.
  - description: Multi-agent role-playing environment referenced in related work; provides reusable agent interaction patterns for extensions.
  - description: Multi-agent LLM framework used to study emergent social behaviors; referenced as a platform for dynamic text-agent interactions.

- CAMEL (Communicative Agents for “Mind” Exploration) · [GitHub](https://github.com/lightaime/camel)
  - description: Multi-agent framework the authors built upon to implement OWL and the experiment pipeline; useful for understanding the underlying agent infrastructure.

- context-plus · [GitHub](https://github.com/Multi-Agent-LLMs/context-plus)
  - description: RAG utility used in the “challenge” ablation to retrieve additional context from Wikipedia.

- LLM-Game-Agent · [GitHub](https://github.com/3DAgentWorld/LLM-Game-Agent)
  - description: Official code release for this EMNLP 2024 paper; contains the multi-agent Avalon framework (memory, analysis, planning, action, response, and experience learning), prompts, and the Python game program used to run all experiments and analyses.

- MaAS (Multi-agent Architecture Search) · [GitHub](https://github.com/bingreeky/MaAS)
  - description: Official code release of the paper; implements the agentic supernet, controller, sampling, and training/evaluation scripts used across all experiments.

- MAD (Multi-Agent Debate) · [GitHub](https://github.com/Skytliang/MAD)
  - description: Hierarchical debate framework with two debaters and a judge; used as a Hierarchical-structure system across tasks and central to analyses where injected errors sometimes improve performance.
  - description: Multi-agent baseline adapted by the authors for open-ended chat comparison in Table 5.

- MAgIC (Multi-Agent in Cognition, Adaptability, Rationality and Collaboration) · [GitHub](https://github.com/cathyxl/MAgIC)
  - description: Official repository for the paper’s benchmark, scenarios, prompts, data, and PGM-aware agent code used to evaluate LLMs in social deduction and game-theory multi-agent settings.

- MALLM (Multi‑Agent Large Language Models) · [GitHub](https://github.com/Multi-Agent-LLMs/mallm)
  - description: Framework used by the authors to run multi‑agent debates (personas, response generators, discussion paradigms, decision protocols) in their experiments.

- MALR (Multi-Agent framework for improving complex Legal Reasoning) · [GitHub](https://github.com/yuanwk99/MALR)
  - description: Official code release for the paper’s multi-agent, non-parametric framework (auto-planner, sub-task agents, adaptive rule-insights, reasoning modules) used to reproduce experiments and ablations.

- MAM (Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis) · [GitHub](https://github.com/yczhou001/MAM)
  - description: Official code release of the paper; implements the role-specialized multi-agent pipeline (General Practitioner, Specialist Team, Radiologist, Medical Assistant, Director), prompts, retrieval, and evaluation setup.

- MAPPO (Multi-Agent PPO)· [GitHub](https://github.com/marlbenchmark/on-policy)
  - description: Reference implementation of multi-agent PPO from Yu et al. (2022); the paper adapts multi-agent PPO to the language domain for co-training agents.
  - description: Cooperative MARL baseline re-implemented per “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games” used for comparison on TDW-Game/TDW-Cook.

- MARBLE (MultiAgentBench) · [GitHub](https://github.com/ulab-uiuc/MARBLE)
  - description: Official codebase and datasets for the paper’s benchmark and multi-agent coordination framework; required to reproduce all experiments across research, Minecraft, database, coding, bargaining, and werewolf scenarios.

- MegaAgent · [GitHub](https://github.com/Xtra-Computing/MegaAgent)
  - description: Official code release of the paper’s large-scale autonomous LLM-based multi-agent system used in all experiments (Gobang development, national policy simulation, and benchmarks).

- MindAgent · [GitHub](https://github.com/OpenGVLab/MindAgent)
  - description: Multi-agent LLM baseline evaluated alongside ReAd-J in DV-RoCoBench and Overcooked-AI.

- MultiAgent_ImplicitBias · [GitHub](https://github.com/MichiganNLP/MultiAgent_ImplicitBias)
  - description: Official code and data release for the paper, including the multi-agent interaction framework, the Scenarios/Fine-tune/Test datasets, prompts, and evaluation scripts used to detect and mitigate implicit gender bias.

- Multi‑Agent Debate (Du et al., 2024) · [GitHub](https://github.com/composable-models/llm_multiagent_debate)
  - description: Canonical MAD baseline implementation; the paper adopts MAD’s default settings to compare fixed‑method debates against DMAD.

- TriageAgent · [GitHub](https://github.com/Lucanyc/TriageAgent)
  - description: Official code and dataset release of the paper; contains the heterogeneous multi-agent framework, prompts, and the first public ESI clinical triage benchmark used in the experiments.

- Hydra-Multi · [GitHub](https://github.com/MIT-SPARK/Hydra-Multi)
  - description: Multi-robot extension of Hydra used as a reference for obtaining region connectivity, semantic mesh, and agent/object state layers in real deployments.

- LLM-Debate · [GitHub](https://github.com/ucl-dark/llm_debate)
  - description: Debate-style multi-agent method; used as both an operator concept and a baseline implementation.

- ReConcile · [GitHub](https://github.com/JustinChihYaoChen/ReConcile)
  - description: Multi-agent consensus baseline adapted by the authors for open-ended chat comparison in Table 5.


## evaluation & benchmark

### Eval Suites

- Promptfoo: LLM evals & red teaming · [GitHub](https://github.com/promptfoo/promptfoo)
- CEval · [GitHub](https://github.com/SJTU-LIT/CEval) · [Website](https://cevalbenchmark.com)
- EmbodiedBench · [GitHub](https://github.com/EmbodiedBench) · [Website](https://embodiedbench.github.io)
  - description: Official benchmark suite and code release with EB‑ALFRED/EB‑Habitat/EB‑Navigation/EB‑Manipulation tasks, unified agent, and evaluation scripts for embodied MLLMs.

- AI Scientist · [GitHub](https://github.com/SakanaAI/AI-Scientist) · [Website](https://sakana.ai/ai-scientist/)
  - description: Baseline single-agent system compared against VIRSCI; the paper aligns settings and evaluates against its LLM-review metric.

- AI-Researcher · [GitHub](https://github.com/NoviScl/AI-Researcher)
  - description: Official release for this paper; contains the LLM ideation agent implementation (retrieval, generation, ranking) and the full human review scores used in the study.

- Aria-UI · [GitHub](https://github.com/aria-ui/Aria-UI)
  - description: Open-source GUI grounding model evaluated as a baseline for element grounding.

- AucArena · [GitHub](https://github.com/jiangjiechen/auc-arena)
  - description: Auction simulation environment used for the First-price Auction experiments; the paper evaluates agents in this arena to measure strategic planning and profit.

- AutoDock Vina · [GitHub](https://github.com/ccsb-scripps/AutoDock-Vina)
  - description: Docking engine on which Smina is based; cited alongside Smina in binding affinity evaluation.

- Needle-in-a-Haystack (original) · [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
  - description: The original long-context stress test that PLUS extends to multi-needle and diversified settings; useful for evaluating context retention.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://microsoft.github.io/autogen/) · [Doc](https://microsoft.github.io/autogen/docs/Getting-Started)
  - description: MAS framework used as a comparison baseline, and source of the Web Browser and Code Executor tools adapted for IoA’s GAIA setup.
  - description: Microsoft’s multi‑agent conversation framework used as an initial agent framework that EVOAGENT can extend; also shown in the paper’s adaptation examples (Appendix D).
  - description: Multi-agent conversation framework used as a comparison baseline in experiments.
  - description: Multi‑agent conversation framework the authors used to manage interacting and judging threads during simulations.
  - description: Multi-agent conversation framework used as a comparison baseline on larger models.
  - description: Multi-agent conversation framework that the paper plugs AgentPrune into for experiments and cost analyses.
  - description: Multi‑agent framework the paper uses to run the hand‑crafted Magnetic‑One system and follows its default settings during experiments.
  - description: Multi-agent conversation framework cited in related work; relevant open-source toolkit practitioners can inspect when extending LLM-agent simulations similar to AgentReview.
  - description: Multi-agent conversation framework whose manager/mediator design is followed for SimClass’s hidden manager agent that selects speakers and actions.
  - description: Multi-agent conversation framework used to build and control agent communications (Chain/Tree/Complete/Random) for the main simulations in this paper.
  - description: Multi-agent conversation framework used as a primary baseline across data analysis, ML, and MATH experiments.
  - description: Open-source multi-agent framework used as one of the agent systems evaluated in the paper for both data analysis and data modeling tasks.
  - description: Open-source multi-agent conversation framework used to implement TRIAGEAGENT’s agent orchestration and group chat workflow.
  - description: Multi-agent LLM framework into which the authors integrated their AgentOptimizer; use this to reproduce the paper’s agent-training within a maintained library.
  - description: Multi-agent LLM framework used as a baseline; the paper compares Flow against AutoGen across coding and document-generation tasks.
  - description: Multi-agent LLM framework used to implement AgentEval; also provides the AutoGen 2-agent and 3-agent baseline systems evaluated in the paper.
  - description: Multi-agent conversation framework used as a baseline in both Gobang and national policy experiments.
  - description: Multi-agent conversation framework cited as a representative MAS platform.
  - description: Multi-agent framework whose evaluation prompts are reused in the paper’s LLM-based answer extraction and judging.
  - description: Multi-agent conversation framework used as one of the three MAS backbones; G-Memory is plugged into AutoGen to evaluate memory augmentation.
  - description: Referenced multi-agent conversation framework; useful for practitioners exploring different infrastructure for agent communication and coordination.

- Baize · [GitHub](https://github.com/project-baize/baize)
  - description: Open-source instruction-tuned baseline model evaluated against WizardLM.

- BGE (FlagEmbedding)· [GitHub](https://github.com/FlagOpen/FlagEmbedding) · [Website](https://huggingface.co/BAAI/bge-base-en)
  - description: Text-embedding model used for retrieving images relevant to generated file contents during data synthesis.
  - description: Retrieval model used in the paper’s RAG‑based few‑shot in‑context learning experiment in the appendix.
  - description: Alternative retriever family evaluated in Appendix E to test robustness across retrieval backbones.
  - description: Multilingual multi-function embeddings used to compute solution representations in the diversity analysis.
  - description: Retriever used for the RAG baselines (top-k and multi-round); BGE m3 embeddings power the retrieval component compared against LONGAGENT.
  - description: Sentence-embedding model (BGE-Large) used to encode and re-rank BM25 candidates to select the optimal precedent and extract related legal articles.

- BLEURT-20 · [GitHub](https://github.com/google-research/bleurt) · [Doc](https://bleurt.readthedocs.io/en/latest/)
  - description: Learned metric used to score translation quality on CommonMT following prior work cited by the paper.

- ChatDev (Puppeteer branch) · [GitHub](https://github.com/OpenBMB/ChatDev/tree/puppeteer)
  - description: Official release of this paper’s orchestration-based multi-agent system; contains code to reproduce results, the orchestrator policy, agent prompts/tools, and evaluation scripts (SRDD data/metrics are hosted in this repo).

- CommonGen-Hard (from Self-Refine) · [GitHub](https://github.com/allenai/self-refine)
  - description: Hard subset of CommonGen introduced by Self-Refine; used to evaluate open-ended generation quality (grammar, fluency, relevance, logic).

- con-nf · [GitHub](https://github.com/leanprover-community/con-nf)
  - description: Formal consistency proof of Quine’s New Foundations; included in the initial curriculum/evaluation.

- Coxeter · [GitHub](https://github.com/NUS-Math-Formalization/coxeter)
  - description: Lean formalization of Coxeter groups; used in evaluation where LeanAgent proved a nontrivial lemma about Coxeter systems.

- DeepFRI · [GitHub](https://github.com/flatironinstitute/DeepFRI)
  - description: Provider of GO annotation tasks (MF/BP/CC) used to evaluate ProSST in multi-label function prediction.

- DeepSeek-V3 · [GitHub](https://github.com/deepseek-ai/DeepSeek-V3) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-V3) · [Doc](https://api-docs.deepseek.com)
  - description: Largest base model evaluated via API in the paper (DeepSeek-V3-671B-Instruct).
  - description: Additional LLM used for generation ablations; evaluated alongside GPT-4o-mini and Qwen to compare simulation quality.

- DeepSeek‑R1 · [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-R1) · [Website](https://www.deepseek.com/)
  - description: Open-source reasoning LLM baseline (Distill-Qwen-32B variant used) evaluated on MMInA single- and multi-hop tasks.
  - description: Reasoning model evaluated as an alternative judge; results reported for both agent‑ and step‑level attribution.
  - description: Teacher model used to generate Chain-of-Thought traces for both PLANNER and EXECUTOR in the CoT experiments.
  - description: Open-source reasoning model used as a MAS-driving LLM in AIME‑2024 evaluations.

- deeptime· [GitHub](https://github.com/deeptime-ml/deeptime) · [Doc](https://deeptime-ml.github.io/deeptime/)
  - description: Library used to perform TICA and project structures onto slow dynamical components for distributional comparisons (JS distance).
  - description: Library used to fit Time-lagged Independent Component Analysis (TICA) for fidelity metrics and visualization.
  - description: Library for dynamical modeling and TICA; used to compute TICA projections for distributional evaluation (JS divergence on TIC components).

- DiffDock · [GitHub](https://github.com/gcorso/DiffDock)
  - description: Docking method used to place ligands (e.g., heme) into designed proteins for function-based design evaluation.

- dill · [GitHub](https://github.com/uqfoundation/dill)
  - description: Python function serialization used by the framework to package actions/evaluators (“code as configuration”).

- DroidBot · [GitHub](https://github.com/honeynet/droidbot)
  - description: UI-guided test input generator for Android; AutoDroid (one of the evaluated agents) relies on DroidBot, which is why the paper evaluates AutoDroid only on single‑app tasks.

- EasyOCR · [GitHub](https://github.com/JaidedAI/EasyOCR) · [Codewiki](https://codewiki.google/github.com/JaidedAI/EasyOCR) · [Doc](https://www.jaided.ai/easyocr/)
  - description: OCR engine used to detect and label interactive text elements on screens for visual prompting and evaluation.
  - description: OCR toolkit used in the paper’s multi-criteria verifier to extract text from reference and generated charts.

- Elasticsearch · [GitHub](https://github.com/elastic/elasticsearch) · [Website](https://www.elastic.co/elasticsearch) · [Doc](https://www.elastic.co/guide/index.html)
  - description: Open-source search engine used as the symbolic retrieval module in the decomposition framework for open-domain QA experiments.

- EvoDiff · [GitHub](https://github.com/salesforce/evodiff)
  - description: Sequence-generation baseline used for unconditional sequence generation comparisons and motif-scaffolding sequence-based evaluation.

- FACTSCORE · [GitHub](https://github.com/google-research/factscore)
  - description: Fine-grained factuality evaluation metric referenced as a technical discriminator option for objective assessments in the framework.

- FILM: Following Instructions in Language with Modular Methods · [GitHub](https://github.com/soyeonm/FILM)
  - description: Modular ALFRED agent architecture whose component tools (finetuned Mask R-CNN detector and Fast Marching Method planner) are the basis for the paper’s multimodal tool-error evaluation.

- FLT (Fermat’s Last Theorem) · [GitHub](https://github.com/ImperialCollegeLondon/FLT)
  - description: Formalization efforts around Fermat’s Last Theorem; part of the initial curriculum/evaluation set.

- Foundation (Formalized Formal Logic) · [GitHub](https://github.com/FormalizedFormalLogic/Foundation)
  - description: Formalized results in logic; part of the initial curriculum/evaluation.

- GCG (llm-attacks) · [GitHub](https://github.com/llm-attacks/llm-attacks)
  - description: Official codebase for the GCG jailbreak baseline used throughout the paper for comparison, including training scripts and evaluation utilities.

- Genie2 · [GitHub](https://github.com/aqlaboratory/genie2)
  - description: Backbone diffusion baseline trained on AFDB subsets; evaluated both at full temperature and with reduced noise according to the paper’s protocol.

- gensim· [GitHub](https://github.com/RaRe-Technologies/gensim)
  - description: Distance metric (implemented in Gensim) used to quantify semantic distance between generated text plans and reference texts.

- GLM-4· [GitHub](https://github.com/THUDM/GLM-4) · [Website](https://chatglm.cn) · [Doc](https://open.bigmodel.cn/dev/api)
  - description: Multimodal GLM model (visual variant GLM-4V) used as an open-source baseline and fine-tuned under the paper’s behavior cloning setup.
  - description: Model family from Zhipu AI used in additional experiments (appendix) as a comparison point on BENCHFORM.
  - description: Additional LLM backend (GLM4-9B) evaluated in the appendix for generalization.

- GraphRAG · [GitHub](https://github.com/microsoft/graphrag) · [Codewiki](https://codewiki.google/github.com/microsoft/graphrag) · [Doc](https://microsoft.github.io/graphrag/)
  - description: Knowledge-graph based RAG framework; used to build the Mind-Map agent for graph construction and Graph-RAG retrieval in the method.

- GROOT · [GitHub](https://github.com/CraftJarvis/GROOT)
  - description: Video-instruction-following agent evaluated on MCU; requires a reference video per task and is integrated via MineStudio.

- Hanabi Learning Environment · [GitHub](https://github.com/deepmind/hanabi-learning-environment) · [Website](https://ai.googleblog.com/2019/06/hanabi-new-challenge-for-reinforcement.html)
  - description: Official environment for the Hanabi Challenge used to run Hanabi Agentic Coordination experiments and evaluate MARL/LLM agents.

- InterProt · [GitHub](https://github.com/etowahadams/interprot) · [Website](https://interprot.com)
  - description: Official codebase and interactive feature visualizer released by the paper; used to train/evaluate TopK SAEs on ESM-2 activations and to visualize latent activations on sequences/structures for interpretation and the human rater study.

- Jericho · [GitHub](https://github.com/microsoft/jericho)
  - description: Text-based interactive fiction environment suite; used as one of the five evaluation tasks to assess agent performance in long-horizon text-game settings.
  - description: Text-game framework used as one of the held-out evaluation environments (with tasks shortened per the paper’s setup).

- Lean4 PDL · [GitHub](https://github.com/M4lvin/lean4-pdl)
  - description: Propositional Dynamic Logic in Lean4; part of the sub-curriculum/evaluation where LeanAgent attempted/proved sorry theorems.

- Lean4Lean · [GitHub](https://github.com/digama0/lean4lean)
  - description: Implementation of the Lean4 kernel in Lean4; part of the initial curriculum/evaluation set.

- LeanAPAP · [GitHub](https://github.com/YaelDillies/LeanAPAP)
  - description: Lean repo on Kelley–Meka bound on Roth numbers; included in the sub-curriculum/evaluation.

- LiteLLM · [GitHub](https://github.com/BerriAI/litellm) · [Doc](https://docs.litellm.ai)
  - description: Wrapper used to call OpenAI embeddings (text-embedding-3-large) during the embedding-based evaluation.

- MAS-Resilience (this paper) · [GitHub](https://github.com/CUHK-ARISE/MAS-Resilience)
  - description: Official code and data release for the paper, including implementations of AUTOTRANSFORM, AUTOINJECT, Challenger, and Inspector, plus adapted prompts and evaluation scripts for all six multi-agent systems and four tasks.

- Mathematics in Lean Source · [GitHub](https://github.com/avigad/mathematics_in_lean_source)
  - description: Source Lean files accompanying the Mathematics in Lean textbook; a major evaluation repo where LeanAgent showed strong progression and highest accuracy gains.

- MAWPS (SingleEQ, SingleOP, AddSub, MultiArith) · [GitHub](https://github.com/MAWPS/MAWPS)
  - description: Repository aggregating classic math word problem sets; subsets are used for out-of-domain evaluation (averaged as MAWPS).

- MDTraj · [GitHub](https://github.com/mdtraj/mdtraj) · [Doc](http://mdtraj.org/latest/)
  - description: Library used for ensemble analysis (e.g., RMSD/RMSF, PCA projections, SASA via Shrake–Rupley) in MD evaluations.

- MiniCPM-V· [GitHub](https://github.com/OpenBMB/MiniCPM-V) · [Codewiki](https://codewiki.google/github.com/OpenBMB/MiniCPM-V) · [Website](https://minicpm.org/)
  - description: One of the two VLM backbones the authors fine-tune with MM‑Traj to build T3‑Agent (MiniCPM‑V‑8.5B).
  - description: Open-source vision-language baseline evaluated on GroundUI; relevant for reproducing model comparisons.
  - description: Open-source multimodal model evaluated as a generalist baseline.
  - description: Compact multimodal baseline tested for element/layout grounding.
  - description: Open-source VLM used by MCU’s AutoEval as an alternative to GPT-4o for multi-dimensional video-based scoring.

- MiniLM · [GitHub](https://github.com/microsoft/unilm/tree/master/minilm)
  - description: Lightweight text embedding model used to embed queries/operators for the controller’s scoring in MaAS.

- NCBI BLAST+ · [GitHub](https://github.com/ncbi/blast) · [Website](https://blast.ncbi.nlm.nih.gov/Blast.cgi)
  - description: Used in the steering analysis to find homologs of top-activating sequences before multiple sequence alignment.
  - description: Sequence similarity searches used for nearest-neighbor identity, novelty analysis, and alignment-based metrics.

- NetworkX · [GitHub](https://github.com/networkx/networkx) · [Doc](https://networkx.org/documentation/stable/)
  - description: Graph library used to build the paper’s graph-based evaluators (checkpoint DAGs) and task graphs.
  - description: Graph library used in experiments that construct and analyze knowledge graphs (e.g., graph-based agents, text–graph alignment metric).

- NeuroKit2 · [GitHub](https://github.com/neuropsychology/NeuroKit)
  - description: Toolbox for neurophysiological signal processing; used to build and evaluate physiology signal-processing tasks.

- OpenMM · [GitHub](https://github.com/openmm/openmm) · [Website](https://openmm.org/) · [Doc](https://openmm.org/documentation.html)
  - description: Molecular simulation toolkit used to compute potential energies and forces, perform protonation/solvation, and run restrained minimization for energy/force guidance.
  - description: Molecular simulation toolkit used for energy minimization and potential energy evaluation of sampled conformations.

- OS-Atlas · [GitHub](https://github.com/OS-Copilot/OS-Atlas)
  - description: Foundation action model for GUI agents; evaluated by the paper for layout/element grounding comparisons.

- Overcooked-AI · [GitHub](https://github.com/HumanCompatibleAI/overcooked_ai)
  - description: Cooperative multi-agent environment used for additional experiments (Cramped Room and Forced Coordination) to evaluate ReAd’s effectiveness.
  - description: Cooperative cooking environment used for Agentic Coordination experiments; the paper uses its PPO/PBT self-play baselines and the human-behavior-cloning agents for cross-play/ZSC.

- Pallatom · [GitHub](https://github.com/levinthal/Pallatom)
  - description: Official code release for the paper’s all-atom protein generative model, including the atom14 representation, training/inference code, and evaluation used throughout the experiments.

- Parsel · [GitHub](https://github.com/ezelikman/Parsel)
  - description: Algorithmic reasoning via composed decompositions; used in the “CodeT+Parsel” HumanEval baselines.

- Perplexica Search Engine · [GitHub](https://github.com/ItzCrazyKns/Perplexica) · [Codewiki](https://codewiki.google/github.com/ItzCrazyKns/Perplexica)
  - description: Open-source web search engine used by the Manager for Online Web Knowledge retrieval during planning.

- PFR (Polynomial Freiman–Ruzsa) · [GitHub](https://github.com/teorth/pfr)
  - description: Lean repository formalizing results around the PFR Conjecture; part of the evaluation set where LeanAgent proved new sorry theorems and analyzed generalization across commits.

- Proteus · [GitHub](https://github.com/Wangchentong/Proteus)
  - description: Official code release for the paper; implements the Proteus diffusion architecture, training, sampling, and evaluation pipelines described in the work.

- ProtST · [GitHub](https://github.com/DeepGraphLearning/ProtST)
  - description: Baseline protein–text retrieval method re-run by the authors with released code and original hyperparameters for comparison.

- Pyserini (BM25) · [GitHub](https://github.com/castorini/pyserini) · [Website](https://pyserini.io/)
  - description: Retrieval toolkit used to access a pre-built Wikipedia index for IoA’s RAG experiments.
  - description: BM25/Anserini-based retrieval toolkit that DSPy supports as a built-in retriever option.
  - description: IR toolkit used to implement BM25 rough retrieval over Legal-KB; the assistant agent first retrieves top candidates with BM25 before re-ranking.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion) · [Website](https://arxiv.org/abs/2303.11366)
  - description: Multi-agent/self-reflection baseline used for comparison; involves an executor LLM with another LLM providing feedback.
  - description: Self-reflection agent baseline used for HumanEval comparisons.
  - description: LLM self-reflection baseline used for comparison with ReAd-J.
  - description: Baseline framework re-implemented (“Reflexion-like”) with verbal self-reflection, used for comparison to CRADLE.
  - description: Technique that inspires DS-Agent’s Debugger component to reflect on execution feedback and iteratively fix code.
  - description: Included in the paper’s collaboration-modes repository; used as one of the MAS reasoning modes.
  - description: Self-reflection/evaluation framework used as a baseline; also integrated for self-reflection in the Collaborative Cooking experiments.
  - description: Strong baseline method compared against CODESIM for basic programming tasks and as a seed method in the second-pass debugging analysis.
  - description: Baseline with self-reflective feedback; compared against Tool-Planner.
  - description: Self-reflection framework used as a dynamic execution/planning baseline; the paper’s memory recollection mechanism is compared against it.
  - description: Planning framework baseline integrated by the authors (denoted ARMAP-R) to compare/test reward-guided planning.
  - description: Baseline method providing self-feedback for agents; compared against in experiments.
  - description: Baseline agent using self-reflection and iterative refinement; used for comparison in experiments.
  - description: Language agents with verbal reinforcement learning referenced in Appendix as an aligned single-agent architecture; useful for adding reflective reasoning to CogMir agents.
  - description: Baseline method that adds self-reflection from past attempts; the paper reimplements it for comparisons.

- rouge· [GitHub](https://github.com/pltrdy/rouge)
  - description: Text summarization metric (ROUGE-L) used to evaluate similarity between generated and reference text plans.

- ROUGE (rouge-score) · [GitHub](https://github.com/google-research/google-research/tree/master/rouge) · [Doc](https://pypi.org/project/rouge-score/)
  - description: ROUGE-L reference-based metric employed in the paper’s automatic evaluation of system outputs.

- SaProt · [GitHub](https://github.com/westlake-repl/SaProt)
  - description: Structure-aware PLM baseline and data resource; the paper follows SaProt’s downstream splits and AFDB retrieval procedure and compares against its models.

- SciLean · [GitHub](https://github.com/lecopivo/SciLean)
  - description: Scientific computing library in Lean; a large evaluation repo where LeanAgent progressively proved theorems ranging from basic algebra to advanced function spaces.

- Self‑Refine · [GitHub](https://github.com/kaistAI/self-refine)
  - description: Iterative self‑reflection baseline evaluated by the paper on both LLMs and MLLMs, contrasted with DMAD’s multi‑agent approach.

- SGLang · [GitHub](https://github.com/sgl-project/sglang) · [Codewiki](https://codewiki.google/github.com/sgl-project/sglang) · [Website](https://sgl-project.github.io/sglang/)
  - description: Fast LLM serving framework used for time‑cost measurements when running Llama‑3.1‑70B‑Instruct in the paper’s efficiency analysis (App. B.1).
  - description: Inference engine the authors extend to build the Multiverse Engine; used for continuous batching and radix attention during serving and evaluation.

- Smina · [GitHub](https://github.com/mwojcikowski/smina)
  - description: Docking/scoring tool used to estimate binding affinity of designed proteins to ligands.

- SymPy · [GitHub](https://github.com/sympy/sympy) · [Website](https://www.sympy.org/en/index.html) · [Doc](https://docs.sympy.org/latest/index.html)
  - description: Symbolic math library repeatedly used in the paper’s tool/functions and solution code (e.g., integrate, solve).
  - description: Python CAS used in generated math functions (e.g., evaluate_expression, solve equations) during MATH experiments.
  - description: Symbolic mathematics library used by the paper for equivalence checking on MATH (answer validation).
  - description: Symbolic math library frequently used in synthesized code blocks for execution during data creation and model inference.

- TabMWP · [GitHub](https://github.com/lupantech/TabMWP)
  - description: Semi-structured (table) math word problems; used for out-of-domain evaluation.

- Text-to-Motion (T2M) · [GitHub](https://github.com/EricGuo5513/text-to-motion)
  - description: Baseline and evaluation toolkit whose pretrained encoders/metrics (R-precision, MM-Dist, FID, Diversity) are used to evaluate MotionLLM.

- ToRA· [GitHub](https://github.com/microsoft/ToRA)
  - description: Open-source math tool-integrated baseline compared against SCIAGENT; both 7B and 13B versions evaluated.
  - description: Tool-augmented math agent referenced under the “LLM as Planner” paradigm; plans natural-language and program/tool steps to solve math problems.
  - description: Tool-integrated reasoning agent with interleaved CoT+code; cited as a closely related baseline and template for interleaving used by MuMath-Code.

- Tree-of-Thoughts (Game of 24) · [GitHub](https://github.com/ysymyth/tree-of-thought-llm)
  - description: Deliberate problem-solving approach; the paper’s case studies mention ToT-style agents within GPTSwarm setups.
  - description: Repository providing Game of 24 evaluation used in the paper; authors follow Yao et al. (2023b) to evaluate with 100 hard puzzles and build the Game24 environment.

- TrueSkill · [GitHub](https://github.com/sublee/trueskill) · [Website](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) · [Doc](https://trueskill.org/)
  - description: Bayesian skill-rating system used to compute rankings/scores in the Auction and Bargaining evaluations.

- Unstructured · [GitHub](https://github.com/Unstructured-IO/unstructured) · [Codewiki](https://codewiki.google/github.com/Unstructured-IO/unstructured) · [Doc](https://unstructured-io.github.io/unstructured/)
  - description: Document parsing library named among the paper’s document-processing tool stack; useful to replicate the Document Processing Worker.
  - description: PDF processing toolkit used to parse papers and extract text and tables for downstream retrieval and extraction.

- Virtual Scientists (VIRSCI) · [GitHub](https://github.com/open-sciencelab/Virtual-Scientists)
  - description: Official code release for this paper; implements the LLM-based multi-agent system (team formation, inter/intra-team discussion, novelty assessment, abstract generation) and evaluation pipeline described in the work.

- WebArena · [GitHub](https://github.com/WebArena-Lab/WebArena) · [Website](https://webarena.dev/)
  - description: Realistic web environment cited for building and evaluating autonomous web agents; relevant as the underlying environment ecosystem for web-agent experiments like those with SeeAct.


### Benchmarks & Leaderboards
- FLASK · [GitHub](https://github.com/kaistAI/FLASK)
- GPQA: A Graduate-Level Google-Proof Q&A Benchmark · [GitHub](https://github.com/idavidrein/GPQA) · [GitHub](https://github.com/idavidrein/gpqa)
- OpenCompass · [GitHub](https://github.com/open-compass/opencompass)
- MMLU (Massive Multitask Language Understanding)· [GitHub](https://github.com/hendrycks/test)
- OpenOOD: Benchmarking Generalized OOD Detection · [GitHub](https://github.com/Jingkang50/OpenOOD)

- SCILEAD (Scientific Leaderboard dataset) / Leaderboard Generation · [GitHub](https://github.com/UKPLab/leaderboard-generation) · [Website](https://www.tudatalib.ulb.tu-darmstadt.de/)
  - description: Official code and data release for the paper; implements PDF parsing, RAG-based TDMR extraction, normalization, and leaderboard construction. The dataset is hosted on TUdatalib as stated in the paper.

- AxCell · [GitHub](https://github.com/paperswithcode/axcell)
  - description: Baseline system for automatic extraction of results from ML papers; used in this work for comparison in TDMR extraction and leaderboard construction.

- rbo (Ranked Biased Overlap) · [GitHub](https://github.com/changyaochen/rbo)
  - description: Library used to compute ranking similarity (Average Overlap) between gold and predicted leaderboards.

- DROP· [GitHub](https://github.com/allenai/drop) · [Website](https://leaderboard.allenai.org/drop)
  - description: Reading comprehension benchmark used to evaluate agents’ F1 on discrete reasoning over passages.
  - description: Reading comprehension dataset requiring discrete reasoning; used as an additional evaluation benchmark (decontextualized subset) for AOP.

- Langfun · [GitHub](https://github.com/google/langfun)
  - description: Google’s agent framework cited in baselines/leaderboard comparisons; relevant for replicating or benchmarking alternative pipelines.

## Simulator
- BrowserGym · [GitHub](https://github.com/webarena-dev/BrowserGym) · [Website](https://webarena.dev) · [Leaderboard](https://webarena.dev/leaderboard)
- BrowserGym (ServiceNow) · [GitHub](https://github.com/ServiceNow/browsergym) · [GitHub](https://github.com/ServiceNow/BrowserGym)
- MiniWoB++ · [GitHub](https://github.com/stanfordnlp/miniwob-plusplus)
- terminal-bench · [GitHub](https://github.com/laude-institute/terminal-bench)
- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop) · [Website](https://webshop-pnlp.github.io)


### Robotics & Embodied Platforms

- AI2-THOR · [GitHub](https://github.com/allenai/ai2thor) · [Website](https://ai2thor.allenai.org/) · [Doc](https://ai2thor.allenai.org/ithor/)
  - description: Interactive 3D simulator used to build EB-ALFRED and EB-Navigation; provides egocentric observations and action execution with textual feedback.
  - description: Interactive 3D household simulator underlying ALFRED; the paper’s embodied experiments and limitations reference AI2-THOR’s affordances when assessing planner/detector failures.
  - description: Interactive 3D environment that underlies ALFWorld; relevant dependency for reproducing the embodied experiments shown in the appendix.

- Genesis (Embodied-AI simulator) · [GitHub](https://github.com/Genesis-EmbodiedAI/Genesis) · [Website](https://genesis-embodied-ai.github.io)
  - description: Physics-based simulator used to build and run all 3D scenes for AnomalyGen; the paper deploys their environments and robot learning in Genesis.

- Habitat 3.0 (AI Habitat)· [GitHub](https://github.com/facebookresearch/habitat-lab) · [Website](https://aihabitat.org/) · [Doc](https://aihabitat.org/docs/intro.html)
  - description: Simulator and framework used for EB-Habitat; the paper evaluates high-level skills and rearrangement tasks within Habitat.
  - description: Simulator and task framework the benchmark is built on; the paper runs all HMRS tasks in Habitat and uses its PDDL integration and navigation stack.
  - description: Simulation platform for human–avatar–robot co-habitation; referenced for embodied Social-AI and human-robot interaction studies.

- IQA (Interactive Question Answering) · [GitHub](https://github.com/allenai/ai2thor-iqa)
  - description: Interactive visual QA environment; text versions leveraged to build embodied interaction trajectories.

- ManiSkill2 · [GitHub](https://github.com/haosulab/ManiSkill2) · [Website](https://maniskill2.github.io)
  - description: Generalizable manipulation benchmark compared against AnomalyGen for diversity.

- Matminer · [GitHub](https://github.com/hackingmaterials/matminer)
  - description: Materials data mining toolkit; adapted for materials feature engineering and benchmarking tasks.

- MCU (Minecraft Universe) · [GitHub](https://github.com/CraftJarvis/MCU)
  - description: Official code release for the paper; includes atomic task lists, LLM-based task configuration, AutoEval (VLM judge), scripts, and MCU-Turbo subset for standardized benchmarking.

- Meta-World · [GitHub](https://github.com/rlworkgroup/metaworld) · [Website](https://meta-world.github.io/)
  - description: Multi-task RL benchmark used as a baseline in diversity comparisons.

- Microsoft Malmo Platform · [GitHub](https://github.com/Microsoft/malmo) · [Website](https://www.microsoft.com/en-us/research/project/project-malmo/)
  - description: Minecraft experimentation platform underpinning many Minecraft research stacks cited by the paper; relevant for reproducing environment backends.

- MineCLIP · [GitHub](https://github.com/MineDojo/MineCLIP)
  - description: Video–language model released with MineDojo; used as a baseline selector in the paper’s ablations for goal selection.

- MineCLIP (from MineDojo)· [GitHub](https://github.com/MineDojo/MineDojo) · [Website](https://minedojo.org)
  - description: Open-ended Minecraft research platform used as one of the evaluation environments (v1.11.2) and task providers; also supplies programmatic tasks referenced in the paper.
  - description: Prior benchmark from which MCU filters and deduplicates tasks; also provides MineCLIP, which MCU uses as an automatic-evaluation baseline.
  - description: CLIP model trained on Minecraft videos; used in MCU as a comparison baseline for automatic trajectory evaluation.

- Mineflayer · [GitHub](https://github.com/PrismarineJS/mineflayer)
  - description: Node.js bot framework for Minecraft; used as the engine enabling text-based interaction and tool execution in the paper’s Minecraft environment.

- MineRL · [GitHub](https://github.com/minerllabs/minerl) · [Doc](https://minerl.readthedocs.io)
  - description: Minecraft research environment used to implement VAB-Minecraft; the paper defines high-level actions and integrates a low-level controller within this framework.
  - description: Minecraft RL platform and dataset used as another evaluation environment (v1.16.5) for testing DEPS with different controllers.

- MineStudio · [GitHub](https://github.com/CraftJarvis/MineStudio) · [Doc](https://arxiv.org/abs/2412.18293)
  - description: The benchmark’s core runtime; MCU’s task initialization, callbacks (commands/summon/reset/record/rewards), unified agent interface, and simulator verification are implemented on top of MineStudio.

- OmniAct · [GitHub](https://github.com/omni-act/omni-act)
  - description: Desktop+web dataset/benchmark cited in comparisons; focuses on action prediction but smaller-scale than UI-Vision.

- OmniGibson · [GitHub](https://github.com/StanfordVL/OmniGibson) · [Website](https://behavior.stanford.edu/)
  - description: High-fidelity household simulator used to build VAB-OmniGibson; the paper adapts OmniGibson scenes/objects and defines high-level actions and judges for embodied tasks.

- OmniParser · [GitHub](https://github.com/microsoft/OmniParser)
  - description: Model for multi-element detection and icon captioning on GUIs; used as a key visual parser to generate high-quality Set-of-Marks.
  - description: Vision parsing tool used in a baseline (“GPT-4 + OmniParser”) for ScreenSpot comparisons reported in the paper.
  - description: Commercial GUI locator baseline (GPT‑4V + Grounding DINO) compared against in grounding and agent tasks.

- Open X-Embodiment (OXE) · [GitHub](https://github.com/google-deepmind/open_x_embodiment) · [Website](https://robotics-transformer-x.github.io/)
  - description: Large-scale robot learning dataset used to pre-train the DiT-based action policy and for SFT/RL data in ThinkAct.

- Voyager · [GitHub](https://github.com/MineDojo/Voyager)
  - description: Embodied LLM agent baseline; its memory ideas and module were used in module combinations during search.
  - description: Baseline agent paradigm adapted (“Voyager-like”) in the paper; CRADLE also borrows its skill-retrieval idea with embeddings.
  - description: Single-agent embodied agent with evolving memory; adapted by the authors as a memory baseline in MAS settings.

- AndroidEnv · [GitHub](https://github.com/google-deepmind/android_env)
  - description: Android RL/emulation environment cited as a related platform; relevant for extending AgentStudio-style agents to mobile settings.

- AndroidEnv · [GitHub](https://github.com/deepmind/android_env)
  - description: Python library used by AndroidWorld to connect agents to the Android OS/emulator and stream observations/actions; cited as the mechanism for device interaction.

- Hydra (3D Scene Graph) · [GitHub](https://github.com/MIT-SPARK/Hydra) · [Website](https://mit-spark.github.io/Hydra/)
  - description: Referenced as the multi-layer scene representation pipeline (L1–L3) informing the textual scene context; cited for deployable multi-robot perception.

- YOLO (You Only Look Once) · [GitHub](https://github.com/pjreddie/darknet) · [Website](https://pjreddie.com/darknet/yolo/)
  - description: Object detection used to provide detection boxes and indices in EB-Manipulation, aiding localization and instruction grounding.

### Web & GUI Environments

- Aguvis · [GitHub](https://github.com/salesforce/AGuVis)
  - description: Open-source pure-vision GUI agent baseline included in element grounding comparisons.

- Aider · [GitHub](https://github.com/Aider-AI/aider) · [Codewiki](https://codewiki.google/github.com/Aider-AI/aider) · [Website](https://aider.chat/)
  - description: Open-source coding assistant; its edit_file-style utilities are incorporated into OpenHands AgentSkills and it serves as a SWE-Bench baseline.

- AutoWebGLM · [GitHub](https://github.com/THUDM/AutoWebGLM)
  - description: Web agent baseline compared against in WebArena results; included for practitioners inspecting alternative open-source agents.
  - description: Baseline LLM web agent referenced for comparison on WebArena.

- Bokeh · [GitHub](https://github.com/bokeh/bokeh) · [Codewiki](https://codewiki.google/github.com/bokeh/bokeh) · [Website](https://bokeh.org) · [Doc](https://docs.bokeh.org/en/latest/)
  - description: Visualization backend specified in some MatPlotBench tasks (e.g., HoloViews+Bokeh chord diagrams) and thus relevant for reproducing those plots.

- ChatArena · [GitHub](https://github.com/chatarena/chatarena)
  - description: Multi-agent language game environment cited as related work; provides open-source infrastructure that can inform alternative simulation designs.

- Deal or No Deal (Negotiation) · [GitHub](https://github.com/facebookresearch/end-to-end-negotiator)
  - description: Official repository for the negotiation task used to implement the Bargaining environment (DealOrNotDeal) in the paper.

- Diffusion Policy (DiT-based policy) · [GitHub](https://github.com/real-stanford/diffusion_policy) · [Website](https://diffusion-policy.cs.columbia.edu)
  - description: Transformer-based action policy architecture used as ThinkAct’s action model and as a baseline (DiT-Policy) in experiments.

- Eureka · [GitHub](https://github.com/eureka-research/eureka) · [Website](https://eureka-research.github.io/)
  - description: Baseline method for LLM-driven reward design/program synthesis; compared against SGA in experiments.

- fastText · [GitHub](https://github.com/facebookresearch/fastText) · [Codewiki](https://codewiki.google/github.com/facebookresearch/fastText) · [Website](https://fasttext.cc)
  - description: Lightweight text classifier used to scale tutorial classification after LLM labeling in AgentTrek’s tutorial filtering stage.

- GPUDrive · [GitHub](https://github.com/Emerge-Lab/gpudrive)
  - description: Official code release of the paper, including the simulator, Dockerfiles, training loops (IPPO), and pre-trained agents used in the experiments.

- HoloViews · [GitHub](https://github.com/holoviz/holoviews) · [Website](https://holoviews.org)
  - description: Visualization library referenced in MatPlotBench queries (e.g., chord diagram with Bokeh backend), which the agent may use to satisfy task requirements.

- html2text · [GitHub](https://github.com/Alir3z4/html2text)
  - description: HTML-to-text conversion library cited among document parsing utilities used by workers for preprocessing web/document content.

- InterProScan · [GitHub](https://github.com/ebi-pf-team/interproscan) · [Website](https://www.ebi.ac.uk/interpro/)
  - description: Functional annotation pipeline used to assess biological relevance and family membership (e.g., SUPERFAMILY, MobiDB).

- Nocturne · [GitHub](https://github.com/facebookresearch/nocturne)
  - description: CPU-based driving simulator used as a primary baseline for speed and training-time comparisons.

- NVIDIA Warp · [GitHub](https://github.com/NVIDIA/warp) · [Doc](https://nvidia.github.io/warp/)
  - description: GPU simulation framework used as the differentiable MPM simulator in the inner-level optimization to compute gradients and feedback.

- Objaverse · [GitHub](https://github.com/allenai/objaverse) · [Website](https://objaverse.allenai.org)
  - description: Large-scale 3D asset repository; used to retrieve auxiliary surrounding objects for scene construction via text and VLM filtering.

- OGGM · [GitHub](https://github.com/OGGM/oggm)
  - description: Open Global Glacier Model; referenced/adapted to construct real-world GIS tasks.

- OpenInterpreter · [GitHub](https://github.com/KillianLucas/open-interpreter) · [Website](https://openinterpreter.com) · [Doc](https://docs.openinterpreter.com)
  - description: Code-first agent used as a baseline on ML and open-ended tasks.

- pycma (CMA-ES) · [GitHub](https://github.com/CMA-ES/pycma) · [Doc](https://cma-es.github.io/)
  - description: Evolution strategy used to train simple controllers in the Recurrent World Models baseline.
  - description: Covariance Matrix Adaptation Evolution Strategy library used to run the CMAES baseline (and CMAES with VED encoding) for comparison.

- Pydantic · [GitHub](https://github.com/pydantic/pydantic) · [Codewiki](https://codewiki.google/github.com/pydantic/pydantic) · [Website](https://pydantic.dev)
  - description: Data modeling/validation library used to define task objects and configurations in CRAB.

- RESEARCHTOWN · [GitHub](https://github.com/ulab-uiuc/research-town)
  - description: Official implementation of the paper’s multi-agent TextGNN simulator for research communities; primary codebase to reproduce all methods and experiments.

- RouteLLM · [GitHub](https://github.com/lm-sys/RouteLLM) · [Website](https://lmsys.org) · [Doc](https://arxiv.org/abs/2406.18665)
  - description: LLM routing baseline compared against MasRouter.

- Search-O1 · [GitHub](https://github.com/THUIR/Search-O1) · [Website](https://arxiv.org/abs/2501.05366)
  - description: Agentic search-enhanced reasoning baseline compared in experiments and web-search ablations.

- SWE-Search (moatless-tree-search) · [GitHub](https://github.com/aorwall/moatless-tree-search) · [Website](https://streamlit.moatless.ai)
  - description: Official code release and interactive demo for the paper’s MCTS + iterative refinement framework; contains the Action/Value/Discriminator agents, search implementation, and visualization to reproduce results.

- UI-TARS · [GitHub](https://github.com/bytedance/UI-TARS)
  - description: Open-source GUI agent used as a primary baseline in all three UI-Vision tasks (element/layout grounding and action prediction).

- Waymax · [GitHub](https://github.com/waymo-research/waymax)
  - description: JAX-based, GPU-accelerated driving simulator; GPUDrive borrows its simplified bicycle model and compares throughput against it.

- WebArena · [GitHub](https://github.com/web-arena-dev/WebArena)
  - description: Realistic web environments; broken into single-step interactions and annotated with rationales.

## Reinforcement Learning
- RecoGym · [GitHub](https://github.com/criteo-research/reco-gym)

- android_world· [GitHub](https://github.com/google-research/android_world)
  - description: Dynamic Android benchmarking environment cited as related work offering reward signals; useful as a complementary environment for extending experiments beyond AndroidLab.
  - description: Official code release for the paper; includes the AndroidWorld benchmark, MobileMiniWoB++ integration, task definitions, environment wrappers, and baseline/agent implementations used in the experiments.
  - description: Dynamic benchmarking environment for autonomous Android agents. Cited as a target for future online evaluation; relevant for extending this work to interactive settings.

- AndroidWorld · [GitHub](https://github.com/google-deepmind/android_world)
  - description: Dynamic benchmarking environment for autonomous Android agents; SPA-Bench integrates M3A/T3A/SeeAct-style agents and adopts AndroidWorld’s action-grounding module.

- AndroidWorld · [GitHub](https://github.com/google-deepmind/android-world)
  - description: Dynamic Android benchmarking environment referenced in the paper’s related work; useful comparison/extension target for mobile agents.

- AndroidWorld · [GitHub](https://github.com/microsoft/AndroidWorld) · [Website](https://arxiv.org/abs/2405.14573)
  - description: Interactive Android environment benchmark; the paper evaluates a vision-only agent and reports success rates vs. M3A.

- ATOM3D · [GitHub](https://github.com/drorlab/atom3d) · [Website](https://atom3d.ai)
  - description: Tasks, standardized splits, and loaders for 3D molecular learning; ProFSA follows ATOM3D’s preprocessing and strict 30%/60% sequence-identity splits for the PDBbind LBA task.

- Carleson · [GitHub](https://github.com/fpvandoorn/carleson)
  - description: Lean repository related to Carleson’s Theorem; included in the sub-curriculum and evaluation.

- DiscoveryWorld · [GitHub](https://github.com/allenai/discoveryworld)
  - description: Virtual environment for automated scientific discovery; used in the “Graph Agent for Discovery” experiments via DiscoveryWorldAPI to evaluate knowledge-graph-augmented agents.

- Gymnasium · [GitHub](https://github.com/Farama-Foundation/Gymnasium) · [Doc](https://gymnasium.farama.org/)
  - description: API standard that GPUDrive adheres to; the paper provides Gymnasium-compatible environments for both torch and jax.

- LatProtRL · [GitHub](https://github.com/haewonc/LatProtRL)
  - description: Official code release for the paper; implements the Variant Encoder-Decoder (VED), latent-space PPO policy, frontier buffer, constrained decoding, and experiment scripts for GFP and AAV optimization.

- MAPoRL · [GitHub](https://github.com/chanwoo-park-official/MAPoRL)
  - description: Official implementation released by the paper; contains the multi-agent debate pipeline, verifier-based reward shaping, and multi-agent PPO training code to reproduce MAPoRL experiments.

- Melting Pot 2.0 · [GitHub](https://github.com/google-deepmind/meltingpot) · [Doc](https://meltingpot.readthedocs.io/en/latest/)
  - description: Multi-agent evaluation benchmark used for all environments in the paper (Running With Scissors, Running With Scissors Arena, Prisoner’s Dilemma, Collaborative Cooking Asymmetric).

- OSWorld / OSWorld-Verified · [GitHub](https://github.com/xlang-ai/OSWorld) · [Website](https://os-world.github.io) · [Doc](https://xlang.ai/blog/osworld-verified)
  - description: Executable Ubuntu desktop benchmark used for the main experiments; also provides the MMAgent baseline implementation that the paper compares against.
  - description: Comprehensive benchmark and VM environment used by the paper to evaluate CRADLE on 369 real-world computer tasks with automatic evaluation scripts.
  - description: Desktop benchmark focusing on online interactions; referenced as related but differing from UI-Vision’s offline, densely annotated setup.
  - description: Real computer environment and execution-based benchmark; used for online evaluation where the agent formats pyautogui code to act.
  - description: Real-computer evaluation benchmark and platform (Windows/Linux/macOS) for multimodal OS agents; used in the survey’s evaluation section as a key computer-use benchmark.
  - description: Real-computer evaluation environment and benchmark used for the main experiments; the paper trains/evaluates AgentStore on OSWorld and also builds OSWorld-Multi on top of it.
  - description: Real-computer benchmark and verified task suite used for the paper’s main online evaluation; authors submit to the public OSWorld-Verified evaluation service.

- PersuasionForGood (P4G) · [GitHub](https://github.com/facebookresearch/ParlAI) · [Doc](https://parl.ai/docs/tasks.html#persuasionforgood)
  - description: Charity persuasion benchmark used for evaluation; the paper uses the test set, and the dataset is available via the ParlAI task implementation.

- PufferLib · [GitHub](https://github.com/PufferAI/pufferlib) · [Doc](https://pufferai.github.io/pufferlib/)
  - description: Library used for the high-throughput IPPO implementation in the end-to-end training benchmarks.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld) · [Website](https://allenai.github.io/ScienceWorld/)
  - description: Interactive scientific reasoning environment used as a benchmark to evaluate EVOAGENT on open‑world multi‑step tasks (§4.2).
  - description: Interactive science tasks benchmark used for evaluating agent reasoning and planning.
  - description: Text-based science lab environments; used as a held-out embodied evaluation with average reward.
  - description: Text-based science environment referenced for several experiment ideas and baselines (e.g., affordance/agent evaluations), indicating relevance for extending the paper’s evaluations.
  - description: Text-based embodied science environment; employed as a benchmark to evaluate procedural reasoning with G-Memory.
  - description: Text-based science experiment environment; used for seen/unseen split evaluations and for generating reward-model training data.
  - description: Interactive science reasoning environment included among held-in tasks for training/evaluation.
  - description: Embodied, long-horizon textual environment used in additional single-agent evaluations to measure performance on complex, decomposable tasks.

- Stable Baselines · [GitHub](https://github.com/hill-a/stable-baselines) · [Doc](https://stable-baselines.readthedocs.io/)
  - description: RL library providing the PPO implementation used to train the policy in LatProtRL.

- Stable-Baselines3 · [GitHub](https://github.com/DLR-RM/stable-baselines3) · [Doc](https://stable-baselines3.readthedocs.io/)
  - description: Alternative (slower) IPPO implementation mentioned as available in the GPUDrive repo.

- STAMP · [GitHub](https://github.com/KatherLab/STAMP)
  - description: End-to-end weakly supervised computational pathology pipeline; TOOLMAKER builds tools for feature extraction and model training (stamp_extract_features, stamp_train_classification_model).

- TabPFN · [GitHub](https://github.com/automl/TabPFN)
  - description: State-of-the-art tabular classification model used as a human-designed baseline for tabular tasks.

- TabPFN · [GitHub](https://github.com/PriorLabs/TabPFN)
  - description: Tabular foundation model and training code; used to train and evaluate predictors in the tabpfn_predict task.
  - description: State-of-the-art tabular classification model referenced as a strong human-level baseline for tabular tasks.

- TextWorld · [GitHub](https://github.com/microsoft/TextWorld) · [Doc](https://microsoft.github.io/TextWorld/)
  - description: Text-based environment underlying ALFWorld; used indirectly when the paper evaluates DEPS on ALFWorld tasks.
  - description: Text-based game engine underlying ALFWorld; cited as the environment foundation relevant for reproducing ALFWorld-based experiments.
  - description: Text-based game framework that ALFWorld is built upon; relevant for reproducing ALFWorld experiments described.
  - description: Text-based game simulator that ALFWorld builds upon; relevant for reproducing the environment setup.

- TextWorldExpress · [GitHub](https://github.com/allenai/textworld-express) · [Doc](https://pypi.org/project/textworld-express/)
  - description: High-performance text-game simulator used for many experiments (e.g., CookingWorld state prediction, action prediction); the paper’s code imports the TextWorldExpress API to generate environments and episodes.

- ThreeDWorld (TDW) · [GitHub](https://github.com/threedworld-mit/tdw) · [Website](https://threedworld.org) · [Doc](https://tdw.readthedocs.io)
  - description: 3D simulator used to instantiate the paper’s embodied multi-agent benchmarks (TDW-Game, TDW-Cook) and collect large-scale rollouts.

- WebRL · [GitHub](https://github.com/THUDM/WebRL)
  - description: Framework used to generate/score trajectories in the synthetic data pipeline; the authors use WebRL-Llama-3.1-70B as the actor and ORM-Llama-3.1-8B as the outcome-supervised reward model for filtering successful trajectories.

- Muon · [GitHub](https://github.com/scverse/muon)
  - description: Multimodal omics analysis framework; used to support multi-omics tasks.


## Agent - Code
### dataset
- Codeforces COTS Dataset · [Dataset](https://huggingface.co/datasets/open-r1/codeforces-cots)
- The Stack v2 Dataset · [Dataset](https://huggingface.co/datasets/bigcode/the-stack-v2)

### Codex & IDE

- Character-LLM · [GitHub](https://github.com/thu-coai/Character-LLM)
  - description: Trainable role-playing agent framework referenced as related work; helpful for persona/role setups akin to CogIdentity in CogMir.

- DIAMOND · [GitHub](https://github.com/bbuchfink/diamond)
  - description: High-throughput protein aligner used to hierarchically cluster proteins at 90/50/30% identity in building PPA-1.

- eofs · [GitHub](https://github.com/ajdawson/eofs)
  - description: EOF analysis library for climate data; used in geoscience tasks for dimensionality reduction/analysis.

- geoplot · [GitHub](https://github.com/ResidentMario/geoplot)
  - description: Geospatial plotting library; used for GIS visualization tasks.

- LDST (LLM-driven Dialogue State Tracking) · [GitHub](https://github.com/WoodScene/LDST)
  - description: TOD baseline referenced in the experiments; provides checkpoints and code for LLM-based state tracking.

- leidenalg (Leiden community detection) · [GitHub](https://github.com/vtraag/leidenalg) · [Doc](https://leidenalg.readthedocs.io/en/stable/)
  - description: Community clustering algorithm applied to the knowledge graph in the Mind-Map agent.

- Madrona Game Engine · [GitHub](https://github.com/madrona-engine/madrona)
  - description: The high-performance ECS simulation engine GPUDrive is built on; provides GPU acceleration, collision checking, and sensor utilities.

- openreview-py · [GitHub](https://github.com/openreview/openreview-py) · [Doc](https://openreview-py.readthedocs.io/en/latest/)
  - description: API and platform used to retrieve ICLR 2020–2023 submissions; provides the paper data the authors use to run simulations.
  - description: Library used to retrieve public ICLR 2024 reviews for REVIEWBENCH construction.

- pymatgen · [GitHub](https://github.com/materialsproject/pymatgen)
  - description: Materials analysis library; code and utilities were adapted for materials-focused computational tasks.

- python-pptx · [GitHub](https://github.com/scanny/python-pptx) · [Doc](https://python-pptx.readthedocs.io/)
  - description: Library used by SlideAgent to create and modify slides via code in the LibreOffice Impress domain.

### gemini cli
- Gemini CLI · [GitHub](https://github.com/google-gemini/gemini-cli)

### Claude Code
- Claude Code · [GitHub](https://github.com/anthropics/claude-code) · [Website](https://www.anthropic.com/claude-code) · [Docs](https://docs.anthropic.com/en/docs/claude-code/overview)

### Others
- SII CLI: Building Next-Generation Cognitive Agentic Intelligence Ecosystem · [GitHub](https://github.com/GAIR-NLP/SII-CLI) · [Website](https://www.opensii.ai) · [Docs](https://www.opensii.ai/cli/docs)
- OpenHands· [GitHub](https://github.com/All-Hands-AI/OpenHands)
- SWE-bench · [GitHub](https://github.com/princeton-nlp/SWE-bench) · [Website](https://www.swebench.com)
- ChatDev· [GitHub](https://github.com/OpenBMB/ChatDev)
- CodeAgent · [GitHub](https://github.com/OpenBMB/CodeAgent)
- AgentCoder · [GitHub](https://github.com/DeepSoftwareAnalytics/AgentCoder)
- MapCoder · [GitHub](https://github.com/tsinghua-fib-lab/MapCoder)
- AutoCodeRover · [GitHub](https://github.com/nus-apr/AutoCodeRover)
- repoforge · [GitHub](https://github.com/sourcegraph/repoforge)
- DS-1000: Data Science Code Generation · [GitHub](https://github.com/HKUNLP/DS-1000)
- BioCoder: Benchmark for Bioinformatics Code Generation · [GitHub](https://github.com/gersteinlab/BioCoder)
- BigCodeBench · [GitHub](https://github.com/bigcode-project/bigcodebench)
- livecodebench · [GitHub](https://github.com/itsnamgyu/livecodebench)
- coffee-gym · [GitHub](https://github.com/kaistAI/coffee-gym)
- SWE-Gym Environment · [GitHub](https://github.com/swe-gym/swe-gym)
- r2e-gym · [GitHub](https://github.com/skyzh/r2e-gym)
- DeepSWE · [GitHub](https://github.com/DeepSWE-AI/DeepSWE)
- swe-rebench · [GitHub](https://github.com/JetBrains-Research/swe-rebench)
- mle-dojo · [GitHub](https://github.com/huggingface/mle-dojo) · [Website](https://mle-bench.github.io)
- agentpack · [GitHub](https://github.com/agentpack-data/agentpack)
- Code-R1: Reproducing R1 for Code with Reliable Rewards · [GitHub](https://github.com/ganler/code-r1)
- kodcode · [GitHub](https://github.com/zhangchenxu/kodcode)
- rstar-coder · [GitHub](https://github.com/microsoft/rstar-coder)
- SWE-Agent· [GitHub](https://github.com/princeton-nlp/SWE-agent)
- self-improving-coder · [GitHub](https://github.com/maximerobeyns/self-improving-coder)
- Awesome Vibe Coding · [GitHub](https://github.com/YuyaoGe/Awesome-Vibe-Coding)

- appium· [GitHub](https://github.com/appium/appium) · [Codewiki](https://codewiki.google/github.com/appium/appium)
  - description: Open-source cross-platform test automation framework used by the authors to control emulators, collect screenshots, and record XML view data during AMEX data collection.

- CAMDA-DILI · [GitHub](https://github.com/anikaliu/CAMDA-DILI)
  - description: Code for DILI prediction; used in chemistry tasks on toxicity modeling.

- cogsci-jnmf · [GitHub](https://github.com/brand-d/cogsci-jnmf)
  - description: Joint NMF code for reasoning analyses; used to build tasks analyzing human reasoning model similarities.

- CONCH · [GitHub](https://github.com/mahmoodlab/CONCH)
  - description: Pathology foundation model repository used as a target codebase for the conch_extract_features task in TM-BENCH.

- CVXPY · [GitHub](https://github.com/cvxpy/cvxpy) · [Doc](https://www.cvxpy.org/)
  - description: Convex optimization library listed among packages used in the generated Python code for certain problems.

- Debate · [GitHub](https://github.com/google-deepmind/debate)
  - description: Google DeepMind repository for debate protocols; included in the initial curriculum list of Lean repositories processed via LeanDojo.

- Decision Protocols (Voting or Consensus?) · [GitHub](https://github.com/lkaesberg/decision-protocols)
  - description: Official code and data release for this paper; contains implementations of the seven decision protocols, AAD/CI methods, and experiment scripts to reproduce results.

- DRAFT · [GitHub](https://github.com/quchangle1/DRAFT)
  - description: Official code release for the paper’s self-driven documentation refinement framework; used to reproduce the experiments and pipelines for Explorer/Analyzer/Rewriter and the termination/diversity mechanisms.

- FlowMap · [GitHub](https://github.com/dcharatan/flowmap)
  - description: Gradient-descent-based camera pose/intrinsic/depth estimation; repository used for the flowmap_overfit_scene task to overfit camera extrinsics.

- G-Designer · [GitHub](https://github.com/yanweiyue/GDesigner)
  - description: Official code release for the paper; contains the implementation of the VGAE-based designer, training/inference scripts, and configs to reproduce the reported multi-agent topology design results.

- Hanabi SAD/OBL (facebookresearch) · [GitHub](https://github.com/facebookresearch/hanabi_SAD)
  - description: Implementations of Simplified Action Decoder (SAD) and Off-Belief Learning (OBL); the paper uses SAD as a self-play baseline and pairs LLM agents with OBL-1/OBL-4 for cross-play.

- Hydra · [GitHub](https://github.com/facebookresearch/hydra)
  - description: Configuration framework used for experiment management and hyperparameter sweeps.

- IPR (Iterative step-level Process Refinement) · [GitHub](https://github.com/WeiminXiong/IPR)
  - description: Official code and data release for the paper; contains the implementation of the IPR framework, training scripts, and resources to reproduce the reported results.

- KPConv (Kernel Point Convolution) · [GitHub](https://github.com/HuguesTHOMAS/KPConv)
  - description: Point cloud convolution implementation used to downsample and process protein surface points within the ProteinINR point encoder pipeline.

- LDB (Debug Like a Human) · [GitHub](https://github.com/TIGER-AI-Lab/LDB)
  - description: External LLM-based debugger used by the authors as a second pass to further improve CODESIM outputs; results reported with and without LDB.

- Megatron-LM · [GitHub](https://github.com/NVIDIA/Megatron-LM)
  - description: Large-scale training framework used for Llama/CodeLlama 70B runs for speed.

- MiniLM (UniLM) · [GitHub](https://github.com/microsoft/unilm) · [Codewiki](https://codewiki.google/github.com/microsoft/unilm) · [Doc](https://arxiv.org/abs/2002.10957)
  - description: Lightweight encoder listed as an alternative module encoder in the collaboration determiner.

- MolPAL · [GitHub](https://github.com/coleygroup/molpal)
  - description: Molecular pool-based active learning framework; code was adapted as a source for computational chemistry tasks.

- pygetwindow · [GitHub](https://github.com/asweigart/pygetwindow) · [Doc](https://pygetwindow.readthedocs.io)
  - description: Used to retrieve foreground/background window titles as part of the observation space.

- RETFound_MAE · [GitHub](https://github.com/rmaphoh/RETFound_MAE)
  - description: Retinal imaging foundation model; target repository for extracting feature vectors in the retfound_feature_vector task.

- STORM · [GitHub](https://github.com/stanford-oval/storm) · [Codewiki](https://codewiki.google/github.com/stanford-oval/storm)
  - description: Agent-based writing system baseline (for FreshWiki/long-form research) compared against Agentic Reasoning.

- syllogistic-nvc · [GitHub](https://github.com/nriesterer/syllogistic-nvc)
  - description: Cognitive modeling code for syllogistic reasoning; adapted for Psychology tasks comparing cognitive models.

- TaskWeaver · [GitHub](https://github.com/microsoft/TaskWeaver) · [Doc](https://microsoft.github.io/TaskWeaver/)
  - description: Code-first agent framework compared as a baseline in ML tasks.

- Tesseract OCR · [GitHub](https://github.com/tesseract-ocr/tesseract) · [Codewiki](https://codewiki.google/github.com/tesseract-ocr/tesseract) · [Doc](https://tesseract-ocr.github.io/tessdoc/)
  - description: Open-source OCR engine used for screen text extraction to create Set-of-Marks.

- TimeCraft (BRIDGE) · [GitHub](https://github.com/microsoft/TimeCraft)
  - description: Official code release for this paper’s BRIDGE framework, including the multi-agent text-template pipeline and the hybrid prototype+text conditioned diffusion model for text-controlled time-series generation.

- UNI · [GitHub](https://github.com/mahmoodlab/UNI)
  - description: Pathology foundation model repository used as the target codebase for the uni_extract_features task.

- Vector Field Network (VFN) · [GitHub](https://github.com/aim-uofa/VFN)
  - description: Official code release for this paper; includes VFN layers and the VFN-Diff, VFN-IF, and VFN-IFE implementations used to reproduce the diffusion and inverse folding experiments.

- XGBoost · [GitHub](https://github.com/dmlc/xgboost) · [Codewiki](https://codewiki.google/github.com/dmlc/xgboost) · [Doc](https://xgboost.readthedocs.io/)
  - description: Gradient boosting library used as a model option in the ML pipelines (e.g., “train predictive models using Random Forest and XGBoost”).
  - description: Gradient-boosting baseline used in the ProSmith-style fusion framework to assess compatibility of protein and molecule representations.

- Xwin-Math · [GitHub](https://github.com/Xwin-LM/Xwin-LM)
  - description: Math-optimized checkpoint used as an alternative Stage-1 initialization in ablations; repository documents Xwin-Math-7B-V1.0.


## Agent - Desktop Automation
- OWL: Optimized Workforce Learning for General Multi-Agent Assistance · [GitHub](https://github.com/camel-ai/owl)
- A2UI · [GitHub](https://github.com/google/A2UI)
- python-mss · [GitHub](https://github.com/BoboTiG/python-mss)
  - description: Screen capture library used to obtain Ubuntu screenshots as observations for desktop-control agents.
- pywinauto · [GitHub](https://github.com/pywinauto/pywinauto) · [Doc](https://pywinauto.readthedocs.io)
  - description: Windows UI Automation wrapper used to build Set-of-Marks representations and extract UI element metadata for desktop agents.

## Deep Research

- Alibaba DeepResearch · [GitHub](https://github.com/Alibaba-NLP/DeepResearch)
- PaperQA2 · [GitHub](https://github.com/Future-House/paper-qa)
- PaperBench · [GitHub](https://github.com/AI-Research-Collab/PaperBench)
- Jr. AI Scientist: Autonomous Scientific Exploration from a Baseline Paper · [GitHub](https://github.com/Agent4Science-UTokyo/Jr.AI-Scientist)
- Kosmos Artifacts for Technical Report · [GitHub](https://github.com/EdisonScientific/kosmos-figures)
- DeepScaleR: Surpassing O1-Preview with a 1.5B Model · [Docs](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)



## Agent - General Science

- arxiv Python package · [GitHub](https://github.com/lukasschwab/arxiv.py) · [Doc](https://pypi.org/project/arxiv/)
  - description: Used to fetch paper metadata (titles, abstracts, IDs) when constructing RESEARCHBENCH.

- Google AI-Co-Scientist
  - 2025-02
  - [Website](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)
  - [arxiv](https://arxiv.org/pdf/2502.18864)

- k-dense ai
  - [website](https://k-dense.ai/)
  - Claude Scientific Skills [GitHub](https://github.com/K-Dense-AI/claude-scientific-skills)

- BLINK (Scalable Zero-shot Entity Linking) · [GitHub](https://github.com/facebookresearch/BLINK)
  - description: Off-the-shelf entity linker used to extract and canonicalize entities from titles/abstracts to construct the entity-centric knowledge store.

- Chronos (KernelSynth) · [GitHub](https://github.com/amazon-science/chronos-forecasting)
  - description: Time-series foundation model and synthetic data baselines; the paper compares synthetic-data-trained forecasting models against KernelSynth data from Chronos.

- ConvNeXt· [GitHub](https://github.com/facebookresearch/ConvNeXt)
  - description: Convolutional visual backbone used to extract image features in the SPHINX Agent setup.

- PyKrige · [GitHub](https://github.com/GeoStat-Framework/PyKrige)
  - description: Geostatistical kriging routines leveraged for GIS interpolation and heat analysis tasks in the geoscience agent stack.

- rasterio · [GitHub](https://github.com/rasterio/rasterio)
  - description: Geospatial raster I/O toolkit used in GIS tasks for reading and processing raster layers.

- Semantic Scholar Python package · [GitHub](https://github.com/danielnsilva/semanticscholar) · [Doc](https://pypi.org/project/semanticscholar/)
  - description: Used to resolve author IDs and collect author publication profiles when building the community graph for science agents.

- DeepPurpose · [GitHub](https://github.com/kexinhuang12345/DeepPurpose)
  - description: Drug–target interaction library; used in DTI/repurposing tasks (e.g., DAVIS) within computational chemistry.

- DeepSDF · [GitHub](https://github.com/facebookresearch/DeepSDF)
  - description: Reference framework for signed distance function sampling and training procedure; the paper follows DeepSDF’s SDF sampling and clamping strategy for ProteinINR.

- FlashAttention · [GitHub](https://github.com/HazyResearch/flash-attention)
  - description: Optimization referenced in the efficiency study; used to compare latency/memory of full attention vs. LONGAGENT’s O(N) chunking pipeline.

- Genie · [GitHub](https://github.com/aqlaboratory/genie)
  - description: Baseline model (equivariant diffusion of oriented residue clouds) used for comparisons in designability, diversity, and efficiency.

- MAmmoTH · [GitHub](https://github.com/TIGER-AI-Lab/MAmmoTH)
  - description: Hybrid instruction-tuned math generalist models; used as a baseline for comparison.

- MetaEuk · [GitHub](https://github.com/soedinglab/metaeuk)
  - description: Eukaryotic gene discovery/annotation resource cited as one of the metagenomic/eukaryotic-focused sources used when assembling PPA-1.

- MoMask · [GitHub](https://github.com/OpenMotionLab/MoMask)
  - description: Token-based motion generation baseline compared in quantitative and qualitative results; discussed regarding its need for manual/estimated length at inference.

- MotionChain · [GitHub](https://github.com/OpenMotionLab/MotionChain)
  - description: Conversational motion controller baseline compared in functionality (multi-turn editing/reasoning) against Motion-Agent.

- ReasoningLM · [GitHub](https://github.com/RUCAIBox/ReasoningLM)
  - description: Structural subgraph reasoning with PLMs for KGQA; used as a competitive baseline (e.g., compared on CWQ).

- TAPAS · [GitHub](https://github.com/google-research/tapas)
  - description: BERT-based table modeling framework cited as an early approach to tabular pre-training and QA in mathematics.

## Agent - Math

### Dataset Math & Competition
- AIMO Validation AIME Dataset · [Dataset](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)
- AIMO Validation AMC Dataset · [Dataset](https://huggingface.co/datasets/AI-MO/aimo-validation-amc)
- GSM8K· [GitHub](https://github.com/openai/grade-school-math)
- MATH· [GitHub](https://github.com/hendrycks/math)


### Solution
- IMO 2025 Problem Solver · [GitHub](https://github.com/lyang36/IMO25) · [Website](https://goo.gle/imo-gold)

1. SciAgent: Tool-augmented Language Models for Scientific Reasoning - EMNLP - 2024 - citation_count 40

86. A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges - ACL - 2025 - citation_count 36 

- Formalisation of Constructable Numbers · [GitHub](https://github.com/Louis-Le-Grand/Formalisation-of-constructable-numbers)
  - description: Formalization of ancient constructible-number problems; part of the sub-curriculum referenced in the paper’s math-agent studies.

- Lean Matrix Cookbook · [GitHub](https://github.com/eric-wieser/lean-matrix-cookbook)
  - description: “Matrix Cookbook” lemmas formalized in Lean; included in the initial curriculum and useful for extending Lean-based math reasoning.

### Math & Science Models

- Auto-GPT · [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
  - description: Autonomous agent baseline evaluated on GAIA in Table 2.
  - description: Open-source agent framework cited as inspiration for stepwise planning; informs the agent’s prompt design and reasoning structure though not used as a direct dependency.
  - description: Popular autonomous agent framework included in the paper’s comparisons and GAIA baseline.
  - description: Cited autonomous agent project; representative baseline framework in the agent ecosystem.

- Awesome-Scientific-Language-Models · [GitHub](https://github.com/yuzhimanhua/Awesome-Scientific-Language-Models)
  - description: Official repository released by this survey; it curates 260+ scientific LLMs, datasets, and benchmarks across fields and modalities that the paper analyzes, useful for reproducing and extending the survey.

- BERTopic · [GitHub](https://github.com/MaartenGr/BERTopic) · [Doc](https://maartengr.github.io/BERTopic/)
  - description: Topic modeling library used in Appendix C to cluster and organize MatSciKB entries into 16 materials-science categories.

- BLOOM · [GitHub](https://github.com/bigscience-workshop/bloom) · [Doc](https://huggingface.co/bigscience/bloom)
  - description: Open multilingual LLM evaluated as a fine-tuned Critic for reward prediction.

- MAmmoTH (MAmmoTH-Coder) · [GitHub](https://github.com/GAIR-NLP/MAmmoTH)
  - description: Open-source math instruction-tuned baseline compared to SCIAGENT; authors evaluate 7B/13B variants.

- MathCoder · [GitHub](https://github.com/OpenGVLab/MathCoder)
  - description: Interleaved CoT+code training/evaluation for math reasoning; referenced as a contemporary approach combining tool use and augmentation.

- MuMath-Code · [GitHub](https://github.com/youweihao-tal/MuMath-Code)
  - description: Official code and data release for this paper; contains MuMath-Code-Data, training/inference scripts (two-stage pipeline), and tool-execution interface for Python.

- PAL (Program-Aided Language Models) · [GitHub](https://github.com/reasoning-machines/pal)
  - description: Early tool-use method that delegates computation to Python; referenced as a representative tool-use baseline and for GSM-Hard.

- PrimeNumberTheoremAnd · [GitHub](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd)
  - description: Repository on the Prime Number Theorem; included in the initial curriculum/evaluation set.

## Agent - Bio
- dMaSIF · [GitHub](https://github.com/LPDI-EPFL/dMaSIF)
  - description: Protein surface learning approach referenced for building end-to-end “chemical color” features from atom categories and distances; informs the surface feature design used in ProteinINR.

- Jellyfish · [GitHub](https://github.com/gmarcais/Jellyfish)
  - description: K-mer counting toolkit used to pre-validate amplicon pools for sufficient sequence diversity before long-read sequencing.

- nnU-Net · [GitHub](https://github.com/MIC-DKFZ/nnUNet)
  - description: Self-configuring biomedical segmentation framework; used in the nnunet_train_model training task.

- SchNetPack (SchNet) · [GitHub](https://github.com/atomistic-machine-learning/schnetpack)
  - description: Reference implementation of SchNet; SchNet is one of the invariant GNN baselines reimplemented within ProteinWorkshop.

### Bio & Protein Models

- AntiBERTy · [GitHub](https://github.com/Graylab/AntiBERTy)
  - description: Antibody language model; used to score sequence naturalness (SeqNat) in antibody evaluations.

- BioBERT · [GitHub](https://github.com/dmis-lab/biobert)
  - description: Biomedical BERT used as a staple baseline/encoder across numerous biomedical tasks listed in the survey.

- BioGPT · [GitHub](https://github.com/microsoft/BioGPT)
  - description: GPT-2–style biomedical generative model featured among decoder-only biomedical LLMs.

- Code Llama · [GitHub](https://github.com/facebookresearch/codellama) · [Website](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
  - description: Open-source LLM baseline and initialization for SCIAGENT-CODER variants; evaluated with and without toolsets.
  - description: Open-source LLM baseline evaluated in MMInA for text-only reasoning on accessibility trees.
  - description: Alternative base LLM (CodeLLaMA-7B/34B) evaluated in ablations to test generalizability.
  - description: Open-source code model used in supplementary evaluations as an alternative agent backbone.
  - description: Open-source code LLM baseline (CodeLlama-34B-Instruct) included in the experimental comparisons.
  - description: Code-focused base models (7B/13B/34B/70B) used as alternative foundations (MuMath-Code-CL variants).

- DNABERT · [GitHub](https://github.com/jerryji1993/DNABERT)
  - description: BERT-style DNA language model highlighted as a Type 1.D example for genomic sequence modeling.

- ESMDiff (Structure Language Models for Protein Conformation Generation) · [GitHub](https://github.com/lujiarui/esmdiff)
  - description: Official code release for this paper; includes training and inference pipelines for SLMs and ESMDiff, data processing, sampling strategies, and evaluation used in the experiments.

- Galactica · [GitHub](https://github.com/paperswithcode/galai) · [Website](https://galactica.org/)
  - description: Scientific LLM baseline evaluated against InstructProtein on understanding tasks.

- GPT‑2· [GitHub](https://github.com/openai/gpt-2) · [Doc](https://huggingface.co/gpt2)
  - description: Decoder-only next-token predictor referenced as the base paradigm for many generative scientific LLMs (e.g., BioGPT).
  - description: Model used to compute sentence-level perplexity (PPL) for assessing the stealthiness of jailbreak prompts.
  - description: Pretrained language model used to compute question perplexity for fluency assessment in the dataset quality analysis.

- InternLM (InternLM2.5) · [GitHub](https://github.com/InternLM/InternLM) · [Website](https://internlm.org/) · [Doc](https://internlm.readthedocs.io/en/latest/)
  - description: Base LLM family from which the authors use the InternLM-7B variant as the backbone for their SPHINX-based SphAgent.
  - description: Referenced LLM family with long-context/chat variants; relevant as alternative backbones for reproducing PRM/MLLM experiments.
  - description: Open-source LLM family; the paper uses InternLM2.5-7B-Chat as a backend model for MindSearch.

- RadFM · [GitHub](https://github.com/chaoyi-wu/RadFM)
  - description: Generalist radiology foundation model serving as a baseline for several medical-imaging agent tasks.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Doc](https://docs.deepwisdom.ai/MetaGPT/latest/)
  - description: Multi‑agent collaboration framework the authors cite as an initial agent framework; EVOAGENT can start from MetaGPT and automatically expand it to a multi-agent system (Appendix D).
  - description: Multi-agent collaborative framework for software/code generation; used as a baseline on HumanEval and other tasks.
  - description: Multi-agent framework cited as a domain-specific reference in the paper’s motivation; useful comparative baseline for software-engineering-style agent pipelines.
  - description: Multi-agent SOP-driven framework used as a Linear-structure baseline; the paper evaluates resilience with MetaGPT across tasks.
  - description: Another SOP-style multi-agent framework referenced in the method section as a comparison point for SimClass’s non-SOP classroom control design.
  - description: Real-world multi-agent software engineering system evaluated under AiTM (roles like Product Manager/Architect/PM/Engineer); the paper also uses its SoftwareDev tasks, noting that “the full version of SoftwareDev is not released yet, we only test on public problems.”
  - description: Multi-agent framework whose tool library is referenced in the programmable node example (e.g., metagpt.tools.libs.data_preprocess.FillMissingValue) for data preprocessing.
  - description: Multi-agent collaborative framework with role assignment and SOPs; used as a comparison baseline.
  - description: Multi-agent collaborative coding framework following SOP-style workflows; evaluated as a baseline against Flow.
  - description: Centralized multi-agent software engineering framework used as a multi-agent comparison baseline.
  - description: Multi-agent collaborative framework cited in related work for MAS.
  - description: Manually designed multi-agent framework cited as a representative baseline MAS.
  - description: MAS framework with SOP-style workflows; its memory variant (MetaGPT-M) is used as a comparison baseline against G-Memory.
  - description: Multi-agent collaborative framework cited in related work; relevant for implementing alternative multi-agent orchestration comparable to CogMir.

- Mistral 7B· [GitHub](https://github.com/mistralai/mistral-src) · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - description: Open-source LLM baseline; also used as a backbone for a SCIAGENT variant fine-tuned on MATHFUNC.
  - description: Sparse MoE LLM (Mixtral 8x7B) evaluated as an alternative backbone in the model comparison study.
  - description: Base LLM used in experiments; models are finetuned and evaluated under the proposed multiagent finetuning pipeline.
  - description: Open LLM family evaluated in CoordinationQA; referenced as one of the model families tested.
  - description: Alternative 7B base LLM evaluated as a drop-in for KG-Agent to assess robustness across backbones.

- OPT · [GitHub](https://github.com/facebookresearch/metaseq)
  - description: Open pre-trained transformer used as the base architecture for InstructProtein and as a comparison baseline.

- ProteinCLAP (components used in reimplementation) — SciBERT · [GitHub](https://github.com/allenai/scibert) · [Website](https://allenai.org/scibert)
  - description: Scientific-domain BERT variant trained on Semantic Scholar; baseline encoder model for many tasks covered by the survey.
  - description: Text encoder used by the authors when re-implementing ProteinCLAP (original code unavailable) for retrieval baselines.

- ProtTrans/ProtBERT· [GitHub](https://github.com/agemagician/ProtTrans) · [Doc](https://huggingface.co/Rostlab/prot_bert)
  - description: Protein encoder (ProtBERT) used by the authors in their ProteinCLAP reimplementation baseline.
  - description: ProtBERT serves as a pretrained backbone combined with RSA for downstream tasks.

- PyEMMA · [GitHub](https://github.com/markovmodel/PyEMMA) · [Website](https://www.pyemma.org/)
  - description: Used to featurize protein structures (pairwise Cα distances) and for preparing inputs to time-lagged independent component analysis (TICA) in evaluation.

- Chroma · [GitHub](https://github.com/generative-biology/chroma)
  - description: Backbone diffusion baseline and the only compared method supporting fold-class–conditional generation; used for conditional comparisons and re-classification analysis.

- IgFold · [GitHub](https://github.com/Graylab/IgFold)
  - description: Antibody structure prediction tool; used to predict CDR-H3 structures in scRMSD and pipeline steps for antibody evaluation.

- AlphaFlow / ESMFlow· [GitHub](https://github.com/bjing2016/alphaflow)
  - description: Official code release for AlphaFLOW/ESMFLOW, including training and sampling under the custom flow-matching framework described in the paper.
  - description: Flow/diffusion models extending folding to ensemble generation; used for multiple-state and distribution prediction with PDB- and MD-fine-tuned checkpoints.

- AlphaFold · [GitHub](https://github.com/deepmind/alphafold)
  - description: Original AlphaFold code and pretrained weights; authors initialize from the CASP14 DeepMind weights and compare against AlphaFold with MSA subsampling.

- AlphaFold2 (incl. IPA) · [GitHub](https://github.com/deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Structure prediction system used to generate wild-type structures for ProteinGym and suggested for obtaining structures for sequence-only datasets.
  - description: Original AlphaFold code and pretrained weights; authors initialize from the CASP14 DeepMind weights and compare against AlphaFold with MSA subsampling.
  - description: Used for baselines (MSA subsampling protocol) and as an alternative initializer; authors ran AF2 pipelines with specific MSA settings.
  - description: Used by the authors only to render protein structure figures in qualitative examples (not required for training/evaluation of ProtT3).
  - description: Structure prediction model augmented with RSA (AlphaFold‑RSA) and compared against standard AlphaFold variants.
  - description: Used in the paper’s in vitro validation to predict structures and pLDDT for selected GFP variants prior to wet-lab testing.
  - description: Structure prediction system used for the pAE interaction-based binding evaluation; SurfPro follows Bennett et al. to compute AF2 pAE interaction scores.
  - description: Source of the IPA frame-based encoder idea and local frame construction referenced by the paper; also used for visualization in the appendix.
  - description: Folding model used as the structural oracle for scTM/scRMSD/pLDDT in several tasks and as a base reference in conformation prediction.
  - description: Architectural inspiration (IPA, triangle modules, template features) for Proteus’ multi-track and triangle mechanisms; referenced extensively in the method design.
  - description: Provides the Invariant Point Attention (IPA) architecture used in encoders/decoders and is the model generating AFDB structures that are filtered into the synthetic dataset.
  - description: High-accuracy protein structure prediction; cited as a scalable upstream structure source to extend ProFSA’s pseudo-pair construction beyond PDB.

- AlphaFold2· [GitHub](https://github.com/google-deepmind/alphafold) · [Codewiki](https://codewiki.google/github.com/google-deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Referenced folding model; AlphaFold predictions are used for comparisons and to inform related baselines (e.g., STR2STR heating/annealing setup).
  - description: Structure prediction method underpinning ColabFold; cited and used for evaluating generated proteins’ structures.
  - description: Source of the Invariant Point Attention (IPA)/Evoformer-style modules and FAPE loss used in DPLM-2’s structure de-tokenizer and tokenizer training.
  - description: Structure prediction system; used via its official repository (v2.3.2) for the MSA-subsampling baseline and for AlphaFlow/ESMFlow configurations.

- antibioticsai · [GitHub](https://github.com/felixjwong/antibioticsai)
  - description: Codebase for explainable antibiotic discovery; referenced/adapted for molecular activity tasks.

- BiomedCLIP · [GitHub](https://github.com/microsoft/BiomedCLIP)
  - description: Vision–language model employed by MMedAgent for closed-set medical image classification via image–text similarity.
  - description: Large-scale biomedical image–text contrastive pre-training resource used for VQA, retrieval, and classification benchmarks in the survey.

- BioPsyKit · [GitHub](https://github.com/mad-lab-fau/BioPsyKit)
  - description: Python package for biopsychological data analysis; used in Psychology/Cognitive Neuroscience tasks.

- Biopython · [GitHub](https://github.com/biopython/biopython) · [Website](https://biopython.org/)
  - description: Used to obtain/reference ground-truth sequences and to evaluate predictions in the PDB task via sequence alignment and identity metrics.
  - description: Bioinformatics toolkit; used for sequence alignment (PairwiseAligner) and peptide bond break evaluations.

- Biotite (P-SEA) · [GitHub](https://github.com/biotite-dev/biotite) · [Website](https://www.biotite-python.org/)
  - description: Library used to compute secondary structure content (P-SEA) of generated proteins for the reported α/β/coil statistics.

- ChemGE (Population-based de novo molecule generation) · [GitHub](https://github.com/tsudalab/ChemGE)
  - description: Population-based molecular design baseline used for comparison on molecular optimization tasks.

- CheXzero · [GitHub](https://github.com/rajpurkarlab/CheXzero)
  - description: Zero-shot chest X‑ray classification method via image–text pre-training, included among biomedical VLP baselines.

- ColabFold· [GitHub](https://github.com/sokrypton/ColabFold) · [Website](https://colabfold.com)
  - description: Used to generate MSAs at inference time (the “ColabFold MMSeqs2 pipeline”) for both PDB and ATLAS evaluations.
  - description: AlphaFold2-based pipeline used to predict structures (pLDDT) of generated sequences to assess foldability.
  - description: Practical AF2 implementation with accelerated MSA search; used by the authors to run AF2 inference during experiments.
  - description: Used to generate MSAs in some comparisons/analyses (e.g., for AlphaFold experiments and MSA coverage analysis).
  - description: Accelerated AlphaFold2 inference and standardized MSA querying; used for AF2 predictions and MSA generation throughout evaluations.
  - description: Fast AlphaFold2/ESMFold inference with online MSA search; used to retrieve MSAs for the MSA-based baselines.

- DSSP · [GitHub](https://github.com/PDB-REDO/dssp) · [Website](https://swift.cmbi.nl/gv/dssp/)
  - description: Secondary structure assignment tool used to filter training data (exclude proteins with >50% loop content).

- Ecology Georeferencing (BIOGR subset) · [GitHub](https://github.com/google-research/ecology-georeferencing)
  - description: Repository hosting a 114-example subset of the BIOGR map georeferencing dataset (images, captions, PDFs, metadata) used in the paper’s multimodal task.

- EigenFold · [GitHub](https://github.com/bjing2016/eigenfold)
  - description: Diffusion-based generative folding model; benchmarked for folding and multiple-state prediction.

- EigenFold · [GitHub](https://github.com/microsoft/eigenfold)
  - description: Non–MD-finetuned baseline for conformation ensemble generation; used for zero-shot equilibrium conformation sampling comparisons.

- ESM (ESM3/ESMFold) · [GitHub](https://github.com/evolutionaryscale/esm) · [Website](https://esm.ai)
  - description: Foundation protein language models; the paper fine-tunes ESM3-1.4B and uses its dVAE structure tokenizer and sequence encoder, and uses ESMFold in baselines/initialization.

- ESM All-Atom (ESM-AA) · [GitHub](https://github.com/zhengkangjie/ESM-AA)
  - description: Official code release for the paper’s multi-scale protein language model; used to reproduce pretraining on mixed protein/molecule data and all downstream evaluations.

- FoldComp · [GitHub](https://github.com/steineggerlab/foldcomp)
  - description: Compression/indexing tool used to store and stream very large AFDB subsets efficiently for training.
  - description: Compression/indexing library for large protein structure sets; the benchmark’s storage‑efficient AFDB/ESM Atlas dataloaders are built on FoldComp.

- FoldFlow · [GitHub](https://github.com/joeybose/foldflow)
  - description: Flow-matching baseline (base/OT/stochastic variants); evaluated using the authors’ public repo with settings matching the paper.

- FoldFlow / FOLDFLOW-2 · [GitHub](https://github.com/DreamFold/FoldFlow)
  - description: Official code release for the FoldFlow family including FOLDFLOW-2; used to train/evaluate the sequence-conditioned SE(3) flow matching model, run baselines/ablations, and reproduce figures and metrics in the paper.

- MoleculeNet · [GitHub](https://github.com/deepchem/deepchem) · [Website](http://moleculenet.org/)
  - description: Standard suite of molecular property prediction benchmarks (e.g., QM7/8/9, HIV, BACE, TOX21) used to evaluate molecular agents and models.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com/)
  - description: Fast structure search and quantized structure tokens used in prior work; ProSST compares against Foldseek-based tokenization in ablations and references foldseek clusters for AFDB.
  - description: Used for structure-based clustering (DFS) and for evaluation (easy-cluster for diversity; easy-search for novelty/TM-score against PDB/AFDB).
  - description: Fast structure search/clustering used to quantify diversity (#clusters) and to cluster structural motifs in the tokenizer analysis.
  - description: Structure search tool used to compute pdbTM and assess novelty of generated proteins by comparing to the PDB database.
  - description: Fast structural search/clustering tool used to cluster AlphaFoldDB and select the 2.27M representative structures for pretraining.
  - description: Fast structure search/clustering; used to compute novelty (Max TM to PDB) and diversity (pairwise TM, clustering) across design tasks.
  - description: Fast structure search tool used to find nearest structural neighbors and for distributional/structural comparisons (e.g., TM-score references).
  - description: Employed for structural clustering/diversity evaluation and for redundancy removal via the easy-cluster pipeline; also used for novelty comparisons to PDB.
  - description: Used with easy-cluster to structurally cluster protein domains when creating train/val/test splits for the Megascale stability alignment experiments.

- FrameDiff (SE(3) diffusion for protein backbones)· [GitHub](https://github.com/jasonkyuyim/se3_diffusion)
  - description: Backbone diffusion baseline; sampled with its public repository and defaults as part of the benchmark.
  - description: Baseline diffusion framework the authors build upon; VFN-Diff replaces FrameDiff’s IPA point-attention with VFN while keeping other components.
  - description: Baseline method for unconditional protein backbone generation; used for comparisons on designability, novelty, and diversity.

- InstructProtein · [GitHub](https://github.com/HICAI-ZJU/InstructProtein)
  - description: Official code release for the paper; contains training/tuning scripts and resources to align human and protein language and reproduce the experiments.

- LLaVA-Med · [GitHub](https://github.com/microsoft/LLaVA-Med)
  - description: Backbone MLLM used by MMedAgent for VQA and dialog; the agent is initialized from LLaVA‑Med 60K-IM and further instruction-tuned.
  - description: Medical adaptation of LLaVA cited as a representative vision–language assistant in the biomedical section.
  - description: Biomedical/clinical adaptation of LLaVA used as a medical LVLM baseline in image-based comparisons.

- Med-Flamingo · [GitHub](https://github.com/snap-stanford/med-flamingo)
  - description: Multimodal medical few-shot learner used as a comparison baseline on the evaluation suite.

- MedICaT · [GitHub](https://github.com/allenai/medicat)
  - description: Biomedical figures, captions, and references dataset; used for vision–language pre-training and VQA tasks in the survey.

- MedMCQA · [GitHub](https://github.com/medmcqa/medmcqa)
  - description: Large-scale multi-subject medical MCQ benchmark cited for evaluating broad medical knowledge of LLM-based agents.

- MedNLI · [GitHub](https://github.com/jgc128/mednli)
  - description: Natural language inference dataset in the clinical domain cited as a key benchmark for evaluating medical reasoning and text understanding of LLM agents.

- MedQA (USMLE)· [GitHub](https://github.com/jind11/MedQA)
  - description: Text QA benchmark used for evaluation of MAM and baselines.
  - description: USMLE-style multiple-choice QA benchmark used in the survey’s referenced systems to assess factual medical knowledge and diagnostic reasoning.

- MedSAM· [GitHub](https://github.com/bowang-lab/MedSAM)
  - description: Medical adaptation of SAM used as the segmentation tool (box-prompted and text-grounded with Grounding DINO + MedSAM) inside MMedAgent.
  - description: Medical segmentation adaptation of SAM; used as the repository for the medsam_inference segmentation task.

- MedSSS · [GitHub](https://github.com/pixas/MedSSS)
  - description: Medical small language model with slow-thinking policy; used to generate responses in the medsss_generate task.

- MoleculeNet· [Website](http://moleculenet.org/) · [GitHub](https://github.com/deepchem/deepchem)
  - description: Open-source chemistry toolkit; multiple benchmark tasks and annotated programs rely on DeepChem models and data utilities.
  - description: Standard suite of molecular property benchmarks (QM7/8/9, HIV, MUV, BACE, BBBP, TOX21, PCBA, SIDER) used to evaluate ESM-AA’s molecular performance.

- OmegaFold · [GitHub](https://github.com/HeliXonProtein/OmegaFold)
  - description: Employed to fold motif-scaffolding generations for success-rate evaluation (pLDDT and motif RMSD).
  - description: Sequence embedding/structure prediction model whose embeddings condition EigenFold; included as it is integral to the EigenFold baseline referenced.
  - description: Single-sequence structure predictor; used to generate embeddings for the EigenFold baseline pipeline.

- OpenFold · [GitHub](https://github.com/aqlaboratory/openfold) · [Website](https://aqlaboratory.github.io/openfold/)
  - description: Open-source reimplementation of AlphaFold used for architecture and training pipeline; authors fine-tune all weights via OpenFold and use it to run baselines.
  - description: Open reimplementation of AlphaFold2; used as an MSA-based folding baseline and backbone for conformation sampling (e.g., MSA-subsampling).

- Papyrus Scaffold Visualizer · [GitHub](https://github.com/martin-sicho/papyrus-scaffold-visualizer)
  - description: Visualization utilities for Papyrus; adapted as source code for chemistry analysis tasks.

- ProGen2 · [GitHub](https://github.com/salesforce/progen)
  - description: Used to compute perplexity as a measure of sequence naturalness in extended unconditional sequence-generation analysis.
  - description: Autoregressive protein LM baseline; also used to compute perplexity for quality evaluation and as a controlled generation baseline.

- ProGen2 fine-tuning helper · [GitHub](https://github.com/hugohrban/ProGen2-finetuning)
  - description: Third-party training scripts referenced for fine-tuning ProGen2 on family-conditioned data for the controllable generation experiments.

- ProGen3 · [GitHub](https://github.com/Profluent-AI/progen3)
  - description: Official code release and model weights for the ProGen3 sparse protein language models introduced in the paper; used for generation, GLM infilling, and alignment experiments.

- ProLIF · [GitHub](https://github.com/chemosim-lab/ProLIF)
  - description: Encodes protein–ligand interactions as fingerprints; used in molecular interaction analysis/visualization tasks.

- ProSST · [GitHub](https://github.com/ai4protein/ProSST)
  - description: Official codebase and pre-trained models released by the paper; implements the structure quantizer, disentangled attention Transformer, training and evaluation pipelines.

- Protein Frame Flow (FrameFlow) · [GitHub](https://github.com/microsoft/protein-frame-flow) · [Doc](https://github.com/microsoft/protein-frame-flow/tree/main/motif_scaffolding)
  - description: SE(3) flow-matching backbone generator; used as a backbone design and motif-scaffolding baseline with its released motif-scaffolding evaluation scripts.

- ProteinGym · [GitHub](https://github.com/OATML-Markslab/ProteinGym) · [Website](https://proteingym.org)
  - description: Benchmark and evaluation scripts used for zero-shot mutation effect prediction; the paper computes Spearman/NDCG/Top-Recall with the provided scripts.
  - description: Large-scale benchmark datasets used for zero-shot and supervised fitness prediction evaluations and alignment experiments.

- ProteinKG25 (from OntoProtein) · [GitHub](https://github.com/zjunlp/OntoProtein) · [Website](https://zjunlp.github.io/project/OntoProtein/)
  - description: Knowledge-graph–derived protein dataset the authors convert into free-text descriptions for captioning and retrieval; used jointly with Swiss-Prot in stage-1 training.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Strong inverse-folding baseline cited and compared in CATH experiments.
  - description: Inverse folding model used to design sequences for generated backbones as part of the designability pipeline.
  - description: Inverse-folding model used as a comparison point and for some evaluation pipelines (e.g., sequence prediction from generated structures).
  - description: Inverse folding baseline used for comparison and for generating candidate binders in the MaSIF comparison.
  - description: Inverse folding model used during evaluation to design sequences for generated backbones and as a pipeline baseline.
  - description: Inverse folding model used to design sequences for generated backbones and to assess designability (scTM/scRMSD) across tasks.
  - description: Inverse folding model used to design sequences for generated backbones when computing designability metrics (8 sequences per backbone).
  - description: Structure-to-sequence model used for structure-only evaluation and co-design metrics (e.g., scPerplexity, scRMSD) and to assess RFDiffusion outputs.
  - description: Inverse folding model used in the evaluation pipeline to design sequences for generated backbones before refolding with ESMFold for self-consistency metrics.
  - description: Fixed‑backbone sequence design baseline used in evaluations (DES-bb) and for comparison against Pallatom’s sequence head.

- ProteinNet · [GitHub](https://github.com/aqlaboratory/proteinnet) · [Website](https://www.proteinnet.org/)
  - description: Dataset used for contact prediction and structure-related evaluations; RSA structural similarity analyses also reference ProteinNet.
  - description: Dataset/splits used for the unsupervised contact prediction evaluation following ESM/TAPE protocols.

- ProteinWorkshop · [GitHub](https://github.com/a-r-j/ProteinWorkshop) · [Website](https://proteins.sh) · [Doc](https://proteins.sh)
  - description: Official codebase and documentation for the benchmark introduced in the paper; includes dataloaders (AFDB/ESM Atlas), models, featurization, tasks, and CLI for pretraining/finetuning.

- Proteína · [GitHub](https://github.com/NVIDIA-Digital-Bio/proteina) · [Website](https://research.nvidia.com/labs/genair/proteina/)
  - description: Official codebase and project page for the paper’s flow-based protein backbone generator, including training, sampling, guidance, metrics, and instructions to reproduce results.

- pubmedqa· [GitHub](https://github.com/pubmedqa/pubmedqa)
  - description: Biomedical research QA dataset used for text-based evaluation.
  - description: Biomedical yes/no QA dataset used as a main benchmark (with the original train/test split) in both the Problem-Solving and Actor–Critic settings.

- PubMedQA · [GitHub](https://github.com/qiaojin/PubMedQA)
  - description: Biomedical research question answering dataset referenced for testing literature-grounded medical reasoning.

- pymed (PubMed querying with Python) · [GitHub](https://github.com/gijswobben/pymed)
  - description: Python library used by the agent’s literature-search tool to query PubMed; the paper cites and uses this to pull and summarize relevant biomedical papers each round.

- Reaction Class Prediction (IEConv Proteins) · [GitHub](https://github.com/phermosilla/IEConv_proteins)
  - description: Dataset and code for enzyme reaction class prediction (EC-based); used for graph‑level classification with 50% sequence identity split.

- RFdiffusion· [GitHub](https://github.com/RosettaCommons/RFdiffusion) · [Website](https://rfdiffusion.github.io/)
  - description: Referenced for Brownian motion on rotation manifolds; used to define the denoising objective for training the structure autoencoder.
  - description: Structure-design baseline compared against DPLM in motif-scaffolding experiments.
  - description: Backbone diffusion baseline; authors ran its official implementation for comparative evaluation.
  - description: Structure-based generative baseline used for benchmarking unconditional structure generation and motif-scaffolding.
  - description: Related diffusion-based protein design system discussed in the paper; not directly compared under identical settings but relevant for extending this line of work.
  - description: Structure-based protein backbone and motif-scaffolding generator; used as a strong backbone-design baseline and to generate de novo backbones for inverse-folding evaluation.
  - description: Strong comparison baseline for backbone generation; the paper matches/exceeds its designability and efficiency without pretraining.
  - description: Structure-generation method providing the functional motif scaffolding benchmark tasks and a fold-conditioned baseline compared against DiMA.
  - description: Strong baseline for protein backbone generation and motif scaffolding; the paper compares FOLDFLOW-2 against this model across designability, novelty, diversity, and conditional tasks.
  - description: State-of-the-art backbone generative baseline; evaluated with the paper’s PMPNN-1 protocol for comparison.

- Rosetta / RosettaAntibodyDesign (RAbD) · [GitHub](https://github.com/RosettaCommons/rosetta) · [Website](https://www.rosettacommons.org/)
  - description: Suite used for side-chain packing, energy minimization, binding energy (InterfaceAnalyzer) and as the source of the RAbD dataset reference in antibody evaluation.

- RoseTTAFold2 · [GitHub](https://github.com/RosettaCommons/RoseTTAFold2)
  - description: MSA-based folding model; included as a single-state folding baseline.

- TorchDrug· [GitHub](https://github.com/DeepGraphLearning/torchdrug) · [Doc](https://torchdrug.ai/docs/tutorials/protein_function.html)
  - description: Framework used to train and evaluate downstream tasks and includes official implementations of GearNet; the paper fine-tunes models and reports metrics within this framework.
  - description: Protein structure encoder architecture (including GearNet-Edge-IEConv) used for structure pre-training and downstream evaluation; implemented and run via TorchDrug.
  - description: ML platform that includes GearNet; cited as a dependency and used for protein‑specific architectures in the benchmark.

## Agent - Engineering
- CadQuery · [GitHub](https://github.com/CadQuery/cadquery) · [Docs](https://cad.onshape.com/FsDoc/)

## Recommendation Systems
- KuaiSim · [GitHub](https://github.com/KuaiRec/KuaiSim)


## Deep Learning
- node2vec: Scalable Feature Learning for Networks · [GitHub](https://github.com/aditya-grover/node2vec) · [Website](https://snap.stanford.edu/node2vec/)
- Mamba: [GitHub](https://github.com/state-spaces/mamba)
- LongT5: Efficient Text-To-Text Transformer for Long Sequences · [GitHub](https://github.com/google-research/longt5)

## World Model
- CLIPort · [GitHub](https://github.com/cliport/cliport) · [Website](https://cliport.github.io)
- ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
- ALFWorld (Princeton Vision Lab) · [GitHub](https://github.com/princeton-vl/ALFWorld)
- Sentence-Transformers· [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net)
- Malmö # · [GitHub](https://github.com/microsoft/malmo)

## Infrastructure
- sglang · [GitHub](https://github.com/sgl-project/sglang)
- vllm · [GitHub](https://github.com/vllm-project/vllm)
- Triton (GPU programming language) · [GitHub](https://github.com/triton-lang/triton) · [CodeWiki](https://codewiki.google/github.com/triton-lang/triton)
- flashinfer: Kernel Library for LLM Serving · [GitHub](https://github.com/flashinfer-ai/flashinfer)
- TIMRUN: Efficient Engine for Long-horizon Reasoning · [GitHub](https://github.com/subconscious-systems/TIMRUN)
- DeepSpeed · [GitHub](https://github.com/microsoft/DeepSpeed)
- FlashAttention-2· [GitHub](https://github.com/Dao-AILab/flash-attention)
- SeqIO · [GitHub](https://github.com/google/seqio)
- T5X · [GitHub](https://github.com/google-research/t5x)
- verl · [GitHub](https://github.com/volcengine/verl)


## Remaining Projects

- ADMET-AI · [GitHub](https://github.com/swansonk14/admet_ai)
  - description: ADMET prediction platform; referenced and adapted to build tasks involving property prediction.

- AHK (Python AutoHotkey wrapper) · [GitHub](https://github.com/spyoungtech/ahk)
  - description: Python wrapper for AutoHotkey used by CRADLE to implement reliable, low-level mouse/keyboard control on Windows.

- EEG2EEG · [GitHub](https://github.com/ZitongLu1996/EEG2EEG)
  - description: Individual-to-individual EEG converters; used to formulate EEG mapping/generation tasks.

- EvoDiff · [GitHub](https://github.com/microsoft/evodiff)
  - description: Discrete diffusion baseline used for comparison and for conditional family-specific generation baselines.
