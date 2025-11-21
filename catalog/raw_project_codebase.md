# Paper Reader v2 Results

This log lists GitHub repositories, official websites, and documentation collected with GPT-5 for
each paper in export_cit_20_tools.jsonl (tag: tool, citation count >= 20).


## 1. SciAgent: Tool-augmented Language Models for Scientific Reasoning - EMNLP - 2024 - citation_count 40 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_40_Main_SciAgent_Tool-augmented_Language_Models_for_Scientific_Reasoning.pdf
- Link: https://aclanthology.org/2024.emnlp-main.880/
- Tags: multiagent, tool, science, agent, biology, protein-function, mathematics, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_40_Main_SciAgent_Tool-augmented_Language_Models_for_Scientific_Reasoning.pdf

### GitHub & Websites

- SCIAGENT (code release) 
  - description: The paper develops open-source tool-augmented agents that plan, retrieve tools, generate code, and execute to solve scientific problems; used for all main results. The PDF states they are open-source but does not provide a URL.

- MATHFUNC (training corpus) 
  - description: Tool-augmented math-focused corpus built by the authors with 31,375 samples and 5,981 documented functions; used to fine-tune SCIAGENT for planning/action and train the retriever.

- SCITOOLBENCH (benchmark) 
  - description: Authors’ evaluation benchmark with 4,250 questions and 2,285 functions across Math, Physics, Chemistry, Finance, and EECS; used for all reported evaluations.

- MATH dataset · [GitHub](https://github.com/hendrycks/math)
  - NIPS 2021
  - description: Base math corpus the authors build upon to synthesize MATHFUNC samples; provides training problems used to teach math skills and tool-use.

- TheoremQA · [GitHub](https://github.com/TIGER-AI-Lab/TheoremQA)
  - EMNLP 2023
  - description: Source of human-annotated questions for SCITOOLBENCH; authors curate and filter questions from this dataset.

- SciBench · [GitHub](https://github.com/mandyyyyii/scibench)
  - ICML 2024 
  - description: Another source of human-annotated scientific questions used to construct SCITOOLBENCH; authors filter and expand questions from it.

- CREATOR Challenge (CREATION Challenge) · [GitHub](https://github.com/amazon-science/creator)
  - status: inaccessible
  - description: External tool-use benchmark re-purposed by the authors to form a global toolset and report additional accuracy results.

- DPR (Dense Passage Retrieval) · [GitHub](https://github.com/facebookresearch/DPR)
  - EMNLP2020
  - description: Retrieval framework the authors follow to train their dense retriever (RoBERTa-base backbone) for function/tool retrieval.

- SimCSE · [GitHub](https://github.com/princeton-nlp/SimCSE)
  - EMNLP 2021
  - description: Alternative retriever baseline evaluated in analysis to study retriever quality on tool retrieval.

- Contriever · [GitHub](https://github.com/facebookresearch/contriever)
  - description: Another alternative dense retriever baseline used in analysis to compare against the authors’ fine-tuned retriever.

- RoBERTa-base · [GitHub](https://github.com/facebookresearch/fairseq) · [Codewiki](https://codewiki.google/github.com/facebookresearch/fairseq) · [Doc](https://huggingface.co/roberta-base)
  - description: Encoder backbone used to train the authors’ dense retriever for function retrieval.

- SymPy · [GitHub](https://github.com/sympy/sympy) · [Website](https://www.sympy.org/en/index.html) · [Doc](https://docs.sympy.org/latest/index.html)
  - description: Symbolic math library repeatedly used in the paper’s tool/functions and solution code (e.g., integrate, solve).

- SciPy · [Website](https://scipy.org/) · [Doc](https://docs.scipy.org/doc/scipy/)
  - description: Scientific computing library referenced in examples (e.g., scipy.integrate); relevant dependency for executing tool-augmented solutions.

- CodeLlama · [GitHub](https://github.com/facebookresearch/codellama)
  - description: Open-source LLM baseline and initialization for SCIAGENT-CODER variants; evaluated with and without toolsets.

- Mistral 7B · [GitHub](https://github.com/mistralai/mistral-src)
  - description: Open-source LLM baseline; also used as a backbone for a SCIAGENT variant fine-tuned on MATHFUNC.

- Llama 3 · [GitHub](https://github.com/meta-llama/llama-models) · [Website](https://ai.meta.com/llama/)
  - description: Open-source LLM baseline; authors fine-tune an 8B version to create SCIAGENT-LLAMA3.

- DeepSeek-Math · [GitHub](https://github.com/deepseek-ai/DeepSeek-Math)
  - description: Math-pretrained LLM baseline; authors fine-tune a 7B version to create SCIAGENT-DEEPMATH.

- ToRA (ToRA-Coder) · [GitHub](https://github.com/microsoft/ToRA)
  - description: Open-source math tool-integrated baseline compared against SCIAGENT; both 7B and 13B versions evaluated.

- MAmmoTH (MAmmoTH-Coder) · [GitHub](https://github.com/GAIR-NLP/MAmmoTH)
  - description: Open-source math instruction-tuned baseline compared to SCIAGENT; authors evaluate 7B/13B variants.

<!-- paper_id: 60126292c0b31dfc8628d99001e057b9f8355000 -->

## 2. Multi-modal Agent Tuning: Building a VLM-Driven Agent for Efficient Tool Usage - ICLR - 2025 - citation_count 27 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_27_Spotlight_Multi-modal_Agent_Tuning_Building_a_VLM-Driven_Agent_for_Efficient_Tool_Usage.pdf
- Link: https://openreview.net/pdf/5023fe68ab67d9e72385045030a9d62d49bf747d.pdf
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_27_Spotlight_Multi-modal_Agent_Tuning_Building_a_VLM-Driven_Agent_for_Efficient_Tool_Usage.pdf

### GitHub & Websites

- Multi-modal Agent Tuning / T3-Agent · [Website](https://mat-agent.github.io)
  - description: Official project page for the paper; central hub for the T3-Agent and MM-Traj resources (code/data/demos referenced by the paper).

- MM-Traj
  - description: The 20K multi-modal tool-usage trajectory dataset introduced in the paper and used to train T3-Agent; released via the project page above.

- GAIA: A Benchmark for General AI Assistants · [Website](https://huggingface.co/datasets/gaia-benchmark/GAIA)
  - description: Evaluation benchmark with multi-modal, tool-use tasks; the paper evaluates T3-Agent on GAIA.

- Open-LLaVA-NeXT-mix1M · [Website](https://huggingface.co/datasets/Lin-Chen/Open-LLaVA-NeXT-mix1M)
  - description: Additional training data mixed in with MM-Traj during VLM tuning, as stated in the training setup.

- BGE m3 Embedding · [GitHub](https://github.com/FlagOpen/FlagEmbedding) · [Website](https://huggingface.co/BAAI/bge-m3)
  - description: Text-embedding model used for retrieving images relevant to generated file contents during data synthesis.

- ShareGPT4V · [GitHub](https://github.com/InternLM/ShareGPT4V)
  - description: Used to produce captions for the collected source images that seed multi-modal file generation.

- AgentLego · [GitHub](https://github.com/InternLM/agentlego)
  - description: Open-source tool API library used as a comparison baseline agent framework in experiments.

- HuggingFace Agents · [Doc](https://huggingface.co/docs/transformers/agents)
  - description: Baseline agent (HF Agent) and tool-calling framework the paper compares against, using the same tool set as T3-Agent.

- LLaVA-NeXT · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
  - description: Open-source VLM baseline controller evaluated by the paper (LLaVA‑NeXT‑8B).

- InternVL2 · [GitHub](https://github.com/OpenGVLab/InternVL)
  - description: Open-source VLM baseline controller evaluated by the paper (InternVL2‑8B).

- MiniCPM‑V · [GitHub](https://github.com/OpenBMB/MiniCPM-V) · [Codewiki](https://codewiki.google/github.com/OpenBMB/MiniCPM-V)
  - description: One of the two VLM backbones the authors fine-tune with MM‑Traj to build T3‑Agent (MiniCPM‑V‑8.5B).

- Qwen2‑VL · [GitHub](https://github.com/QwenLM/Qwen2-VL)
  - description: The other VLM backbone the authors fine-tune with MM‑Traj to build T3‑Agent (Qwen2‑VL‑7B).

- Qwen 1.5 · [GitHub](https://github.com/QwenLM/Qwen) · [Codewiki](https://codewiki.google/github.com/QwenLM/Qwen)
  - description: Open-source LLM baseline controller (Qwen1.5‑72B‑chat) used for comparison.

- LLaMA 3 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open-source LLM baseline controller (LLaMA‑3‑70B‑Instruct) used for comparison.

- OWL‑ViT · [GitHub](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)
  - description: Open-vocabulary object localization model used as a real executable tool in T3‑Agent.

- Stable Diffusion · [GitHub](https://github.com/CompVis/stable-diffusion)
  - description: Image generation model used as a tool within T3‑Agent.

- InstructPix2Pix · [GitHub](https://github.com/timothybrooks/instruct-pix2pix)
  - description: Image editing model used as a tool within T3‑Agent.

- DSFD (Dual Shot Face Detector) · [GitHub](https://github.com/sfzhang15/DSFD)
  - description: Face detection model used as a tool within T3‑Agent.

- COCO (Common Objects in Context) · [Website](https://cocodataset.org)
  - description: One of eight image sources the authors compile to seed multi-modal file generation.

- ChartQA · [GitHub](https://github.com/vis-nlp/ChartQA)
  - description: Chart question answering dataset used as an image source for building the image pool in data synthesis.

- Segment Anything (SAM) · [GitHub](https://github.com/facebookresearch/segment-anything) · [Codewiki](https://codewiki.google/github.com/facebookresearch/segment-anything) · [Website](https://segment-anything.com)
  - description: Dataset/model cited as an image source to diversify the pool of images for MM‑Traj synthesis.

- TextVQA · [Website](https://textvqa.org)
  - description: Dataset used as an image source for the collected pool in multi-modal file generation.

- CelebA (Web‑Celebrity) · [Website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - description: Face attribute/image dataset included among the eight image sources used for data synthesis.

- Google Landmarks Dataset v2 (Web‑Landmark) · [GitHub](https://github.com/cvdfoundation/google-landmark)
  - description: Landmark recognition/retrieval dataset used as an image source in the image pool.

- WikiArt · [Website](https://www.wikiart.org)
  - description: Art image collection used as an image source to enrich the diversity of MM‑Traj’s image pool.

- GPT‑4o mini · [Doc](https://cdn.openai.com/gpt-4o-system-card.pdf)
  - description: Closed-source model used extensively in the pipeline (query/file/trajectory generation and verification) and as baseline controller in comparisons.

<!-- paper_id: 616a8e2e315bbb3d50e597fc95ba84e7b0638a38 -->

## 3. ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models - NAACL - 2025 - citation_count 94 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_NAACL_94_Long_ResearchAgent_Iterative_Research_Idea_Generation_over_Scientific_Literature_with_Large_Language_Models.pdf
- Link: https://aclanthology.org/2025.naacl-long.342/
- Tags: multiagent, tool, science, agent, biology, protein-function, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_NAACL_94_Long_ResearchAgent_Iterative_Research_Idea_Generation_over_Scientific_Literature_with_Large_Language_Models.pdf

### GitHub & Websites

- ResearchAgent · [GitHub](https://github.com/JinheonBaek/ResearchAgent)
  - description: Official code release for the paper; implements the ResearchAgent pipeline, prompts, and evaluation setup for iterative research idea generation with literature, entity knowledge store, and reviewing agents.

- Semantic Scholar Academic Graph API · [Website](https://www.semanticscholar.org/product/api)
  - description: Source used to collect papers, citations, and abstracts (post–May 1, 2023) to build the benchmark and retrieve related literature for the agent.

- BLINK (Scalable Zero-shot Entity Linking) · [GitHub](https://github.com/facebookresearch/BLINK)
  - description: Off-the-shelf entity linker used to extract and canonicalize entities from titles/abstracts to construct the entity-centric knowledge store.

- OpenAI GPT-4 · [Website](https://openai.com/research/gpt-4) · [Doc](https://platform.openai.com/docs/models#gpt-4)
  - description: Primary LLM used to generate ideas and to act as the judge in model-based evaluations.

- OpenAI GPT-3.5 · [Doc](https://platform.openai.com/docs/models#gpt-3-5)
  - description: Additional LLM evaluated in ablation analyses to assess model choice sensitivity.

- Meta Llama 3 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open LLM used in auxiliary experiments comparing different backbones within the ResearchAgent framework.

- Mixtral of Experts (Mistral AI) · [GitHub](https://github.com/mistralai/mistral-src) · [Website](https://mistral.ai/)
  - description: Sparse MoE LLM (Mixtral 8x7B) evaluated as an alternative backbone in the model comparison study.

- Qwen1.5 (Alibaba Cloud) · [GitHub](https://github.com/QwenLM/Qwen) · [Codewiki](https://codewiki.google/github.com/QwenLM/Qwen) · [Website](https://qwenlm.ai/)
  - description: Open LLM family tested as another backbone to analyze ResearchAgent’s robustness across models.

- Label Studio · [GitHub](https://github.com/HumanSignal/label-studio) · [Website](https://labelstud.io/)
  - description: Annotation platform used to collect expert human evaluations and induce human-aligned criteria for reviewing agents.

<!-- paper_id: 51b7b3ad7645a69e3c1c80cae69473b8bd472f67 -->

## 4. ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery - ICLR - 2025 - citation_count 48 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_48_Poster_ScienceAgentBench_Toward_Rigorous_Assessment_of_Language_Agents_for_Data-Driven_Scientific_Discovery.pdf
- Link: https://openreview.net/pdf/01bb45041a6c8c75021cdf0e1835d645398f660d.pdf
- Tags: multiagent, tool, science, agent, biology, protein-function, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_48_Poster_ScienceAgentBench_Toward_Rigorous_Assessment_of_Language_Agents_for_Data-Driven_Scientific_Discovery.pdf

### GitHub & Websites

- ScienceAgentBench · [Website](https://osu-nlp-group.github.io/ScienceAgentBench/)
  - description: Official project page for the benchmark introduced in the paper; hosts tasks, datasets, and evaluation resources needed to reproduce results.

- OpenHands (CodeAct) · [GitHub](https://github.com/All-Hands-AI/OpenHands)
  - description: Generalist code-generation/SE agent framework used as a baseline (CodeAct v1.9) in the paper’s evaluations.

- pipreqs · [GitHub](https://github.com/bndr/pipreqs)
  - description: Tool the authors use to infer a program’s Python dependencies when setting up execution environments for evaluation.

- pip-tools · [GitHub](https://github.com/jazzband/pip-tools)
  - description: Used to resolve and install Python dependencies for each generated program during evaluation.

- Llama 3.1 · [Website](https://ai.meta.com/llama/)
  - description: Open-weight LLM (70B and 405B) evaluated under direct prompting, OpenHands CodeAct, and self-debug frameworks.

- Mistral Large 2 · [Website](https://mistral.ai/news/mistral-large-2407)
  - description: Proprietary LLM (123B) evaluated as a backbone model across the three agent frameworks.

- GPT-4o · [Website](https://openai.com/index/hello-gpt-4o)
  - description: Proprietary LLM evaluated as a backbone model and also used as an automated figure-quality judge.

- Claude 3.5 Sonnet · [Website](https://www.anthropic.com/news/claude-3-5-sonnet)
  - description: Proprietary LLM that achieved the best performance under the self-debug framework in the paper.

- DeepChem · [GitHub](https://github.com/deepchem/deepchem)
  - description: Open-source chemistry toolkit; multiple benchmark tasks and annotated programs rely on DeepChem models and data utilities.

- MolPAL · [GitHub](https://github.com/coleygroup/molpal)
  - description: Molecular pool-based active learning framework; code was adapted as a source for computational chemistry tasks.

- ADMET-AI · [GitHub](https://github.com/swansonk14/admet_ai)
  - description: ADMET prediction platform; referenced and adapted to build tasks involving property prediction.

- Papyrus Scaffold Visualizer · [GitHub](https://github.com/martin-sicho/papyrus-scaffold-visualizer)
  - description: Visualization utilities for Papyrus; adapted as source code for chemistry analysis tasks.

- Papyrus Scripts · [GitHub](https://github.com/OlivierBeq/Papyrus-scripts)
  - description: Scripts around the Papyrus dataset; used to construct chemistry tasks and analyses.

- BioPsyKit · [GitHub](https://github.com/mad-lab-fau/BioPsyKit)
  - description: Python package for biopsychological data analysis; used in Psychology/Cognitive Neuroscience tasks.

- pymatgen · [GitHub](https://github.com/materialsproject/pymatgen)
  - description: Materials analysis library; code and utilities were adapted for materials-focused computational tasks.

- NeuroKit2 · [GitHub](https://github.com/neuropsychology/NeuroKit)
  - description: Toolbox for neurophysiological signal processing; used to build and evaluate physiology signal-processing tasks.

- syllogistic-nvc · [GitHub](https://github.com/nriesterer/syllogistic-nvc)
  - description: Cognitive modeling code for syllogistic reasoning; adapted for Psychology tasks comparing cognitive models.

- cogsci-jnmf · [GitHub](https://github.com/brand-d/cogsci-jnmf)
  - description: Joint NMF code for reasoning analyses; used to build tasks analyzing human reasoning model similarities.

- MAST-ML · [GitHub](https://github.com/uw-cmg/MAST-ML)
  - description: Materials Simulation Toolkit for ML; adapted to construct materials data-driven tasks.

- EEG2EEG · [GitHub](https://github.com/ZitongLu1996/EEG2EEG)
  - description: Individual-to-individual EEG converters; used to formulate EEG mapping/generation tasks.

- geoplot · [GitHub](https://github.com/ResidentMario/geoplot)
  - description: Geospatial plotting library; used for GIS visualization tasks.

- MODNet · [GitHub](https://github.com/ppdebreuck/modnet)
  - description: Materials property prediction toolkit; adapted for materials modeling/feature selection tasks.

- GeoPandas · [GitHub](https://github.com/geopandas/geopandas)
  - description: Core geospatial dataframe library; widely used across the GIS tasks in the benchmark.

- DeepPurpose · [GitHub](https://github.com/kexinhuang12345/DeepPurpose)
  - description: Drug–target interaction library; used in DTI/repurposing tasks (e.g., DAVIS) within computational chemistry.

- antibioticsai · [GitHub](https://github.com/felixjwong/antibioticsai)
  - description: Codebase for explainable antibiotic discovery; referenced/adapted for molecular activity tasks.

- Iris (SciTools) · [GitHub](https://github.com/SciTools/iris)
  - description: Python library for meteorological and climate data; used to support GIS/climate-related analysis tasks.

- OGGM · [GitHub](https://github.com/OGGM/oggm)
  - description: Open Global Glacier Model; referenced/adapted to construct real-world GIS tasks.

- Scanpy · [GitHub](https://github.com/scverse/scanpy)
  - description: Single-cell analysis toolkit; used for bioinformatics tasks involving scRNA-seq processing and visualization.

- scvi-tools · [GitHub](https://github.com/scverse/scvi-tools)
  - description: Probabilistic deep generative models for single-cell omics; adapted in single-cell analysis tasks.

- Muon · [GitHub](https://github.com/scverse/muon)
  - description: Multimodal omics analysis framework; used to support multi-omics tasks.

- Scirpy · [GitHub](https://github.com/scverse/scirpy)
  - description: TCR single-cell repertoire analysis; used for immunology-related bioinformatics tasks.

- PyKrige · [GitHub](https://github.com/GeoStat-Framework/PyKrige)
  - description: Geostatistical kriging in Python; used in GIS interpolation/heat analysis tasks.

- Predicting-Activity-by-Machine-Learning · [GitHub](https://github.com/psa-lab/predicting-activity-by-machine-learning)
  - description: QSAR/ML workflows; adapted for activity prediction and feature interpretation tasks.

- ProLIF · [GitHub](https://github.com/chemosim-lab/ProLIF)
  - description: Encodes protein–ligand interactions as fingerprints; used in molecular interaction analysis/visualization tasks.

- CAMDA-DILI · [GitHub](https://github.com/anikaliu/CAMDA-DILI)
  - description: Code for DILI prediction; used in chemistry tasks on toxicity modeling.

- eofs · [GitHub](https://github.com/ajdawson/eofs)
  - description: EOF analysis library for climate data; used in geoscience tasks for dimensionality reduction/analysis.

- rasterio · [GitHub](https://github.com/rasterio/rasterio)
  - description: Geospatial raster I/O; used in GIS tasks for reading and processing raster layers.

- Matminer · [GitHub](https://github.com/hackingmaterials/matminer)
  - description: Materials data mining toolkit; adapted for materials feature engineering and benchmarking tasks.

- Deep Sea Corals (NOAA) dataset · [Website](https://www.gbif.org/dataset/df8e3fb8-3da7-4104-a866-748f6da20a3c)
  - description: Public dataset referenced for GIS/biology mapping tasks (e.g., coral and sponge distribution); used as real data in benchmark tasks.

- Solve-Geosolutions transform 2022
  - description: Repository listed among adapted sources for GIS-related tasks (CC-BY-3.0-AU); used to build geospatial processing workflows.

<!-- paper_id: 45b7c7448768b51b6dbd9b76495c9cd9d110bd91 -->

## 5. Learning to Use Tools via Cooperative and Interactive Agents - EMNLP - 2024 - citation_count 41 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_41_Findings_Learning_to_Use_Tools_via_Cooperative_and_Interactive_Agents.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.624/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_41_Findings_Learning_to_Use_Tools_via_Cooperative_and_Interactive_Agents.pdf

### GitHub & Websites

- ToolBench / ToolLLM · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolllm.github.io)
  - description: Primary dataset and baseline framework used in the paper. The authors sample tasks for SPAN distillation and evaluation, and compare against ToolLLM’s DFSDT decision module.

- RestGPT / RestBench · [GitHub](https://github.com/Yifan-Song793/RestGPT)
  - description: Benchmark provider for two subsets (TMDB and Spotify) used in experiments; also a multi-agent baseline compared by the authors.

- The Movie Database (TMDB) API · [Website](https://www.themoviedb.org) · [Doc](https://developer.themoviedb.org/docs)
  - description: Tool/APIs invoked in RestBench-TMDB; the paper’s agents execute these endpoints during evaluation.

- Spotify Web API · [Website](https://developer.spotify.com) · [Doc](https://developer.spotify.com/documentation/web-api)
  - description: Tool/APIs invoked in RestBench-Spotify; used by the agents for music-related tasks.

- RapidAPI · [Website](https://rapidapi.com) · [Doc](https://docs.rapidapi.com/)
  - description: API platform cited as a source of tools used in experiments, ensuring reproducible access to third-party APIs.

- OpenAI GPT-3.5 Turbo / GPT-4 · [Website](https://openai.com/chatgpt) · [Doc](https://platform.openai.com/docs/models)
  - description: GPT-3.5-turbo is the main backbone for experiments; GPT-4 is used to synthesize specialized trajectories for the SPAN distillation dataset.

- Mixtral-8x7B (Mistral) · [GitHub](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) · [Website](https://huggingface.co/mistralai)
  - description: Open-source LLM backbone used for the paper’s open-source experiments and SPAN instruction tuning.

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient tuning method employed to train the three specialized agents during SPAN distillation.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Single-agent baseline prompting method (reasoning + acting) against which CONAGENTS is compared; also run with multiple attempts (ReAct@N).

- Chameleon (Plug-and-Play Compositional Reasoning) · [GitHub](https://github.com/lupantech/Chameleon)
  - description: Baseline system that plans multi-step tool use; included for comparison in the experiments.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Multi-agent/self-reflection baseline used for comparison; involves an executor LLM with another LLM providing feedback.

<!-- paper_id: 966ba2acfe0700c2410efe15ed1b6c25340b7a95 -->

## 6. Multi-agent Architecture Search via Agentic Supernet - ICML - 2025 - citation_count 51 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICML_51_Oral_Multi-agent_Architecture_Search_via_Agentic_Supernet.pdf
- Link: https://openreview.net/pdf/46a781e93da44fdacc085588e4eeb8edc5f20439.pdf
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICML_51_Oral_Multi-agent_Architecture_Search_via_Agentic_Supernet.pdf

### GitHub & Websites

- MaAS (Multi-agent Architecture Search) · [GitHub](https://github.com/bingreeky/MaAS)
  - description: Official code release of the paper; implements the agentic supernet, controller, sampling, and training/evaluation scripts used across all experiments.

- ADAS (Automated Design of Agentic Systems) · [GitHub](https://github.com/ShengranHu/ADAS)
  - description: Baseline/related toolkit; the paper adapts part of ADAS’s textual-gradient implementation for operator updates and compares against ADAS in experiments.

- AgentSquare · [GitHub](https://github.com/tsinghua-fib-lab/agentsquare)
  - description: Baseline for agent search; the paper adapts code snippets from this repo for textual gradient and includes AgentSquare as a comparison baseline.

- GPTSwarm · [GitHub](https://github.com/SchmidhuberAI/GPTSwarm)
  - description: Multi-agent optimization framework used as an automated baseline in experiments.

- AutoAgents · [GitHub](https://github.com/Link-AGI/AutoAgents)
  - description: Automatic multi-agent generation framework; used as a comparison baseline.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Multi-agent collaboration framework; included as a hand-crafted multi-agent baseline.

- LLM-Debate · [GitHub](https://github.com/ucl-dark/llm_debate)
  - description: Debate-style multi-agent method; used as both an operator concept and a baseline implementation.

- ComplexCoT (Complexity-Based Prompting) · [GitHub](https://github.com/FranxYao/Complexity-Based-Prompting)
  - description: Baseline for prompting; the paper follows its official implementation for ComplexCoT.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting operator referenced as a building block in the MaAS operator set.

- Auto-GPT · [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
  - description: Autonomous agent baseline evaluated on GAIA in Table 2.

- TapeAgents · [GitHub](https://github.com/ServiceNow/TapeAgents)
  - description: Agent development and optimization framework; used as a baseline on GAIA.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Grade-school math reasoning benchmark; used for evaluation (train/test split 1:4).

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: Mathematical problem-solving benchmark; used for evaluation and cost analysis.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code generation benchmark; used to measure pass@1 and to visualize sampled workflows.

- MBPP · [GitHub](https://github.com/google-research/google-research/tree/master/mbpp)
  - description: Python program synthesis benchmark; used to evaluate code generation performance.

- GAIA (General AI Assistant Benchmark) · [GitHub](https://github.com/gaia-benchmark/GAIA) · [Website](https://huggingface.co/datasets/gaia-benchmark/GAIA)
  - description: Multi-domain tool-use benchmark (web, files, multimodal); main benchmark for tool-use experiments and baselines.

- MiniLM · [GitHub](https://github.com/microsoft/unilm/tree/master/minilm)
  - description: Lightweight text embedding model used to embed queries/operators for the controller’s scoring in MaAS.

- Sentence-BERT · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net/)
  - description: Sentence embedding toolkit mentioned as an alternative lightweight encoder for the controller’s embedding function.

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.ai/)
  - description: Open-source LLM backbone (Qwen-2.5-72B-Instruct) accessed via API; used for evaluations and transfer experiments.

- Llama 3.1 · [GitHub](https://github.com/meta-llama/llama-models) · [Website](https://www.llama.com/)
  - description: Open-source LLM backbone (llama-3.1-70B) used in cross-model evaluations and transferability tests.

- GPT-4o mini · [Website](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence) · [Doc](https://platform.openai.com/docs/models#gpt-4o-mini)
  - description: Primary closed-source LLM backbone accessed via API for all baselines and MaAS experiments.

<!-- paper_id: 01f257a840d271d766d741f911bb1078240fbbdf -->

## 7. MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents - ACL - 2025 - citation_count 41 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_41_Long_MultiAgentBench_Evaluating_the_Collaboration_and_Competition_of_LLM_agents.pdf
- Link: https://aclanthology.org/2025.acl-long.421/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_41_Long_MultiAgentBench_Evaluating_the_Collaboration_and_Competition_of_LLM_agents.pdf

### GitHub & Websites

- MARBLE (MultiAgentBench) · [GitHub](https://github.com/ulab-uiuc/MARBLE)
  - description: Official codebase and datasets for the paper’s benchmark and multi-agent coordination framework; required to reproduce all experiments across research, Minecraft, database, coding, bargaining, and werewolf scenarios.

- Mineflayer · [GitHub](https://github.com/PrismarineJS/mineflayer)
  - description: Node.js bot framework for Minecraft; used as the engine enabling text-based interaction and tool execution in the paper’s Minecraft environment.

- OpenAI API (Function Calling) · [Website](https://www.openai.com) · [Doc](https://platform.openai.com/docs/guides/gpt/function-calling)
  - description: GPT-3.5-turbo-0125 and GPT-4o-mini are used as agent backends with function-calling to operate tools and environments; this documentation is needed to replicate model-tool interactions.

- Together AI Inference · [Website](https://www.together.ai)
  - description: Hosting/provider used to access open-source Llama 3.x models for agent runs; reproducing the setup requires an inference endpoint like Together.

- PostgreSQL · [Website](https://www.postgresql.org) · [Doc](https://www.postgresql.org/docs/)
  - description: Core database engine for the Database Environment; agents diagnose root causes by querying PostgreSQL system views.

- Docker · [Website](https://www.docker.com) · [Doc](https://docs.docker.com/)
  - description: Used to run the PostgreSQL database for the Database Environment in a reproducible containerized setup.

- Amazon Products Dataset 2023 (1.4M products) · [Website](https://www.kaggle.com/datasets/asaniczka/amazon-products-dataset-2023)
  - description: Real-world product dataset used to construct 100 bargaining scenarios (price, rating, etc.); required to recreate the bargaining environment inputs.

- ResearchTown
  - description: Simulator/dataset source for research collaboration tasks; the benchmark selects 100 ML/AI papers and author profiles from ResearchTown to form the Research scenario.

- VillagerAgent
  - description: Multi-agent Minecraft framework from which the paper adapts its Minecraft environment and the 100 target structures used for evaluation.

<!-- paper_id: 8998b3895cce8b206098197478d3d9bc3add321c -->

## 8. BLADE: Benchmarking Language Model Agents for Data-Driven Science - EMNLP - 2024 - citation_count 35 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_35_Findings_BLADE_Benchmarking_Language_Model_Agents_for_Data-Driven_Science.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.815/
- Tags: multiagent, tool, science, agent, biology, protein-function, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_35_Findings_BLADE_Benchmarking_Language_Model_Agents_for_Data-Driven_Science.pdf

### GitHub & Websites

- BLADE · [GitHub](https://github.com/behavioral-data/BLADE)
  - description: Official repository for the paper’s benchmark, including the 12 datasets, expert-annotated ground-truth decision space, prompts, automatic evaluation modules, and baseline agent code to reproduce and extend results.

- Arquero · [GitHub](https://github.com/uwdata/arquero) · [Website](https://idl.uw.edu/arquero/)
  - description: JavaScript data-wrangling library whose verb taxonomy informed BLADE’s transform-verb set and graph-based matching for data transformations.

- Vega-Lite · [GitHub](https://github.com/vega/vega-lite) · [Website](https://vega.github.io/vega-lite/)
  - description: Grammar of interactive graphics cited as a basis for transform abstractions; BLADE’s transformation representation references verbs/concepts aligned with Vega/Vega-Lite.

- ReAct (Reason + Act) · [GitHub](https://github.com/ysymyth/ReAct) · [Website](https://react-lm.github.io/)
  - description: Agent framework used to build the paper’s iterative notebook-interacting baseline agent for analysis generation and exploration.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Doc](https://python.langchain.com/v0.1/docs/)
  - description: Used for structured outputs via the Pydantic output parser when collecting and evaluating agent generations (e.g., JSON schemas for conceptual variables and transforms).

- LIDA (Language Interface for Data Analysis) · [Doc](https://arxiv.org/abs/2303.02927)
  - description: The paper uses LIDA’s data summarizer to generate dataset schema JSONs provided to models during prompting.

- statsmodels · [GitHub](https://github.com/statsmodels/statsmodels) · [Doc](https://www.statsmodels.org/)
  - description: Statistical modeling library (e.g., OLS, GLM/logit) imported in the sandbox environment and used in the example/modeling code for agents.

- pandas · [GitHub](https://github.com/pandas-dev/pandas) · [Codewiki](https://codewiki.google/github.com/pandas-dev/pandas) · [Doc](https://pandas.pydata.org/)
  - description: Core data-frame and transformation library used throughout agent transform code and the evaluation environment.

- NumPy · [GitHub](https://github.com/numpy/numpy) · [Doc](https://numpy.org/)
  - description: Numerical computing dependency used in transformation/modeling code within the sandbox.

- SciPy · [GitHub](https://github.com/scipy/scipy) · [Codewiki](https://codewiki.google/github.com/scipy/scipy) · [Doc](https://scipy.org/)
  - description: Scientific computing routines imported in the notebook environment supporting analysis.

- scikit-learn · [GitHub](https://github.com/scikit-learn/scikit-learn) · [Codewiki](https://codewiki.google/github.com/scikit-learn/scikit-learn) · [Doc](https://scikit-learn.org/)
  - description: ML utilities imported in the sandbox environment available to agents during analysis.

- Matplotlib · [GitHub](https://github.com/matplotlib/matplotlib) · [Doc](https://matplotlib.org/)
  - description: Plotting library imported in the environment for exploratory analysis steps during agent runs.

- Seaborn · [GitHub](https://github.com/mwaskom/seaborn) · [Codewiki](https://codewiki.google/github.com/mwaskom/seaborn) · [Doc](https://seaborn.pydata.org/)
  - description: Statistical visualization library imported in the environment for data exploration during agent analysis.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code-generation benchmark used by the paper for cross-benchmark comparison with BLADE to analyze correlations between coding ability and analysis-generation performance.

<!-- paper_id: 3cabe88c2ff476d35cd179525c7dfe474b4b75bb -->

## 9. Internet of Agents: Weaving a Web of Heterogeneous Agents for Collaborative Intelligence - ICLR - 2025 - citation_count 56 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_56_Spotlight_Internet_of_Agents_Weaving_a_Web_of_Heterogeneous_Agents_for_Collaborative_Intelligence.pdf
- Link: https://openreview.net/pdf/1006483e763807a740f78d0096898fc8d8a8424b.pdf
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_56_Spotlight_Internet_of_Agents_Weaving_a_Web_of_Heterogeneous_Agents_for_Collaborative_Intelligence.pdf

### GitHub & Websites

- Internet of Agents (IoA) · [GitHub](https://github.com/OpenBMB/IoA)
  - description: Official code release of the paper’s framework, including the server/client implementations, message protocol, team formation, and conversation flow control used in all experiments.

- AutoGPT · [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
  - description: Third-party autonomous agent integrated into IoA and also used as a baseline; IoA orchestrates AutoGPT within teams on open-ended tasks.

- Open Interpreter · [GitHub](https://github.com/OpenInterpreter/open-interpreter)
  - description: Third-party agent integrated into IoA and used as a baseline; IoA coordinates it for tool-use and coding in the open-ended instruction benchmark.

- AutoGen · [GitHub](https://github.com/microsoft/autogen)
  - description: MAS framework used as a comparison baseline, and source of the Web Browser and Code Executor tools adapted for IoA’s GAIA setup.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Doc](https://python.langchain.com/v0.1/docs/integrations/tools/wikidata/)
  - description: IoA adapts LangChain’s Wikidata tool to provide a Wikidata searcher agent for GAIA tasks.

- PyTube · [GitHub](https://github.com/pytube/pytube)
  - description: Used in IoA’s GAIA configuration to download YouTube video transcripts as part of the YouTube agent’s toolset.

- Whisper · [GitHub](https://github.com/openai/whisper) · [Codewiki](https://codewiki.google/github.com/openai/whisper)
  - description: Employed when YouTube videos lack transcripts; IoA’s YouTube tool uses Whisper to transcribe audio to text.

- Milvus · [GitHub](https://github.com/milvus-io/milvus) · [Codewiki](https://codewiki.google/github.com/milvus-io/milvus) · [Website](https://milvus.io/)
  - description: Vector database used by the IoA server’s Agent Registry for similarity-based agent discovery.

- SQLite · [Website](https://www.sqlite.org/)
  - description: Lightweight database used by IoA clients to store agent contacts, group info, and task management data locally.

- Pyserini · [GitHub](https://github.com/castorini/pyserini) · [Website](https://pyserini.io/)
  - description: Retrieval toolkit used to access a pre-built Wikipedia index for IoA’s RAG experiments.

- GAIA: A Benchmark for General AI Assistants · [Website](https://doi.org/10.48550/arXiv.2311.12983)
  - description: Benchmark used to evaluate IoA on heterogeneous tool use; IoA integrates ReAct-style agents (browser, code, Wikidata, YouTube) to solve GAIA tasks.

- RoCoBench (RoCo: Dialectic Multi-Robot Collaboration) · [Website](https://doi.org/10.48550/arXiv.2307.04738)
  - description: Embodied multi-agent benchmark used to assess IoA with heterogeneous observation/action spaces; IoA clients output action plans in the required format.

- Apollo’s Oracle: Retrieval-Augmented Reasoning in Multi-Agent Debates · [Website](https://api.semanticscholar.org/CorpusID:266149845)
  - description: RAG baseline referenced for settings and comparison; IoA’s RAG experiments include homogeneous/heterogeneous evidence configurations following this line of work.

- TriviaQA · [Website](http://nlp.cs.washington.edu/triviaqa/)
  - description: QA dataset used in IoA’s RAG evaluations to measure answer accuracy with retrieval coordination.

- Natural Questions (NQ) · [GitHub](https://github.com/google-research-datasets/natural-questions)
  - description: Open-domain QA dataset used in RAG experiments to evaluate IoA’s multi-agent retrieval and synthesis.

- HotpotQA · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used to test IoA’s collaborative retrieval and reasoning.

- 2WikiMultiHopQA · [GitHub](https://github.com/Alab-NII/2WikiMultiHopQA)
  - description: Multi-hop QA dataset used in IoA’s RAG experiments to assess reasoning over multiple pieces of evidence.

- The Alignment Handbook · [GitHub](https://github.com/huggingface/alignment-handbook)
  - description: Training recipes used to fine-tune Llama 3 8B as IoA’s communication LLM (following the SFT config referenced in the paper).

- Llama 3 · [GitHub](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)
  - description: Base model whose 8B variant was fine-tuned to act as IoA’s communication layer LLM in additional experiments.

<!-- paper_id: 3eb714c5e226d8d223f80b135cf5bbd9f8c37e02 -->

## 10. EvoAgent: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms - NAACL - 2025 - citation_count 50 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NAACL_50_Long_EvoAgent_Towards_Automatic_Multi-Agent_Generation_via_Evolutionary_Algorithms.pdf
- Link: https://aclanthology.org/2025.naacl-long.315/
- Tags: multiagent, tool, science, agent, biology, protein-function, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NAACL_50_Long_EvoAgent_Towards_Automatic_Multi-Agent_Generation_via_Evolutionary_Algorithms.pdf

### GitHub & Websites

- EvoAgent · [Website](https://evo-agent.github.io)
  - description: Official project page for the paper; hosts resources to reproduce the evolutionary multi‑agent system (prompts, examples, and links referenced as “Resources are available at https://evo-agent.github.io/.”).

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT)
  - description: Multi‑agent collaboration framework the authors cite as an initial agent framework; EVOAGENT can start from MetaGPT and automatically expand it to a multi‑agent system (Appendix D).

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen)
  - description: Microsoft’s multi‑agent conversation framework used as an initial agent framework that EVOAGENT can extend; also shown in the paper’s adaptation examples (Appendix D).

- CAMEL · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel)
  - description: Communicative role‑playing agents framework; the paper demonstrates EVOAGENT can automatically generate roles to use within CAMEL instead of manual role design (Appendix D).

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Baseline multi‑agent framework compared against EVOAGENT in Table 1; discussed as a human‑designed pipeline that EVOAGENT seeks to automate.

- AutoAgents · [GitHub](https://github.com/Link-AGI/AutoAgents)
  - description: Baseline automatic agent‑generation framework compared in Table 1; the paper contrasts its fixed human‑designed architecture with EVOAGENT’s EA‑driven generation.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld)
  - description: Interactive scientific reasoning environment used as a benchmark to evaluate EVOAGENT on open‑world multi‑step tasks (§4.2).

- MMMU (Massive Multi‑discipline Multimodal Understanding) · [GitHub](https://github.com/MMMU-Benchmark/MMMU) · [Website](https://mmmu-benchmark.github.io)
  - description: Multimodal benchmark used to test EVOAGENT with GPT‑4V and Gemini‑Pro on multiple‑choice validation questions (§4.1).

- TravelPlanner · [Website](https://arxiv.org/abs/2402.01622)
  - description: Real‑world complex planning benchmark used to evaluate EVOAGENT’s planning performance and constraint satisfaction (§4.3, App. A.5).

- ReAct · [GitHub](https://github.com/ysymyth/react)
  - description: Reasoning‑and‑acting prompting method used in ScienceWorld experiments and as a baseline in TravelPlanner (§4.2, §4.3).

- SGLang · [GitHub](https://github.com/sgl-project/sglang) · [Codewiki](https://codewiki.google/github.com/sgl-project/sglang)
  - description: Fast LLM serving framework used for time‑cost measurements when running Llama‑3.1‑70B‑Instruct in the paper’s efficiency analysis (App. B.1).

- Azure OpenAI Service (GPT‑3.5, GPT‑4) · [Website](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
  - description: API endpoint the authors used to run GPT‑3.5 and GPT‑4 (version 2024‑02‑15‑preview) for experiments (App. A.2).

- Google Gemini API (Gemini‑Pro) · [Doc](https://ai.google.dev/gemini-api)
  - description: API used to obtain Gemini‑Pro results in the paper’s experiments (App. A.2).

- Llama 2 · [Website](https://ai.meta.com/llama/)
  - description: Open LLM family; LLama2‑13B‑Chat is used as one of the backbone models for NLP task evaluations (§4.1).

- Llama 3/3.1 (Meta Llama) · [GitHub](https://github.com/meta-llama/llama)
  - description: The time‑efficiency study deploys Llama‑3.1‑70B‑Instruct with SGLang to compare single‑ vs multi‑agent runtimes (App. B.1).

- Mistral‑7B · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - description: Backbone model used as one setting in the TravelPlanner evaluation (§4.3).

<!-- paper_id: b031522b71df27dac92c1c62b97ac1685cefd732 -->

## 11. AgentStudio: A Toolkit for Building General Virtual Agents - ICLR - 2025 - citation_count 32 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_32_Poster_AgentStudio_A_Toolkit_for_Building_General_Virtual_Agents.pdf
- Link: https://openreview.net/pdf/474fb13c084ca1d640c138556e983c3c43addabb.pdf
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_32_Poster_AgentStudio_A_Toolkit_for_Building_General_Virtual_Agents.pdf

### GitHub & Websites

- AgentStudio · [Website](https://ltzheng.github.io/agent-studio)
  - description: Official project page for the paper; hosts the toolkit (environment + tools), online benchmark tasks, and links to datasets (GroundUI, IDMBench, CriticBench) needed to reproduce and extend the work.

- GroundUI (dataset) · [Website](https://ltzheng.github.io/agent-studio)
  - description: Paper’s UI-grounding dataset (18K samples; 1K eval subset) curated via AgentStudio tools and reorganized from prior sources; used to evaluate single-step GUI grounding and localization.

- IDMBench (dataset) · [Website](https://ltzheng.github.io/agent-studio)
  - description: Paper’s benchmark for labeling actions from videos (IDM-Single and IDM-Multiple) built from Mind2Web, AITW, VisualWebArena, and AgentStudio trajectories; evaluates learning-from-video ability.

- CriticBench (dataset) · [Website](https://ltzheng.github.io/agent-studio)
  - description: Paper’s benchmark for success detection on multi-step trajectories across devices; supports development of general critic models and auto-evaluation.

- Google Workspace APIs · [Doc](https://developers.google.com/workspace)
  - description: Official API documentation; the AgentStudio environment and tasks require calling Google services (Docs, Drive, Gmail, Sheets, Slides, Calendar, Forms) via credentials and tokens as shown in the paper’s system prompt.

- Visual Studio Code (VS Code) · [Website](https://code.visualstudio.com) · [Doc](https://code.visualstudio.com/docs)
  - description: Used as a target application in AgentStudio’s benchmark tasks (e.g., opening files, editing settings) for GUI/API interactions.

- LibreOffice · [Website](https://www.libreoffice.org) · [Doc](https://help.libreoffice.org)
  - description: Open-source office suite used to create/evaluate spreadsheet, document, and presentation tasks in the benchmark.

- GIMP · [Website](https://www.gimp.org) · [Doc](https://www.gimp.org/docs/)
  - description: Open-source image editor used in AgentStudio’s GUI benchmarks to test agents on professional image-editing operations.

- Docker · [Doc](https://docs.docker.com)
  - description: The paper’s implementation runs the real-computer environment in Docker for a lightweight, reproducible setup.

- Mind2Web · [GitHub](https://github.com/OSU-NLP/Mind2Web)
  - description: Web task dataset/environment used by the paper to source trajectories for IDMBench and CriticBench.

- AndroidEnv · [GitHub](https://github.com/google-deepmind/android_env)
  - description: Android RL/emulation environment cited as a related platform; relevant for extending AgentStudio-style agents to mobile settings.

- AndroidWorld · [GitHub](https://github.com/google-deepmind/android-world)
  - description: Dynamic Android benchmarking environment referenced in the paper’s related work; useful comparison/extension target for mobile agents.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Prior web interaction environment benchmark referenced as a domain-specific baseline; contrasts with AgentStudio’s broader action/observation spaces.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Realistic web environment referenced for comparison; AgentStudio extends beyond web-only action spaces.

- SeeClick · [GitHub](https://github.com/TencentARC/SeeClick)
  - description: Specialized GUI grounding model evaluated on GroundUI; a strong open-source baseline practitioners can inspect for improving UI localization.

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent)
  - description: Open-source VLM agent baseline evaluated on GroundUI; relevant for reproducing baseline results and comparisons.

- CogVLM2 · [GitHub](https://github.com/THUDM/CogVLM2)
  - description: Open-source VLM (including Llama3-chat-19B variant) evaluated on GroundUI; a baseline implementation for UI grounding comparisons.

- MiniCPM-Llama3-V (MiniCPM-V) · [GitHub](https://github.com/OpenBMB/MiniCPM-V) · [Codewiki](https://codewiki.google/github.com/OpenBMB/MiniCPM-V)
  - description: Open-source vision-language baseline evaluated on GroundUI; relevant for reproducing model comparisons.

- Qwen-VL-Chat · [GitHub](https://github.com/QwenLM/Qwen-VL)
  - description: Open-source vision-language baseline evaluated on GroundUI, IDMBench, and CriticBench; useful for extending experiments with open models.

<!-- paper_id: 8a54f8ae4fcacc19f8f999229c0f5abc0167f4f2 -->

## 12. Improving Multi-Agent Debate with Sparse Communication Topology - EMNLP - 2024 - citation_count 55 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_55_Findings_Improving_Multi-Agent_Debate_with_Sparse_Communication_Topology.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.427/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_55_Findings_Improving_Multi-Agent_Debate_with_Sparse_Communication_Topology.pdf

### GitHub & Websites

- MATH (Measuring Mathematical Problem Solving) · [GitHub](https://github.com/hendrycks/math) · [Dataset](https://huggingface.co/datasets/hendrycks/Math)
  - description: Challenging competition-style math problems; the paper evaluates MAD variants on the algebra linear 1d composed sub-task.

- GSM8K (Grade School Math) · [GitHub](https://github.com/openai/grade-school-math) · [Website](https://openai.com/research/gsm8k) · [Dataset](https://huggingface.co/datasets/gsm8k)
  - description: Grade school math word problems; used as a primary text reasoning benchmark for comparing sparse vs. fully-connected MAD and baselines.

- MathVista · [GitHub](https://github.com/lupantech/MathVista) · [Website](https://mathvista.github.io/)
  - description: Visual mathematical reasoning benchmark; used to test multimodal MAD (with GPT-4o) and measure accuracy/cost trade-offs under different topologies.

- Anthropic HH-RLHF (Helpful–Harmless) · [Dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf)
  - description: Human preference data for helpfulness and harmlessness; used for alignment labeling experiments and AI Labeler Alignment evaluation.

- OpenAI API (GPT-3.5, GPT-4o) · [Website](https://platform.openai.com/docs) · [Doc](https://platform.openai.com/docs/models/gpt-3-5-turbo) · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Models and API used to instantiate agents for text (GPT-3.5) and multimodal (GPT-4o) reasoning within the MAD framework.

- Mistral 7B · [GitHub](https://github.com/mistralai) · [Website](https://mistral.ai) · [Model Card](https://huggingface.co/mistralai/Mistral-7B-v0.1)
  - description: Open LLM used to instantiate agents for alignment labeling tasks; the paper compares performance/cost of MAD topologies using this model.

<!-- paper_id: c04e6bab808c0d5f0344159190ed21a20039c4e9 -->

## 13. Small LLMs Are Weak Tool Learners: A Multi-LLM Agent - EMNLP - 2024 - citation_count 82 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_82_Main_Small_LLMs_Are_Weak_Tool_Learners_A_Multi-LLM_Agent.pdf
- Link: https://aclanthology.org/2024.emnlp-main.929/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_82_Main_Small_LLMs_Are_Weak_Tool_Learners_A_Multi-LLM_Agent.pdf

### GitHub & Websites

- Multi-LLM-Agent (α-UMi) · [GitHub](https://github.com/X-PLUG/Multi-LLM-Agent)
  - description: Official code release for the paper’s multi-LLM agent with GLPFT training, prompts for planner/caller/summarizer, and evaluation scripts.

- ToolBench / ToolLLaMA / ToolEval · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://openbmb.github.io/ToolBench/)
  - description: Benchmark, datasets, baselines (ToolLLaMA), and evaluation toolkit (including ReAct and DFSDT/ToolEval) for API-tool use; the paper trains and evaluates α-UMi on ToolBench and uses its real-time and static evaluation protocols.

- RapidAPI Hub · [Website](https://rapidapi.com/hub) · [Doc](https://docs.rapidapi.com/)
  - description: Marketplace of real-world APIs used by ToolBench; the paper’s real-time evaluation executes API calls via RapidAPI.

- ToolAlpaca · [GitHub](https://github.com/thunlp/ToolAlpaca)
  - description: Simulated tool-use benchmark with 3k cases; used as an additional evaluation set in the paper.

- MATH dataset · [GitHub](https://github.com/hendrycks/math)
  - description: Math problem benchmark; the paper collects trajectories (via GPT-3.5/4) for training and evaluates α-UMi with a program-aided agent on MATH.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math) · [Doc](https://huggingface.co/datasets/gsm8k)
  - description: Grade school math word problems; used to collect training trajectories and evaluate α-UMi’s program-aided math reasoning.

- Llama 2 (LLaMA-2-chat-7B/13B) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama)
  - description: Backbone open-source chat models used to instantiate the planner, caller, and summarizer and for single-LLM baselines.

- DeepSpeed (ZeRO Stage 3) · [GitHub](https://github.com/microsoft/DeepSpeed) · [Website](https://www.deepspeed.ai/)
  - description: Distributed training library; the paper uses ZeRO Stage 3 to speed up fine-tuning of the backbones.

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient fine-tuning method; used for the Multi-LLMone-stage (LoRA) and α-UMi (LoRA) baselines compared against full fine-tuning.

- ReAct (Reason + Act) · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-acting agent prompting framework; the paper builds agents with ReAct for ToolBench real-time evaluations.

- ModelScope-Agent · [GitHub](https://github.com/modelscope/modelscope-agent) · [Website](https://modelscope.cn/)
  - description: Open-source agent framework; the paper adopts its static evaluation idea to compare model outputs with annotated references for ToolBench.

- OpenAI API (GPT-3.5/GPT-4) · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Proprietary LLMs used as baselines and to generate execution trajectories for MATH/GSM8K training data.

- Claude 2 (Anthropic) · [Website](https://www.anthropic.com/claude)
  - description: Proprietary LLM baseline included in the paper’s real-time ToolBench evaluation.

<!-- paper_id: ff61aef2fef3a235bfaa123158a990c4f5f27d1a -->

## 14. MMedAgent: Learning to Use Medical Tools with Multi-modal Agent - EMNLP - 2024 - citation_count 63 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/agent-tools/2024_EMNLP_63_Findings_MMedAgent_Learning_to_Use_Medical_Tools_with_Multi-modal_Agent.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.510/
- Tags: multiagent, tool, science, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/agent-tools/2024_EMNLP_63_Findings_MMedAgent_Learning_to_Use_Medical_Tools_with_Multi-modal_Agent.pdf

### GitHub & Websites

- MMedAgent · [GitHub](https://github.com/Wangyixinxin/MMedAgent)
  - description: Official code and web UI for the paper’s multi-modal medical agent, including the instruction-tuning data and tool integration needed to reproduce the system.

- LLaVA-Med · [GitHub](https://github.com/microsoft/LLaVA-Med)
  - description: Backbone MLLM used by MMedAgent for VQA and dialog; the agent is initialized from LLaVA‑Med 60K-IM and further instruction-tuned.

- Grounding DINO · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
  - description: Open-set detector that the authors fine-tune to create a medical grounding tool for object detection/localization across modalities.

- MedSAM (Segment Anything in Medical Images) · [GitHub](https://github.com/bowang-lab/MedSAM)
  - description: Medical adaptation of SAM used as the segmentation tool (box-prompted and text-grounded with Grounding DINO + MedSAM) inside MMedAgent.

- BiomedCLIP · [GitHub](https://github.com/microsoft/BiomedCLIP)
  - description: Vision–language model employed by MMedAgent for closed-set medical image classification via image–text similarity.

- ChatCAD · [Website](https://arxiv.org/abs/2302.07257)
  - description: Open-source tool for chest X‑ray medical report generation; integrated by MMedAgent as the MRG component.

- ChatCAD+ · [Website](https://arxiv.org/abs/2406.20015)
  - description: Retrieval-augmented medical assistant used by MMedAgent for RAG; retrieves from a medical dictionary (Merck Manual) to enrich responses.

- FastChat · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat)
  - description: Serving framework used to host the agent, tools, and web UI for interactive demos in the paper.

- FLARE2021 (Fast and Low GPU Abdomen CT Organ Segmentation Challenge) · [Website](https://flare21.grand-challenge.org/)
  - description: Abdomen CT segmentation dataset used to prepare training data (converted to boxes) for fine-tuning the medical Grounding DINO.

- WORD (Abdominal Organ Segmentation from CT) · [GitHub](https://github.com/HiLab-git/WORD)
  - description: Large-scale CT organ segmentation dataset used to generate grounding labels and to evaluate segmentation-related tools.

- BRATS (Brain Tumor Segmentation) · [Website](https://www.med.upenn.edu/cbica/brats2020/)
  - description: MRI brain tumor segmentation dataset whose masks are converted to bounding boxes to train the medical grounding tool.

- Montgomery County X-ray Set (NLM TB Chest X-rays) · [Website](https://ceb.nlm.nih.gov/repositories/tuberculosis-chest-x-ray-image-data-sets/)
  - description: Chest X-ray dataset used to derive detection labels for medical grounding and related tasks.

- VinDr-CXR · [Website](https://physionet.org/content/vindr-cxr/1.0.0/)
  - description: Annotated chest X‑ray dataset providing findings/lesions; used to build box labels for the grounding tool.

- CellSeg (Multi-modality Cell Segmentation Challenge) · [Website](https://cellseg.grand-challenge.org/)
  - description: Histology/cell segmentation benchmark used to create training labels (converted to boxes) for the medical grounding model.

- MIMIC‑CXR-JPG · [Website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
  - description: Large public chest X‑ray dataset used by ChatCAD; supports the paper’s medical report generation component.

- Merck Manual (Professional Edition) · [Website](https://www.merckmanuals.com/professional)
  - description: External medical knowledge base used by ChatCAD+ for retrieval-augmented generation within MMedAgent.

- MS COCO · [Website](https://cocodataset.org/)
  - description: Natural image dataset included during Grounding DINO fine-tuning to preserve general object detection ability.

- Flickr30k Entities · [GitHub](https://github.com/BryanPlummer/flickr30k_entities) · [Website](https://bryanplummer.com/Flickr30kEntities/)
  - description: Phrase-to-region grounding dataset used alongside COCO when fine-tuning Grounding DINO to retain open-set grounding skills.

- LLaVA · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA)
  - description: General MLLM baseline compared against MMedAgent; also informs the instruction format for tool use via prior LLaVA work.

- Qwen-VL-Chat · [GitHub](https://github.com/QwenLM/Qwen-VL)
  - description: Open-source vision–language chat model used as a strong baseline in experiments.

- Yi (01.ai) · [GitHub](https://github.com/01-ai/Yi)
  - description: Open foundation model family providing the Yi‑VL‑34B baseline evaluated against MMedAgent.

- Med-Flamingo · [GitHub](https://github.com/snap-stanford/med-flamingo)
  - description: Multimodal medical few-shot learner used as a comparison baseline on the evaluation suite.

- RadFM · [GitHub](https://github.com/chaoyi-wu/RadFM)
  - description: Generalist radiology foundation model serving as a baseline for several tasks in the paper.

<!-- paper_id: 8d36a67a6d7571f4dfd88c8f2df21f97de10eb4d -->

## 15. MMInA: Benchmarking Multihop Multimodal Internet Agents - ACL - 2025 - citation_count 30 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ACL_30_Findings_MMInA_Benchmarking_Multihop_Multimodal_Internet_Agents.pdf
- Link: https://aclanthology.org/2025.findings-acl.703/
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ACL_30_Findings_MMInA_Benchmarking_Multihop_Multimodal_Internet_Agents.pdf
- Token Usage: input 19585, output 5796, total 25381

### GitHub & Websites

- MMInA · [GitHub](https://github.com/shulin16/MMInA)
  - description: Official code and data release for the MMInA benchmark; contains tasks, evaluation protocol, and scripts to run and assess multihop multimodal Internet agents.

- VisualWebArena (VWA) · [GitHub](https://github.com/web-arena-x/visualwebarena) · [Website](https://visualwebarena.github.io)
  - description: Realistic visual web-browsing benchmark; MMInA follows VWA’s condensed action space and multimodal web interaction setup.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Suite of realistic web environments; MMInA adopts WebArena-style browser/display settings and evaluation practices for single-hop tasks.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Simulated e-commerce environment and agent baseline; used in MMInA as a heuristic web-agent baseline (queries adapted via GPT-3.5).

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent)
  - description: Visual language model for GUI/web agents; evaluated as a multimodal agent baseline on MMInA.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Doc](https://playwright.dev)
  - description: Browser automation framework; MMInA uses Playwright to render pages, extract accessibility trees, and execute the 12 summarized actions.

- BLIP-2 (via LAVIS) · [GitHub](https://github.com/salesforce/LAVIS) · [Doc](https://lavis.readthedocs.io/en/latest/)
  - description: Vision-language captioning used by MMInA to generate image captions for the “caption-augmented” text-only baselines.

- OpenAI API (GPT-3.5/4/4V/4o) · [Doc](https://platform.openai.com/docs)
  - description: Proprietary LLM/LMM backbones used for reasoning and evaluation (including fuzzy matching with GPT-3.5) in MMInA experiments.

- Google Gemini API (Gemini-Pro / Gemini-Pro-Vision) · [Doc](https://ai.google.dev/gemini-api)
  - description: Proprietary multimodal models used as agent backbones and in memory-augmented variants in MMInA.

- CodeLLaMA · [GitHub](https://github.com/facebookresearch/codellama)
  - description: Open-source LLM baseline evaluated in MMInA for text-only reasoning on accessibility trees.

- DeepSeek-R1 · [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-R1)
  - description: Open-source reasoning LLM baseline (Distill-Qwen-32B variant used) evaluated on MMInA single- and multi-hop tasks.

- Kiwix (offline Wikipedia access) · [Website](https://www.kiwix.org) · [Website](https://library.kiwix.org/viewer#wikipedia_en_all_maxi_2024-01/A/User%3AThe_other_Kiwix_guy/Landing)
  - description: Offline/packaged Wikipedia used by MMInA to provide a stable Wikipedia browsing source within the evolving-web setup.

- Trip.com Car Hire · [Website](https://www.trip.com/carhire/)
  - description: Real-world website used as a car rental domain in MMInA multihop tasks.

- Momondo · [Website](https://www.momondo.com/)
  - description: Real-world flight-search website used for flight booking hops in MMInA.

- Trip.com Hotels · [Website](https://www.trip.com/hotels/)
  - description: Real-world hotel booking website used for accommodation-related hops in MMInA.

- Eventbrite · [Website](https://www.eventbrite.com/)
  - description: Real-world event search site used in MMInA tasks requiring event discovery.

- Twitter · [Website](https://twitter.com/home)
  - description: Social media site included among MMInA’s real websites for information-seeking hops.

- Amazon · [Website](https://www.amazon.com/)
  - description: E-commerce site used in MMInA for shopping-related actions and comparisons.

- YouTube · [Website](https://www.youtube.com/)
  - description: Video platform used in MMInA for hops involving video search and viewing.

- Time Out · [Website](https://www.timeout.com/)
  - description: Local discovery/food guide website used in MMInA for finding restaurants and attractions.

- XE Currency Converter · [Website](https://www.xe.com/)
  - description: Currency conversion site used in MMInA to support finance-related hops (e.g., exchange rates).

- Nomadic Matt · [Website](https://www.nomadicmatt.com)
  - description: Travel guide site used in MMInA tasks for destination research and planning.

- Allrecipes · [Website](https://www.allrecipes.com/)
  - description: Recipe website used in MMInA for cooking/recipe retrieval hops.

- Trip.com Trains · [Website](https://www.trip.com/trains/)
  - description: Train booking site used in MMInA to cover transportation planning beyond flights.

- OneStopMarket (offline standalone website)
  - description: An offline shopping site used in MMInA (due to challenges of fetching images from some live pages); included as part of the benchmark’s real/evolving website mix.

<!-- paper_id: 34fc623e43f60b73827f4d0259013b96aeddfe6f -->

## 16. SMART: Self-Aware Agent for Tool Overuse Mitigation - ACL - 2025 - citation_count 25 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_25_Findings_SMART_Self-Aware_Agent_for_Tool_Overuse_Mitigation.pdf
- Link: https://aclanthology.org/2025.findings-acl.239/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_25_Findings_SMART_Self-Aware_Agent_for_Tool_Overuse_Mitigation.pdf
- Token Usage: input 21320, output 5015, total 26335

### GitHub & Websites

- Open-SMARTAgent · [GitHub](https://github.com/qiancheng0/Open-SMARTAgent)
  - description: Official code release for SMART, including SMART-ER dataset, training/inference scripts, and SMARTAgent checkpoints used throughout the paper.

- MATH (Mathematical Reasoning Dataset) · [GitHub](https://github.com/hendrycks/math)
  - description: Source dataset for the Math domain in SMART-ER; provides problems and solutions used to compose tool-free and tool-needed reasoning steps.

- GSM8K (Grade School Math) · [GitHub](https://github.com/openai/grade-school-math)
  - description: Used for preliminary tool-overuse analysis and as an out-of-distribution evaluation benchmark for SMARTAgent.

- FreshQA (FreshLLMs) · [GitHub](https://github.com/google-research/google-research/tree/master/freshqa)
  - description: Provides fast-changing vs. slow-changing factual questions; used to build the Time domain of SMART-ER and evaluate temporal tool use.

- Serper API (Web Search) · [Website](https://serper.dev) · [Doc](https://serper.dev/api)
  - description: Real-time search backend employed for the Search tool in SMART-ER pipelines and SMARTAgent inference.

- OpenAI GPT-4o · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Auxiliary model used for data decomposition/annotation and as a baseline in experiments; also simulates “AskUser” responses in controlled setups.

- Meta Llama 3.1 Instruct Models · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) · [Doc](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  - description: Base backbones fine-tuned to create SMARTAgent variants (8B and 70B) following the SMART training recipe.

- Meta Llama 3.3-70B-Instruct · [Doc](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
  - description: Additional backbone used in Appendix evaluations to compare against the Llama-3.1-70B-based SMARTAgent.

- Mistral Instruct Models · [Website](https://mistral.ai) · [Doc](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) · [Doc](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
  - description: Base backbones (7B, Nemo 12B) fine-tuned to obtain SMARTAgent; also includes the 24B variant used in main results. Additional model card: https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501.

- AgentGPT · [GitHub](https://github.com/reworkd/AgentGPT) · [Website](https://agentgpt.reworkd.ai)
  - description: LM-driven agent baseline used in preliminary tool-overuse experiments to show unnecessary tool invocation behavior.

- XAgent · [GitHub](https://github.com/OpenBMB/XAgent)
  - description: Agent framework evaluated in the preliminary study; used to demonstrate tool overuse patterns on GSM8K-style tasks.

- PEFT (LoRA) · [GitHub](https://github.com/huggingface/peft) · [Codewiki](https://codewiki.google/github.com/huggingface/peft) · [Doc](https://huggingface.co/docs/peft/index)
  - description: Parameter-efficient fine-tuning library used to train SMARTAgent (LoRA rank/alpha settings detailed in the paper).

- Stanford Alpaca (Instruction Format) · [GitHub](https://github.com/tatsu-lab/stanford_alpaca) · [Codewiki](https://codewiki.google/github.com/tatsu-lab/stanford_alpaca)
  - description: The instruction-following (Instruction-Input-Output) format adopted for supervised fine-tuning of SMARTAgent.

- AMC (American Mathematics Competitions) · [Website](https://www.maa.org/math-competitions/amc)
  - description: Additional evaluation benchmark (AMC 2023) used in the appendix to test SMARTAgent on more challenging math reasoning tasks.

<!-- paper_id: e0c33d7096a3767b595d2c3375be6daebd8dbc0d -->

## 17. Agent S: An Open Agentic Framework that Uses Computers Like a Human - ICLR - 2025 - citation_count 85 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-computer-use/2025_ICLR_85_Poster_Agent_S_An_Open_Agentic_Framework_that_Uses_Computers_Like_a_Human.pdf
- Link: https://openreview.net/pdf/5c0a5b17c744fe619f72841610ad18eb2216c723.pdf
- Tags: multiagent, tool, science, agent-computer-use
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-computer-use/2025_ICLR_85_Poster_Agent_S_An_Open_Agentic_Framework_that_Uses_Computers_Like_a_Human.pdf
- Token Usage: input 24453, output 3885, total 28338

### GitHub & Websites

- Agent S · [GitHub](https://github.com/simular-ai/Agent-S)
  - description: Official code release for the paper’s agentic framework, including the Experience-Augmented Hierarchical Planning, Agent-Computer Interface (ACI), memory modules, and evaluation scripts for OSWorld and WindowsAgentArena.

- OSWorld · [GitHub](https://github.com/xlang-ai/OSWorld)
  - description: Executable Ubuntu desktop benchmark used for the main experiments; also provides the MMAgent baseline implementation that the paper compares against.

- WindowsAgentArena · [GitHub](https://github.com/microsoft/WindowsAgentArena)
  - description: Windows OS benchmark used to evaluate cross-OS generalization; includes the NAVI baseline referenced in the paper.

- PaddleOCR · [GitHub](https://github.com/PaddlePaddle/PaddleOCR) · [Codewiki](https://codewiki.google/github.com/PaddlePaddle/PaddleOCR) · [Doc](https://paddleocr.readthedocs.io/en/latest/)
  - description: OCR toolkit used to augment the accessibility tree with screenshot text for improved GUI grounding in the ACI.

- Perplexica Search Engine · [GitHub](https://github.com/ItzCrazyKns/Perplexica) · [Codewiki](https://codewiki.google/github.com/ItzCrazyKns/Perplexica)
  - description: Open-source web search engine used by the Manager for Online Web Knowledge retrieval during planning.

- OpenAI Embeddings (text-embedding-3-small) · [Doc](https://platform.openai.com/docs/guides/embeddings)
  - description: Embedding model used for similarity-based retrieval in both Narrative and Episodic memories.

- OpenAI GPT-4o · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: One of the backbone MLLMs used to run Agent S and report results on OSWorld and WindowsAgentArena.

- Anthropic Claude 3.5 Sonnet · [Doc](https://docs.anthropic.com/en/docs/models-overview)
  - description: Alternative backbone MLLM used for Agent S in OSWorld experiments.

- SWE-agent (Agent-Computer Interface for software engineering) · [GitHub](https://github.com/princeton-nlp/SWE-agent) · [Website](https://swe-agent.com)
  - description: Prior ACI design for software engineering agents that inspired Agent S’s ACI abstraction for GUI control, relevant for practitioners extending ACI concepts.

<!-- paper_id: 5e6799c927dd3ed46b338801da12115b8236746e -->

## 18. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks - ICML - 2025 - citation_count 42 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICML_42_Spotlightposter_G-Designer_Architecting_Multi-agent_Communication_Topologies_via_Graph_Neural_Networks.pdf
- Link: https://openreview.net/pdf/ac2d69ee4852bf5e47910fa3e4da09a7322def18.pdf
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICML_42_Spotlightposter_G-Designer_Architecting_Multi-agent_Communication_Topologies_via_Graph_Neural_Networks.pdf
- Token Usage: input 22163, output 5111, total 27274

### GitHub & Websites

- G-Designer · [GitHub](https://github.com/yanweiyue/GDesigner)
  - description: Official code release for the paper; contains the implementation of the VGAE-based designer, training/inference scripts, and configs to reproduce the reported multi-agent topology design results.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework used as a comparison baseline in experiments.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Doc](https://docs.deepwisdom.ai/metagpt/)
  - description: Multi-agent collaborative framework for software/code generation; used as a baseline on HumanEval and other tasks.

- MMLU (Measuring Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test)
  - description: General reasoning benchmark used for evaluation; the paper reports accuracy on MMLU subsets and the full benchmark.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Math word problem benchmark used for evaluation; the paper reports substantial gains on GSM8K.

- SVAMP · [GitHub](https://github.com/arkilpatel/SVAMP)
  - description: Math reasoning dataset used for evaluation; the paper compares accuracy and token cost on SVAMP.

- AQuA · [GitHub](https://github.com/deepmind/AQuA)
  - description: Multiple-choice math word problems dataset (AQuA-RAT) used for evaluation.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code generation benchmark used for evaluation; the paper reports pass@1 and compares against multi-agent baselines.

- Sentence-Transformers: all-MiniLM-L6-v2 · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [Doc](https://www.sbert.net/)
  - description: Lightweight text-embedding model used as the NodeEncoder to embed agent profiles and task text.

- OpenAI API (GPT-4 / GPT-3.5) · [GitHub](https://github.com/openai/openai-python) · [Codewiki](https://codewiki.google/github.com/openai/openai-python) · [Doc](https://platform.openai.com/docs/api-reference)
  - description: LLM API used to run agents and generate profiles; the paper evaluates systems primarily with gpt-4-1106-preview and gpt-3.5-turbo-0125.

<!-- paper_id: 0db8ee4c82cb700af1f96df72a8218cb3511c2d9 -->

## 19. AgentSense: Benchmarking Social Intelligence of Language Agents through Interactive Scenarios - NAACL - 2025 - citation_count 20 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_20_Long_AgentSense_Benchmarking_Social_Intelligence_of_Language_Agents_through_Interactive_Scenarios.pdf
- Link: https://aclanthology.org/2025.naacl-long.257/
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_20_Long_AgentSense_Benchmarking_Social_Intelligence_of_Language_Agents_through_Interactive_Scenarios.pdf
- Token Usage: input 27957, output 3553, total 31510

### GitHub & Websites

- AgentSense · [GitHub](https://github.com/ljcleo/agent_sense)
  - description: Official code and data release for the paper’s benchmark; includes scenario templates, prompts, simulation/evaluation scripts to reproduce AgentSense.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://autogen.readthedocs.io/)
  - description: Multi‑agent conversation framework the authors used to manage interacting and judging threads during simulations.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Doc](https://docs.vllm.ai/)
  - description: LLM serving system the paper used to deploy all open‑source models for experiments.

- Internet Movie Script Database (IMSDb) · [Website](https://imsdb.com)
  - description: Source of movie/TV scripts from which the authors extracted scenes and built scenario templates.

- Meta Llama (Llama 2/3) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open‑source baseline models evaluated (Llama‑2‑7b/13b/70b‑Chat; Llama‑3‑8b/70b‑Instruct).

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io)
  - description: Open‑source models used both as agent backbones (7b/14b/72b) and as judge models.

- Mistral‑7B‑Instruct · [Website](https://mistral.ai) · [Doc](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  - description: Open‑source baseline model evaluated in the benchmark.

- OpenAI GPT‑4o / GPT‑3.5‑turbo · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs/models)
  - description: Proprietary models used as agents and third‑party judges; GPT‑4o also powers parts of the automated scenario construction.

- Sotopia · [GitHub](https://github.com/sotopia-lab/sotopia) · [Website](https://sotopia-lab.github.io/)
  - description: Related interactive social‑intelligence benchmark cited and compared in the paper’s discussion/table; useful for practitioners to inspect complementary evaluation settings.

<!-- paper_id: 2787130200766f799773b4435f86524b3082c45c -->

## 20. MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning - ACL - 2025 - citation_count 27 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4RL/2025_ACL_27_Long_MAPoRL_Multi-Agent_Post-Co-Training_for_Collaborative_Large_Language_Models_with_Reinforcement_Learning.pdf
- Link: https://aclanthology.org/2025.acl-long.1459/
- Tags: multiagent, tool, science, rl
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4RL/2025_ACL_27_Long_MAPoRL_Multi-Agent_Post-Co-Training_for_Collaborative_Large_Language_Models_with_Reinforcement_Learning.pdf
- Token Usage: input 35510, output 4336, total 39846

### GitHub & Websites

- MAPoRL · [GitHub](https://github.com/chanwoo-park-official/MAPoRL)
  - description: Official implementation released by the paper; contains the multi-agent debate pipeline, verifier-based reward shaping, and multi-agent PPO training code to reproduce MAPoRL experiments.

- GSM8K (Grade School Math 8K) · [GitHub](https://github.com/openai/grade-school-math) · [Website](https://huggingface.co/datasets/gsm8k)
  - description: Math word problem dataset and verifier training resources; used to train the verifier and evaluate MAPoRL on mathematical reasoning.

- ANLI (Adversarial NLI) · [GitHub](https://github.com/facebookresearch/anli) · [Website](https://huggingface.co/datasets/anli)
  - description: Adversarial natural language inference dataset; used to train a verifier and evaluate MAPoRL on logical inference.

- QLoRA · [GitHub](https://github.com/artidoro/qlora)
  - description: Efficient low-rank finetuning for quantized LLMs; used to fine-tune base models and verifiers in MAPoRL due to compute constraints.

- MAPPO (Multi-Agent PPO, “on-policy” codebase) · [GitHub](https://github.com/marlbenchmark/on-policy)
  - description: Reference implementation of multi-agent PPO from Yu et al. (2022); the paper adapts multi-agent PPO to the language domain for co-training agents.

- Phi-3-mini-128k-instruct · [GitHub](https://github.com/microsoft/Phi-3CookBook) · [Website](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
  - description: Microsoft’s 3.4B instruction-tuned model used as the main base LLM and verifier backbone for MAPoRL training and evaluation.

- Qwen2.5-3B-Instruct · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
  - description: Qwen model used as an alternative/heterogeneous agent in experiments assessing multi-model collaboration with MAPoRL.

- Llama 3 8B Instruct · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com/)
  - description: Meta’s Llama 3 8B instruction-tuned model used as another heterogeneous agent to study co-training across models under MAPoRL.

<!-- paper_id: aa89c6bf86486e180833037333555e3492b15c8e -->

## 21. EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction - NAACL - 2025 - citation_count 95 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_NAACL_95_Long_EASYTOOL_Enhancing_LLM-based_Agents_with_Concise_Tool_Instruction.pdf
- Link: https://aclanthology.org/2025.naacl-long.44/
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_NAACL_95_Long_EASYTOOL_Enhancing_LLM-based_Agents_with_Concise_Tool_Instruction.pdf
- Token Usage: input 24320, output 6165, total 30485

### GitHub & Websites

- EASYTOOL (in JARVIS) · [GitHub](https://github.com/microsoft/JARVIS/tree/main/easytool)
  - description: Official code release from the paper; provides scripts to transform tool documentation into concise tool instructions and reproduce the reported experiments.

- HuggingGPT (JARVIS) · [GitHub](https://github.com/microsoft/JARVIS) · [Codewiki](https://codewiki.google/github.com/microsoft/JARVIS) · [Website](https://hugginggpt.github.io/)
  - description: The broader project repo used in the paper (cited as HFmodels) for tool/model integration; the EASYTOOL code lives under this repo.

- ToolBench / ToolLLM · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolllm.github.io/)
  - description: Dataset and toolkit of real-world REST APIs used extensively for evaluation (I1/I2/I3 subsets, DFSDT baseline, retriever); the paper converts its tool docs with EASYTOOL and benchmarks performance.

- RestGPT / RestBench · [GitHub](https://github.com/Yifan-Song793/RestGPT)
  - description: Framework and benchmark for connecting LLMs to real-world RESTful APIs; the paper evaluates EASYTOOL on the TMDB subset of RestBench and compares against RestGPT baselines.

- RapidAPI · [Website](https://rapidapi.com/)
  - description: Source of many API tools in ToolBench; reproducing ToolBench-style experiments often requires RapidAPI access.

- The Movie Database (TMDB) API · [Doc](https://developer.themoviedb.org/reference/intro)
  - description: Official API used as the tool set for the TMDB subset of RestBench where EASYTOOL is evaluated on tool-path planning.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Baseline prompting algorithm combining reasoning and acting; used in comparisons on ToolBench and RestBench.

- LLMLingua · [GitHub](https://github.com/microsoft/LLMLingua)
  - description: Prompt compression toolkit used as a comparison in the appendix; the paper shows naive compression can harm tool-use versus EASYTOOL.

- tiktoken · [GitHub](https://github.com/openai/tiktoken) · [Codewiki](https://codewiki.google/github.com/openai/tiktoken)
  - description: Tokenizer used for analysis (cl100k_base) to measure documentation/instruction token lengths.

- Gorilla · [GitHub](https://github.com/ShishirPatil/gorilla) · [Website](https://gorilla.cs.berkeley.edu/)
  - description: LLM connected to massive APIs; its dataset/documentation statistics are referenced in the paper’s analysis of tool documentation.

- ToolAlpaca · [GitHub](https://github.com/tangqiaoyu/ToolAlpaca)
  - description: Simulated tool-learning dataset referenced for documentation statistics and context in the paper.

- Vicuna (via FastChat) · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat) · [Website](https://lmsys.org/blog/2023-03-30-vicuna/)
  - description: Open-source chat model used as a backbone (Vicuna-13B/30B) in several baselines and comparisons.

- Llama 3/3.1 (Meta) · [Website](https://ai.meta.com/llama/)
  - description: The paper evaluates Llama‑3.1‑8B‑Instruct and Llama‑3.1‑70B‑Instruct as backbones; practitioners need the official weights/access to reproduce results.

- BERT (Google Research) · [GitHub](https://github.com/google-research/bert) · [Codewiki](https://codewiki.google/github.com/google-research/bert)
  - description: Used as the dense retriever baseline in ToolBench for tool retrieval comparisons.

- OpenAI Function Calling · [Website](https://openai.com/blog/function-calling-and-other-api-updates)
  - description: API capability used to implement tool invocation for OpenAI models in the baselines, as detailed in the paper’s appendix.

<!-- paper_id: bf21281f3faba32b275012bc90b10e7e988b8867 -->

## 22. Automated Design of Agentic Systems - ICLR - 2025 - citation_count 105 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_105_Poster_Automated_Design_of_Agentic_Systems.pdf
- Link: https://openreview.net/pdf/116ed45ae3d0873b16f5fdcab7f1fa4f12253e6d.pdf
- Tags: multiagent, tool, science, agent, biology, protein-function, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_105_Poster_Automated_Design_of_Agentic_Systems.pdf
- Token Usage: input 37703, output 3764, total 41467

### GitHub & Websites

- Automated Design of Agentic Systems (ADAS) · [GitHub](https://github.com/ShengranHu/ADAS)
  - description: Official code release for the paper’s Meta Agent Search algorithm and framework; used to generate, evaluate, and archive discovered agents across all experiments.

- Abstraction and Reasoning Corpus (ARC) · [GitHub](https://github.com/fchollet/ARC) · [Website](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)
  - description: Visual reasoning dataset used as the core case study; agents must learn programmatic transformations from examples and predict test outputs.

- DROP (Discrete Reasoning Over Paragraphs) · [GitHub](https://github.com/allenai/drop) · [Website](https://leaderboard.allenai.org/drop)
  - description: Reading comprehension benchmark used to evaluate agents’ F1 on discrete reasoning over passages.

- MGSM (Multilingual Grade School Math) · [GitHub](https://github.com/google-research/google-research/tree/master/mgsm)
  - description: Multilingual math word-problem benchmark used as a primary search domain and source of agents for cross-domain transfer.

- MMLU (Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test) · [Website](https://mmlu.ai)
  - description: Multi-task QA benchmark used to assess general reasoning and as a transfer target for agents discovered on math.

- GPQA (Graduate-level Google-Proof Q&A) · [Doc](https://huggingface.co/datasets/IdavidRein/GPQA)
  - description: Hard science multiple-choice benchmark (e.g., GPQA-Diamond) used to evaluate agents’ scientific reasoning.

- GSM8K (Grade School Math) · [GitHub](https://github.com/openai/grade-school-math) · [Doc](https://huggingface.co/datasets/gsm8k)
  - description: Math word-problem benchmark used as a held-out domain to test transfer of math agents discovered via MGSM.

- GSM-Hard · [GitHub](https://github.com/FranxYao/GSM8K-Hard)
  - description: Hard subset of GSM8K used as another held-out math domain to evaluate transfer performance.

- SVAMP · [GitHub](https://github.com/arkilpatel/SVAMP)
  - description: Math word-problem dataset used in additional transfer evaluations from MGSM-discovered agents.

- ASDiv · [GitHub](https://github.com/chaochun/nlu-asdiv-dataset)
  - description: Arithmetic reasoning dataset used in further math transfer experiments.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Website](https://www.langchain.com/) · [Doc](https://python.langchain.com/docs/)
  - description: Open-source agent framework cited as a natural substrate for code-space ADAS (the paper suggests future searches could build on such toolkits).

- OpenAI Simple Evals · [GitHub](https://github.com/openai/simple-evals)
  - description: Evaluation utilities referenced for prompting/evaluation practice (e.g., DROP one-shot style), relevant for reproducing the paper’s evaluation setup.

<!-- paper_id: c9537f656e7d9713fd4108ce7bf512290f48e562 -->

## 23. AgentSquare: Automatic LLM Agent Search in Modular Design Space - ICLR - 2025 - citation_count 44 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_44_Poster_AgentSquare_Automatic_LLM_Agent_Search_in_Modular_Design_Space.pdf
- Link: https://openreview.net/pdf/c06588c16619cd2e27dd12043d72f687442e36c5.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_44_Poster_AgentSquare_Automatic_LLM_Agent_Search_in_Modular_Design_Space.pdf
- Token Usage: input 30687, output 5450, total 36137

### GitHub & Websites

- AgentSquare · [GitHub](https://github.com/tsinghua-fib-lab/AgentSquare)
  - description: Official implementation of the paper’s modular agent search framework (AgentSquare) and standardized module interfaces for Planning, Reasoning, Tool Use, and Memory.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop) · [Website](https://webshop-pnlp.github.io/)
  - description: E-commerce web interaction benchmark used as one of the six evaluation tasks.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld) · [Website](https://alfworld.github.io/)
  - description: Text-based embodied household tasks benchmark used for evaluation.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld) · [Website](https://allenai.github.io/ScienceWorld/)
  - description: Interactive science tasks benchmark used for evaluating agent reasoning and planning.

- TravelPlanner · [GitHub](https://github.com/TIGER-AI-Lab/TravelPlanner)
  - description: Real-world tool-use planning benchmark used as one of the tool-oriented evaluation tasks.

- M3ToolEval
  - description: Multi-tool interaction evaluation suite referenced as M3ToolEval in experiments (Wang et al., 2024b); used as one of the six benchmarks.

- PDDL tasks (AgentBoard)
  - description: PDDL game-style planning tasks referenced via AgentBoard (Ma et al., 2024); used as one of the six benchmarks.

- ADAS (Automated Design of Agentic Systems)
  - description: Prior agent-search method whose official codebase was used/modified; also used as a comparison baseline in experiments.

- OPRO (LLMs as Optimizers) · [GitHub](https://github.com/google-deepmind/opro)
  - description: Prompt-search baseline method used for comparison; LLM-based optimizer for instruction/prompt refinement.

- Tree of Thoughts (ToT) · [GitHub](https://github.com/princeton-nlp/tree-of-thought-llm)
  - description: Reasoning strategy baseline; also influences module designs within the Reasoning module space.

- Self-Refine · [GitHub](https://github.com/madaan/self-refine)
  - description: Hand-crafted agent baseline that iteratively refines outputs using self-feedback.

- HuggingGPT (JARVIS) · [GitHub](https://github.com/microsoft/JARVIS) · [Codewiki](https://codewiki.google/github.com/microsoft/JARVIS)
  - description: Tool-use agent baseline coordinating models/tools via Hugging Face; included among hand-crafted baselines.

- Voyager · [GitHub](https://github.com/MineDojo/Voyager)
  - description: Embodied LLM agent baseline; its memory ideas and module were used in module combinations during search.

- Generative Agents · [GitHub](https://github.com/joonspk-research/generative_agents) · [Codewiki](https://codewiki.google/github.com/joonspk-research/generative_agents)
  - description: Single-agent simulation framework baseline; its memory module was reused in discovered high-performing combinations.

- OPENAGI · [GitHub](https://github.com/agiresearch/OpenAGI)
  - description: Tool-use/agent framework baseline for comparison.

- ToolBench · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Tool-use dataset/tooling referenced in the paper’s ToolUse module design (e.g., retrieval-based tool selection in derived modules).

- Toolformer · [GitHub](https://github.com/TimoSchick/Toolformer)
  - description: Tool-use method referenced for multi-candidate tool invocation and voting strategies used in evolved ToolUse modules.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain)
  - description: Workflow framework cited as an LLM tooling baseline; also relevant for reproducing memory/vector DB components used in modules.

- BabyAGI · [GitHub](https://github.com/yoheinakajima/babyagi)
  - description: Minimal agentic workflow example cited as related engineering resource for building LLM agent systems.

- Chroma (ChromaDB) · [GitHub](https://github.com/chroma-core/chroma) · [Codewiki](https://codewiki.google/github.com/chroma-core/chroma)
  - description: Vector database used in memory-module code examples (hierarchical memory storage/retrieval) to persist and retrieve experiences.

<!-- paper_id: f9da7cfe8d403f66ab844b08577deeff6d1b1170 -->

## 24. Caution for the Environment: Multimodal Agents are Susceptible to Environmental Distractions - ACL - 2025 - citation_count 42 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_42_Long_Caution_for_the_Environment_Multimodal_Agents_are_Susceptible_to_Environmental_Distractions.pdf
- Link: https://aclanthology.org/2025.acl-long.1087/
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_42_Long_Caution_for_the_Environment_Multimodal_Agents_are_Susceptible_to_Environmental_Distractions.pdf
- Token Usage: input 20396, output 5168, total 25564

### GitHub & Websites

- EnvDistraction · [GitHub](https://github.com/xbmxb/EnvDistraction)
  - description: Official code release for this paper. Contains scripts to simulate environmental distractions in GUIs, implement the three working patterns, run evaluations, and reproduce results on the proposed dataset.

- OpenAI GPT-4o (vision-capable) · [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Generalist MLLM evaluated as a GUI agent and also used to help generate dataset content; accessed via API in experiments.

- OpenAI GPT-4V (Vision) · [Doc](https://platform.openai.com/docs/guides/vision)
  - description: Vision-enabled GPT-4 variant evaluated under the paper’s working patterns as a generalist GUI agent.

- Claude 3.5 Sonnet · [Website](https://www.anthropic.com/news/claude-3-5-sonnet) · [Doc](https://docs.anthropic.com/en/docs/models/overview)
  - description: Anthropic’s multimodal model evaluated as a generalist agent via API.

- GLM-4v (ZhipuAI) · [Website](https://open.bigmodel.cn/) · [Doc](https://bigmodel.cn/dev/api)
  - description: Vision-capable GLM model evaluated as a generalist agent through the ZhipuAI API.

- Qwen-VL-Plus (DashScope API) · [Website](https://qwenlm.github.io/) · [Doc](https://dashscope.aliyun.com/api-reference/qwen-vl)
  - description: Proprietary API model from Qwen used as a generalist MLLM baseline in experiments.

- Qwen-VL-Chat · [GitHub](https://github.com/QwenLM/Qwen-VL) · [Website](https://qwenlm.github.io/)
  - description: Open-source multimodal Qwen model evaluated as a generalist baseline.

- MiniCPM-Llama3-V2.5 (MiniCPM-V) · [GitHub](https://github.com/OpenBMB/MiniCPM-V) · [Codewiki](https://codewiki.google/github.com/OpenBMB/MiniCPM-V) · [Website](https://minicpm.org/)
  - description: Open-source multimodal model evaluated as a generalist baseline.

- LLaVA v1.6 · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io/)
  - description: Open-source vision-language model evaluated as a generalist baseline.

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent)
  - description: Specialist GUI agent evaluated for coordinate-based action prediction in the paper’s tasks.

- SeeClick · [Doc](https://arxiv.org/abs/2401.10935)
  - description: Specialist GUI grounding/agent model evaluated as a baseline; used for coordinate-based action prediction.

- Amazon Reviews Dataset · [Website](https://nijianmo.github.io/amazon/index.html)
  - description: Product review corpus used to simulate a BM25-based recommendation retriever in the Recommendation subset.

- Google Custom Search JSON API · [Doc](https://developers.google.com/custom-search/v1/overview)
  - description: Used to fetch real search results for constructing the Search subset by mixing true and fake results.

- Discord (product docs) · [Website](https://discord.com/) · [Doc](https://support.discord.com/hc/en-us)
  - description: Platform used to create the Chat subset; goals and actions are derived with reference to the Discord manual/interface.

- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory) · [Codewiki](https://codewiki.google/github.com/hiyouga/LLaMA-Factory)
  - description: Training framework used for faithfulness-improvement experiments (LoRA + DPO) on Llama-based models.

- Stanford Alpaca · [GitHub](https://github.com/tatsu-lab/stanford_alpaca) · [Codewiki](https://codewiki.google/github.com/tatsu-lab/stanford_alpaca)
  - description: Instruction dataset used to synthesize a pseudo-preference dataset for DPO training to bias models toward following the user goal.

- Meta Llama 3.1 8B Instruct · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  - description: Base model fine-tuned (with LoRA + DPO) in the paper’s faithfulness improvement experiment.

- Gradio · [GitHub](https://github.com/gradio-app/gradio) · [Codewiki](https://codewiki.google/github.com/gradio-app/gradio) · [Website](https://www.gradio.app/)
  - description: Used to deploy and run open-source MLLMs locally during evaluations.

<!-- paper_id: ee745d4c5e9ff890731866f54b6b333bb3a4eb57 -->

## 25. AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration - ACL - 2025 - citation_count 21 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_21_Long_AgentDropout_Dynamic_Agent_Elimination_for_Token-Efficient_and_High-Performance_LLM-Based_Multi-Agent_Collaboration.pdf
- Link: https://aclanthology.org/2025.acl-long.1170/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_21_Long_AgentDropout_Dynamic_Agent_Elimination_for_Token-Efficient_and_High-Performance_LLM-Based_Multi-Agent_Collaboration.pdf
- Token Usage: input 25993, output 3795, total 29788

### GitHub & Websites

- AgentDropout · [GitHub](https://github.com/wangzx1219/AgentDropout)
  - description: Official code release for the paper; contains the implementation of Node Dropout and Edge Dropout, training/evaluation scripts, and configs to reproduce the reported results.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai) · [Doc](https://docs.vllm.ai)
  - description: High-throughput LLM inference engine used by the authors to run Llama3 and Qwen2.5 experiments.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://microsoft.github.io/autogen)
  - description: Multi-agent conversation framework used as a comparison baseline on larger models.

- AgentVerse · [GitHub](https://github.com/THUDM/AgentVerse) · [Website](https://agentverse.ai)
  - description: Multi-agent collaboration framework compared as a baseline in the experiments.

- AgentPrune
  - description: Prior SOTA method for pruning redundant communications in MAS; the authors compare against it and reuse its agent configuration files.

- Meta Llama 3 · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: One of the base LLMs used to build MAS for evaluation (Meta-Llama3-8B-Instruct).

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io)
  - description: Another base LLM used (Qwen2.5-72B-Instruct) in the experiments.

- DeepSeek-V3 · [GitHub](https://github.com/deepseek-ai/DeepSeek-V3) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-V3) · [Doc](https://api-docs.deepseek.com)
  - description: Largest base model evaluated via API in the paper (DeepSeek-V3-671B-Instruct).

- MMLU (Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test) · [Website](https://hendrycks.com/test)
  - description: Benchmark used to evaluate general reasoning.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math) · [Website](https://paperswithcode.com/dataset/gsm8k)
  - description: Math word-problem dataset for evaluating mathematical reasoning.

- AQuA-RAT · [GitHub](https://github.com/deepmind/AQuA)
  - description: Multiple-choice math reasoning dataset used in evaluation.

- SVAMP · [GitHub](https://github.com/arkilpatel/SVAMP)
  - description: Math word-problem dataset assessing robustness to variations; used in evaluation.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code generation benchmark used to measure program synthesis performance.

<!-- paper_id: ee552989a03693a441863af4c29dc594bfcd1ab5 -->

## 26. BioDiscoveryAgent: An AI Agent for Designing Genetic Perturbation Experiments - ICLR - 2025 - citation_count 44 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_44_Poster_BioDiscoveryAgent_An_AI_Agent_for_Designing_Genetic_Perturbation_Experiments.pdf
- Link: https://openreview.net/pdf/dcfab1feaba7d38761a21e709247e9d97d4b98e2.pdf
- Tags: multiagent, tool, science, agent, biology, protein-function, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_44_Poster_BioDiscoveryAgent_An_AI_Agent_for_Designing_Genetic_Perturbation_Experiments.pdf
- Token Usage: input 55475, output 4700, total 60175

### GitHub & Websites

- BioDiscoveryAgent · [GitHub](https://github.com/snap-stanford/BioDiscoveryAgent)
  - description: Official code release from the paper; includes prompts, agent implementation, tool integrations, and scripts to reproduce the closed-loop experiment design experiments.

- pymed (PubMed querying with Python) · [GitHub](https://github.com/gijswobben/pymed)
  - description: Python library used by the agent’s literature-search tool to query PubMed; the paper cites and uses this to pull and summarize relevant biomedical papers each round.

- PubMed API (Entrez E‑utilities) · [Website](https://pubmed.ncbi.nlm.nih.gov/) · [Doc](https://www.ncbi.nlm.nih.gov/books/NBK25500/)
  - description: Underlying API for retrieving biomedical literature; BioDiscoveryAgent issues literature queries (via pymed) to ground gene recommendations with citations.

- Reactome Pathway Knowledgebase · [Website](https://reactome.org) · [Doc](https://reactome.org/dev/content-service)
  - description: Pathway database used by the agent’s “gene search” tool for enrichment analysis; the agent expands candidate genes by finding those in pathways enriched among previous hits.

- KEGG (Kyoto Encyclopedia of Genes and Genomes) · [Website](https://www.kegg.jp/) · [Doc](https://www.kegg.jp/kegg/rest/keggapi.html)
  - description: Curated pathway resource used for pathway enrichment and pathway-based gene expansion in baseline/human strategies and tools described in the paper.

- ARCHS4 · [Website](https://maayanlab.cloud/archs4/) · [Doc](https://maayanlab.cloud/archs4/help.html)
  - description: RNA‑seq compendium and API used by the agent to retrieve correlated genes and shared tissue expression profiles when constructing tool-assisted gene suggestions.

- DepMap (Cancer Dependency Map) · [Website](https://depmap.org/portal/) · [Doc](https://depmap.org/portal/download/)
  - description: Source of gene co‑essentiality profiles referenced for computing gene similarity/dissimilarity in the agent’s gene-search tools.

- Anthropic Claude API · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/claude)
  - description: LLMs (Claude 3.5 Sonnet, Claude 3 Haiku, etc.) used as the core reasoning engine and as the critic agent in multiple experimental settings.

- OpenAI API · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: LLMs (GPT‑3.5‑Turbo, GPT‑4o, o1‑mini, o1‑preview) evaluated as alternative cores for BioDiscoveryAgent to compare performance and tool-use effects.

- Auto‑GPT · [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
  - description: Open-source agent framework cited as inspiration for stepwise planning; informs the agent’s prompt design and reasoning structure though not used as a direct dependency.

<!-- paper_id: bba1d4de76b8330bea5f73ddeb99da4e01c02e16 -->

## 27. Scaling Large-Language-Model-based Multi-Agent Collaboration - ICLR - 2025 - citation_count 123 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_123_Poster_Scaling_Large-Language-Model-based_Multi-Agent_Collaboration.pdf
- Link: https://openreview.net/pdf/ae4a2f1ceb38887964f97458f71ef3306ae433cb.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_123_Poster_Scaling_Large-Language-Model-based_Multi-Agent_Collaboration.pdf
- Token Usage: input 24309, output 3778, total 28087

### GitHub & Websites

- ChatDev · MACNET branch · [GitHub](https://github.com/OpenBMB/ChatDev/tree/macnet)
  - description: Official implementation released by this paper; contains MACNET code to build DAG-based multi-agent topologies and run experiments, along with scripts and resources used in evaluation (including the SRDD benchmark/metrics from the ChatDev project).

- AutoGPT · [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
  - description: Baseline system; an autonomous LLM agent framework with planning and tool use against which MACNET is compared.

- AgentVerse · [GitHub](https://github.com/THUDM/AgentVerse)
  - description: Baseline multi-agent framework that assembles expert agents in chained/hierarchical structures; used for comparison in experiments.

- GPTSwarm (Language Agents as Optimizable Graphs) · [Website](https://arxiv.org/abs/2402.16823)
  - description: Baseline method that models agents as computational graphs with optimizable nodes/edges; used for comparison.

- MMLU · [GitHub](https://github.com/hendrycks/test) · [Doc](https://huggingface.co/datasets/cais/mmlu)
  - description: Evaluation dataset for multitask language understanding; used to measure MACNET accuracy on multiple-choice reasoning.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code-generation benchmark with unit tests; used to evaluate pass@k for MACNET.

- CommonGen-Hard (from Self-Refine) · [GitHub](https://github.com/allenai/self-refine)
  - description: Hard subset of CommonGen introduced by Self-Refine; used to evaluate open-ended generation quality (grammar, fluency, relevance, logic).

- OpenAI API (GPT-3.5) · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Inference backend used for interactive reasoning in MACNET experiments (GPT-3.5 as the default model).

<!-- paper_id: 208d489c73ebf182faa974191355fb2505ce8da5 -->

## 28. OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation - NeurIPS - 2025 - citation_count 61 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NeurIPS_61_Poster_OWL_Optimized_Workforce_Learning_for_General_Multi-Agent_Assistance_in_Real-World_Task_Automation.pdf
- Link: https://openreview.net/pdf/9ea2d3d5cf7f874c7669ab5c3f1270eb3bc794d1.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NeurIPS_61_Poster_OWL_Optimized_Workforce_Learning_for_General_Multi-Agent_Assistance_in_Real-World_Task_Automation.pdf
- Token Usage: input 40896, output 5735, total 46631

### GitHub & Websites

- OWL (Optimized Workforce Learning / WORKFORCE) · [GitHub](https://github.com/camel-ai/owl)
  - description: Official code release for the paper, including the multi-agent WORKFORCE framework, OWL training pipelines, tools, prompts, and assets to reproduce results and extend the system.

- CAMEL (Communicative Agents for “Mind” Exploration) · [GitHub](https://github.com/lightaime/camel)
  - description: Multi-agent framework the authors built upon to implement OWL and the experiment pipeline; useful for understanding the underlying agent infrastructure.

- GAIA Benchmark (General AI Assistant) · [Website](https://gaia-benchmark.github.io/)
  - description: The main evaluation benchmark used in the paper; provides task definitions, guidelines, and leaderboard information to reproduce evaluation.

- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory) · [Codewiki](https://codewiki.google/github.com/hiyouga/LLaMA-Factory)
  - description: Training framework the authors used to run SFT and DPO; includes scripts and docs for efficient post-training of LLMs.

- Qwen2.5-32B-Instruct · [Website](https://qwenlm.github.io/) · [Doc](https://qwen.readthedocs.io/) · [GitHub](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
  - description: Base planner model that the authors fine-tune with OWL; required to replicate planner training and ablations.

- HotpotQA · [Website](https://hotpotqa.github.io/)
  - description: One of the training datasets used for planner SFT/DPO to teach multi-hop web reasoning.

- WikiTableQuestions · [GitHub](https://github.com/ppasupat/WikiTableQuestions)
  - description: Training dataset used for planner curriculum on table reasoning and operations.

- Infinity-MM
  - description: Multimodal instruction dataset cited and used in the training curriculum to expose the planner to multimodal orchestration.

- AG2 (AgentOS for AI Agents) · [GitHub](https://github.com/ag2ai/ag2) · [Doc](https://docs.ag2.ai/)
  - description: Open-source agent framework referenced in related work; a useful baseline/toolkit for practitioners comparing orchestration approaches.

- Langfun · [GitHub](https://github.com/google/langfun)
  - description: Google’s agent framework cited in baselines/leaderboard comparisons; relevant for replicating or benchmarking alternative pipelines.

- smolagents (Hugging Face Agents) · [GitHub](https://github.com/huggingface/smolagents) · [Codewiki](https://codewiki.google/github.com/huggingface/smolagents) · [Website](https://huggingface.co/blog/open-deep-research)
  - description: Open-source library and “Open Deep Research” baseline referenced in experiments; provides agents and search/browsing baselines compared against WORKFORCE.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT)
  - description: Multi-agent framework cited as a domain-specific reference in the paper’s motivation; useful comparative baseline for software-engineering-style agent pipelines.

- Trase Agent · [Website](https://www.trasesystems.com/)
  - description: Proprietary/open platform listed among evaluated/compared agent frameworks on GAIA; relevant for baseline reproduction.

- H2O.ai h2oGPTe Agent · [Website](https://h2o.ai/platform/enterprise-h2ogpte/#AgenticAI)
  - description: Commercial agentic system included in GAIA leaderboard comparisons; useful as a reference point for closed-source baselines.

- Microsoft Multi-Agent Experiment v0.1 (GAIA) · [Website](https://aka.ms/gaia_multiagent_v01_march_1st)
  - description: Public multi-agent baseline referenced in the paper’s comparison set; provides scripts and settings for GAIA tasks.

- Unstructured · [GitHub](https://github.com/Unstructured-IO/unstructured) · [Codewiki](https://codewiki.google/github.com/Unstructured-IO/unstructured)
  - description: Document parsing library named among the paper’s document-processing tool stack; useful to replicate the Document Processing Worker.

- Firecrawl · [GitHub](https://github.com/mendableai/firecrawl)
  - description: Web crawling/extraction tool mentioned as part of the paper’s document/web processing toolkit; aids in robust content extraction.

- html2text · [GitHub](https://github.com/Alir3z4/html2text)
  - description: HTML-to-text conversion library cited among document parsing utilities used by workers for preprocessing web/document content.

- CAMEL-AI.org · [Website](https://camel-ai.org/)
  - description: The open-source organization behind OWL and WORKFORCE; project hub and community entry point for updates, issues, and extension efforts.

<!-- paper_id: dead4faccd5d800c3e06ea378e1da468cae7afd3 -->

## 29. Windows Agent Arena: Evaluating Multi-Modal OS Agents at Scale - ICML - 2025 - citation_count 79 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_79_Poster_Windows_Agent_Arena_Evaluating_Multi-Modal_OS_Agents_at_Scale.pdf
- Link: https://openreview.net/pdf/649fdc07b1256b087a6db62c7d4274d3b265c9d2.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_79_Poster_Windows_Agent_Arena_Evaluating_Multi-Modal_OS_Agents_at_Scale.pdf
- Token Usage: input 42002, output 5769, total 47771

### GitHub & Websites

- WINDOWS AGENT ARENA (official release)
  - description: The paper states that the code will be open-sourced but does not provide a URL in the PDF. Quote: “Github link will be available after paper review.”

- OSWorld · [Website](https://os-world.github.io)
  - description: Prior benchmark the authors build upon and adapt for Windows; they follow its general structure and evaluation approach while extending it to Windows OS.

- dockur/windows · [GitHub](https://github.com/dockur/windows) · [Codewiki](https://codewiki.google/github.com/dockur/windows)
  - description: Windows-in-Docker image the authors adapt to deploy a Windows 11 VM within their Docker-based benchmark infrastructure.

- QEMU · [Website](https://www.qemu.org) · [Doc](https://www.qemu.org/documentation/)
  - description: Hypervisor used to run the Windows 11 VM inside the containerized setup described in the benchmark infrastructure.

- KVM (Kernel-based Virtual Machine) · [Website](https://www.linux-kvm.org/page/Main_Page)
  - description: Linux virtualization support leveraged for nested virtualization when running the Windows VM at scale.

- Flask · [GitHub](https://github.com/pallets/flask) · [Codewiki](https://codewiki.google/github.com/pallets/flask) · [Website](https://flask.palletsprojects.com)
  - description: Python web server used as the bridge inside the VM to receive commands, execute them, and return observations/files.

- pywinauto · [GitHub](https://github.com/pywinauto/pywinauto) · [Doc](https://pywinauto.readthedocs.io)
  - description: Library used to query the Windows UI Automation (UIA) tree to build Set-of-Marks and extract UI element metadata.

- pygetwindow · [GitHub](https://github.com/asweigart/pygetwindow) · [Doc](https://pygetwindow.readthedocs.io)
  - description: Used to retrieve foreground/background window titles as part of the observation space.

- pyperclip · [GitHub](https://github.com/asweigart/pyperclip) · [Doc](https://pyperclip.readthedocs.io)
  - description: Used to read and write clipboard contents (text/image descriptions) for agent observations/actions.

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui) · [Doc](https://pyautogui.readthedocs.io)
  - description: Forms part of the action space and initial state setup scripts for mouse/keyboard automation within the VM.

- Tesseract OCR · [GitHub](https://github.com/tesseract-ocr/tesseract) · [Codewiki](https://codewiki.google/github.com/tesseract-ocr/tesseract) · [Doc](https://tesseract-ocr.github.io/tessdoc/)
  - description: Open-source OCR engine used for screen text extraction to create Set-of-Marks.

- pytesseract · [GitHub](https://github.com/madmaze/pytesseract)
  - description: Python wrapper around Tesseract used in the pipeline to obtain OCR text from screenshots.

- Grounding DINO · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
  - description: Open-set detector used for icon/image detection to augment Set-of-Marks for visual grounding.

- OmniParser · [GitHub](https://github.com/microsoft/OmniParser)
  - description: Model for multi-element detection and icon captioning on GUIs; used as a key visual parser to generate high-quality Set-of-Marks.

- Azure Machine Learning · [Doc](https://learn.microsoft.com/azure/machine-learning/) · [Website](https://azure.microsoft.com/products/machine-learning)
  - description: Cloud service the authors use to parallelize benchmark jobs across compute instances for fast evaluation at scale.

- Azure Blob Storage · [Doc](https://learn.microsoft.com/azure/storage/blobs/storage-blobs-overview)
  - description: Used to host the Windows 11 snapshot and collect outputs/logs during cloud evaluation.

- LibreOffice · [Website](https://www.libreoffice.org) · [Doc](https://documentation.libreoffice.org/)
  - description: Office suite (Writer/Calc) installed in the VM and used to define many document and spreadsheet manipulation tasks.

- VLC media player · [Website](https://www.videolan.org/vlc/)
  - description: Media player used for numerous video/audio manipulation tasks and settings configuration in the benchmark.

- Visual Studio Code · [GitHub](https://github.com/microsoft/vscode) · [Codewiki](https://codewiki.google/github.com/microsoft/vscode) · [Website](https://code.visualstudio.com)
  - description: IDE targeted by several tasks (extensions, settings, editing) within the Windows environment.

- Microsoft Edge · [Website](https://www.microsoft.com/edge)
  - description: Browser used for web tasks (privacy settings, PWA install, homepage/search setup) within the benchmark.

- Google Chrome · [Website](https://www.google.com/chrome/)
  - description: Alternative browser used for browsing, privacy, and settings tasks in the task suite.

- 7-Zip · [Website](https://www.7-zip.org/)
  - description: Compression utility referenced in tasks (e.g., creating password-protected archives) to evaluate file-management workflows.

<!-- paper_id: 39c5b07079e3b92aca9629d7cce694f276b9d9c6 -->

## 30. WebDancer: Towards Autonomous Information Seeking Agency - NeurIPS - 2025 - citation_count 54 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NeurIPS_54_Poster_WebDancer_Towards_Autonomous_Information_Seeking_Agency.pdf
- Link: https://openreview.net/pdf/7c886fbc63b09377d123254d93907b41820d72d7.pdf
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NeurIPS_54_Poster_WebDancer_Towards_Autonomous_Information_Seeking_Agency.pdf
- Token Usage: input 27835, output 7134, total 34969

### GitHub & Websites

- Tongyi DeepResearch (WebDancer code and demo) · [GitHub](https://github.com/Alibaba-NLP/DeepResearch) · [Codewiki](https://codewiki.google/github.com/Alibaba-NLP/DeepResearch)
  - description: Official release accompanying the paper; contains code and demos for WebDancer, including data synthesis (CRAWLQA/E2HQA), ReAct-style agent, SFT cold-start, and DAPO-based RL training.

- Qwen-Agent · [GitHub](https://github.com/QwenLM/Qwen-Agent)
  - description: Agent toolkit used to implement the ReAct-based system and tool-calling; the paper builds WebDancer on top of this framework and trains with chatml formatting.

- VERL (vLLM Efficient Reinforcement Learning) · [GitHub](https://github.com/vllm-project/verl)
  - description: RL framework used to run on-policy rollouts and optimize with the DAPO algorithm; the authors state they implement VERL to support RL algorithm and rollouts.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai)
  - description: High-throughput LLM serving engine (PagedAttention) underpinning VERL; enables efficient rollouts and training in the RL stage referenced by the paper.

- WebWalker · [Website](https://arxiv.org/abs/2501.07572)
  - description: Evaluation benchmark for deep web traversal used in the paper to assess WebDancer (680 examples; Pass@1/Pass@3/Cons@3 reported).

- GAIA
  - description: Core benchmark for general AI assistants; WebDancer is evaluated on GAIA’s text-only split using LLM-as-judge metrics.

- BrowseComp-zh · [Website](https://arxiv.org/abs/2504.19314)
  - description: Chinese browsing benchmark used as an additional evaluation to measure WebDancer’s web information-seeking ability (Pass@1/Pass@3).

- WebThinker · [GitHub](https://github.com/RUC-NLPIR/WebThinker)
  - description: Open-source baseline and concurrent agentic framework (SFT+RL) reproduced and compared against in experiments.

- SimpleDeepSearcher · [GitHub](https://github.com/RUCAIBox/SimpleDeepSearcher)
  - description: Closely related open-source implementation for deep information seeking via trajectory synthesis; cited as a comparison/alternative to RL-based training.

- OpenAI Deep Research (system card) · [Website](https://cdn.openai.com/deep-research-system-card.pdf)
  - description: Closed-source agentic system used as a reference point and comparison in results; documents the end-to-end RL-trained deep research agent.

- QwQ-32B (reasoning model) · [Website](https://qwenlm.github.io/blog/qwq-32b/)
  - description: Reasoning LRM used by the authors for Long-CoT trajectory sampling and as a backbone in some configurations.

- DeepResearcher · [Website](https://arxiv.org/abs/2504.03160)
  - description: Related open research on scaling deep research via RL in real web environments; cited in related work as a closely aligned direction.

- SimpleQA · [Website](https://openai.com/index/introducing-simpleqa/)
  - description: Source style for constructing E2HQA; the authors iteratively rewrite SimpleQA-style questions to synthesize harder multi-step queries.

<!-- paper_id: 84ac4173ac8cd7cfea0262ceedc0796d0ea4b0e2 -->

## 31. ToolGen: Unified Tool Retrieval and Calling via Generation - ICLR - 2025 - citation_count 25 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_25_Poster_ToolGen_Unified_Tool_Retrieval_and_Calling_via_Generation.pdf
- Link: https://openreview.net/pdf/b5d464a0c1f8e39ed945666ae1468185132c7754.pdf
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_25_Poster_ToolGen_Unified_Tool_Retrieval_and_Calling_via_Generation.pdf
- Token Usage: input 28535, output 3662, total 32197

### GitHub & Websites

- ToolGen · [GitHub](https://github.com/Reason-Wang/ToolGen)
  - description: Official code and data release for the paper; provides training scripts for tool virtualization, memorization, retrieval, and end-to-end agent tuning, plus evaluation assets to reproduce results.

- ToolBench (ToolLLM) · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Real-world tool/dataset repository with ~47k APIs and trajectories; used to build ToolGen’s tool set, retrieval pairs, and agent-tuning data, and provides the ToolRetriever baseline.

- StableToolBench · [GitHub](https://github.com/THUDM/StableToolBench)
  - description: Benchmark and evaluation framework used for end-to-end tool-use assessment (SoPR/SoWR) with stabilized tasks and GPT-4-based simulation for failed tools.

- Llama 3 · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://llama.meta.com/)
  - description: Base foundation model (Llama‑3‑8B) used for ToolGen training with the Llama‑3 chat template.

- DeepSpeed · [GitHub](https://github.com/microsoft/DeepSpeed) · [Website](https://www.deepspeed.ai/)
  - description: Training system used with ZeRO‑3 to fine-tune ToolGen efficiently across GPUs.

- FlashAttention / FlashAttention‑2 · [GitHub](https://github.com/Dao-AILab/flash-attention) · [Codewiki](https://codewiki.google/github.com/Dao-AILab/flash-attention)
  - description: Fast attention kernels employed during training/inference to reduce memory and speed up ToolGen’s agent experiments.

- OpenAI API (GPT‑3.5, GPT‑4/GPT‑4o, text-embedding‑3‑large) · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Used for baselines and utilities—GPT‑3.5 as an agent baseline, text‑embedding‑3‑large for embedding similarity, GPT‑4/GPT‑4o for long‑context comparison and evaluation setup.

- OpenHermes‑2.5 (instruction-following dataset) · [Website](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
  - description: General instruction-tuning data integrated with ToolGen to preserve broad instruction-following ability (ToolGen‑Instruct).

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io/blog/qwen2.5/)
  - description: Alternative base models (1.5B–14B) used to study ToolGen’s scaling across model sizes.

<!-- paper_id: b6aca622200ad7f722c667058d1dd664946d241b -->

## 32. CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents - ACL - 2025 - citation_count 28 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ACL_28_Findings_CRAB_Cross-environment_Agent_Benchmark_for_Multimodal_Language_Model_Agents.pdf
- Link: https://aclanthology.org/2025.findings-acl.1113/
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ACL_28_Findings_CRAB_Cross-environment_Agent_Benchmark_for_Multimodal_Language_Model_Agents.pdf
- Token Usage: input 36560, output 3385, total 39945

### GitHub & Websites

- CRAB (Cross-environment Agent Benchmark) · [GitHub](https://github.com/camel-ai/crab) · [Doc](https://github.com/camel-ai/crab/blob/main/crab-benchmark-v0/README.md)
  - description: Official repository for the paper’s framework and benchmark; includes code, evaluator graphs, environment interface, and reproducibility instructions used to build and run CRAB Benchmark-v0.

- CRAB Benchmark-v0 (dataset and tasks) · [GitHub](https://github.com/camel-ai/crab/tree/main/crab-benchmark-v0) · [Doc](https://github.com/camel-ai/crab/blob/main/crab-benchmark-v0/README.md)
  - description: The released suite of 120 cross-environment tasks (Ubuntu + Android) with subtask templates and graph evaluators; used in the paper’s evaluations.

- Google Android Emulator · [Website](https://developer.android.com/studio/run/emulator)
  - description: The Android device simulator used to instantiate the smartphone environment for CRAB.

- Android Debug Bridge (ADB) · [Website](https://developer.android.com/tools/adb)
  - description: Command-line tool to control the Android emulator; used to implement Android-side actions and evaluators.

- QEMU/KVM · [Website](https://www.qemu.org)
  - description: Virtualization stack used to run the reproducible Ubuntu desktop environment for the benchmark.

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui)
  - description: Python library for mouse/keyboard automation on the Ubuntu VM; used to implement desktop interaction actions.

- python-mss · [GitHub](https://github.com/BoboTiG/python-mss)
  - description: Screen capture library used to obtain Ubuntu screenshots as observations.

- EasyOCR · [GitHub](https://github.com/JaidedAI/EasyOCR) · [Codewiki](https://codewiki.google/github.com/JaidedAI/EasyOCR)
  - description: OCR engine used to detect and label interactive text elements on screens for visual prompting and evaluation.

- GroundingDINO · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
  - description: Open-set object detector used to locate interactive icons (icon/logo prompts) as part of the visual prompt generation.

- XFeat (Accelerated Features) · [GitHub](https://github.com/verlab/accelerated_features)
  - description: Lightweight feature matching toolkit used in evaluator functions for image matching on Ubuntu tasks.

- NetworkX · [GitHub](https://github.com/networkx/networkx) · [Website](https://networkx.org)
  - description: Graph library used to build the paper’s graph-based evaluators (checkpoint DAGs) and task graphs.

- dill · [GitHub](https://github.com/uqfoundation/dill)
  - description: Python function serialization used by the framework to package actions/evaluators (“code as configuration”).

- Pydantic · [GitHub](https://github.com/pydantic/pydantic) · [Codewiki](https://codewiki.google/github.com/pydantic/pydantic) · [Website](https://pydantic.dev)
  - description: Data modeling/validation library used to define task objects and configurations in CRAB.

- Ubuntu 22.04.4 LTS ISO · [Website](https://releases.ubuntu.com/jammy/ubuntu-22.04.4-desktop-amd64.iso)
  - description: Base OS image used to provision the Ubuntu virtual machine environment for the benchmark.

- Android Studio · [Website](https://developer.android.com/studio)
  - description: Tooling used to create the predefined Google Pixel device profiles and manage the Android emulator for CRAB.

<!-- paper_id: 0fd5fb95058d42a7566d6cf69f6112b23e3adcc1 -->

## 33. SPA-Bench: A Comprehensive Benchmark for SmartPhone Agent Evaluation - ICLR - 2025 - citation_count 37 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICLR_37_Spotlight_SPA-Bench_A_Comprehensive_Benchmark_for_SmartPhone_Agent_Evaluation.pdf
- Link: https://openreview.net/pdf/d6e724ea54d8555e1cf78adf21853e302b21f716.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICLR_37_Spotlight_SPA-Bench_A_Comprehensive_Benchmark_for_SmartPhone_Agent_Evaluation.pdf
- Token Usage: input 43897, output 5504, total 49401

### GitHub & Websites

- SPA-Bench · [Website](https://ai-agents-2030.github.io/SPA-Bench/)
  - description: Official project page for the benchmark introduced in the paper, including tasks, the plug-and-play agent framework, and the automated evaluation pipeline for smartphone agents.

- AndroidWorld · [GitHub](https://github.com/google-deepmind/android_world)
  - description: Dynamic benchmarking environment for autonomous Android agents; the paper integrates M3A/T3A/SeeAct-style agents and explicitly adopts AndroidWorld’s action-grounding module for SeeAct within SPA-Bench.

- GUI Odyssey · [GitHub](https://github.com/OpenGVLab/GUI-Odyssey)
  - description: Cross-app GUI navigation dataset and tasks; SPA-Bench references its task taxonomy and draws most English cross-app tasks from it.

- UIAutomator2 · [GitHub](https://github.com/openatx/uiautomator2)
  - description: Android UI automation library; used by the authors to execute end-to-end actions for fine-tuned agents that lack direct Android interaction capabilities.

- DroidBot · [GitHub](https://github.com/honeynet/droidbot)
  - description: UI-guided test input generator for Android; AutoDroid (one of the evaluated agents) relies on DroidBot, which is why the paper evaluates AutoDroid only on single‑app tasks.

- PaddleOCR · [GitHub](https://github.com/PaddlePaddle/PaddleOCR) · [Codewiki](https://codewiki.google/github.com/PaddlePaddle/PaddleOCR)
  - description: OCR toolkit used in SPA-Bench’s single-app success detection (coarse stage) to extract screen text for key-component matching.

- ADBKeyBoard · [GitHub](https://github.com/senzhk/ADBKeyBoard)
  - description: Android input method for ADB text entry; the paper installs it to enable reliable Chinese text input for several agents (e.g., AppAgent, SeeAct, M3A, T3A).

- Android Emulator · [Doc](https://developer.android.com/studio/run/emulator)
  - description: Official Android emulator documentation; SPA-Bench runs agents on snapshot-based emulators for scalable, consistent multi-device testing.

- OpenAI GPT-4o · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Multimodal LLM used both as the core decision model for several agentic baselines and as the evaluator in the automated success-detection pipeline. 

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent) · [Website](https://cogvlm.github.io)
  - description: Visual-language model for GUI agents; included as an “agent-as-a-model” baseline integrated and evaluated in SPA-Bench.

<!-- paper_id: 30005325950b25248d4c825e48feea5aed160ea9 -->

## 34. On the Resilience of LLM-Based Multi-Agent Collaboration with Faulty Agents - ICML - 2025 - citation_count 25 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICML_25_Poster_On_the_Resilience_of_LLM-Based_Multi-Agent_Collaboration_with_Faulty_Agents.pdf
- Link: https://openreview.net/pdf/9e2edbde1577fb0c18c09cfef297de459fecb239.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICML_25_Poster_On_the_Resilience_of_LLM-Based_Multi-Agent_Collaboration_with_Faulty_Agents.pdf
- Token Usage: input 27134, output 6581, total 33715

### GitHub & Websites

- MAS-Resilience (this paper) · [GitHub](https://github.com/CUHK-ARISE/MAS-Resilience)
  - description: Official code and data release for the paper, including implementations of AUTOTRANSFORM, AUTOINJECT, Challenger, and Inspector, plus adapted prompts and evaluation scripts for all six multi-agent systems and four tasks.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Doc](https://docs.deepwisdom.ai/MetaGPT/)
  - description: Multi-agent SOP-driven framework used as a Linear-structure baseline; the paper evaluates resilience with MetaGPT across tasks.

- Self-collaboration (Self-collab)
  - description: Three-role linear coding system (analyzer/coder/tester) used as a Linear-structure baseline for code generation experiments.

- CAMEL · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel) · [Website](https://www.camel-ai.org)
  - description: Two-agent “User–Assistant” flat collaboration framework; used as a Flat-structure baseline across tasks and as a target for AUTOTRANSFORM/AUTOINJECT case studies.

- Solo-Performance-Prompting (SPP) · [GitHub](https://github.com/GAIR-NLP/solo-performance-prompting)
  - description: Flat-structure coding setup via multi-persona prompting of a single model; used as a Flat baseline on code generation.

- MAD (Multi-Agent Debate) · [GitHub](https://github.com/Skytliang/MAD)
  - description: Hierarchical debate framework with two debaters and a judge; used as a Hierarchical-structure system across tasks and central to analyses where injected errors sometimes improve performance.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Hierarchical multi-agent framework with dynamic recruitment; used as a Hierarchical-structure baseline in all tasks.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code generation benchmark used to evaluate Pass@1 and study error-type and error-rate effects under AUTOINJECT/AUTOTRANSFORM.

- CIAR (Counter-Intuitive Arithmetic Reasoning) · [GitHub](https://github.com/Skytliang/CIAR)
  - description: Math reasoning dataset with hidden traps; used to measure system resilience on math problem solving.

- CommonMT
  - description: Commonsense-sensitive MT benchmark; the paper samples 100 Lexical cases and evaluates with BLEURT-20 to assess translation resilience.

- FairEval
  - description: Human-annotated win/tie/lose dataset for LLM-as-judge evaluation; used to test resilience on text evaluation.

- BLEURT-20 · [GitHub](https://github.com/google-research/bleurt) · [Doc](https://bleurt.readthedocs.io/en/latest/)
  - description: Learned metric used to score translation quality on CommonMT following prior work cited by the paper.

- OpenAI API (GPT-3.5, GPT-4o) · [Website](https://platform.openai.com/docs)
  - description: Backbone LLMs for all agents and for running AUTOTRANSFORM/AUTOINJECT; temperature set to zero in experiments.

- GPTSwarm · [GitHub](https://github.com/microsoft/gptswarm)
  - description: Graph-based multi-agent framework referenced for appendix experiments (star vs. complete topologies) to validate that hierarchical oversight improves robustness.

<!-- paper_id: c474f9b87013301b66cc921a2f7a3918d506d684 -->

## 35. OpenWebVoyager: Building Multimodal Web Agents via Iterative Real-World Exploration, Feedback and Optimization - ACL - 2025 - citation_count 26 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_26_Long_OpenWebVoyager_Building_Multimodal_Web_Agents_via_Iterative_Real-World_Exploration,_Feedback_and_Optimization.pdf
- Link: https://aclanthology.org/2025.acl-long.1336/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_26_Long_OpenWebVoyager_Building_Multimodal_Web_Agents_via_Iterative_Real-World_Exploration,_Feedback_and_Optimization.pdf
- Token Usage: input 21966, output 6309, total 28275

### GitHub & Websites

- OpenWebVoyager · [GitHub](https://github.com/MinorJerry/OpenWebVoyager)
  - description: Official release of the paper including code and data for training the multimodal web agent, running the imitation-learning and exploration–feedback–optimization cycles, and reproducing results.

- WebVoyager · [Website](https://arxiv.org/abs/2401.13919)
  - description: Prior framework and online Selenium-based environment the paper adopts/extends for real-world web navigation, and the source of the WebVoyager test set used for evaluation.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Real-world web navigation dataset; the paper uses 37 of its websites to create training queries and evaluates on its cross-task and cross-website test splits.

- Selenium · [GitHub](https://github.com/SeleniumHQ/selenium) · [Codewiki](https://codewiki.google/github.com/SeleniumHQ/selenium) · [Website](https://www.selenium.dev/) · [Doc](https://www.selenium.dev/documentation/)
  - description: Browser automation toolkit used to implement the online web environment and execute actions (click, type, scroll, restart) during trajectory collection.

- Idefics2-8B-Instruct · [Website](https://huggingface.co/HuggingFaceM4/idefics2-8b-instruct) · [Doc](https://huggingface.co/docs/transformers/main/en/model_doc/idefics2)
  - description: Open-source large multimodal model serving as the backbone of OpenWebVoyager; finetuned with IL and iterative optimization on multimodal trajectories.

- Self-Instruct · [GitHub](https://github.com/yizhongw/self-instruct) · [Website](https://arxiv.org/abs/2212.10560)
  - description: Procedure used to synthesize new web task queries in each exploration–feedback–optimization cycle.

- sentence-transformers (all-mpnet-base-v2) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
  - description: Embedding model/library used to compute query similarity and filter near-duplicate queries when generating task sets.

- OpenAI GPT-4o · [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs)
  - description: Closed-source multimodal model used in two roles: to sample expert trajectories for imitation learning (WebVoyager-4o) and as an automatic evaluator to accept/reject explored trajectories.

- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory) · [Codewiki](https://codewiki.google/github.com/hiyouga/LLaMA-Factory) · [Website](https://llamafactory.ai)
  - description: Training framework the paper uses for additional backbone experiments (e.g., finetuning Qwen2.5-VL).

- Qwen2.5-VL-7B-Instruct · [GitHub](https://github.com/QwenLM/Qwen-VL) · [Website](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
  - description: Alternative open-source LVLM evaluated as a backbone in the appendix to verify the generality of the method across models.

- LLaVA-OneVision · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io/blog/2024-08-28-llava-onevision/)
  - description: Another open-source LVLM baseline/backbone the paper experiments with to compare/improve visual grounding for web agents.

<!-- paper_id: c099f982a63a92d4891703f94fb5fd0ec6b64b1f -->

## 36. Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems - ICLR - 2025 - citation_count 46 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_46_Poster_Cut_the_Crap_An_Economical_Communication_Pipeline_for_LLM-based_Multi-Agent_Systems.pdf
- Link: https://openreview.net/pdf/490fff420d6359d82b48829601ac0ed820e335b4.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_46_Poster_Cut_the_Crap_An_Economical_Communication_Pipeline_for_LLM-based_Multi-Agent_Systems.pdf
- Token Usage: input 47557, output 5006, total 52563

### GitHub & Websites

- AgentPrune · [GitHub](https://github.com/yanweiyue/AgentPrune)
  - description: Official code release for the paper’s economical multi-agent communication pruning framework; used to reproduce AgentPrune and its integrations.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework that the paper plugs AgentPrune into for experiments and cost analyses.

- GPTSwarm · [GitHub](https://github.com/metauto-ai/GPTSwarm)
  - description: Optimizable-graph multi-agent framework used as a strong baseline and as a backbone combined with AgentPrune; authors note minor code modifications for fair comparison.

- DyLAN · [GitHub](https://github.com/SALT-NLP/DyLAN)
  - description: Dynamic LLM-agent network optimizing temporal communication; used as a temporal-communication baseline in experiments.

- LLM-Blender · [GitHub](https://github.com/yuchenlin/LLM-Blender)
  - description: Ensemble framework with PairRanker and GenFuser; treated as a spatial message-passing baseline in the paper.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Self-reflection agent baseline used for HumanEval comparisons.

- Parsel · [GitHub](https://github.com/ezelikman/Parsel)
  - description: Algorithmic reasoning via composed decompositions; used in the “CodeT+Parsel” HumanEval baselines.

- Tree of Thoughts (ToT) · [GitHub](https://github.com/ysymyth/tree-of-thought-llm)
  - description: Deliberate problem-solving approach; the paper’s case studies mention ToT-style agents within GPTSwarm setups.

- MMLU (Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test)
  - description: General reasoning benchmark used for evaluation and cost/performance plots.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math) · [Website](https://huggingface.co/datasets/gsm8k)
  - description: Math word problem dataset used to evaluate mathematical reasoning and token costs.

- SVAMP · [GitHub](https://github.com/arkilpatel/SVAMP)
  - description: Math reasoning dataset; part of the paper’s benchmark suite.

- AQuA-RAT (AQuA) · [GitHub](https://github.com/deepmind/AQuA)
  - description: Multi-choice math reasoning dataset; used in experiments.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Function-level code generation benchmark used for Pass@1 evaluation and cost comparisons.

- OpenAI API · [Website](https://platform.openai.com/docs/api-reference)
  - description: API used to access GPT-3.5 and GPT-4 models throughout all experiments and evaluations.

<!-- paper_id: aebfeb42bbd155c1541a67fddf0a6e2bc5d6ae34 -->

## 37. Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents - ACL - 2025 - citation_count 20 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_20_Findings_Explorer_Scaling_Exploration-driven_Web_Trajectory_Synthesis_for_Multimodal_Web_Agents.pdf
- Link: https://aclanthology.org/2025.findings-acl.326/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_20_Findings_Explorer_Scaling_Exploration-driven_Web_Trajectory_Synthesis_for_Multimodal_Web_Agents.pdf
- Token Usage: input 24020, output 4628, total 28648

### GitHub & Websites

- Explorer (this paper) · [Website](https://osu-nlp-group.github.io/Explorer/)
  - description: Official project page for Explorer, the exploration-driven web trajectory synthesis framework and dataset introduced in the paper; serves as the hub for resources to reproduce the work.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Real-world web agent dataset/benchmark used for offline evaluation and fine-tuning; the paper also evaluates on its online variant Mind2Web-Live.

- MiniWob++ · [GitHub](https://github.com/stanfordnlp/miniwob-plusplus)
  - description: Classic web interaction benchmark used in the paper’s zero-shot evaluation to test low-level GUI skills.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Doc](https://playwright.dev/)
  - description: Browser automation framework the authors use to execute actions and collect screenshots, HTML, and accessibility trees during trajectory synthesis.

- Tranco Top Sites List · [Website](https://tranco-list.eu/)
  - description: Public web ranking used by the authors to seed popular URLs for large-scale exploration.

- Similarweb Top Sites · [Website](https://www.similarweb.com/)
  - description: Source of high-traffic website URLs used as seeds to diversify trajectory generation.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL)
  - description: Vision-language model backbone; the 7B variant is fine-tuned to produce Explorer-7B and used in several evaluations.

- Phi-3.5 Vision · [Website](https://aka.ms/phi3) · [Doc](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
  - description: Multimodal small language model serving as the backbone for Explorer-4B; fine-tuned on the synthesized trajectories.

- DeBERTa · [GitHub](https://github.com/microsoft/DeBERTa)
  - description: Pretrained model used (as in prior work) to generate candidate elements for Multimodal-Mind2Web evaluation inputs.

- SeeAct · [GitHub](https://github.com/OSU-NLP-Group/SeeAct)
  - description: Web agent baseline compared in the paper; uses planning plus grounding with GPT-4V and serves as a strong reference implementation.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev/)
  - description: Realistic web environment referenced in related work; useful for practitioners exploring alternative evaluation/simulation setups for web agents.

<!-- paper_id: 26acd2d4b5d2bed7e25250d3f2f222b5611ee8ef -->

## 38. Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems - ICML - 2025 - citation_count 30 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICML_30_Spotlightposter_Which_Agent_Causes_Task_Failures_and_When_On_Automated_Failure_Attribution_of_LLM_Multi-Agent_Systems.pdf
- Link: https://openreview.net/pdf/ff66e9a98409dd3de0dd970b8433fae8c26a3674.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICML_30_Spotlightposter_Which_Agent_Causes_Task_Failures_and_When_On_Automated_Failure_Attribution_of_LLM_Multi-Agent_Systems.pdf
- Token Usage: input 21496, output 4461, total 25957

### GitHub & Websites

- Who&When (dataset and code)
  - description: Official release from this paper containing 184 annotated failure logs from 127 multi‑agent systems, plus prompts and evaluation code. The abstract states: “Code and dataset are available in the public repository.”

- AG2 (AgentGym/AG2) · [GitHub](https://github.com/ag2ai/ag2)
  - description: Agent framework used in the paper to auto‑generate multi‑agent systems via the CaptainAgent algorithm; provides team formation, tools, and orchestration used to build the algorithm‑generated portion of Who&When.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi‑agent framework the paper uses to run the hand‑crafted Magnetic‑One system and follows its default settings during experiments.

- Magnetic‑One (hand‑crafted multi‑agent system)
  - description: A mature generalist multi‑agent system included as one of the system types in Who&When; used to collect failure logs for annotation and evaluation.

- GAIA (General AI Assistant benchmark)
  - description: Benchmark providing real‑world assistant tasks; the paper samples validation instances to generate/collect failure logs for Who&When.

- AssistantBench
  - description: Benchmark of realistic, time‑consuming web tasks; the paper uses its validation set to evaluate systems and gather failure logs included in Who&When.

- OpenAI GPT‑4o · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Primary LLM used as both the base model for generated systems (GPT‑4o 2024‑08‑01‑preview) and as the judge in failure‑attribution methods.

- OpenAI GPT‑4‑turbo / GPT‑4o‑mini · [Doc](https://platform.openai.com/docs/models)
  - description: Additional OpenAI models evaluated to test method robustness across LLMs.

- OpenAI o1 · [Doc](https://platform.openai.com/docs/models#o1)
  - description: Strong reasoning model evaluated for automated failure attribution; compared against GPT‑4o and others.

- DeepSeek R1 · [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-R1)
  - description: Reasoning model evaluated as an alternative judge; results reported for both agent‑ and step‑level attribution.

- Llama 3.1 (8B/70B) · [Website](https://llama.meta.com/)
  - description: Open‑source baseline models used to evaluate the attribution methods across different LLM families.

- Qwen 2.5 (7B/72B) · [GitHub](https://github.com/QwenLM/Qwen2.5)
  - description: Open‑source baseline models used in the paper to test generality of the proposed attribution methods across model families.

<!-- paper_id: a0d37ec77dc2acddb223d9a7ac4f23ca10e2908f -->

## 39. Towards Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions - EMNLP - 2024 - citation_count 31 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_31_Findings_Towards_Implicit_Bias_Detection_and_Mitigation_in_Multi-Agent_LLM_Interactions.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.545/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_31_Findings_Towards_Implicit_Bias_Detection_and_Mitigation_in_Multi-Agent_LLM_Interactions.pdf
- Token Usage: input 27140, output 3905, total 31045

### GitHub & Websites

- MultiAgent_ImplicitBias · [GitHub](https://github.com/MichiganNLP/MultiAgent_ImplicitBias)
  - description: Official code and data release for the paper, including the multi-agent interaction framework, the Scenarios/Fine-tune/Test datasets, prompts, and evaluation scripts used to detect and mitigate implicit gender bias.

- Mistral-7B-Instruct-v0.1 · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) · [Doc](https://docs.mistral.ai/models/)
  - description: Open-source LLM checkpoint used as one of the agents and for fine-tuning experiments; the Hugging Face model card provides weights and usage details.

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers/index)
  - description: Library used via the “Hugging Face interface” to run and fine-tune Mistral-7B-Instruct; required to reproduce local fine-tuning and inference.

- Azure OpenAI Service · [Website](https://learn.microsoft.com/azure/ai-services/openai/) · [Doc](https://learn.microsoft.com/azure/ai-services/openai/how-to/fine-tune)
  - description: Cloud API used for inference with GPT‑4 and GPT‑3.5 (gpt‑35‑turbo) and to fine-tune GPT‑3.5; the docs cover model access and fine-tuning steps matching the paper’s setup.

- OpenAI GPT‑3.5 Turbo Fine‑tuning · [Website](https://openai.com/index/gpt-3-5-turbo-fine-tuning-and-api-updates/) · [Doc](https://platform.openai.com/docs/guides/fine-tuning)
  - description: Official OpenAI resources referenced for GPT‑3.5 fine‑tuning and API usage; relevant for practitioners reproducing the fine‑tune of gpt‑35‑turbo (via OpenAI or Azure endpoints).

<!-- paper_id: b92ec2ef54e4df2d08cbc66e4dda3e37b6362dbd -->

## 40. COMBO: Compositional World Models for Embodied Multi-Agent Cooperation - ICLR - 2025 - citation_count 30 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_30_Poster_COMBO_Compositional_World_Models_for_Embodied_Multi-Agent_Cooperation.pdf
- Link: https://openreview.net/pdf/54baa5d646fd9a8bbe3d264abfa7e975492da99d.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_30_Poster_COMBO_Compositional_World_Models_for_Embodied_Multi-Agent_Cooperation.pdf
- Token Usage: input 25926, output 4429, total 30355

### GitHub & Websites

- COMBO (Compositional World Models for Embodied Multi-Agent Cooperation) · [Website](https://umass-embodied-agi.github.io/COMBO/)
  - description: Official project page with videos/demos for the paper’s method, benchmarks (TDW-Game, TDW-Cook), and qualitative results.

- ThreeDWorld (TDW) · [GitHub](https://github.com/threedworld-mit/tdw) · [Website](https://threedworld.org) · [Doc](https://tdw.readthedocs.io)
  - description: 3D simulator used to instantiate the paper’s embodied multi-agent benchmarks (TDW-Game, TDW-Cook) and collect large-scale rollouts.

- AVDC (Learning to Act from Actionless Videos through Dense Correspondences) · [GitHub](https://github.com/apple/ml-avdc)
  - description: Video-diffusion codebase the authors build upon (with architectural modifications) to implement their compositional world model.

- LLaVA (Large Language and Vision Assistant) · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io)
  - description: Vision-language model fine-tuned and used to implement COMBO’s planning sub-modules (Action Proposer, Intent Tracker, Outcome Evaluator).

- T5 (Text-To-Text Transfer Transformer) · [GitHub](https://github.com/google-research/text-to-text-transfer-transformer) · [Doc](https://github.com/google-research/text-to-text-transfer-transformer#readme)
  - description: The paper uses a T5-XXL encoder to preprocess text action prompts for the video diffusion model.

- BERT · [GitHub](https://github.com/google-research/bert) · [Codewiki](https://codewiki.google/github.com/google-research/bert)
  - description: Used to encode textual actions in the Recurrent World Models baseline (VAE + MDN-RNN).

- DDIM (Denoising Diffusion Implicit Models) · [GitHub](https://github.com/ermongroup/ddim)
  - description: Sampling method used across experiments to generate videos from the compositional diffusion world model.

- MAPPO (Multi-Agent PPO) · [GitHub](https://github.com/marlbenchmark/on-policy)
  - description: Cooperative MARL baseline re-implemented per “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games” used for comparison on TDW-Game/TDW-Cook.

- World Models (VAE + MDN-RNN controller) · [GitHub](https://github.com/hardmaru/WorldModels)
  - description: Baseline implementation framework the authors adapt for a recurrent world model comparator.

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient fine-tuning method used to finetune LLaVA for COMBO’s planning sub-modules.

- GroundingDINO · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
  - description: Suggested alternative for identifying action-relevant regions when setting agent-dependent loss scaling without reachability info (Appendix B).

- Segment Anything 2 (SAM 2) · [GitHub](https://github.com/facebookresearch/segment-anything-2)
  - description: Suggested tool to segment relevant regions to adjust loss scaling for the video diffusion training when reachability masks are unavailable (Appendix B).

- CMA-ES (pycma) · [GitHub](https://github.com/CMA-ES/pycma)
  - description: Evolution strategy used to train simple controllers in the Recurrent World Models baseline.

<!-- paper_id: 33f421e1c921f10e608ae6249be4b79633909d28 -->

## 41. Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools - ACL - 2025 - citation_count 32 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_32_Long_Agentic_Reasoning_A_Streamlined_Framework_for_Enhancing_LLM_Reasoning_with_Agentic_Tools.pdf
- Link: https://aclanthology.org/2025.acl-long.1383/
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_32_Long_Agentic_Reasoning_A_Streamlined_Framework_for_Enhancing_LLM_Reasoning_with_Agentic_Tools.pdf
- Token Usage: input 17188, output 6326, total 23514

### GitHub & Websites

- Agentic Reasoning (official code and data)
  - description: Official resources for this paper; the authors state “Our code and data are publicly available.” (links not provided in the PDF).

- GraphRAG · [GitHub](https://github.com/microsoft/graphrag) · [Codewiki](https://codewiki.google/github.com/microsoft/graphrag) · [Doc](https://microsoft.github.io/graphrag/)
  - description: Knowledge-graph based RAG framework; used to build the Mind-Map agent for graph construction and Graph-RAG retrieval in the method.

- DeepSeek-R1 (Reasoning model) · [Website](https://www.deepseek.com/)
  - description: Primary reasoning LLM used throughout experiments and tool-calling; serves as the base model that Agentic Reasoning augments.

- DeepSeek-V3 (LLM for agents) · [Website](https://www.deepseek.com/)
  - description: Used for the Web-Search agent’s query breakdown and RAG, and for Mind-Map graph construction/retrieval.

- QwQ-32B (Qwen-QwQ) · [Website](https://qwenlm.github.io/blog/qwq-32b-preview/)
  - description: Alternative open-source reasoning model evaluated with Agentic Reasoning to show generality across bases.

- Bing Web Search API · [Doc](https://learn.microsoft.com/bing/search-apis/bing-web-search/overview)
  - description: Search service used by the Web-Search agent to retrieve top web pages.

- Cohere ReRank 3.5 · [Doc](https://docs.cohere.com/docs/rerank)
  - description: Re-ranking model used to score and filter retrieved web pages in the Web-Search agent.

- Claude 3.5 Sonnet · [Doc](https://docs.anthropic.com/claude)
  - description: Coding LLM used by the Coding agent to write and execute code, returning results in natural language.

- Python 3.11 · [Website](https://docs.python.org/3.11/)
  - description: Runtime used by the Coding agent to execute generated code.

- leidenalg (Leiden community detection) · [GitHub](https://github.com/vtraag/leidenalg) · [Doc](https://leidenalg.readthedocs.io/en/stable/)
  - description: Community clustering algorithm applied to the knowledge graph in the Mind-Map agent.

- Hugging Face Transformers Agents (Agent Toolbox) · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers/agents)
  - description: External agent toolbox used in ablation (7-tool setup) to compare against the proposed three-agent design.

- LangChain Agents · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Doc](https://python.langchain.com/docs/modules/agents/)
  - description: Large agent/toolbox baseline (109 tools) used in ablations to study tool quantity vs. quality.

- GPQA · [GitHub](https://github.com/Idavidrein/gpqa) · [Website](https://huggingface.co/datasets/Idavidrein/gpqa)
  - description: Graduate-level science QA benchmark used for evaluation (Physics, Chemistry, Biology).

- GAIA Benchmark · [GitHub](https://github.com/GAIA-benchmark/GAIA) · [Website](https://huggingface.co/datasets/GAIA-benchmark/GAIA)
  - description: General AI assistant benchmark (Levels 1–3) used to evaluate agentic reasoning, browsing, and tool-use.

- FreshWiki · [Website](https://huggingface.co/datasets/stanford-oval/FreshWiki)
  - description: Dataset of recent Wikipedia articles used to evaluate deep research and long-form generation.

- STORM · [GitHub](https://github.com/stanford-oval/storm) · [Codewiki](https://codewiki.google/github.com/stanford-oval/storm)
  - description: Agent-based writing system baseline (for FreshWiki/long-form research) compared against Agentic Reasoning.

- Search-O1 · [GitHub](https://github.com/THUIR/Search-O1) · [Website](https://arxiv.org/abs/2501.05366)
  - description: Agentic search-enhanced reasoning baseline compared in experiments and web-search ablations.

- MemGPT · [GitHub](https://github.com/cpacker/MemGPT)
  - description: Long-term memory framework used as a comparative memory strategy in the appendix ablation.

<!-- paper_id: 9f88f56b592e1c1b8d012affec24ed6ca52d9e2b -->

## 42. Mixture-of-Agents Enhances Large Language Model Capabilities - ICLR - 2025 - citation_count 235 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_235_Spotlight_Mixture-of-Agents_Enhances_Large_Language_Model_Capabilities.pdf
- Link: https://openreview.net/pdf/6ec35d5989095ecc687734c8a3549468e2ce33d9.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_235_Spotlight_Mixture-of-Agents_Enhances_Large_Language_Model_Capabilities.pdf
- Token Usage: input 21952, output 6513, total 28465

### GitHub & Websites

- Mixture-of-Agents (MoA) · [GitHub](https://github.com/togethercomputer/moa)
  - description: Official code release for the paper; includes prompts and evaluation scripts to reproduce the Mixture-of-Agents results reported (footnote link in the paper).

- Together AI Inference API · [Website](https://api.together.ai) · [Doc](https://docs.together.ai/docs/inference)
  - description: Inference endpoint used to run all open-source LLMs in the paper (the text cites the Together playground and models pages for running and pricing).

- AlpacaEval 2.0 · [GitHub](https://github.com/tatsu-lab/alpaca_eval) · [Website](https://tatsu-lab.github.io/alpaca_eval/)
  - description: Primary benchmark used (length-controlled win rate) to evaluate MoA and baselines.

- Arena-Hard · [Dataset](https://huggingface.co/datasets/lmarena/arena-hard) · [Website](https://lmsys.org/blog/2024-06-21-arena-hard/)
  - description: Hard instruction-following benchmark used for evaluation and budget analysis.

- MT-Bench · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat) · [Website](https://lmsys.org/blog/2023-06-22-mt-bench/)
  - description: Multi-turn evaluation benchmark (and evaluation scripts via FastChat) used to score models; authors report turn-based scores.

- FLASK · [GitHub](https://github.com/kaistAI/FLASK)
  - description: Fine-grained alignment skill evaluation benchmark used to break down MoA performance across skills.

- Qwen (Qwen1.5 family) · [GitHub](https://github.com/QwenLM/Qwen) · [Codewiki](https://codewiki.google/github.com/QwenLM/Qwen) · [Website](https://qwenlm.ai)
  - description: Major open-source LLMs used as proposers and as the final aggregator (e.g., Qwen1.5-110B-Chat, Qwen1.5-72B-Chat).

- WizardLM · [GitHub](https://github.com/nlpxucan/WizardLM)
  - description: Open-source LLM family (e.g., WizardLM-2-8x22B) used as proposers and analyzed for proposer/aggregator specialization.

- Llama 3 (70B Instruct and family) · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: Open-source Meta models used as proposers/aggregators in MoA and in small-model experiments.

- Mixtral-8x22B v0.1 · [Website](https://mistral.ai) · [Doc](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1)
  - description: Open-source MoE model used among MoA proposers/aggregators.

- DBRX Instruct · [Website](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) · [Doc](https://huggingface.co/databricks/dbrx-instruct)
  - description: Open LLM from Databricks/Mosaic used as one of the MoA agents.

- OpenAI GPT-4o · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Proprietary model used as the final aggregator in a MoA variant and as reference in benchmarks/cost plots.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: Math reasoning benchmark used in the paper’s reasoning evaluations and MoA layer-depth ablation.

- BIG-bench Hard (BBH) · [GitHub](https://github.com/suzgunmirac/BIG-bench-Hard)
  - description: Challenging subset of BIG-bench used to assess reasoning improvements with MoA.

- MMLU · [GitHub](https://github.com/hendrycks/test)
  - description: Knowledge and reasoning benchmark used in the paper’s reasoning evaluation suite.

- CommonsenseQA 2.0 (CSQA2) · [Website](https://allenai.org/data/csqa2) · [Doc](https://leaderboard.allenai.org/csqa2)
  - description: Commonsense reasoning dataset used to test MoA’s reasoning performance.

- MAD (Multi-Agent Debate) · [GitHub](https://github.com/Skytliang/MAD)
  - description: Multi-agent baseline adapted by the authors for open-ended chat comparison in Table 5.

- ReConcile · [GitHub](https://github.com/JustinChihYaoChen/ReConcile)
  - description: Multi-agent consensus baseline adapted by the authors for open-ended chat comparison in Table 5.

- RapidFuzz python-Levenshtein · [GitHub](https://github.com/rapidfuzz/python-Levenshtein)
  - description: Library used to compute Levenshtein similarity in the paper’s correlation analyses (Appendix B).

- Gemma 2 (9B) · [Website](https://ai.google.dev/gemma) · [Doc](https://huggingface.co/google/gemma-2-9b)
  - description: Small open-source model used in the small-model MoA experiments as an aggregator and proposer.

- Llama 3.1 8B Instruct · [Doc](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  - description: Small model variant used in small-model MoA evaluations.

- Mistral-7B-Instruct-v0.3 · [Doc](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  - description: Small instruct model used in the small-model MoA experiments.

- Qwen1.5-7B-Chat · [Doc](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)
  - description: Small Qwen model used as a proposer in small-model MoA evaluations.

<!-- paper_id: 2b3ad2fdd9d2013119232ee49e6d21eb08474b74 -->

## 43. Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration - ACL - 2025 - citation_count 46 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ACL_46_Findings_Towards_Efficient_LLM_Grounding_for_Embodied_Multi-Agent_Collaboration.pdf
- Link: https://aclanthology.org/2025.findings-acl.84/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ACL_46_Findings_Towards_Efficient_LLM_Grounding_for_Embodied_Multi-Agent_Collaboration.pdf
- Token Usage: input 44039, output 4077, total 48116

### GitHub & Websites

- ReAd (Reinforced Advantage for LLM Grounding) · [Website](https://read-llm.github.io/)
  - description: Official project page for the paper’s method; hosts results and resources for reproducing the ReAd framework and the DV-RoCoBench variants used in the paper.

- RoCo / RoCoBench · [GitHub](https://github.com/roco-llm/roco) · [Website](https://roco-llm.github.io/)
  - description: Multi-robot collaboration system and benchmark that the paper extends into DV-RoCoBench and also uses as a strong baseline (RoCo) and simulator for tabletop tasks.

- Overcooked-AI · [GitHub](https://github.com/HumanCompatibleAI/overcooked_ai)
  - description: Cooperative multi-agent environment used for additional experiments (Cramped Room and Forced Coordination) to evaluate ReAd’s effectiveness.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Closed-loop LLM reasoning-and-acting baseline compared against ReAd-J in the experiments.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: LLM self-reflection baseline used for comparison with ReAd-J.

- MindAgent · [GitHub](https://github.com/OpenGVLab/MindAgent)
  - description: Multi-agent LLM baseline evaluated alongside ReAd-J in DV-RoCoBench and Overcooked-AI.

- OpenAI GPT-4 Turbo · [Website](https://openai.com/api/) · [Doc](https://platform.openai.com/docs/models#gpt-4-turbo)
  - description: Closed-source LLM used as the base policy µ for all primary experiments in the paper.

- Llama 3.1 70B Instruct · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open-source LLM evaluated in an extended experiment to show ReAd’s applicability beyond closed-source models.

- BERT · [GitHub](https://github.com/google-research/bert) · [Codewiki](https://codewiki.google/github.com/google-research/bert)
  - description: Pretrained language encoder used in the critic network to extract textual features for value/advantage estimation.

<!-- paper_id: e1b62c7ee4e22ab63e3b0c9968563e6675833e36 -->

## 44. AgentReview: Exploring Peer Review Dynamics with LLM Agents - EMNLP - 2024 - citation_count 52 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_52_Main_AgentReview_Exploring_Peer_Review_Dynamics_with_LLM_Agents.pdf
- Link: https://aclanthology.org/2024.emnlp-main.70/
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_52_Main_AgentReview_Exploring_Peer_Review_Dynamics_with_LLM_Agents.pdf
- Token Usage: input 24169, output 3410, total 27579

### GitHub & Websites

- AgentReview · [GitHub](https://github.com/Ahren09/AgentReview) · [Website](https://agentreview.github.io/)
  - description: Official codebase and project page for the paper’s LLM-agent simulation framework of peer review; primary resource to reproduce experiments and access assets described in the paper.

- OpenReview (openreview-py) · [GitHub](https://github.com/openreview/openreview-py) · [Website](https://openreview.net/)
  - description: API and platform used to retrieve ICLR 2020–2023 submissions; provides the paper data the authors use to run simulations.

- OpenAI GPT-4 API · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: The gpt-4-1106-preview model is the backbone for all agents (reviewers, authors, ACs); required to replicate the simulation runs and prompts.

- LlamaIndex · [GitHub](https://github.com/run-llama/llama_index) · [Codewiki](https://codewiki.google/github.com/run-llama/llama_index) · [Website](https://www.llamaindex.ai/)
  - description: Used in Appendix A.5 to extract and match major comments between human and LLM-generated reviews during validation analyses.

- BERTScore · [GitHub](https://github.com/Tiiiger/bert_score)
  - description: Metric used to quantify similarity between reviews and meta-reviews when evaluating AC involvement strategies.

- Sentence-Transformers (Sentence-BERT) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net/)
  - description: Used to compute sentence embedding similarity between reviews and meta-reviews as part of content-level analysis.

- AutoGPT · [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
  - description: Referenced agent-based modeling framework; cited as foundational to the LLM-agent paradigm that AgentReview builds upon and useful for extensions to multi-agent setups.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework cited in related work; relevant open-source toolkit practitioners can inspect when extending LLM-agent simulations similar to AgentReview.

- ChatArena · [GitHub](https://github.com/chatarena/chatarena)
  - description: Multi-agent language game environment cited as related work; provides open-source infrastructure that can inform alternative simulation designs.

<!-- paper_id: 9348b7b95982d0a675a767e92c23647aa6915a94 -->

## 45. Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers - ICLR - 2025 - citation_count 236 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_236_Poster_Can_LLMs_Generate_Novel_Research_Ideas_A_Large-Scale_Human_Study_with_100+_NLP_Researchers.pdf
- Link: https://openreview.net/pdf/ba931cac8ba9b4d58275b6fc0d74bc21f4ffc882.pdf
- Tags: multiagent, tool, science, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_236_Poster_Can_LLMs_Generate_Novel_Research_Ideas_A_Large-Scale_Human_Study_with_100+_NLP_Researchers.pdf
- Token Usage: input 58000, output 3562, total 61562

### GitHub & Websites

- AI-Researcher · [GitHub](https://github.com/NoviScl/AI-Researcher)
  - description: Official release for this paper; contains the LLM ideation agent implementation (retrieval, generation, ranking) and the full human review scores used in the study.

- Semantic Scholar API · [Website](https://www.semanticscholar.org/product/api) · [Doc](https://api.semanticscholar.org/api-docs/)
  - description: Paper-retrieval backend used in the agent’s RAG stage; the agent issues function calls to this API to fetch papers, references, and metadata for grounding idea generation.

- Sentence-Transformers · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Doc](https://www.sbert.net/) · [Doc](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - description: Library and model used for idea de-duplication; the study encodes ideas with all-MiniLM-L6-v2 and filters pairs by cosine similarity.

- Anthropic Claude 3.5 Sonnet · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/)
  - description: Primary backbone LLM for the ideation agent (retrieval planning, idea generation, and pairwise ranking).

- OpenAI GPT-4o · [Website](https://openai.com/index/gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Comparative model used in ablations for idea diversity and ranking accuracy.

- OpenAI o1-mini · [Website](https://openai.com/index/oi/) · [Doc](https://platform.openai.com/docs/models#o1)
  - description: Additional baseline model evaluated in the paper’s diversity ablations.

- Meta Llama 3.1 (405B Instruct) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/blog/meta-llama-3-1/) · [Doc](https://llama.meta.com/)
  - description: Open-weight model used in ablations comparing idea diversity across different base LLMs.

<!-- paper_id: 110f5dc6d5bfe67138d64c261d6851c727021d1f -->

## 46. GPUDrive: Data-driven, multi-agent driving simulation at 1 million FPS - ICLR - 2025 - citation_count 22 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_22_Poster_GPUDrive_Data-driven,_multi-agent_driving_simulation_at_1_million_FPS.pdf
- Link: https://openreview.net/pdf/58416eb8dcfad96ca7cb5ada85252ab5f1d42c94.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_22_Poster_GPUDrive_Data-driven,_multi-agent_driving_simulation_at_1_million_FPS.pdf
- Token Usage: input 18804, output 4364, total 23168

### GitHub & Websites

- GPUDrive · [GitHub](https://github.com/Emerge-Lab/gpudrive)
  - description: Official code release of the paper, including the simulator, Dockerfiles, training loops (IPPO), and pre-trained agents used in the experiments.

- Waymo Open Motion Dataset (WOMD) · [Website](https://waymo.com/open/data/motion/)
  - description: Primary dataset used by GPUDrive; provides maps, human trajectories, and road objects for over 100k traffic scenarios.

- Hugging Face Datasets (GPUDrive-preprocessed WOMD JSONs) · [Website](https://huggingface.co/datasets) · [Doc](https://huggingface.co/docs/datasets)
  - description: The paper states the preprocessed WOMD JSON files used to initialize the simulator are hosted on the Hugging Face Datasets hub.

- Madrona Game Engine · [GitHub](https://github.com/madrona-engine/madrona)
  - description: The high-performance ECS simulation engine GPUDrive is built on; provides GPU acceleration, collision checking, and sensor utilities.

- Waymax · [GitHub](https://github.com/waymo-research/waymax)
  - description: JAX-based, GPU-accelerated driving simulator; GPUDrive borrows its simplified bicycle model and compares throughput against it.

- Nocturne · [GitHub](https://github.com/facebookresearch/nocturne)
  - description: CPU-based driving simulator used as a primary baseline for speed and training-time comparisons.

- Gymnasium · [GitHub](https://github.com/Farama-Foundation/Gymnasium) · [Doc](https://gymnasium.farama.org/)
  - description: API standard that GPUDrive adheres to; the paper provides Gymnasium-compatible environments for both torch and jax.

- PufferLib · [GitHub](https://github.com/PufferAI/pufferlib) · [Doc](https://pufferai.github.io/pufferlib/)
  - description: Library used for the high-throughput IPPO implementation in the end-to-end training benchmarks.

- Stable-Baselines3 · [GitHub](https://github.com/DLR-RM/stable-baselines3) · [Doc](https://stable-baselines3.readthedocs.io/)
  - description: Alternative (slower) IPPO implementation mentioned as available in the GPUDrive repo.

- PyTorch · [GitHub](https://github.com/pytorch/pytorch) · [Doc](https://pytorch.org/docs/stable/)
  - description: One of the two backends for GPUDrive’s Gymnasium environments and training loops.

- JAX · [GitHub](https://github.com/google/jax) · [Codewiki](https://codewiki.google/github.com/google/jax) · [Doc](https://jax.readthedocs.io/)
  - description: Second backend for GPUDrive’s Gymnasium environments; also used by comparison simulators like Waymax.

- nanobind · [GitHub](https://github.com/wjakob/nanobind)
  - description: C++/Python binding tool used to expose GPUDrive’s C++ engine through a Pythonic interface.

<!-- paper_id: 847146644e2df7b586beb1a7d0425984ccb02e1c -->

## 47. Large Language Model-based Human-Agent Collaboration for Complex Task Solving - EMNLP - 2024 - citation_count 35 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_35_Findings_Large_Language_Model-based_Human-Agent_Collaboration_for_Complex_Task_Solving.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.72/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_35_Findings_Large_Language_Model-based_Human-Agent_Collaboration_for_Complex_Task_Solving.pdf
- Token Usage: input 22656, output 3210, total 25866

### GitHub & Websites

- ReHAC · [GitHub](https://github.com/XueyangFeng/ReHAC)
  - description: Official code and released datasets for the paper’s Reinforcement Learning-based Human-Agent Collaboration method; primary resource to reproduce experiments and train the collaboration policy.

- HotpotQA · [GitHub](https://github.com/hotpotqa/hotpot) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used for real and simulated human–agent collaboration experiments and evaluation in the paper.

- StrategyQA · [GitHub](https://github.com/allenai/strategyqa) · [Website](https://allenai.org/data/strategyqa)
  - description: Implicit reasoning QA dataset used for training/evaluation under the ReAct framework and GPT-4-simulated human setting.

- InterCode · [GitHub](https://github.com/princeton-nlp/InterCode) · [Website](https://princeton-nlp.github.io/InterCode/)
  - description: Interactive coding benchmark (SQL split used) providing the environment and “Try Again” prompting setup for coding experiments.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reason+Act agent framework the paper adopts for QA tasks; provides action space (Search/Lookup/Finish) and Wikipedia-API tooling used in their setup.

- Reflexion · [GitHub](https://github.com/noahshinn/reflexion)
  - description: Agent prompting framework with verbal reinforcement learning; used to test ReHAC’s generalization across prompt frameworks.

- PEFT (Parameter-Efficient Fine-Tuning) · [GitHub](https://github.com/huggingface/peft) · [Codewiki](https://codewiki.google/github.com/huggingface/peft)
  - description: LoRA-based fine-tuning toolkit used to train the Llama2 collaboration policy model.

- Llama 2–7B (Hugging Face) · [Doc](https://huggingface.co/meta-llama/Llama-2-7b-hf) · [Website](https://ai.meta.com/llama/)
  - description: Base model used as the collaboration policy π_collab with LoRA tuning in most experiments.

- Llama 2–13B (Hugging Face) · [Doc](https://huggingface.co/meta-llama/Llama-2-13b-hf) · [Website](https://ai.meta.com/llama/)
  - description: Larger variant used for scaling analysis of the collaboration policy model.

- OpenAI ChatGPT / GPT-4 API · [Doc](https://platform.openai.com/docs)
  - description: Closed-source LLMs used as the task policy (ChatGPT gpt-3.5-turbo-0613) and as simulated humans (GPT-4 gpt-4-0613) for data collection and evaluation.

- Wikipedia API (MediaWiki) · [Doc](https://www.mediawiki.org/wiki/API:Main_page)
  - description: Web API backing the ReAct tool environment (Search/Lookup) used in QA experiments.

<!-- paper_id: a70f0f9b9b9dc7d5caadcb23a551ea4213727548 -->

## 48. AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML - ICML - 2025 - citation_count 44 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICML_44_Poster_AutoML-Agent_A_Multi-Agent_LLM_Framework_for_Full-Pipeline_AutoML.pdf
- Link: https://openreview.net/pdf/e11df054e7e2b676ce0272b6b5318b01b7327210.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICML_44_Poster_AutoML-Agent_A_Multi-Agent_LLM_Framework_for_Full-Pipeline_AutoML.pdf
- Token Usage: input 50895, output 6862, total 57757

### GitHub & Websites

- AutoML-Agent · [GitHub](https://github.com/deepauto-ai/automl-agent)
  - description: Official code release for the paper’s multi-agent full-pipeline AutoML framework; used to reproduce all methods, planning, execution, and verification components.

- AutoGluon · [GitHub](https://github.com/autogluon/autogluon) · [Website](https://auto.gluon.ai)
  - description: State-of-the-art AutoML toolkit used as a baseline (Tabular, TimeSeries, Multimodal variants) in the experiments.

- DS-Agent
  - description: LLM-based data science agent used as a comparison baseline; the paper reproduces results with the official source code and case banks.

- SELA: Tree-search enhanced LLM agents for AutoML · [Website](https://arxiv.org/abs/2410.17238)
  - description: Training-based MCTS AutoML agent compared against AutoML-Agent for speed and performance in additional analysis.

- Time-Series-Library (TSLib) · [GitHub](https://github.com/thuml/Time-Series-Library)
  - description: Repository providing the Weather and Electricity forecasting datasets and evaluation setup used in the experiments.

- PyTorch Geometric · [GitHub](https://github.com/pyg-team/pytorch_geometric) · [Codewiki](https://codewiki.google/github.com/pyg-team/pytorch_geometric) · [Doc](https://pytorch-geometric.readthedocs.io)
  - description: Library used to load Planetoid graph datasets (Cora, Citeseer) and implement node classification baselines per the paper’s setup.

- Gradio · [GitHub](https://github.com/gradio-app/gradio) · [Codewiki](https://codewiki.google/github.com/gradio-app/gradio) · [Website](https://www.gradio.app)
  - description: Deployment toolkit used by the Operation Agent to produce web endpoints for trained models.

- PyTorch · [Website](https://pytorch.org) · [Doc](https://pytorch.org/docs/stable/index.html)
  - description: Core deep learning framework used to implement training, fine-tuning, and evaluation pipelines across tasks.

- scikit-learn · [GitHub](https://github.com/scikit-learn/scikit-learn) · [Codewiki](https://codewiki.google/github.com/scikit-learn/scikit-learn) · [Doc](https://scikit-learn.org/stable/)
  - description: Classic ML toolkit used for tabular baselines, metrics, and utilities (e.g., KMeans, train/test split) in generated pipelines.

- OpenAI API (GPT‑4o, GPT‑3.5‑turbo) · [Website](https://platform.openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Closed-source LLMs used as backbone models for agents and zero-shot baselines; the paper specifies model versions and API usage.

- Mixtral‑8x7B‑Instruct‑v0.1 · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Instruction-tuned LLM used as the Prompt Agent backbone after additional instruction tuning.

- WizardLM / Evol‑Instruct · [GitHub](https://github.com/nlpxucan/WizardLM)
  - description: Method used to automatically generate ~2.3K instruction–response pairs for tuning the Prompt Agent’s JSON parsing.

- iTransformer · [GitHub](https://github.com/thuml/iTransformer)
  - description: Strong time-series forecasting model used as a human-designed baseline for the Weather/Electricity tasks.

- TabPFN · [GitHub](https://github.com/automl/TabPFN)
  - description: State-of-the-art tabular classification model used as a human baseline for tabular tasks.

- Papers with Code · [Website](https://paperswithcode.com)
  - description: External knowledge source used in the retrieval-augmented planning (RAP) stage to summarize SOTA models and practices.

- Kaggle (hub, data, notebooks) · [Website](https://www.kaggle.com)
  - description: Data and notebook hub used both for dataset sourcing and retrieval-augmented planning; also hosts human-baseline notebooks cited in the paper.

- Hugging Face Hub · [Website](https://huggingface.co)
  - description: Model and dataset repository used as a retrieval source by the Data and Model Agents when searching for assets.

- UCI Machine Learning Repository · [Website](https://archive.ics.uci.edu)
  - description: Source used to retrieve the Higher Education Students Performance dataset via ucimlrepo as specified in the tasks.

- Butterfly Image Classification (dataset) · [Website](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification)
  - description: Image classification dataset used in experiments; one of the two computer vision benchmarks.

- Shopee-IET (competition data) · [Website](https://www.kaggle.com/competitions/demo-shopee-iet-competition/data)
  - description: Clothing-category image dataset used for classification experiments.

- Ecommerce Text Classification (dataset) · [Website](https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification)
  - description: Four-class text classification dataset used in the NLP experiments.

- Banana Quality (dataset) · [Website](https://www.kaggle.com/datasets/l3llff/banana/data)
  - description: Tabular binary classification dataset used for evaluating tabular pipelines.

- Crop Price Prediction (dataset) · [Website](https://www.kaggle.com/datasets/varshitanalluri/crop-price-prediction-dataset)
  - description: Tabular regression dataset used to evaluate regression pipelines (RMSLE).

- Higher Education Students Performance (dataset) · [Website](https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation)
  - description: UCI dataset repurposed for clustering (RI metric) in the experiments.

- Butterfly baseline notebook (ResNet-18) · [Website](https://www.kaggle.com/code/mohamedhassanali/butterfly-classify-pytorch-pretrained-model-acc-99/notebook)
  - description: Kaggle notebook used as a human-designed baseline for the Butterfly image classification task.

- E-commerce text baseline (Word2Vec + XGBoost) · [Website](https://www.kaggle.com/code/sugataghosh/e-commerce-text-classification-tf-idf-word2vec#Word2Vec-Hyperparameter-Tuning)
  - description: Kaggle notebook used as a human baseline for the Ecommerce text classification task.

- Textual Entailment baseline (XLM‑RoBERTa) · [Website](https://www.kaggle.com/code/vbookshelf/basics-of-bert-and-xlm-roberta-pytorch)
  - description: Kaggle notebook used as a human baseline for the textual entailment task.

- Crab Age regression baseline · [Website](https://www.kaggle.com/code/shatabdi5/crab-age-regression)
  - description: Kaggle notebook used as a human baseline for the Crab Age regression dataset.

- Crop yield regression baseline · [Website](https://www.kaggle.com/code/mahmoudmagdyelnahal/crop-yield-prediction-99/notebook)
  - description: Kaggle notebook used as a human baseline for crop price/yield regression.

- Unsupervised clustering baseline (KMeans) · [Website](https://www.kaggle.com/code/samuelcortinhas/tps-july-22-unsupervised-clustering)
  - description: Kaggle notebook approach referenced for human baseline in clustering tasks (RI metric).

<!-- paper_id: efde8940a0b924e93d35184c4a1e8f9670b94fe7 -->

## 49. ToolACE: Winning the Points of LLM Function Calling - ICLR - 2025 - citation_count 87 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_87_Poster_ToolACE_Winning_the_Points_of_LLM_Function_Calling.pdf
- Link: https://openreview.net/pdf/9c7c53cc6199348d235063c044442216d84429c4.pdf
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_87_Poster_ToolACE_Winning_the_Points_of_LLM_Function_Calling.pdf
- Token Usage: input 23938, output 5479, total 29417

### GitHub & Websites

- ToolACE Releases (Team-ACE) · [Website](https://huggingface.co/Team-ACE)
  - description: Official release by the paper; provides ToolACE-8B checkpoints and a subset of the synthesized function-calling data used in experiments.

- Berkeley Function-Calling Leaderboard (BFCL) · [Website](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
  - description: Primary benchmark and evaluation framework used in the paper (AST, Executable, Relevance/Irrelevance, Multi-turn); all main results are reported on BFCL.

- API-Bank
  - description: Secondary benchmark used to evaluate planning/retrieval/calling; the paper reports “Call” and “Retrieval+Call” accuracy on API-Bank.

- Llama 3 / Llama 3.1 · [GitHub](https://github.com/meta-llama/llama3) · [Doc](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)
  - description: Base model family; the authors fine-tune LLaMA‑3.1‑8B‑Instruct as their main backbone.

- Qwen 1.5 · [GitHub](https://github.com/QwenLM/Qwen) · [Codewiki](https://codewiki.google/github.com/QwenLM/Qwen) · [Website](https://huggingface.co/Qwen)
  - description: Alternative backbone models used for scaling and backbone studies (0.5B–7B Chat variants).

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter‑efficient fine‑tuning method employed to train ToolACE models.

- Gorilla · [GitHub](https://github.com/ShishirPatil/gorilla) · [Website](https://gorilla.cs.berkeley.edu/)
  - description: Tool‑use baseline family and related ecosystem; BFCL originates from the Gorilla team and Gorilla‑OpenFunctions‑v2 is a comparison baseline.

- ToolLLM / ToolBench · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Widely used tool‑learning dataset and system; used as a comparison baseline and in additional data training comparisons in the paper.

- ToolAlpaca · [GitHub](https://github.com/tangqiaoyu/ToolAlpaca)
  - description: Simulated tool‑learning dataset/baseline cited in the paper’s comparisons of API coverage and capabilities.

- Functionary · [Website](https://functionary.meetkai.com)
  - description: Function‑calling baseline model evaluated on BFCL and compared against ToolACE‑8B.

- FlagEmbedding (BGE models) · [GitHub](https://github.com/FlagOpen/FlagEmbedding) · [Website](https://huggingface.co/BAAI/bge-base-en)
  - description: Retrieval model used in the paper’s RAG‑based few‑shot in‑context learning experiment in the appendix.

- JSON Schema · [Website](https://json-schema.org/)
  - description: Specification referenced for rule‑based verification; the rule checker validates API definitions against JSON Schema-style structure and constraints.

<!-- paper_id: 0350636522997217df53553ddf3e472338bca97b -->

## 50. Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains - ICLR - 2025 - citation_count 51 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_51_Poster_Multiagent_Finetuning_Self_Improvement_with_Diverse_Reasoning_Chains.pdf
- Link: https://openreview.net/pdf/2df12554b353f4c99981918b614fa32dd8d0239c.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_51_Poster_Multiagent_Finetuning_Self_Improvement_with_Diverse_Reasoning_Chains.pdf
- Token Usage: input 28020, output 5023, total 33043

### GitHub & Websites

- Multiagent Finetuning: Self Improvement with Diverse Reasoning Chains · [Website](https://llm-multiagent-ft.github.io)
  - description: Official project page for this ICLR 2025 paper; hosts an overview, results, and resources for reproducing the multiagent finetuning method.

- GSM8K (Grade School Math) · [GitHub](https://github.com/openai/grade-school-math)
  - description: Math word-problem dataset used for finetuning and evaluation; the paper trains on 500 examples and tests on held-out sets.

- MATH Dataset · [GitHub](https://github.com/hendrycks/math)
  - description: Competition-level math dataset used extensively for training and evaluation (including multiple finetuning iterations and zero-shot tests).

- MMLU (Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test)
  - description: General knowledge and reasoning benchmark used for additional evaluation in the appendix.

- Multiagent Debate (Du et al., 2023) · [Website](https://arxiv.org/abs/2305.14325)
  - description: Debate framework that the paper adopts to generate data and conduct inference-time aggregation (majority vote across rounds).

- Phi-3 Mini (128K) Instruct · [GitHub](https://github.com/microsoft/Phi-3CookBook) · [Website](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct)
  - description: One of the base open-source LLMs the paper finetunes end-to-end for multiagent generation and critic roles.

- Mistral 7B Instruct v0.2 · [GitHub](https://github.com/mistralai/mistral-src) · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - description: Base LLM used in experiments; models are finetuned and evaluated under the proposed multiagent finetuning pipeline.

- Llama 3 8B Instruct · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com/) · [Doc](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
  - description: Base LLM used for finetuning and evaluation, including multi-iteration studies and broader benchmarks.

- OpenAI GPT-3.5 Turbo (and Fine-tuning API) · [Website](https://openai.com/blog/chatgpt) · [Doc](https://platform.openai.com/docs/models/gpt-3-5-turbo) · [Doc](https://platform.openai.com/docs/guides/fine-tuning)
  - description: Proprietary model finetuned via OpenAI’s API in the paper to demonstrate applicability beyond open-source models.

- T5 (Text-to-Text Transfer Transformer) · [GitHub](https://github.com/google-research/text-to-text-transfer-transformer) · [Website](https://t5.dev/)
  - description: Used as a held-out encoder (T5-3B) to compute embedding-based diversity metrics for the analysis.

- Gemma 2 · [Website](https://ai.google.dev/gemma)
  - description: Used (2B variant) to estimate likelihoods and compute KL-divergence/likelihood-based diversity metrics in the analysis.

<!-- paper_id: e1078aaf47cd70c95e05b11d29b2cfbb3b4f0168 -->

## 51. MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration - EMNLP - 2024 - citation_count 50 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_50_Main_MAgIC_Investigation_of_Large_Language_Model_Powered_Multi-Agent_in_Cognition,_Adaptability,_Rationality_and_Collaboration.pdf
- Link: https://aclanthology.org/2024.emnlp-main.416/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_50_Main_MAgIC_Investigation_of_Large_Language_Model_Powered_Multi-Agent_in_Cognition,_Adaptability,_Rationality_and_Collaboration.pdf
- Token Usage: input 28820, output 3514, total 32334

### GitHub & Websites

- MAgIC (Multi-Agent in Cognition, Adaptability, Rationality and Collaboration) · [GitHub](https://github.com/cathyxl/MAgIC)
  - description: Official repository for the paper’s benchmark, scenarios, prompts, data, and PGM-aware agent code used to evaluate LLMs in social deduction and game-theory multi-agent settings.

- OpenAI GPT models (GPT-3.5-turbo, GPT-4, GPT-4-turbo, GPT o1) · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs/models)
  - description: Commercial LLMs used as challenger and defender agents in the experiments; the benchmark relies on OpenAI’s API to run these models under fixed temperatures and settings.

- Anthropic Claude 2 · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/claude)
  - description: Evaluated as one of the LLM baselines within the benchmark; accessed via Claude API to measure the seven abilities and win rates.

- Google PaLM 2 · [Website](https://ai.google/discover/palm2) · [Doc](https://ai.google.dev/)
  - description: Included as a commercial LLM baseline in the evaluation; practitioners can reproduce results using Google’s PaLM API/Vertex AI access.

- Cohere Platform (Cohere Command models) · [Website](https://cohere.com/) · [Doc](https://docs.cohere.com/)
  - description: Another evaluated LLM provider; models are invoked via Cohere’s API as challenger agents in the benchmark.

- Llama 2–70B (Chat) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
  - description: Open-source baseline model assessed by the benchmark; practitioners can load the 70B chat model to replicate the Llama-2-70B results reported in the paper.

<!-- paper_id: 72273f7a050529fc71c7d45c0256d2b9754f56bb -->

## 52. LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models - NAACL - 2025 - citation_count 30 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_30_Findings_LLM-Coordination_Evaluating_and_Analyzing_Multi-agent_Coordination_Abilities_in_Large_Language_Models.pdf
- Link: https://aclanthology.org/2025.findings-naacl.448/
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_30_Findings_LLM-Coordination_Evaluating_and_Analyzing_Multi-agent_Coordination_Abilities_in_Large_Language_Models.pdf
- Token Usage: input 25500, output 5677, total 31177

### GitHub & Websites

- Overcooked-AI · [GitHub](https://github.com/HumanCompatibleAI/overcooked_ai)
  - description: Cooperative cooking environment used for Agentic Coordination experiments; the paper uses its PPO/PBT self-play baselines and the human-behavior-cloning agents for cross-play/ZSC.

- Overcooked-AI Human Trajectories (BC dataset) · [GitHub](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/overcooked_ai/data/human_data)
  - description: Human gameplay dataset bundled with Overcooked-AI; the paper uses BC agents trained on this data as human proxies and as partners/opponents in zero-shot coordination.

- Hanabi Learning Environment · [GitHub](https://github.com/deepmind/hanabi-learning-environment) · [Website](https://ai.googleblog.com/2019/06/hanabi-new-challenge-for-reinforcement.html)
  - description: Official environment for the Hanabi Challenge used to run Hanabi Agentic Coordination experiments and evaluate MARL/LLM agents.

- Hanabi SAD/OBL (facebookresearch) · [GitHub](https://github.com/facebookresearch/hanabi_SAD)
  - description: Implementations of Simplified Action Decoder (SAD) and Off-Belief Learning (OBL); the paper uses SAD as a self-play baseline and pairs LLM agents with OBL-1/OBL-4 for cross-play.

- ReAct (Reason+Act) · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting prompting framework; cited as one of the advanced reasoning strategies explored in the paper’s LLM agent scaffolding.

- Vicuna · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat) · [Website](https://lmsys.org/blog/2023-03-30-vicuna/)
  - description: Open-source chat model family evaluated on the CoordinationQA benchmark.

- Mistral 7B · [Website](https://mistral.ai/news/announcing-mistral-7b/) · [GitHub](https://github.com/mistralai/mistral-src)
  - description: Open LLM family evaluated in CoordinationQA; referenced as one of the model families tested.

- Mixtral 8x7B · [Website](https://mistral.ai/news/mixtral-of-experts/) · [Doc](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Mixture-of-Experts open model used in both agentic and CoordinationQA evaluations.

- Llama 2 · [GitHub](https://github.com/facebookresearch/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open foundation/chat models evaluated on the CoordinationQA suite.

- OpenAI GPT models (GPT-4-turbo, GPT-4o, GPT-3.5-turbo) · [Doc](https://platform.openai.com/docs/models)
  - description: Proprietary LLMs used as the primary agents and for QA evaluations (including ablations like ToM reasoning and self-verification).

<!-- paper_id: 7f0d1740e74ce36424d64d608270077b64dfe7c0 -->

## 53. Middleware for LLMs: Tools Are Instrumental for Language Agents in Complex Environments - EMNLP - 2024 - citation_count 45 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_EMNLP_45_Main_Middleware_for_LLMs_Tools_Are_Instrumental_for_Language_Agents_in_Complex_Environments.pdf
- Link: https://aclanthology.org/2024.emnlp-main.436/
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_EMNLP_45_Main_Middleware_for_LLMs_Tools_Are_Instrumental_for_Language_Agents_in_Complex_Environments.pdf
- Token Usage: input 20655, output 5734, total 26389

### GitHub & Websites

- Middleware for LLMs (this paper) · [GitHub](https://github.com/OSU-NLP/Middleware)
  - description: Official code release implementing the middleware tool framework, error-feedback, and decoupled-generation schemes used throughout the paper.

- KBQA-AGENT (dataset) · [Website](https://huggingface.co/datasets/OSU-NLP/KBQA-Agent)
  - description: Official test set curated by the authors for complex KBQA with tool action annotations; used to evaluate agents over Freebase.

- SQLite · [Website](https://www.sqlite.org/) · [Doc](https://www.sqlite.org/docs.html)
  - description: The paper uses SQLite as the standard database engine to execute SQL and to implement database-side tools (e.g., error feedback from sqlite3).

- Virtuoso (OpenLink) · [Website](https://virtuoso.openlinksw.com/) · [Doc](https://docs.openlinksw.com/virtuoso/)
  - description: Employed as the KB query engine for Freebase; backs the KB tool executions (e.g., get_neighbors, intersection, count).

- Freebase · [Website](https://developers.google.com/freebase)
  - description: The target knowledge base used for KBQA tasks in the paper; all KB tools operate over Freebase entities, relations, and attributes.

- BIRD: Big Bench for Large-Scale Database-Grounded Text-to-SQL · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/BIRD) · [Website](https://bird-bench.github.io/)
  - description: Primary database benchmark used without oracle knowledge; evaluates agents’ ability to navigate real DB content via the middleware tools.

- GRAILQA · [GitHub](https://github.com/dki-lab/GrailQA)
  - description: One of the KBQA sources used to construct KBQA-AGENT; contributes complex multi-hop Freebase questions.

- ComplexWebQuestions · [Website](https://www.tau-nlp.org/compwebq)
  - description: Another KBQA source used in building KBQA-AGENT; provides compositional web-based questions mapped to Freebase.

- GraphQuestions (GRAPHQ) · [Website](http://www.cs.cmu.edu/~ark/GraphQuestions/)
  - description: KBQA source dataset included in KBQA-AGENT curation; supplies graph-structured Freebase questions.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: The paper uses ReAct as the backbone prompting framework for reasoning-acting loops, on top of which their error-feedback and decoupled-generation are built.

- Sentence-Transformers (Sentence-BERT) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net/)
  - description: Used in analysis to retrieve top-K KB triples by semantic similarity when testing “direct prompting with sampled triples.”

- OpenAI GPT-3.5/4 · [Website](https://platform.openai.com/docs/models)
  - description: Closed-source LLMs used as main backbones for agent reasoning and tool use in all experiments.

- Llama 2 (Meta) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open-source chat models (7B/13B) evaluated as alternative agent backbones within the middleware framework.

- Mistral 7B Instruct v0.2 · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - description: Open-source LLM evaluated as a backbone in the middleware framework for both DB and KB tasks.

- Mixtral 8x7B Instruct v0.1 · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Open-source mixture-of-experts LLM evaluated as another backbone under the proposed middleware setup.

<!-- paper_id: ac2fc8c5d4a1f44464f1415ea3dd3ed45398b9d9 -->

## 54. TrustAgent: Towards Safe and Trustworthy LLM-based Agents - EMNLP - 2024 - citation_count 40 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_40_Findings_TrustAgent_Towards_Safe_and_Trustworthy_LLM-based_Agents.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.585/
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_40_Findings_TrustAgent_Towards_Safe_and_Trustworthy_LLM-based_Agents.pdf
- Token Usage: input 18535, output 5668, total 24203

### GitHub & Websites

- TrustAgent · [GitHub](https://github.com/agiresearch/TrustAgent)
  - description: Official code and data release for the paper’s Agent-Constitution framework implementing pre-, in-, and post-planning safety strategies, plus the synthetic datasets and evaluation scripts used in the experiments.

- Contriever (facebook/contriever-msmarco) · [Doc](https://huggingface.co/facebook/contriever-msmarco)
  - description: Dense retriever model used in TrustAgent’s in-planning stage to dynamically fetch the top-k relevant safety regulations for prompt conditioning at each plan step.

- OpenAI GPT-4 / GPT-3.5 API · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Closed-source LLM backbones (GPT-4-1106-preview and GPT-3.5-turbo-1106) used for planning and safety inspection; reproductions require access via the OpenAI API.

- Anthropic Claude (Claude-2, Claude-instant) · [Website](https://www.anthropic.com/) · [Doc](https://docs.anthropic.com/claude)
  - description: Closed-source LLM backbones evaluated as planners in the study; access via Anthropic’s API is needed to replicate those results.

- Mixtral-8x7B-Instruct · [Website](https://mistral.ai/) · [Doc](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Open-source LLM backbone used in experiments for long-context planning; the Hugging Face model card provides downloads and usage instructions.

- ToolEmu (LM-emulated Sandbox and Dataset)
  - description: Referenced framework and benchmark from which the paper adopts everyday and finance tasks, and whose LM-emulation idea underpins TrustAgent’s tool-execution sandboxing.

- FTC: CAN-SPAM Act—A Compliance Guide for Business · [Website](https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business)
  - description: Regulation source used to construct housekeeping/email safety rules within the Agent Constitution.

- Housekeeping Safety Training and Tips (Polo & Tweed) · [Website](https://poloandtweed.com/blog/housekeeping-safety-training-and-tips)
  - description: Practical safety guidance referenced to build housekeeping domain regulations for TrustAgent.

- ADT: Financial Safety—Protect Yourself from Risks · [Website](https://www.adt.com/resources/financial-safety-tips)
  - description: Resource used to draft finance-domain safety regulations (e.g., avoiding overdrafts, secure transactions).

- Health.gov: Use Medicines Safely · [Website](https://health.gov/myhealthfinder/healthy-living/safety/use-medicines-safely)
  - description: Official guidance used to form medicine-domain safety regulations (dosage, interactions, labeling).

- NIA: Taking Medicines Safely as You Age · [Website](https://www.nia.nih.gov/health/medicines-and-medication-management/taking-medicines-safely-you-age)
  - description: Source for medicine safety rules addressing age-related considerations and safe medication practices.

- FDA: Safe Food Handling · [Website](https://www.fda.gov/food/buy-store-serve-safe-food/safe-food-handling)
  - description: Official FDA guidelines used to construct food-domain safety regulations (cross-contamination, storage).

- USDA FSIS: Food Safety Basics · [Website](https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/steps-keep-food-safe)
  - description: USDA resource informing food safety regulations in the Agent Constitution (clean, separate, cook, chill).

<!-- paper_id: c9db4ccaf91d0d2e44cb6c6b5b77e25b887739c8 -->

## 55. Simulating Classroom Education with LLM-Empowered Agents - NAACL - 2025 - citation_count 105 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_105_Long_Simulating_Classroom_Education_with_LLM-Empowered_Agents.pdf
- Link: https://aclanthology.org/2025.naacl-long.520/
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_105_Long_Simulating_Classroom_Education_with_LLM-Empowered_Agents.pdf
- Token Usage: input 19191, output 4360, total 23551

### GitHub & Websites

- SimClass · [GitHub](https://github.com/THU-MAIC/SimClass)
  - description: Official code and service released by the paper for their LLM-based multi-agent classroom simulator, including role agents, the session controller, prompts, and the online system used in experiments.

- ZhipuAI GLM-4 (ChatGLM family) · [Website](https://www.zhipuai.cn/) · [Doc](https://open.bigmodel.cn/dev/api)
  - description: The backbone LLM used in the deployed SimClass system (model name glm-4); accessed via ZhipuAI’s API.

- OpenAI GPT-4 (gpt-4-vision-preview, gpt-4-turbo) · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Used for ablation systems (GPT-4V) and for labeling FIAS interaction categories (GPT-4 Turbo) during evaluation.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework whose manager/mediator design is followed for SimClass’s hidden manager agent that selects speakers and actions.

- ChatDev · [GitHub](https://github.com/OpenBMB/ChatDev)
  - description: SOP-style multi-agent framework cited in the method section as a contrast to SimClass’s dynamic session controller (referenced when discussing systems with standardized operating procedures).

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Doc](https://docs.deepwisdom.ai/MetaGPT/latest/)
  - description: Another SOP-style multi-agent framework referenced in the method section as a comparison point for SimClass’s non-SOP classroom control design.

<!-- paper_id: a7a4aafe038f0c78b8b5320b537bdd3fb8c2b28d -->

## 56. Cradle: Empowering Foundation Agents Towards General Computer Control - ICML - 2025 - citation_count 62 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICML_62_Poster_Cradle_Empowering_Foundation_Agents_Towards_General_Computer_Control.pdf
- Link: https://openreview.net/pdf/3d1e188a8d22526e5f74b4daa4e464094b008106.pdf
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICML_62_Poster_Cradle_Empowering_Foundation_Agents_Towards_General_Computer_Control.pdf
- Token Usage: input 70873, output 4221, total 75094

### GitHub & Websites

- CRADLE · [Website](https://baai-agents.github.io/Cradle)
  - description: Official project page for CRADLE with code and video demos; this is the authors’ release for reproducing the framework in the General Computer Control (GCC) setting.

- OSWorld · [GitHub](https://github.com/xlang-ai/OSWorld)
  - description: Comprehensive benchmark and VM environment used by the paper to evaluate CRADLE on 369 real-world computer tasks with automatic evaluation scripts.

- Grounding DINO · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
  - description: Open-set object detector used by CRADLE’s Information Gathering module to provide bounding boxes for GUI/object grounding.

- Segment Anything (SAM) · [GitHub](https://github.com/facebookresearch/segment-anything) · [Codewiki](https://codewiki.google/github.com/facebookresearch/segment-anything) · [Website](https://segment-anything.com/)
  - description: Segmentation model used to augment screenshots (SAM2SOM-style overlays) for robust GUI element grounding in CRADLE.

- PyDirectInput · [GitHub](https://github.com/learncodebygaming/pydirectinput)
  - description: Input library used by CRADLE to send low-level keyboard events compatible with DirectX-based games.

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui) · [Doc](https://pyautogui.readthedocs.io/)
  - description: Cross‑platform GUI automation library used for keyboard/mouse control (especially in OSWorld and standard desktop apps).

- AHK (Python AutoHotkey wrapper) · [GitHub](https://github.com/spyoungtech/ahk)
  - description: Python wrapper for AutoHotkey used by CRADLE to implement reliable, low-level mouse/keyboard control on Windows.

- VideoSubFinder · [Website](https://sourceforge.net/projects/videosubfinder/)
  - description: Tool used to extract keyframes and on-screen text cues from gameplay videos to support CRADLE’s Information Gathering.

- OpenAI GPT‑4o · [Website](https://openai.com/index/hello-gpt-4o/)
  - description: Backbone multimodal model powering CRADLE’s perception and reasoning across all tasks.

- OpenAI Realtime API · [Website](https://openai.com/index/introducing-the-realtime-api/)
  - description: Referenced as a path to reduce LMM latency; relevant for practitioners optimizing CRADLE’s interaction loop.

- OpenAI text-embedding-ada-002 · [Website](https://openai.com/index/new-and-improved-embedding-model/)
  - description: Embedding model used by CRADLE for skill retrieval in procedural memory (similar to Voyager-style retrieval).

- Claude 3 (Opus) · [Website](https://www.anthropic.com/news/claude-3-family)
  - description: Alternative backbone model evaluated as a baseline variant of CRADLE.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Baseline framework re-implemented (“ReAct-like”) for comparison; provides the reasoning+acting prompting pattern.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Baseline framework re-implemented (“Reflexion-like”) with verbal self-reflection, used for comparison to CRADLE.

- Voyager · [GitHub](https://github.com/MineDojo/Voyager)
  - description: Baseline agent paradigm adapted (“Voyager-like”) in the paper; CRADLE also borrows its skill-retrieval idea with embeddings.

- Set‑of‑Mark (SoM) · [GitHub](https://github.com/CASIA-IVA-Lab/SoM)
  - description: Visual grounding prompting method used for OSWorld baselines (GPT‑4o+SoM) to compare against CRADLE’s image-only augmentation.

- AppleScript Language Guide · [Doc](https://developer.apple.com/library/archive/documentation/AppleScript/Conceptual/AppleScriptLangGuide/introduction/ASLR_intro.html)
  - description: Referenced OS-level scripting used by CRADLE to switch windows and control focus on macOS during experiments.

- Microsoft DirectX Graphics · [Doc](https://learn.microsoft.com/en-us/windows/win32/directx)
  - description: Platform note explaining why standard GUI libraries can fail in modern games; relevant for reproducing CRADLE’s low-level input approach.

<!-- paper_id: 52976286324d79c3349cf1cd101aa3a8832d2954 -->

## 57. KG-Agent: An Efficient Autonomous Agent Framework for Complex Reasoning over Knowledge Graph - ACL - 2025 - citation_count 53 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_53_Long_KG-Agent_An_Efficient_Autonomous_Agent_Framework_for_Complex_Reasoning_over_Knowledge_Graph.pdf
- Link: https://aclanthology.org/2025.acl-long.468/
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_53_Long_KG-Agent_An_Efficient_Autonomous_Agent_Framework_for_Complex_Reasoning_over_Knowledge_Graph.pdf
- Token Usage: input 23285, output 6201, total 29486

### GitHub & Websites

- KG-Agent (official code and data)
  - description: The authors state “Our code and data will be publicly released.” This would be the primary resource to reproduce the agent, toolbox, executor, and instruction-tuning data used in the paper.

- Freebase Knowledge Graph · [Website](https://developers.google.com/freebase)
  - description: One of the target KGs on which KG-Agent reasons (used for WebQSP, CWQ, GrailQA, WQ-Freebase); the authors report using the entire Freebase KG (about 1.9B triples).

- Wikidata · [Website](https://www.wikidata.org) · [Doc](https://www.wikidata.org/wiki/Wikidata:Database_download)
  - description: Another target KG for KG-Agent (used for KQA Pro, NQ-Wiki, TQ-Wiki); the authors use a partial dump (around 3B triples).

- WebQuestionsSP (WebQSP) · [Website](https://www.microsoft.com/en-us/download/details.aspx?id=52763)
  - description: In-domain KGQA benchmark over Freebase; used for instruction data synthesis and evaluation.

- ComplexWebQuestions (CWQ) · [Website](https://www.tau-nlp.org/compwebq)
  - description: In-domain KGQA benchmark over Freebase with complex, multi-hop questions; used for instruction tuning and evaluation.

- GrailQA · [GitHub](https://github.com/dki-lab/GrailQA) · [Website](https://dki-lab.github.io/GrailQA/)
  - description: In-domain KGQA dataset emphasizing generalization (i.i.d., compositional, zero-shot) over Freebase; used for instruction synthesis and evaluation.

- KQA Pro · [Website](https://thunlp.github.io/KQAPro/)
  - description: In-domain KGQA dataset over Wikidata with explicit compositional programs; used both to synthesize code-based instructions and for evaluation.

- MetaQA
  - description: Movie-domain KGQA dataset used to test transfer to a domain-specific KG (one-shot setup on 1/2/3-hop subsets).

- WebQuestions (WQ)
  - description: Open-domain QA dataset; the paper evaluates on a WQ-Freebase subset to test zero-shot generalization with KG grounding.

- Natural Questions (NQ) · [Website](https://ai.google.com/research/NaturalQuestions)
  - description: Open-domain QA benchmark; the authors evaluate on an NQ-Wiki subset in zero-shot mode.

- TriviaQA (TQ) · [Website](http://nlp.cs.washington.edu/triviaqa/)
  - description: Open-domain QA benchmark; the authors evaluate on a TQ-Wiki subset in zero-shot mode.

- StructGPT · [GitHub](https://github.com/RUCAIBox/StructGPT)
  - description: A synergy-augmented LLM framework to reason over structured data; used as a strong comparison baseline and speed reference.

- UniKGQA · [GitHub](https://github.com/RUCAIBox/UniKGQA)
  - description: Unified retrieval-and-reasoning framework for multi-hop KGQA; used as a baseline on Freebase-based datasets.

- ReasoningLM · [GitHub](https://github.com/RUCAIBox/ReasoningLM)
  - description: Structural subgraph reasoning with PLMs for KGQA; used as a competitive baseline (e.g., compared on CWQ).

- FlashAttention · [GitHub](https://github.com/Dao-AILab/flash-attention) · [Codewiki](https://codewiki.google/github.com/Dao-AILab/flash-attention)
  - description: Dependency used during training for memory-efficient exact attention.

- DeepSpeed · [GitHub](https://github.com/microsoft/DeepSpeed)
  - description: Dependency used to facilitate efficient LLM training for instruction tuning.

- Llama 2 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Primary base LLM (LLaMA2-7B) fine-tuned to build KG-Agent; also used in scaling/ablation experiments.

- Code Llama · [GitHub](https://github.com/facebookresearch/codellama)
  - description: Alternative base LLM (CodeLLaMA-7B/34B) evaluated in ablations to test generalizability.

- Mistral 7B · [GitHub](https://github.com/mistralai/mistral-src) · [Website](https://mistral.ai/news/announcing-mistral-7b/)
  - description: Alternative 7B base LLM evaluated as a drop-in for KG-Agent to assess robustness across backbones.

- Phi-2 · [Website](https://huggingface.co/microsoft/phi-2)
  - description: Alternative 3B base LLM evaluated for smaller-scale ablation of KG-Agent.

- OpenAI API (ChatGPT, GPT-4, Davinci-003) · [Doc](https://platform.openai.com/docs)
  - description: Closed-source LLMs used for comparison baselines and evaluated with the February API versions.

- AutoGPT · [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
  - description: Referenced as a representative autonomous agent framework with memory and tool use; cited in related work to contrast with KG-Agent’s open-source 7B setup.

<!-- paper_id: fc3c717987218662f49243e2be6bacc093dd47d8 -->

## 58. CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning - ICLR - 2025 - citation_count 20 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_20_Poster_CURIE_Evaluating_LLMs_On_Multitask_Scientific_Long_Context_Understanding_and_Reasoning.pdf
- Link: https://openreview.net/pdf/cdb4a7c6aa15fcb43b966794be08258eaafd8c1d.pdf
- Tags: multiagent, tool, science, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_20_Poster_CURIE_Evaluating_LLMs_On_Multitask_Scientific_Long_Context_Understanding_and_Reasoning.pdf
- Token Usage: input 46101, output 4424, total 50525

### GitHub & Websites

- CURIE Benchmark · [GitHub](https://github.com/google/curie)
  - description: Official release of the CURIE benchmark with data, prompts, evaluation code (LLMSim/LMScore), and model outputs; primary resource to reproduce and extend all experiments.

- Ecology Georeferencing (BIOGR subset) · [GitHub](https://github.com/google-research/ecology-georeferencing)
  - description: Repository hosting a 114-example subset of the BIOGR map georeferencing dataset (images, captions, PDFs, metadata) used in the paper’s multimodal task.

- Error Correction Zoo (EC Zoo) · [GitHub](https://github.com/errorcorrectionzoo/eczoo_data) · [Website](https://errorcorrectionzoo.org/)
  - description: Open-source repository and site for EC code entries; the QECC task asks models to produce YAML entries following this template and schema.

- S2ORC: The Semantic Scholar Open Research Corpus · [GitHub](https://github.com/allenai/s2orc)
  - description: Paper sources for multiple CURIE tasks (e.g., DFT, MPV, HFD/HFE, GEO) were selected from S2ORC; useful to replicate paper selection and extend the benchmark.

- Atomic Simulation Environment (ASE) · [GitHub](https://gitlab.com/ase/ase) · [Doc](https://wiki.fysik.dtu.dk/ase/)
  - description: Python library used in the DFT-C task to set up, run, and analyze DFT workflows; models were prompted to generate ASE-based code to reproduce calculations.

- Biopython · [GitHub](https://github.com/biopython/biopython) · [Website](https://biopython.org/)
  - description: Used to obtain/reference ground-truth sequences and to evaluate predictions in the PDB task via sequence alignment and identity metrics.

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Source of the 3D structures used in the PDB sequence reconstruction task; practitioners can fetch additional structures to extend evaluation.

- Materials Project · [Website](https://materialsproject.org/) · [Doc](https://docs.materialsproject.org/methodology/materials-methodology/electronic-structure-accuracy-of-band-structures)
  - description: Referenced in DFT-S fields (e.g., mp-id) and as background for band-structure accuracy; relevant when enriching DFT structure metadata.

- OpenAI GPT-4o · [Website](https://openai.com/index/hello-gpt-4o/)
  - description: Closed-weight baseline evaluated across CURIE tasks; required to reproduce the GPT-4o results or use as an LLM judge variant.

- Anthropic Claude 3 (Opus/Sonnet/Haiku) · [Doc](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)
  - description: Closed-weight baseline family used in evaluations; Opus reported as a top performer on CURIE.

- Google Gemini models (1.0 Pro, 1.5 Pro/Flash, 2.0 Flash) · [Doc](https://ai.google.dev/gemini-api/docs/models/gemini)
  - description: Closed-weight baselines with long-context and multimodal support; several variants evaluated extensively across tasks.

- Cohere Command R+ · [Website](https://cohere.com/blog/command-r-plus-microsoft-azure)
  - description: Open-weight baseline (RAG-optimized) included in evaluations; competitive on several extraction/aggregation tasks (e.g., QECC, DFT, MPV).

- Mixtral of Experts (Mistral) · [Website](https://mistral.ai/news/mixtral-of-experts/)
  - description: Open-weight baseline model evaluated on CURIE; useful for reproducing open-model results.

- LongLLaMA · [GitHub](https://github.com/CStanKonrad/long_llama)
  - description: Open-source long-context model included as a baseline; used to test long-context comprehension on CURIE.

<!-- paper_id: 6ad54e99b9ce6fb7d9d312aa0cc753ba873cfff6 -->

## 59. UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction - ICML - 2025 - citation_count 23 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICML_23_Poster_UI-Vision_A_Desktop-centric_GUI_Benchmark_for_Visual_Perception_and_Interaction.pdf
- Link: https://openreview.net/pdf/dea9d6856a1edbb3e8eb8d5f5cf27826da64fab6.pdf
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICML_23_Poster_UI-Vision_A_Desktop-centric_GUI_Benchmark_for_Visual_Perception_and_Interaction.pdf
- Token Usage: input 39489, output 4785, total 44274

### GitHub & Websites

- UI-Vision · [Website](https://uivision.github.io)
  - description: Official project page for the UI-Vision desktop GUI benchmark introduced in the paper; hosts dataset/benchmark information and links for reproducing evaluations.

- UI-Vision (project site repo) · [GitHub](https://github.com/uivision/uivision.github.io)
  - description: GitHub repository for the project website; referenced as the official site and the paper states code and processing scripts will be released alongside the benchmark for reproducibility.

- UI-TARS · [GitHub](https://github.com/bytedance/UI-TARS)
  - description: Open-source GUI agent used as a primary baseline in all three UI-Vision tasks (element/layout grounding and action prediction).

- UGround · [GitHub](https://github.com/OSU-NLP/UGround)
  - description: Universal GUI grounding model used as a competitive baseline and as a grounding component in planner+grounder ablations reported by the paper.

- SeeClick · [GitHub](https://github.com/OpenGVLab/SeeClick)
  - description: GUI grounding model evaluated as an open-source baseline for element and layout grounding.

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent)
  - description: Vision-language GUI agent evaluated as an open-source baseline on UI-Vision tasks.

- ShowUI · [GitHub](https://github.com/ShowLab/ShowUI)
  - description: Vision-language-action GUI agent included as an open-source baseline, especially for action prediction.

- OS-Atlas · [GitHub](https://github.com/OS-Copilot/OS-Atlas)
  - description: Foundation action model for GUI agents; evaluated by the paper for layout/element grounding comparisons.

- InternVL (InternVL2 series) · [GitHub](https://github.com/OpenGVLab/InternVL)
  - description: Open-source VLMs (InternVL2/2.5) used as VLM baselines for UI grounding and layout tasks.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL)
  - description: Open-source VLM baseline evaluated for element and layout grounding.

- Qwen2.5-VL · [GitHub](https://github.com/QwenLM/Qwen2.5-VL)
  - description: Newer Qwen VLM variant also evaluated as an open-source baseline for grounding.

- MiniCPM-V · [GitHub](https://github.com/OpenBMB/MiniCPM-V) · [Codewiki](https://codewiki.google/github.com/OpenBMB/MiniCPM-V)
  - description: Compact multimodal baseline tested for element/layout grounding.

- Aria-UI · [GitHub](https://github.com/aria-ui/Aria-UI)
  - description: Open-source GUI grounding model evaluated as a baseline for element grounding.

- Aguvis · [GitHub](https://github.com/salesforce/AGuVis)
  - description: Open-source pure-vision GUI agent baseline included in element grounding comparisons.

- GPT-4o · [Website](https://openai.com/index/hello-gpt-4o/)
  - description: Closed-source VLM baseline; also used for generating functional descriptions in the dataset construction (element grounding).

- Claude 3.5/3.7 Sonnet · [Website](https://www.anthropic.com/claude)
  - description: Closed-source VLM baselines evaluated for grounding and layout tasks.

- Gemini 1.5/2.0 · [Website](https://ai.google.dev/gemini-api)
  - description: Closed-source VLM baselines; Gemini-1.5-Pro achieved top layout grounding performance and was analyzed in action prediction.

- Llama 3 (Llama 3.3-70B) · [Website](https://ai.meta.com/llama/)
  - description: Used in dataset creation to cluster UI elements and generate layout regions/descriptions for the Layout Grounding task.

- Selenium · [Website](https://www.selenium.dev/)
  - description: Web automation framework cited to contrast the lack of standardized desktop automation APIs; contextual background for benchmark motivation.

- Playwright · [Website](https://playwright.dev/)
  - description: Web automation toolkit referenced as a contrast to desktop agents; part of motivation for building a desktop-focused benchmark.

- MiniWoB++ · [GitHub](https://github.com/miniwob/miniwob-plusplus)
  - description: Classic web GUI benchmark referenced for context; contrasts with UI-Vision’s desktop focus.

- Mind2Web · [GitHub](https://github.com/OSU-NLP/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Web agent dataset/benchmark cited and compared in related work; UI-Vision targets desktop instead of web.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Realistic web environment for autonomous agents; referenced as a web benchmark contrasting with desktop UI-Vision.

- WorkArena · [Website](https://workarena.ai/)
  - description: Benchmark of knowledge-work web tasks; cited in related work as web-focused compared to UI-Vision’s desktop scope.

- OmniAct · [GitHub](https://github.com/omni-act/omni-act)
  - description: Desktop+web dataset/benchmark cited in comparisons; focuses on action prediction but smaller-scale than UI-Vision.

- OSWorld · [GitHub](https://github.com/xlang-ai/OSWorld)
  - description: Desktop benchmark focusing on online interactions; referenced as related but differing from UI-Vision’s offline, densely annotated setup.

- VideoGUI · [GitHub](https://github.com/ShowLab/VideoGUI)
  - description: Benchmark from instructional videos for GUI automation; cited as a smaller-scale desktop dataset compared to UI-Vision.

- ScreenSpot · [GitHub](https://github.com/Sea-Snell/ScreenSpot)
  - description: GUI grounding dataset cited as prior work focusing on grounding rather than full desktop interaction; contrasted with UI-Vision’s broader tasks.

- ScreenSpot-Pro · [GitHub](https://github.com/Sea-Snell/ScreenSpot-Pro)
  - description: Professional high-resolution GUI grounding dataset referenced as prior grounding-focused work; UI-Vision expands to layout and actions.

- AndroidEnv · [GitHub](https://github.com/google-research/android_env)
  - description: RL platform for Android; cited among mobile/online environments to contrast with UI-Vision’s desktop offline benchmark.

<!-- paper_id: 6254219f2e508f938becaa5920e68f361e489b73 -->

## 60. Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark - ICML - 2025 - citation_count 74 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_74_Oral_Can_MLLMs_Reason_in_Multimodality_EMMA_An_Enhanced_MultiModal_ReAsoning_Benchmark.pdf
- Link: https://openreview.net/pdf/2ed8cc3801e22b8ada2e3d50fdf6cf6f5713f938.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_74_Oral_Can_MLLMs_Reason_in_Multimodality_EMMA_An_Enhanced_MultiModal_ReAsoning_Benchmark.pdf
- Token Usage: input 63424, output 5760, total 69184

### GitHub & Websites

- EMMA Benchmark (Enhanced MultiModal ReAsoning) · [Website](https://emma-benchmark.github.io/)
  - description: Official project homepage for EMMA introduced in this paper; hosts benchmark information, download links, and updates needed to reproduce the evaluation.

- MathVista · [Website](https://mathvista.github.io)
  - description: Source benchmark used in EMMA’s math curation and filtering pipeline to collect visual-math problems.

- MMMU · [GitHub](https://github.com/MMMU-Benchmark/MMMU) · [Website](https://mmmu-benchmark.github.io/)
  - description: Multi-discipline multimodal benchmark from which EMMA sources physics/chemistry items after applying stricter filtering.

- OlympiadBench · [Website](https://olympiadbench.github.io/)
  - description: Source of multimodal physics problems used in EMMA’s data curation (filtered to retain genuinely multimodal reasoning).

- EXAMS-V · [GitHub](https://github.com/mbzuai-nlp/EXAMS)
  - description: Multilingual, multi-discipline multimodal exam benchmark; EMMA filters its physics/chemistry problems for inclusion.

- RAVEN · [GitHub](https://github.com/WellyZhang/RAVEN)
  - description: Visual reasoning dataset used to supplement EMMA’s math Pattern Inference category with inherently multi-hop visual reasoning tasks.

- Learn AP Physics · [Website](https://www.learnapphysics.com)
  - description: Public physics problem source used to manually collect additional multimodal physics questions for EMMA.

- Khan Academy (Science) · [Website](https://www.khanacademy.org/science/)
  - description: Public educational resource used to source additional physics problems that meet EMMA’s multimodal criteria.

- RDKit · [GitHub](https://github.com/rdkit/rdkit) · [Website](https://www.rdkit.org/)
  - description: Open-source cheminformatics toolkit used to analyze molecular properties from SMiCRM and generate new chemistry questions (structure recognition/bond counting).

- CharXiv · [Website](https://charxiv.github.io)
  - description: Benchmark of real-world charts used as “seed visualizations” to construct EMMA’s coding tasks (code↔vis and modification).

- Matplotlib Example Gallery · [Website](https://matplotlib.org/stable/gallery/index.html)
  - description: Official gallery used to gather and reproduce advanced visualization seeds that underpin EMMA’s coding task construction.

- Matplotlib · [Website](https://matplotlib.org/)
  - description: Visualization library used to implement all EMMA coding questions (Python/matplotlib-based renderings).

- Seaborn · [Website](https://seaborn.pydata.org/)
  - description: Complementary plotting library used in EMMA coding questions.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm)
  - description: Inference/serving library used to speed up multi-sample generations during EMMA’s filtering and evaluation.

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers)
  - description: Library used to load and run open-source MLLMs during EMMA’s experiments.

- Qwen2‑VL‑72B‑Instruct · [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct)
  - description: Open-source MLLM baseline evaluated on EMMA; link to model family repo and model card for replication.

- QVQ‑72B‑Preview · [Website](https://qwenlm.github.io/blog/qvq-72b-preview/) · [Doc](https://huggingface.co/Qwen/QVQ-72B-Preview)
  - description: Vision-reasoning model evaluated on EMMA-mini; official announcement and model card.

- LLaVA‑OneVision · [GitHub](https://github.com/haotian-liu/LLaVA-OneVision) · [Doc](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf)
  - description: Open-source MLLM baseline used in EMMA evaluations; repo and evaluated checkpoint.

- InternVL2 · [GitHub](https://github.com/OpenGVLab/InternVL) · [Doc](https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B)
  - description: Open-source MLLM baseline; EMMA evaluates the Llama3‑76B variant.

- InternVL2.5 · [GitHub](https://github.com/OpenGVLab/InternVL) · [Doc](https://huggingface.co/OpenGVLab/InternVL2_5-78B)
  - description: Updated open-source MLLM baseline evaluated on EMMA.

- GPT‑4o · [Website](https://platform.openai.com) · [Doc](https://openai.com/index/hello-gpt-4o/)
  - description: Proprietary MLLM baseline used for evaluation and image captioning in filtering; accessed via OpenAI API.

- Claude 3.5 Sonnet · [Website](https://www.anthropic.com/news/claude-3-5-sonnet)
  - description: Proprietary MLLM baseline evaluated on EMMA with direct and CoT prompting.

- Gemini 2.0 Flash · [Website](https://ai.google.dev/)
  - description: Google model baseline evaluated on EMMA (direct and CoT test-time scaling).

- Gemini 2.0 Flash Thinking · [Website](https://ai.google.dev/) · [Doc](https://ai.google.dev/gemini-api/docs/thinking-mode)
  - description: Reasoning-enabled variant achieving the best overall EMMA scores; used also as a reward model in Best-of-N/Tournament selection.

- OpenAI o1 · [Website](https://openai.com/index/learning-to-reason-with-llms/) · [Doc](https://chatgpt.com/)
  - description: OpenAI reasoning model evaluated on EMMA-mini; serves as a strong coding baseline.

- Qwen2.5‑Math‑RM‑72B · [Doc](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B)
  - description: Specialized math reward model tested for Best‑of‑N selection in EMMA’s test-time scaling experiments.

- SMiCRM (Mechanistic Molecular Images) 
  - description: Dataset used to construct EMMA’s chemistry tasks (structure recognition and reaction simulation) by extracting molecules and properties; paired with RDKit for question generation.

- MathVision
  - description: Multimodal math dataset used as a source for EMMA’s math problems before applying the paper’s stricter caption-based filtering.

<!-- paper_id: 23c3bf6a61ee9a1f03fa0f5e16ab286f7f040487 -->

## 61. Agent Workflow Memory - ICML - 2025 - citation_count 68 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3context-engineering/2025_ICML_68_Poster_Agent_Workflow_Memory.pdf
- Link: https://openreview.net/pdf/c7ec3d6eaa775686f3b8750477e00bc7a65034e3.pdf
- Tags: multiagent, tool, science, context-engineering
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3context-engineering/2025_ICML_68_Poster_Agent_Workflow_Memory.pdf
- Token Usage: input 20356, output 3425, total 23781

### GitHub & Websites

- Agent Workflow Memory (AWM) · [GitHub](https://github.com/zorazrw/agent-workflow-memory)
  - description: Official code release for this paper; contains the offline/online workflow induction pipeline, prompts, and evaluation scripts for WebArena and Mind2Web.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Realistic web agent benchmark and environment used for evaluation; the paper follows its accessibility-tree observation format and reports both overall and per-site results, including a cross-template subset.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Dataset and benchmark for web navigation generalization; used for both offline (train-induced workflows) and online AWM evaluation. The repo also provides the MindAct baseline components (e.g., element filtering) compared against AWM.

- BrowserGym · [GitHub](https://github.com/ServiceNow/BrowserGym) · [Doc](https://browsergym.github.io/)
  - description: Web-agent framework adopted for baselines; the paper uses its default action space and compares to a BrowserGym variant with accessibility-tree-only webpage representations.

<!-- paper_id: c68cc84ec7808d7bbd5686a6bd1393752a9d8e8d -->

## 62. AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials - ICLR - 2025 - citation_count 44 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICLR_44_Spotlight_AgentTrek_Agent_Trajectory_Synthesis_via_Guiding_Replay_with_Web_Tutorials.pdf
- Link: https://openreview.net/pdf/e95d923ccea15b1bab268aeeb8b3845547e3dafe.pdf
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICLR_44_Spotlight_AgentTrek_Agent_Trajectory_Synthesis_via_Guiding_Replay_with_Web_Tutorials.pdf
- Token Usage: input 19289, output 5388, total 24677

### GitHub & Websites

- AgentTrek · [Website](https://agenttrek.github.io)
  - description: Official project page for the paper; hosts the project overview and links to resources for reproducing AgentTrek’s tutorial-to-trajectory pipeline and models.

- RedPajama-Data · [GitHub](https://github.com/togethercomputer/RedPajama-Data)
  - description: Open corpus used as the starting source to mine GUI-like tutorials; AgentTrek’s tutorial harvesting and filtering are performed on this dataset.

- BrowserGym · [GitHub](https://github.com/ServiceNow/BrowserGym)
  - description: Web-agent environment used for guided replay; AgentTrek executes standardized tutorials in Chromium via BrowserGym to collect trajectories.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Doc](https://playwright.dev)
  - description: Browser automation toolkit used for precise web actions and for logging reproducible traces (DOM snapshots, network, action sequences) during guided replay.

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui) · [Doc](https://pyautogui.readthedocs.io)
  - description: Used to implement pixel-level actions for the vision-based agent by mapping higher-level Playwright actions to screen-coordinate interactions.

- fastText · [GitHub](https://github.com/facebookresearch/fastText) · [Codewiki](https://codewiki.google/github.com/facebookresearch/fastText) · [Website](https://fasttext.cc)
  - description: Lightweight text classifier used to scale tutorial classification after LLM labeling in AgentTrek’s tutorial filtering stage.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena)
  - description: Benchmark used to evaluate the text-based web agent trained with AgentTrek trajectories.

- Mind2Web · [GitHub](https://github.com/OSU-NLP/Mind2Web) · [Dataset](https://huggingface.co/datasets/osunlp/Mind2Web)
  - description: Dataset/benchmark used for evaluation (and cited for statistics); AgentTrek reports results on the multimodal extension and compares against HTML and HTML+Image settings.

- ScreenAgent (dataset) · [GitHub](https://github.com/niuzaisheng/ScreenAgent/tree/main/data/ScreenAgent/train)
  - description: Dataset referenced in the dataset comparison and statistics; used as a point of comparison for scale/steps in AgentTrek’s analysis.

- WebShop (eto-sft-trajectory) · [Dataset](https://huggingface.co/datasets/agent-eto/eto-sft-trajectory)
  - description: Dataset cited for data-scale comparisons in Appendix; used to contextualize AgentTrek vs. existing trajectory resources.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://qwenlm.github.io)
  - description: Vision-language model fine-tuned with AgentTrek multimodal trajectories to build the vision-based web agent.

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io)
  - description: LLM family fine-tuned on AgentTrek text trajectories (AXTree + Playwright actions) to build the text-based web agent.

- SeeAct · [GitHub](https://github.com/OSU-NLP/SeeAct)
  - description: Baseline method referenced for multimodal planning on Mind2Web; used for comparison in the evaluation tables.

- OmniParser · [GitHub](https://github.com/microsoft/OmniParser)
  - description: Vision parsing tool used in a baseline (“GPT-4 + OmniParser”) for ScreenSpot comparisons reported in the paper.

- AutoWebGLM · [GitHub](https://github.com/THUDM/AutoWebGLM)
  - description: Web agent baseline compared against in WebArena results; included for practitioners inspecting alternative open-source agents.

<!-- paper_id: 744ecc70e9e6b31b43f519cf8c3affdcc2cda8ef -->

## 63. HoneyComb: A Flexible LLM-Based Agent System for Materials Science - EMNLP - 2024 - citation_count 47 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_47_Findings_HoneyComb_A_Flexible_LLM-Based_Agent_System_for_Materials_Science.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.192/
- Tags: multiagent, tool, science, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_47_Findings_HoneyComb_A_Flexible_LLM-Based_Agent_System_for_Materials_Science.pdf
- Token Usage: input 18859, output 4215, total 23074

### GitHub & Websites

- HoneyComb · [GitHub](https://github.com/BangLab-UdeM-Mila/NLP4MatSci-HoneyComb)
  - description: Official code release for the paper; contains the LLM-based agent system with MatSciKB (knowledge base), ToolHub (general and materials-science tools), retriever components, and evaluation setups for reproducing results.

- MaScQA (Materials Science QA) · [Website](https://arxiv.org/abs/2308.09115)
  - description: Evaluation dataset derived from GATE materials science exams used to benchmark HoneyComb; the paper reports results on MaScQA.

- SciQ (referred to as SciQA in the paper) · [Website](https://allenai.org/data/sciq)
  - description: Multiple-choice science question dataset used both as an evaluation benchmark and as a knowledge source (support passages) within MatSciKB.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Doc](https://python.langchain.com/)
  - description: Framework used to implement HoneyComb’s unified tool interface and agent-tool integrations (ToolHub).

- Contriever · [GitHub](https://github.com/facebookresearch/contriever)
  - description: Dense retrieval model used in HoneyComb’s hybrid retriever (combined with BM25) for semantically re-ranking candidate knowledge entries and tool outputs.

- BERTopic · [GitHub](https://github.com/MaartenGr/BERTopic) · [Doc](https://maartengr.github.io/BERTopic/)
  - description: Topic modeling library used in Appendix C to cluster and organize MatSciKB entries into 16 materials-science categories.

- arXiv API · [Doc](https://arxiv.org/help/api)
  - description: Public API used to populate MatSciKB with materials science papers and to enable the Arxiv Search general tool in ToolHub.

- Wikipedia API · [Doc](https://www.mediawiki.org/wiki/API:Main_page)
  - description: API used to ingest and query Wikipedia materials-science content for MatSciKB and the Wikipedia Search tool.

- YouTube Data API · [Doc](https://developers.google.com/youtube/v3)
  - description: Referenced as the backend for the YouTube Search tool in ToolHub to access up-to-date multimedia explanatory content.

- OpenAI API (GPT-3.5, GPT-4) · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Commercial LLMs used as backends/baselines and as the reasoning engine in some HoneyComb configurations.

- Llama 2 / Llama 3 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/) · [Doc](https://llama.meta.com/docs)
  - description: Open-source LLM baselines evaluated with and without HoneyComb to demonstrate framework improvements.

<!-- paper_id: a970be54c4df5f04c3fe65b7414e0c2879c55909 -->

## 64. Red-Teaming LLM Multi-Agent Systems via Communication Attacks - ACL - 2025 - citation_count 41 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_41_Findings_Red-Teaming_LLM_Multi-Agent_Systems_via_Communication_Attacks.pdf
- Link: https://aclanthology.org/2025.findings-acl.349/
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_41_Findings_Red-Teaming_LLM_Multi-Agent_Systems_via_Communication_Attacks.pdf
- Token Usage: input 22715, output 3351, total 26066

### GitHub & Websites

- AiTM (Agent-in-the-Middle) · [GitHub](https://github.com/PengfeiHePower/AiTM)
  - description: Official code release from this paper implementing the AiTM communication attack, including prompts and experiment scripts for AutoGen/CAMEL structures and real-world tests.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework used to build and control agent communications (Chain/Tree/Complete/Random) for the main simulations in this paper.

- CAMEL · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel) · [Website](https://www.camel-ai.org/) · [Doc](https://camel-ai.github.io/camel/)
  - description: Role-playing multi-agent framework used as a second implementation platform for all communication structures and experiments.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT)
  - description: Real-world multi-agent software engineering system evaluated under AiTM (roles like Product Manager/Architect/PM/Engineer); the paper also uses its SoftwareDev tasks, noting that “the full version of SoftwareDev is not released yet, we only test on public problems.”

- ChatDev · [GitHub](https://github.com/OpenBMB/ChatDev)
  - description: Real-world multi-agent software development framework evaluated under AiTM by intercepting agents (CEO/CPO/CTO/Programmer) across phases.

- MMLU (Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test) · [Website](https://huggingface.co/datasets/cais/mmlu)
  - description: Benchmark used for multiple-choice tasks (biology and physics subsets) in both targeted-behavior and DoS attack evaluations.

- HumanEval · [GitHub](https://github.com/openai/human-eval) · [Website](https://huggingface.co/datasets/openai_humaneval)
  - description: Code generation benchmark where the paper evaluates targeted injection (adding a safety_check function) and DoS attacks.

- MBPP (Mostly Basic Python Problems) · [GitHub](https://github.com/google-research/google-research/tree/master/mbpp) · [Website](https://huggingface.co/datasets/mbpp)
  - description: Python coding benchmark used to measure AiTM’s targeted behavior injection and DoS attack success rates.

- OpenAI GPT models (GPT-4o, GPT-4, GPT-3.5-turbo) · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Black-box LLM backends powering both the system agents and the adversarial agent across experiments; model variants are compared to study attack effectiveness.

<!-- paper_id: 4669474df8ee4985a95c43c0ee54d621c0a639e1 -->

## 65. HiAgent: Hierarchical Working Memory Management for Solving Long-Horizon Agent Tasks with Large Language Model - ACL - 2025 - citation_count 27 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3context-engineering/2025_ACL_27_Long_HiAgent_Hierarchical_Working_Memory_Management_for_Solving_Long-Horizon_Agent_Tasks_with_Large_Language_Model.pdf
- Link: https://aclanthology.org/2025.acl-long.1575/
- Tags: multiagent, tool, science, context-engineering
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3context-engineering/2025_ACL_27_Long_HiAgent_Hierarchical_Working_Memory_Management_for_Solving_Long-Horizon_Agent_Tasks_with_Large_Language_Model.pdf
- Token Usage: input 23492, output 4144, total 27636

### GitHub & Websites

- HiAgent · [GitHub](https://github.com/HiAgent2024/HiAgent)
  - description: Official code release for the paper; contains the hierarchical working-memory agent implementation, prompts, and scripts needed to reproduce the results reported in the paper.

- AgentBoard · [Website](https://arxiv.org/abs/2401.13178)
  - description: Analytical evaluation board used as the base implementation for experiments; the paper runs five long-horizon tasks (Blocksworld, Gripper, Tyreworld, Barman, Jericho) and uses AgentBoard’s metrics and environments.

- Jericho · [GitHub](https://github.com/microsoft/jericho)
  - description: Text-based interactive fiction environment suite; used as one of the five evaluation tasks to assess agent performance in long-horizon text-game settings.

- OpenAI API (GPT-4 Turbo) · [Doc](https://platform.openai.com/docs)
  - description: The paper uses GPT-4 (gpt-4-turbo) via the OpenAI API as both the agent policy and the observation summarization model; reproducing results requires this API.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting agent framework cited by the paper as compatible with HiAgent’s hierarchical memory management; useful for practitioners looking to integrate the proposed memory mechanism into existing agent paradigms.

<!-- paper_id: a7fb4245b412f0e54ec26d5973f041d52c83c0ad -->

## 66. AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories - EMNLP - 2024 - citation_count 24 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4llm-tech/2024_EMNLP_24_Findings_AgentBank_Towards_Generalized_LLM_Agents_via_Fine-Tuning_on_50000+_Interaction_Trajectories.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.116/
- Tags: multiagent, tool, science, llm-tech
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4llm-tech/2024_EMNLP_24_Findings_AgentBank_Towards_Generalized_LLM_Agents_via_Fine-Tuning_on_50000+_Interaction_Trajectories.pdf
- Token Usage: input 22698, output 6076, total 28774

### GitHub & Websites

- AgentBank (dataset) · [Website](https://huggingface.co/datasets/Solaris99/AgentBank)
  - description: Official release of the 50k+ agent-environment interaction trajectory dataset introduced in the paper; used to train and evaluate SAMOYED.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting framework the paper adopts for agent tasking, where the model emits rationale before each action.

- FastChat · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat)
  - description: Training/evaluation framework used by the authors (with MT-Bench); leveraged for efficient fine-tuning and benchmarking.

- PyTorch FSDP · [Doc](https://pytorch.org/docs/stable/fsdp.html)
  - description: Fully Sharded Data Parallel used by the authors for efficient distributed training.

- ShareGPT (instruction data) · [Website](https://sharegpt.com/)
  - description: Generalist instruction-following data mixed into training (10%) to alleviate catastrophic forgetting and improve generalization.

- WizardCoder / Evol-CodeAlpaca (code data) · [GitHub](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder)
  - description: Code-focused Evol-Instruct dataset used as 10% of the training mixture to boost code/agent planning skills.

- HotpotQA · [GitHub](https://github.com/hotpotqa/hotpotqa) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset adapted into a multi-turn search-tool environment for AgentBank.

- StrategyQA · [GitHub](https://github.com/allenai/strategyqa) · [Website](https://allenai.org/data/strategyqa)
  - description: Implicit reasoning QA used as a reasoning task with a search tool in AgentBank.

- TriviaQA · [GitHub](https://github.com/mandarjoshi90/triviaqa) · [Website](http://nlp.cs.washington.edu/triviaqa/)
  - description: Compositional QA dataset integrated into AgentBank with a search interface.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math) · [Website](https://huggingface.co/datasets/openai/gsm8k)
  - description: Math word problems; official solution paths were reformatted into interaction trajectories.

- MathQA · [Website](https://huggingface.co/datasets/math_qa)
  - description: Multiple-choice math dataset; adapted with a Python tool environment for AgentBank.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: Challenging math competition problems; turned into interactive tasks using Python and Wikipedia tools.

- InterCode (IC-SQL / IC-Bash) · [GitHub](https://github.com/princeton-nlp/InterCode)
  - description: Interactive coding benchmark; IC-SQL is a held-in programming task and IC-Bash is used as a held-out evaluation.

- APPS · [GitHub](https://github.com/hendrycks/apps)
  - description: Python coding benchmark; instances were reformatted into trajectories for programming skill training.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Functional code synthesis benchmark; used to build programming trajectories.

- MBPP · [GitHub](https://github.com/google-research/google-research/tree/master/mbpp)
  - description: Beginner-level Python tasks; used to construct programming trajectories and evaluation.

- Mind2Web · [GitHub](https://github.com/OSU-NLP/Mind2Web)
  - description: Generalist web agent dataset; used as a held-in web task with step success metrics.

- WebArena · [GitHub](https://github.com/web-arena-dev/WebArena)
  - description: Realistic web environments; broken into single-step interactions and annotated with rationales.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Simulated e-commerce website; used with exploration and answer-forcing to create shopping trajectories and evaluate average reward.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
  - description: Text-based embodied environments paralleling ALFRED; human gold trajectories and reformatted interactions used for training/eval.

- RoomR (Visual Room Rearrangement) · [GitHub](https://github.com/allenai/roomr)
  - description: Embodied rearrangement tasks; gold action sequences constructed via heuristic DFS and used in AgentBank.

- IQA (Interactive Question Answering) · [GitHub](https://github.com/allenai/ai2thor-iqa)
  - description: Interactive visual QA environment; text versions leveraged to build embodied interaction trajectories.

- Bamboogle · [GitHub](https://github.com/ofirpress/Bamboogle)
  - description: Held-out reasoning benchmark requiring compositional web search; used to test generalization.

- TheoremQA · [GitHub](https://github.com/wenhuchen/TheoremQA)
  - description: Held-out theorem-driven QA in math/science; used to evaluate math/generalization with Python and Wikipedia tools.

- MiniWoB++ · [GitHub](https://github.com/miniwob/miniwob-plusplus)
  - description: Diverse web interaction tasks; used as a held-out web benchmark for generalization.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld)
  - description: Text-based science lab environments; used as a held-out embodied evaluation with average reward.

- AgentBench · [GitHub](https://github.com/THUDM/AgentBench)
  - description: External agent evaluation suite; the paper reports additional results on this benchmark for reference.

- AlpacaEval 2 · [GitHub](https://github.com/tatsu-lab/alpaca_eval)
  - description: General capability benchmark used to measure catastrophic forgetting when fine-tuning.

- MMLU (Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test)
  - description: General knowledge benchmark used to assess whether mixture training preserves general abilities.

<!-- paper_id: b60a9a78caaf06fbdbf8ee91ed9416efa0e6c3c4 -->

## 67. Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning - NeurIPS - 2025 - citation_count 20 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4RL/2025_NeurIPS_20_Poster_Improving_Retrieval-Augmented_Generation_through_Multi-Agent_Reinforcement_Learning.pdf
- Link: https://openreview.net/pdf/0bc82f5b29d7122b8b021ba8fad9f4e5ce031dd8.pdf
- Tags: multiagent, tool, science, rl
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4RL/2025_NeurIPS_20_Poster_Improving_Retrieval-Augmented_Generation_through_Multi-Agent_Reinforcement_Learning.pdf
- Token Usage: input 29503, output 5575, total 35078

### GitHub & Websites

- MMOA-RAG · [GitHub](https://github.com/chenyiqun/MMOA-RAG)
  - description: Official code release of the paper’s multi-agent reinforcement learning framework for joint optimization of RAG modules (Query Rewriter, Selector, Generator) using MAPPO; used for all experiments.

- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory) · [Codewiki](https://codewiki.google/github.com/hiyouga/LLaMA-Factory) · [Doc](https://llamafactory.readthedocs.io/)
  - description: Training toolkit whose PPO codebase the authors build upon to implement MAPPO-based optimization and SFT for MMOA-RAG.

- HotpotQA · [GitHub](https://github.com/hotpotqa/hotpot) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used as a primary benchmark for training, evaluation, and ablations.

- 2WikiMultihopQA · [GitHub](https://github.com/Alab-NII/2wikimultihop)
  - description: Multi-hop QA dataset used to evaluate the proposed method’s generalization and performance.

- AmbigQA · [GitHub](https://github.com/shmsw25/ambigqa) · [Website](https://nlp.cs.washington.edu/ambigqa/)
  - description: Ambiguous open-domain QA dataset used for single-hop evaluation and out-of-domain tests.

- Contriever · [GitHub](https://github.com/facebookresearch/contriever)
  - description: Dense retriever used as the fixed first-stage retrieval model in the main experiments.

- FlagEmbedding (BGE) · [GitHub](https://github.com/FlagOpen/FlagEmbedding) · [Doc](https://huggingface.co/BAAI/bge-base-en-v1.5)
  - description: Alternative retriever family evaluated in Appendix E to test robustness across retrieval backbones.

- E5 · [GitHub](https://github.com/microsoft/unilm/tree/master/e5) · [Doc](https://huggingface.co/intfloat/e5-large-v2)
  - description: Text-embedding retriever family used as another alternative retriever in additional experiments.

- Meta Llama 3 8B Instruct · [Website](https://llama.meta.com) · [Doc](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - description: Backbone LLM used to implement the Query Rewriter, Selector, and Generator agents for all baselines and MMOA-RAG.

- SELF-RAG · [GitHub](https://github.com/AkariAsai/self-rag)
  - description: Public implementation used as a comparison baseline in the experiments.

<!-- paper_id: 5e5c2b08231678dcb33b1f00bc2dd5bb7045bcd5 -->

## 68. ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities - NAACL - 2025 - citation_count 75 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_75_Findings_ToolSandbox_A_Stateful,_Conversational,_Interactive_Evaluation_Benchmark_for_LLM_Tool_Use_Capabilities.pdf
- Link: https://aclanthology.org/2025.findings-naacl.65/
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_75_Findings_ToolSandbox_A_Stateful,_Conversational,_Interactive_Evaluation_Benchmark_for_LLM_Tool_Use_Capabilities.pdf
- Token Usage: input 24092, output 6719, total 30811

### GitHub & Websites

- TOOLSANDBOX · [GitHub](https://github.com/apple/ToolSandbox)
  - description: Official release of the paper’s stateful, conversational, interactive tool-use evaluation framework with 1032 scenarios, Python tool implementations, user-simulator prompts, and milestone/minefield evaluation.

- Python 3.9.19 (code.InteractiveConsole) · [Doc](https://docs.python.org/3.9/)
  - description: The benchmark runs in a Python-native environment and executes tool calls via code.InteractiveConsole; this is the official language/library documentation referenced for implementation details.

- RapidAPI
  - [Website](https://rapidapi.com/)
  - description: Several TOOLSANDBOX tools wrap carefully selected RapidAPI endpoints (e.g., weather, stock) to provide realistic external information sources used during evaluation.

- ToolBench / ToolLLM / ToolEval · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Referenced benchmark/dataset and evaluation suite for tool-use LLMs; used in the paper’s comparisons (ToolEval) and to contextualize differences in task design and evaluation.

- API-Bank
  - description: Cited benchmark of tool-augmented LLMs used for comparison; includes tools and tasks but does not study implicit state dependencies as emphasized by TOOLSANDBOX.

- ToolTalk
  - description: Referenced conversational tool-use evaluation benchmark relied upon for comparison; evaluates off-policy dialog trajectories unlike TOOLSANDBOX’s on-policy setup.

- τ-bench
  - description: Referenced benchmark for tool-agent-user interaction; used for comparison to highlight TOOLSANDBOX’s flexibility with multiple valid trajectories and error correction.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena)
  - description: Related web-interaction benchmark cited in related work to contrast web-browsing agents with TOOLSANDBOX’s stateful tool-use focus.

- MiniWoB++ · [GitHub](https://github.com/miniwob/miniwob-plusplus)
  - description: Classic GUI/web task environment cited among web-agent benchmarks; provides context for agent evaluation beyond function calling.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Referenced web shopping environment used to evaluate grounded web agents; included for practitioners looking at adjacent agent-eval settings.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Generalist web-agent benchmark referenced in related work; useful for broader comparisons with TOOLSANDBOX’s tool-use evaluation.

- AgentBench · [GitHub](https://github.com/THUDM/AgentBench)
  - description: Generalist agent benchmark suite cited to situate tool-use evaluation within broader multi-task agent testing.

- Gorilla · [GitHub](https://github.com/ShishirPatil/gorilla)
  - description: Tool-API calling LLM referenced as an open-source baseline; paper notes limitations consuming tool responses in conversational settings.

- CodeAct · [GitHub](https://github.com/xingyaoww/CodeAct)
  - description: Referenced agent approach incorporating executable code actions; included as a closely related open-source implementation for practitioners.

- GPT-4o · [Website](https://openai.com/index/hello-gpt-4o/)
  - description: Proprietary model used both as an evaluated agent and as the LLM-based user simulator in TOOLSANDBOX experiments.

- Hermes-2-Pro-Mistral-7B · [Website](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B)
  - description: Open-source model evaluated by the paper; Hugging Face model card for reproducing model-side experiments.

- Cohere Command R (c4ai-command-r-v01) · [Website](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
  - description: Open-source model evaluated by the paper; authors note limitations consuming tool responses when using the released weights/prompt templates.

<!-- paper_id: 7738e909d563d84fbd4ab5cb6aacf62c84fe2ab9 -->

## 69. Data Interpreter: An LLM Agent For Data Science - ACL - 2025 - citation_count 129 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ACL_129_Findings_Data_Interpreter_An_LLM_Agent_For_Data_Science.pdf
- Link: https://aclanthology.org/2025.findings-acl.1016/
- Tags: multiagent, tool, science, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ACL_129_Findings_Data_Interpreter_An_LLM_Agent_For_Data_Science.pdf
- Token Usage: input 30328, output 5948, total 36276

### GitHub & Websites

- Data Interpreter (this paper)
  - description: The paper states “We will release the code upon publication.” indicating no public code or project page is available at the time of writing.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework used as a primary baseline across data analysis, ML, and MATH experiments.

- OpenHands · [GitHub](https://github.com/All-Hands-AI/OpenHands)
  - description: Generalist software/agent platform compared as a baseline for ML and open-ended tasks; noted for event streaming and sandboxing.

- OpenInterpreter · [GitHub](https://github.com/KillianLucas/open-interpreter) · [Website](https://openinterpreter.com) · [Doc](https://docs.openinterpreter.com)
  - description: Code-first agent used as a baseline on ML and open-ended tasks.

- TaskWeaver · [GitHub](https://github.com/microsoft/TaskWeaver) · [Doc](https://microsoft.github.io/TaskWeaver/)
  - description: Code-first agent framework compared as a baseline in ML tasks.

- XAgent · [GitHub](https://github.com/OpenBMB/XAgent)
  - description: Autonomous agent framework included as a baseline in the ML benchmark.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT)
  - description: Multi-agent framework whose tool library is referenced in the programmable node example (e.g., metagpt.tools.libs.data_preprocess.FillMissingValue) for data preprocessing.

- MATH Dataset · [GitHub](https://github.com/hendrycks/math)
  - description: Benchmark for mathematical reasoning; the paper evaluates level-5 problems in several categories and compares against MathChat/AutoGen.

- Kaggle · [Website](https://www.kaggle.com)
  - description: Source of the ML-Benchmark datasets and MLE-Bench(-Lite) tasks used in the paper’s machine learning evaluations.

- scikit-learn · [Website](https://scikit-learn.org/) · [Doc](https://scikit-learn.org/stable/)
  - description: Core ML toolkit used in action graphs (e.g., IsolationForest, MinMaxScaler, confusion matrix, classification report) and model training/evaluation.

- XGBoost · [GitHub](https://github.com/dmlc/xgboost) · [Codewiki](https://codewiki.google/github.com/dmlc/xgboost) · [Doc](https://xgboost.readthedocs.io/)
  - description: Gradient boosting library used as a model option in the ML pipelines (e.g., “train predictive models using Random Forest and XGBoost”).

- PaddleOCR · [GitHub](https://github.com/PaddlePaddle/PaddleOCR) · [Codewiki](https://codewiki.google/github.com/PaddlePaddle/PaddleOCR)
  - description: OCR toolkit explicitly required in the open-ended OCR tasks; used to extract fields and amounts from images.

- rembg · [GitHub](https://github.com/danielgatis/rembg) · [Codewiki](https://codewiki.google/github.com/danielgatis/rembg)
  - description: Image background removal tool used in the open-ended “Image Background Removal” tasks.

- Stable Diffusion · [GitHub](https://github.com/CompVis/stable-diffusion)
  - description: Text-to-image generation model referenced in open-ended tasks (T2I) for image synthesis.

<!-- paper_id: 01e863776846ebd1a9a7acc4a9ca24217f953aa2 -->

## 70. EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents - ICML - 2025 - citation_count 64 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_64_Oral_EmbodiedBench_Comprehensive_Benchmarking_Multi-modal_Large_Language_Models_for_Vision-Driven_Embodied_Agents.pdf
- Link: https://openreview.net/pdf/b9e775a028b2a809c09d3c36562f179b9cac55a4.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_64_Oral_EmbodiedBench_Comprehensive_Benchmarking_Multi-modal_Large_Language_Models_for_Vision-Driven_Embodied_Agents.pdf
- Token Usage: input 64286, output 4687, total 68973

### GitHub & Websites

- EmbodiedBench · [Website](https://embodiedbench.github.io) · [GitHub](https://github.com/EmbodiedBench)
  - description: Official code, benchmark tasks, and datasets released by the paper; includes the unified agent, evaluation scripts, EB-ALFRED/EB-Habitat/EB-Navigation/EB-Manipulation environments, and the auto task-generation script for EB-Navigation.

- AI2-THOR · [GitHub](https://github.com/allenai/ai2thor) · [Website](https://ai2thor.allenai.org/) · [Doc](https://ai2thor.allenai.org/ithor/)
  - description: Interactive 3D simulator used to build EB-ALFRED and EB-Navigation; provides egocentric observations and action execution with textual feedback.

- ALFRED · [GitHub](https://github.com/askforalfred/alfred) · [Website](https://askforalfred.com/)
  - description: Household instruction-following dataset and tasks on AI2-THOR; EB-ALFRED is developed from ALFRED with simulator fixes and multi-instance support.

- Habitat 2.0 / Habitat-Lab · [GitHub](https://github.com/facebookresearch/habitat-lab) · [Website](https://aihabitat.org/) · [Doc](https://aihabitat.org/docs/)
  - description: Simulator and framework used for EB-Habitat; the paper evaluates high-level skills and rearrangement tasks within Habitat.

- Language Rearrangement Benchmark (Habitat) · [Website](https://aihabitat.org/datasets/rearrangement/)
  - description: Dataset/task suite adapted to form EB-Habitat; authors reorganize its subsets (base, commonsense, complex, visual, spatial, long-horizon) for capability-oriented evaluation.

- CoppeliaSim (formerly V-REP) · [Website](https://www.coppeliarobotics.com/) · [Doc](https://www.coppeliarobotics.com/helpFiles/en/)
  - description: Robotics simulator used to implement EB-Manipulation with a 7-DoF Franka Panda arm and discretized low-level action control.

- VLMbench · [GitHub](https://github.com/UM-ARM-Lab/VLMbench)
  - description: Compositional vision-language manipulation benchmark that EB-Manipulation extends; provides categories such as pick-place, stacking, shape sorting, and wiping.

- YOLO (You Only Look Once) · [GitHub](https://github.com/pjreddie/darknet) · [Website](https://pjreddie.com/darknet/yolo/)
  - description: Object detection used to provide detection boxes and indices in EB-Manipulation, aiding localization and instruction grounding.

- YCB Object and Model Set · [Website](https://www.ycbbenchmarks.com/)
  - description: Object models used in Habitat rearrangement tasks; EB-Habitat leverages YCB assets for evaluation scenes and objects.

- ReplicaCAD · [Website](https://aihabitat.org/datasets/replica_cad/)
  - description: Habitat-compatible CAD reconstructions used as scene assets in EB-Habitat to support rearrangement and navigation in realistic environments.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai/) · [Doc](https://docs.vllm.ai/)
  - description: High-throughput LLM serving used by the authors to run open-source vision-language models locally for benchmarking.

- LMDeploy · [GitHub](https://github.com/InternLM/lmdeploy) · [Doc](https://lmdeploy.readthedocs.io/)
  - description: Toolkit used by the authors to deploy and serve open-source MLLMs during evaluation.

- OpenAI GPT-4o API · [Website](https://platform.openai.com/docs/models#gpt-4o)
  - description: Proprietary MLLM accessed via API; evaluated as one of the main baselines across all four EmbodiedBench environments.

- Anthropic Claude 3.5/3.7 · [Website](https://www.anthropic.com/api)
  - description: Proprietary MLLMs evaluated on EmbodiedBench; accessed via Anthropic API for high- and low-level embodied tasks.

- Google Gemini 1.5/2.0 · [Website](https://ai.google.dev/gemini-api/docs)
  - description: Proprietary MLLMs evaluated in the benchmark; used via Gemini API for planning and perception across tasks.

<!-- paper_id: 6fbb3ed823526ac050b610d353ea91a8515f7e69 -->

## 71. AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents - ACL - 2025 - citation_count 37 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_37_Long_AndroidLab_Training_and_Systematic_Benchmarking_of_Android_Autonomous_Agents.pdf
- Link: https://aclanthology.org/2025.acl-long.107/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_37_Long_AndroidLab_Training_and_Systematic_Benchmarking_of_Android_Autonomous_Agents.pdf
- Token Usage: input 28532, output 4881, total 33413

### GitHub & Websites

- AndroidLab · [GitHub](https://github.com/THUDM/Android-Lab)
  - description: Official release from the paper. Provides the unified Android agent environment (XML and SoM modes), the 138-task benchmark with prebuilt AVD images, evaluation scripts/metrics, and the Android Instruct data and annotation tool needed to reproduce results and fine-tune models.

- VisualAgentBench (VAB-Mobile) · [GitHub](https://github.com/THUDM/VisualAgentBench)
  - description: Benchmark suite from the authors where parts of AndroidLab’s SoM modes are included as the VAB-Mobile component; useful for inspecting the SoM grounding and comparing across visual agent settings referenced by the paper.

- AndroidEnv · [GitHub](https://github.com/google-research/android_env)
  - description: Interactive Android RL platform whose action space the paper draws from; relevant dependency/background for the defined Tap/Swipe/Type/Long-Press/Home/Back action space used in AndroidLab.

- AndroidWorld · [GitHub](https://github.com/google-research/android_world)
  - description: Dynamic Android benchmarking environment cited as related work offering reward signals; useful as a complementary environment for extending experiments beyond AndroidLab.

- AppAgent · [GitHub](https://github.com/RUC-GSAI/AppAgent)
  - description: Smartphone agent baseline whose action space the paper adopts alongside AndroidEnv; helpful for practitioners to compare agent formulations and prompts.

- Android in the Wild (AITW) · [Website](https://research.google/resources/datasets/android-in-the-wild/)
  - description: Large-scale Android GUI dataset cited and used as an academic seed source during task derivation/expansion; relevant dataset for pretraining or augmenting Android operation data.

- Snips NLU Benchmark · [GitHub](https://github.com/snipsco/nlu-benchmark)
  - description: Public intent/slot dataset (Coucke et al., 2018) used by the paper as an additional seed source for generating realistic executable instructions during Android Instruct data construction.

- ADB Keyboard · [GitHub](https://github.com/senzhk/ADBKeyBoard)
  - description: Open-source Android IME used in the paper’s annotation tool to programmatically input text (ADB keyboard ON) for recording/type actions consistently during dataset creation.

- Android Debug Bridge (ADB) · [Doc](https://developer.android.com/tools/adb)
  - description: Official Android developer documentation for ADB, which AndroidLab relies on for device control, XML/screenshot capture, time/geolocation setting, and automated evaluation.

- Android Accessibility Service · [Doc](https://developer.android.com/guide/topics/ui/accessibility/service)
  - description: Official documentation; the paper reimplements XML retrieval via Accessibility Service to avoid ADB UI-tree timeouts and capture stable page structures during annotation.

- Android Virtual Device (AVD) · [Doc](https://developer.android.com/studio/run/managing-avds)
  - description: Official Android Studio AVD documentation; AndroidLab packages deterministic AVD images (fixed device/time/location/app states) to ensure reproducible benchmarking.

<!-- paper_id: 6e9f7bb3f647b31bc907715f3b5dae29cef8a2c5 -->

## 72. RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts - ICML - 2025 - citation_count 52 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_52_Spotlightposter_RE-Bench_Evaluating_frontier_AI_R&D_capabilities_of_language_model_agents_against_human_experts.pdf
- Link: https://openreview.net/pdf/4ecca478c42f4d707b818055092b92bfb1094b98.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_52_Spotlightposter_RE-Bench_Evaluating_frontier_AI_R&D_capabilities_of_language_model_agents_against_human_experts.pdf
- Token Usage: input 43935, output 5049, total 48984

### GitHub & Websites

- RE-Bench Environments (ai-rd-tasks) · [GitHub](https://github.com/METR/ai-rd-tasks)
  - description: Official release of the seven RE‑Bench AI R&D environments, including starting solutions, scoring functions, and reference solutions used for all experiments in the paper.

- RE-Bench Agent Trajectories · [Website](https://transcripts.metr.org)
  - description: Official repository of agent transcripts/trajectories from the benchmark runs, enabling analysis and replication of agent behaviors reported in the paper.

- Vivaria (Agent Evaluation Platform) · [Website](https://vivaria.metr.org)
  - description: The open-source platform the authors used to provision secure VMs with H100 GPUs for both human and agent runs.

- AIDE (Agent Scaffold) · [Website](https://www.weco.ai/blog/technical-report)
  - description: Agent scaffolding framework used by the authors (with minor adaptations) for several evaluations; selected because it performed well on MLE‑Bench.

- Modular Agent (METR scaffold) · [Doc](https://metr.github.io/autonomy-evals-guide/openai-o1-preview-report/)
  - description: General-purpose agent scaffold employed by the authors for one set of runs; this page documents METR’s agent setup and evaluation approach.

- LLM Foundry · [GitHub](https://github.com/mosaicml/llm-foundry) · [Doc](https://docs.mosaicml.com/projects/llm-foundry)
  - description: Training/finetuning framework used to build the “Optimize LLM Foundry” environment where agents speed up a finetuning script without changing behavior.

- nanoGPT · [GitHub](https://github.com/karpathy/nanoGPT) · [Codewiki](https://codewiki.google/github.com/karpathy/nanoGPT)
  - description: Lightweight GPT training codebase used as the foundation for the “Fix Embedding” and “Scaling Law Experiment” environments.

- Triton (GPU programming language) · [GitHub](https://github.com/triton-lang/triton) · [Codewiki](https://codewiki.google/github.com/triton-lang/triton) · [Website](https://triton-lang.org)
  - description: GPU kernel framework (v2.3.1) that agents used to write custom kernels in the “Optimize a Kernel” environment.

- Replicate API · [Website](https://replicate.com) · [Doc](https://replicate.com/docs)
  - description: In “Finetune GPT‑2 for QA,” the evaluation script calls the Replicate API to get judgments from Llama‑3‑8B‑Instruct.

- OpenWebText · [Website](https://skylion007.github.io/OpenWebTextCorpus/)
  - description: Dataset used for training/evaluating loss in multiple environments (e.g., Fix Embedding, Restricted Architecture MLM, Scaling Law Experiment).

- Stanford Alpaca · [GitHub](https://github.com/tatsu-lab/stanford_alpaca) · [Codewiki](https://codewiki.google/github.com/tatsu-lab/stanford_alpaca)
  - description: Dataset/model reference used to construct the comparison baseline “gpt‑2 (small) finetuned on Stanford Alpaca” in the QA finetuning environment.

- OpenAI o1-preview · [Doc](https://openai.com/index/openai-o1-system-card/)
  - description: Frontier model evaluated as an agent in the benchmark.

- Claude 3.5 Sonnet · [Doc](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf)
  - description: Frontier model evaluated as an agent (two versions) across scaffolds and time budgets.

<!-- paper_id: 5dbd9f9fd231863dda17d9e3caff2edb99a113e2 -->

## 73. SWE-Search: Enhancing Software Agents with Monte Carlo Tree Search and Iterative Refinement - ICLR - 2025 - citation_count 46 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_ICLR_46_Poster_SWE-Search_Enhancing_Software_Agents_with_Monte_Carlo_Tree_Search_and_Iterative_Refinement.pdf
- Link: https://openreview.net/pdf/fd21b53183d7f37c46833653eed82df55976adf4.pdf
- Tags: multiagent, tool, science, agent-coding
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_ICLR_46_Poster_SWE-Search_Enhancing_Software_Agents_with_Monte_Carlo_Tree_Search_and_Iterative_Refinement.pdf
- Token Usage: input 29162, output 4084, total 33246

### GitHub & Websites

- SWE-Search (moatless-tree-search) · [GitHub](https://github.com/aorwall/moatless-tree-search) · [Website](https://streamlit.moatless.ai)
  - description: Official code release and interactive demo for the paper’s MCTS + iterative refinement framework; contains the Action/Value/Discriminator agents, search implementation, and visualization to reproduce results.

- moatless-tools · [GitHub](https://github.com/aorwall/moatless-tools)
  - description: Base agent framework extended by the authors (“Moatless-Adapted”) to support tree-structured states, backtracking, and test running; used as the primary baseline and foundation for SWE-Search.

- SWE-bench (incl. SWE-bench Lite) · [GitHub](https://github.com/princeton-nlp/SWE-bench) · [Website](https://www.swe-bench.com)
  - description: Benchmark dataset and harness for repository-level issue resolution; the paper evaluates on SWE-bench Lite and runs tests using Docker images built from the SWE-bench library.

- OpenAI GPT-4o / GPT-4o-mini · [Website](https://platform.openai.com/docs)
  - description: Closed-source LLMs used as backbones in experiments for the Action/Value/Discriminator agents; API access required to reproduce these runs.

- Qwen2.5 (Qwen2.5-72B-Instruct) · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io)
  - description: Open-source LLM used as one of the main models in experiments; enables reproducing the open-weight model results reported.

- Llama 3.1 (70B Instruct) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open-weight Meta model evaluated by the paper; provides instructions and weights access for replicating the Llama-based runs.

- DeepSeek-Coder V2.5 · [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-Coder)
  - description: Open-source code-focused LLM corresponding to the DeepSeek-V2.5 model used in the experiments; repository provides model access and usage guidance for reproduction.

<!-- paper_id: 8a381d4da1c5dc9837ef22afd5f47f7c567c00be -->

## 74. VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents - ICLR - 2025 - citation_count 57 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICLR_57_Poster_VisualAgentBench_Towards_Large_Multimodal_Models_as_Visual_Foundation_Agents.pdf
- Link: https://openreview.net/pdf/bee7975a3c8aa69c07f6b3df50d3f4b390af6700.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICLR_57_Poster_VisualAgentBench_Towards_Large_Multimodal_Models_as_Visual_Foundation_Agents.pdf
- Token Usage: input 74355, output 6004, total 80359

### GitHub & Websites

- VisualAgentBench (VAB) · [GitHub](https://github.com/THUDM/VisualAgentBench)
  - description: Official release of the benchmark, environments, training trajectories, prompts, and evaluation code for all five settings (VAB-OmniGibson, VAB-Minecraft, VAB-AndroidLab, VAB-WebArena-Lite, VAB-CSS).

- OmniGibson · [GitHub](https://github.com/StanfordVL/OmniGibson) · [Website](https://behavior.stanford.edu/)
  - description: High-fidelity household simulator used to build VAB-OmniGibson; the paper adapts OmniGibson scenes/objects and defines high-level actions and judges for embodied tasks.

- BEHAVIOR-1K · [Website](https://behavior.stanford.edu/)
  - description: Source benchmark from which activity prototypes and BDDL goals are referenced for VAB-OmniGibson task design and evaluation.

- NVIDIA Omniverse · [Website](https://www.nvidia.com/en-us/omniverse/)
  - description: Underlying simulation platform required by OmniGibson; relevant for running the embodied environment used in VAB.

- MineRL · [GitHub](https://github.com/minerllabs/minerl) · [Doc](https://minerl.readthedocs.io)
  - description: Minecraft research environment used to implement VAB-Minecraft; the paper defines high-level actions and integrates a low-level controller within this framework.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Interactive web environment the paper adapts and cleans to form VAB-WebArena-Lite; used with SoM annotations and automated via Playwright.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Doc](https://playwright.dev/python/docs/intro)
  - description: Browser automation toolkit used to write program-based solvers and to execute web actions when collecting training trajectories and during evaluation in Web tasks.

- Android Virtual Device (AVD) · [Doc](https://developer.android.com/studio/run/managing-avds)
  - description: Android emulator used to create reproducible smartphone environments for VAB-AndroidLab tasks and interactive evaluation.

- InternVL/InternVL2 · [GitHub](https://github.com/OpenGVLab/InternVL)
  - description: Open-source LMMs fine-tuned/evaluated as baselines in VAB; practitioners can reproduce the open-model experiments and training.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://qwenlm.github.io)
  - description: Open LMM baseline fine-tuned on VAB trajectories; used for comparison across all environments.

- GLM-4/GLM-4V · [GitHub](https://github.com/THUDM/GLM-4)
  - description: Multimodal GLM model (visual variant GLM-4V) used as an open-source baseline and fine-tuned under the paper’s behavior cloning setup.

- LLaVA (incl. LLaVA-1.5 and LLaVA-NeXT) · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io)
  - description: Open LMM baselines the authors fine-tune on VAB’s multi-environment trajectories for agent evaluation.

- CogVLM · [GitHub](https://github.com/THUDM/CogVLM)
  - description: Open visual-language model used as a baseline; fine-tuned with VAB data in experiments.

- CogVLM2 · [GitHub](https://github.com/THUDM/CogVLM2)
  - description: Newer CogVLM variant evaluated as an open-source baseline on VAB after multitask fine-tuning.

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent)
  - description: GUI-focused visual-language agent used as an open-source baseline and fine-tuned on VAB trajectories.

- Qwen-VL · [GitHub](https://github.com/QwenLM/Qwen-VL)
  - description: Earlier Qwen multimodal model used as an open baseline for VAB fine-tuning and evaluation.

<!-- paper_id: f1f63620e87facef02234e82864c4b8adee081ec -->

## 75. AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents - ICLR - 2025 - citation_count 145 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICLR_145_Poster_AndroidWorld_A_Dynamic_Benchmarking_Environment_for_Autonomous_Agents.pdf
- Link: https://openreview.net/pdf/47ef762908227ecf3c9f07aea77ea162a772b501.pdf
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICLR_145_Poster_AndroidWorld_A_Dynamic_Benchmarking_Environment_for_Autonomous_Agents.pdf
- Token Usage: input 32609, output 3263, total 35872

### GitHub & Websites

- ANDROIDWORLD · [GitHub](https://github.com/google-research/android_world)
  - description: Official code release for the paper; includes the AndroidWorld benchmark, MobileMiniWoB++ integration, task definitions, environment wrappers, and baseline/agent implementations used in the experiments.

- AndroidEnv · [GitHub](https://github.com/deepmind/android_env)
  - description: Python library used by AndroidWorld to connect agents to the Android OS/emulator and stream observations/actions; cited as the mechanism for device interaction.

- Android Studio / Android Emulator · [Website](https://developer.android.com/studio) · [Doc](https://developer.android.com/studio/run/emulator)
  - description: Required runtime environment; AndroidWorld is designed to run on the freely available Android Emulator packaged with Android Studio.

- Android Debug Bridge (adb) · [Doc](https://developer.android.com/tools/adb)
  - description: System tool AndroidWorld uses to manipulate and inspect device state (file system, app databases, settings) for durable, programmatic reward checks.

- F-Droid · [Website](https://f-droid.org/)
  - description: Open-source Android app repository from which fixed app versions are sourced and installed to ensure reproducibility of AndroidWorld’s tasks.

- MiniWoB++ · [GitHub](https://github.com/stanfordnlp/miniwob-plusplus)
  - description: Parameterizable web UI benchmark whose tasks are integrated into AndroidWorld as MobileMiniWoB++; serves as the web task suite used in the paper’s experiments.

- SeeAct · [GitHub](https://github.com/OSU-NLP-Group/SeeAct)
  - description: Baseline web agent adapted by the authors to the Android setting for comparison; their implementation follows the SeeActchoice variant with Android-specific actions.

<!-- paper_id: d1b2eaf7aaebbd3d847272da04be180e35c7b68b -->

## 76. Agents' Room: Narrative Generation through Multi-step Collaboration - ICLR - 2025 - citation_count 30 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_30_Poster_Agents'_Room_Narrative_Generation_through_Multi-step_Collaboration.pdf
- Link: https://openreview.net/pdf/28c4bdc08fd7e4c2b2b7308c42d0447cbf1a0620.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_30_Poster_Agents'_Room_Narrative_Generation_through_Multi-step_Collaboration.pdf
- Token Usage: input 38227, output 5450, total 43677

### GitHub & Websites

- TELL ME A STORY (Agents’ Room) · [GitHub](https://github.com/google-deepmind/tell_me_a_story)
  - description: Official release from the paper including the high-quality prompts and human-written stories dataset, splits, and evaluation metrics/scripts used to assess long-form narratives.

- Gemini API (Gemini 1.5 Flash / Pro / Ultra) · [Website](https://cloud.google.com/apis) · [Doc](https://ai.google.dev/gemini-api/docs)
  - description: Google’s LLM platform used throughout the work—Gemini 1.5 Flash as the backbone for all agents and baselines, Gemini Ultra as the teacher model for synthetic data generation, and Gemini 1.5 Pro as the automatic evaluator.

- Gemma 2 · [Website](https://ai.google.dev/gemma)
  - description: Open language model family used in additional experiments (Appendix F) to demonstrate the framework with a smaller, publicly available backbone.

- WikiPlots · [GitHub](https://github.com/markriedl/WikiPlots)
  - description: Public dataset of plot summaries referenced for dataset scale/characteristics comparison against TELL ME A STORY.

- ROCStories / Story Cloze Test · [Website](https://www.cs.rochester.edu/nlp/rocstories/)
  - description: Short commonsense story corpus referenced in the paper’s dataset comparison table to contextualize narrative benchmarks.

- BERTScore · [GitHub](https://github.com/Tiiiger/bert_score)
  - description: Reference-based evaluation metric used to automatically compare generated stories with human references.

- ROUGE (rouge-score) · [GitHub](https://github.com/google-research/google-research/tree/master/rouge) · [Doc](https://pypi.org/project/rouge-score/)
  - description: ROUGE-L reference-based metric employed in the paper’s automatic evaluation of system outputs.

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient fine-tuning method used to train the end-to-end baselines and specialized agents on limited data.

<!-- paper_id: 5a4d1add108f2e6d8ccd9f6ac94ab3e0335db540 -->

## 77. OpenHands: An Open Platform for AI Software Developers as Generalist Agents - ICLR - 2025 - citation_count 270 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_ICLR_270_Poster_OpenHands_An_Open_Platform_for_AI_Software_Developers_as_Generalist_Agents.pdf
- Link: https://openreview.net/pdf/95990590797cff8b93c33af989ecf4ac58bde9bb.pdf
- Tags: multiagent, tool, science, agent-coding
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_ICLR_270_Poster_OpenHands_An_Open_Platform_for_AI_Software_Developers_as_Generalist_Agents.pdf
- Token Usage: input 31705, output 7749, total 39454

### GitHub & Websites

- OpenHands · [GitHub](https://github.com/All-Hands-AI/OpenHands)
  - description: Official code release of the paper (MIT licensed) including agent implementations, sandboxed runtime, skills library, multi-agent delegation, UI, and integrated benchmarks for reproduction.

- BrowserGym · [GitHub](https://github.com/ServiceNow/BrowserGym)
  - description: Web task automation environment whose DSL and action set OpenHands uses for BrowseInteractiveAction and web-agent evaluation.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Website](https://playwright.dev/)
  - description: Headless Chromium driver used by OpenHands’ runtime browser to execute BrowserGym actions.

- IPython/Jupyter · [GitHub](https://github.com/ipython/ipython) · [Codewiki](https://codewiki.google/github.com/ipython/ipython) · [Website](https://ipython.org)
  - description: Provides the interactive Python server OpenHands uses for IPythonRunCellAction to execute agent-generated code.

- Docker · [Website](https://www.docker.com/) · [Doc](https://docs.docker.com/)
  - description: OpenHands runs all actions inside a Docker-sandboxed OS; required to reproduce the runtime environment.

- SWE-Agent · [GitHub](https://github.com/princeton-nlp/SWE-agent)
  - description: Software-engineering agent whose Agent-Computer Interface and file-editing utilities are adapted in OpenHands’ AgentSkills and used as a SWE baseline.

- Aider · [GitHub](https://github.com/Aider-AI/aider) · [Codewiki](https://codewiki.google/github.com/Aider-AI/aider) · [Website](https://aider.chat/)
  - description: Open-source coding assistant; its edit_file-style utilities are incorporated into OpenHands AgentSkills and it serves as a SWE-Bench baseline.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev/)
  - description: Realistic, self-hostable web-agent benchmark; integrated for OpenHands web evaluation and baseline comparison.

- MiniWoB++ · [GitHub](https://github.com/google-deepmind/miniwob-plusplus)
  - description: Interactive synthetic web tasks benchmark; used to evaluate OpenHands’ browsing agent.

- SWE-bench · [GitHub](https://github.com/princeton-nlp/SWE-bench) · [Website](https://www.swe-bench.com/)
  - description: Real-world GitHub issue fixing benchmark (and Lite subset) used for OpenHands SWE evaluations.

- HumanEvalFix · [Website](https://huggingface.co/datasets/bigcode/humanevalpack)
  - description: Bug-fixing version of HumanEval from BigCode; OpenHands evaluates multi-turn self-debugging on its Python split.

- BIRD (Text-to-SQL) · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) · [Website](https://bird-bench.github.io/)
  - description: Large-scale, realistic text-to-SQL benchmark; integrated to assess OpenHands’ database-grounded code generation.

- Gorilla (APIBench) · [GitHub](https://github.com/ShishirPatil/gorilla) · [Website](https://gorilla.cs.berkeley.edu/)
  - description: API calling benchmark used to test OpenHands’ ability to select and call software APIs.

- GAIA · [GitHub](https://github.com/GAIA-benchmark/GAIA) · [Website](https://gaia-benchmark.github.io/)
  - description: General AI assistant benchmark spanning tool use, browsing, and multimodality; OpenHands evaluates GPTSwarm and agents on GAIA.

- GPQA · [GitHub](https://github.com/idavidrein/gpqa) · [Website](https://gpqa.github.io/)
  - description: Graduate-level, Google-proof QA benchmark; OpenHands evaluates tool-augmented reasoning on GPQA subsets.

- AgentBench · [GitHub](https://github.com/THUDM/AgentBench)
  - description: Multi-domain agent evaluation suite; OpenHands uses the OS (bash) subset for code-grounded system interaction.

- MINT · [GitHub](https://github.com/xingyaoww/MINT-benchmark)
  - description: Multi-turn interaction benchmark with tool use and simulated language feedback; OpenHands evaluates math and code subsets.

- ProofWriter · [GitHub](https://github.com/allenai/proofwriter) · [Website](https://allenai.org/data/proofwriter)
  - description: Deductive reasoning dataset; OpenHands evaluates long-hop logical reasoning with tool assistance.

- Moatless Tools · [GitHub](https://github.com/aorwall/moatless-tools)
  - description: Open-source SWE toolkit used as a comparison baseline against OpenHands on SWE-Bench Lite.

- Auto-GPT · [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
  - description: Popular autonomous agent framework included in the paper’s comparisons and GAIA baseline.

- LangChain / LangGraph · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain)
  - description: General agent/tooling framework cited in related work; discussed as an alternative for building agent workflows comparable to OpenHands.

- Mozilla Accessibility Tree (Docs) · [Doc](https://developer.mozilla.org/en-US/docs/Glossary/Accessibility_tree)
  - description: Documentation referenced for browser observations (DOM/accessibility trees) returned by OpenHands’ runtime browser.

<!-- paper_id: 1d07e5b6f978cf69c0186f3d5f434fa92d471e46 -->

## 78. DSBench: How Far Are Data Science Agents from Becoming Data Science Experts? - ICLR - 2025 - citation_count 52 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_52_Poster_DSBench_How_Far_Are_Data_Science_Agents_from_Becoming_Data_Science_Experts.pdf
- Link: https://openreview.net/pdf/c7598f99c29d589965e824c0fc1c1d64fc079c1e.pdf
- Tags: multiagent, tool, science, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICLR_52_Poster_DSBench_How_Far_Are_Data_Science_Agents_from_Becoming_Data_Science_Experts.pdf
- Token Usage: input 44905, output 3178, total 48083

### GitHub & Websites

- DSBench · [GitHub](https://github.com/LiqiangJing/DSBench)
  - description: Official repository released by the paper; contains the data and code for DSBench, enabling reproduction of the benchmark and experiments.

- ModelOff (Financial Modeling World Championship) · [Website](https://corporatefinanceinstitute.com/resources/financial-modeling/modeloff-guide/)
  - description: Source of the 466 real-world data analysis tasks used in DSBench; the paper scraped and curated questions and files from ModelOff challenges.

- Kaggle · [Website](https://www.kaggle.com/)
  - description: Source of the 74 machine learning competitions used for DSBench data modeling tasks; training/test splits and metrics are derived from Kaggle competitions.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://microsoft.github.io/autogen/)
  - description: Open-source multi-agent framework used as one of the agent systems evaluated in the paper for both data analysis and data modeling tasks.

- OpenAI Assistants Code Interpreter · [Doc](https://platform.openai.com/docs/assistants/tools/code-interpreter) · [Website](https://platform.openai.com/)
  - description: Closed-source agent runtime evaluated as a baseline; authors upload data via the Assistants API and use the Code Interpreter tool to execute code and generate submissions.

- pandas · [GitHub](https://github.com/pandas-dev/pandas) · [Codewiki](https://codewiki.google/github.com/pandas-dev/pandas) · [Doc](https://pandas.pydata.org/)
  - description: Python data analysis library used in preprocessing; authors convert spreadsheets to text for LLM baselines and handle data files during experiments.

- LLaVA · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io/)
  - description: Open-source LVLM evaluated as a model-only baseline on DSBench data analysis tasks.

- Meta Llama 3 · [Website](https://ai.meta.com/llama/)
  - description: LLM family (Llama3-8B/70B) evaluated as open-source model-only baselines.

- OpenAI GPT models (GPT-3.5/4/4o/4o-mini) · [Doc](https://platform.openai.com/docs/models)
  - description: Closed-source LLMs/LVLMs used as primary baselines and as the backbone of agent systems (AutoGen and Code Interpreter).

- Google Gemini · [Website](https://ai.google.dev/) · [Doc](https://ai.google.dev/gemini-api)
  - description: Closed-source LVLM evaluated as a model-only baseline on DSBench.

- Anthropic Claude · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/claude)
  - description: Closed-source LLM evaluated as a model-only baseline on DSBench.

<!-- paper_id: 395c978221a21ee47c84a40a2ef11fb4d012fca1 -->

## 79. Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance - ICLR - 2025 - citation_count 30 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_30_Poster_Proactive_Agent_Shifting_LLM_Agents_from_Reactive_Responses_to_Active_Assistance.pdf
- Link: https://openreview.net/pdf/ba670e591dd33509822a7a0360e6a84be1debcd5.pdf
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_30_Poster_Proactive_Agent_Shifting_LLM_Agents_from_Reactive_Responses_to_Active_Assistance.pdf
- Token Usage: input 22885, output 3678, total 26563

### GitHub & Websites

- ProactiveBench (dataset) 
  - description: The paper’s benchmark and training corpus of 6,790 synthesized events plus 233 real-world test events across Coding, Writing, and Daily Life; used to fine-tune and evaluate proactive behavior of LLM agents. 

- LLaMA‑3.1‑8B‑Proactive (fine-tuned checkpoint) 
  - description: Authors fine-tune LLaMA‑3.1‑8B‑Instruct on ProactiveBench to improve proactive task prediction; used as a main model in experiments. 

- Qwen2‑7B‑Proactive (fine-tuned checkpoint) 
  - description: Authors fine-tune Qwen2‑7B‑Instruct on ProactiveBench; achieves the best reported F1 on the benchmark. 

- Reward Model for Acceptance Prediction 
  - description: A classifier fine-tuned (based on LLaMA‑3.1‑8B‑Instruct) on 1,760 human-annotated entries to simulate user judgments and automatically score agent proactivity.

- ActivityWatch · [GitHub](https://github.com/ActivityWatch/activitywatch) · [Website](https://activitywatch.net/) · [Doc](https://docs.activitywatch.net/en/latest/)
  - description: Open-source activity tracker used by the authors to build their monitoring software for collecting real-world keyboard/mouse, browser, and app-usage events that seed scenarios and example events in the Environment Gym.

- Meta Llama 3.1 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/blog/meta-llama-3/) · [Doc](https://llama.meta.com/)
  - description: Base open-source LLM family; the authors fine-tune LLaMA‑3.1‑8B‑Instruct both as the proactive agent and as the reward model, and also evaluate baseline performance.

- Qwen2 · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://qwenlm.github.io/)
  - description: Open-source LLM family from Alibaba; Qwen2‑7B‑Instruct is used as a baseline and fine-tuned to create the “Qwen2‑7B‑Proactive” model.

- OpenAI GPT‑4o · [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs/)
  - description: Proprietary model used to generate scenarios/events in the Environment Gym, to act as the user agent in data generation, and as a strong baseline in evaluation.

- OpenAI GPT‑4o‑mini · [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs/)
  - description: Smaller OpenAI model evaluated as a baseline and in multi-prediction settings on ProactiveBench.

- Anthropic Claude 3 Sonnet · [Website](https://www.anthropic.com/news/claude-3-family) · [Doc](https://docs.anthropic.com/)
  - description: Proprietary baseline model evaluated on ProactiveBench.

- Anthropic Claude 3.5 Sonnet · [Website](https://www.anthropic.com/news/claude-3-5-sonnet) · [Doc](https://docs.anthropic.com/)
  - description: Newer proprietary baseline model evaluated on ProactiveBench.

- OpenAI Embeddings (Text-Ada-Embedding) · [Doc](https://platform.openai.com/docs/guides/embeddings)
  - description: Used to embed model-predicted tasks and select a diverse set (by cosine distance) for human annotation when constructing the reward-model dataset.

<!-- paper_id: fbed2564c89622cbe6c75a5df4aab037965c0bc3 -->

## 80. TriageAgent: Towards Better Multi-Agents Collaborations for Large Language Model-Based Clinical Triage - EMNLP - 2024 - citation_count 21 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/multiagent/2024_EMNLP_21_Findings_TriageAgent_Towards_Better_Multi-Agents_Collaborations_for_Large_Language_Model-Based_Clinical_Triage.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.329/
- Tags: multiagent, tool, science, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/multiagent/2024_EMNLP_21_Findings_TriageAgent_Towards_Better_Multi-Agents_Collaborations_for_Large_Language_Model-Based_Clinical_Triage.pdf
- Token Usage: input 21908, output 3293, total 25201

### GitHub & Websites

- TriageAgent · [GitHub](https://github.com/Lucanyc/TriageAgent)
  - description: Official code and dataset release of the paper; contains the heterogeneous multi-agent framework, prompts, and the first public ESI clinical triage benchmark used in the experiments.

- Emergency Severity Index (ESI) Handbook v4 · [Website](https://media.emscimprovement.center/documents/ESI_Handbook2125.pdf)
  - description: Authoritative triage guideline used for dataset construction and as the primary knowledge source in the RAG component.

- Microsoft AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Open-source multi-agent conversation framework used to implement TRIAGEAGENT’s agent orchestration and group chat workflow.

- OpenAI GPT-3.5 Turbo · [Website](https://openai.com/index/gpt-3-5-turbo-fine-tuning-and-api-updates/) · [Doc](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  - description: One of the backbone LLMs for zero-shot experiments; the paper specifies temperatures, iterations, and other API settings.

- OpenAI GPT-4 · [Website](https://openai.com/index/gpt-4/) · [Doc](https://platform.openai.com/docs/models#gpt-4)
  - description: Primary backbone LLM used for zero-shot evaluation and main results comparisons.

- Llama 2 · [Website](https://llama.meta.com/llama2/)
  - description: Open foundation model family used in appendix experiments as an alternative backbone for comparison.

- Llama 3 · [Website](https://llama.meta.com/llama3/)
  - description: Model family additionally evaluated (including fine-tuning) to compare against GPT models on the triage task.

- PubMed · [Website](https://pubmed.ncbi.nlm.nih.gov/)
  - description: External biomedical knowledge source tested in ablation as an alternative retrieval tool for the agents.

- Wikipedia · [Website](https://www.wikipedia.org/)
  - description: General-purpose external knowledge source evaluated in ablation for retrieval-augmented evidence.

<!-- paper_id: 1fda4c0f505ad99437957cc154156aea47c6e102 -->

## 81. FlowBench: Revisiting and Benchmarking Workflow-Guided Planning for LLM-based Agents - EMNLP - 2024 - citation_count 26 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2024_EMNLP_26_Findings_FlowBench_Revisiting_and_Benchmarking_Workflow-Guided_Planning_for_LLM-based_Agents.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.638/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2024_EMNLP_26_Findings_FlowBench_Revisiting_and_Benchmarking_Workflow-Guided_Planning_for_LLM-based_Agents.pdf
- Token Usage: input 25220, output 3869, total 29089

### GitHub & Websites

- FlowBench · [GitHub](https://github.com/Justherozen/FlowBench)
  - description: Official repository for the FlowBench benchmark, including benchmark data, workflow knowledge in multiple formats (text/code/flowchart), evaluation scripts, and prompts used in the paper.

- Alibaba DAMO-ConvAI (FlowBench mirror/hub) · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI)
  - description: Alibaba Research organization hub that hosts conversational AI resources; the paper points here as an official location for accessing FlowBench.

- Mermaid · [GitHub](https://github.com/mermaid-js/mermaid) · [Codewiki](https://codewiki.google/github.com/mermaid-js/mermaid) · [Website](https://mermaid.js.org) · [Doc](https://mermaid.js.org/intro/)
  - description: Diagramming and visual programming syntax used to encode flowchart-format workflow knowledge (Markdown Mermaid) throughout FlowBench.

- OpenAI API (GPT-4o/GPT-4-Turbo/GPT-3.5-Turbo) · [Website](https://platform.openai.com) · [Doc](https://platform.openai.com/docs) 
  - description: Closed-source LLM APIs used for inference, function-calling/tool usage formatting, user simulation, and automatic evaluation in the paper.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning+acting prompting framework; used as the inference template for agent planning (thought, action, observation) in FlowBench experiments.

- WikiHow Dataset · [GitHub](https://github.com/mahnazkoupaee/WikiHow-Dataset) · [Website](https://www.wikihow.com)
  - description: Professional knowledge corpus cited as a source for constructing and organizing workflow-related knowledge in FlowBench.

- Zapier · [Website](https://zapier.com) · [Doc](https://help.zapier.com/hc/en-us)
  - description: Automation/workflow website referenced as an online source of workflow knowledge during the benchmark’s knowledge curation.

- Qwen2 (Instruct models) · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://qwenlm.github.io)
  - description: Open-source LLMs (Qwen2-7B-Instruct, Qwen2-72B-Instruct) used in additional experiments to demonstrate FlowBench’s applicability to open-source models.

- KnowAgent · [GitHub](https://github.com/zjunlp/KnowAgent)
  - description: Knowledge-augmented agent baseline referenced and compared in the paper’s discussion of workflow-guided planning.

- ProAgent · [GitHub](https://github.com/THUDM/ProAgent)
  - description: Agentic process automation system using code/control flows; listed as a related workflow-guided agent baseline in the benchmark comparison.

<!-- paper_id: b49707c75c9f29ea020111e7a6f3af28c7061729 -->

## 82. AMEX: Android Multi-annotation Expo Dataset for Mobile GUI Agents - ACL - 2025 - citation_count 64 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_64_Findings_AMEX_Android_Multi-annotation_Expo_Dataset_for_Mobile_GUI_Agents.pdf
- Link: https://aclanthology.org/2025.findings-acl.110/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_64_Findings_AMEX_Android_Multi-annotation_Expo_Dataset_for_Mobile_GUI_Agents.pdf
- Token Usage: input 17271, output 5302, total 22573

### GitHub & Websites

- AMEX (Android Multi-annotation EXpo) dataset
  - description: Official dataset introduced by the paper containing 104K Android screenshots with three levels of annotations (element grounding, screen/element functionality descriptions, and instruction–action chains). It is the primary resource the authors curate and use to train and evaluate their SPHINX Agent baselines.

- ANDROIDCONTROL
  - [GitHub](https://github.com/microsoft/AndroidControl)
  - description: Large-scale Android GUI-control dataset and benchmark. Used in the paper to train and evaluate SphAgent and to measure gains from adding AMEX Level I/II data.

- Android in the Wild (AITW)
  - [GitHub](https://github.com/google-research/google-research/tree/master/android_in_the_wild)
  - description: Large-scale dataset of Android device control episodes with a shared action space. Used as a benchmark to train and evaluate agents and to mix with AMEX for improved performance.

- AndroidWorld
  - [GitHub](https://github.com/google-research/android_world)
  - description: Dynamic benchmarking environment for autonomous Android agents. Cited as a target for future online evaluation; relevant for extending this work to interactive settings.

- ScreenSpot
  - description: Benchmark dataset for UI element grounding on mobile/desktop/web screenshots. Used by the authors to evaluate the element grounding ability of their SphAgent trained on AMEX Levels I/II.

- Appium
  - [GitHub](https://github.com/appium/appium) · [Codewiki](https://codewiki.google/github.com/appium/appium) · [Website](https://appium.io) · [Doc](https://appium.io/docs/en/latest/)
  - description: Open-source cross-platform test automation framework used by the authors to control emulators, collect screenshots, and record XML view data during AMEX data collection.

- Genymotion
  - [Website](https://www.genymotion.com/)
  - description: Android emulator platform used as one of the environments to collect screenshots and XMLs for AMEX.

- Android Virtual Device (AVD)
  - [Doc](https://developer.android.com/studio/run/managing-avds)
  - description: Official Android emulator/AVD tooling used to run emulators during AMEX data collection.

- GPT-4o (OpenAI)
  - [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: LMM used to generate and cross-check screen/element functionality descriptions and prompts in AMEX Level II.

- Gemini 1.5 Pro (Google)
  - [Doc](https://ai.google.dev/gemini-api/docs/models/gemini#gemini-1.5-pro)
  - description: LMM used alongside GPT-4o to generate and verify AMEX Level II screen and element functionality descriptions.

- InternLM
  - [GitHub](https://github.com/InternLM/InternLM)
  - description: Base LLM family from which the authors use the InternLM-7B variant as the backbone for their SPHINX-based SphAgent.

- DINOv2
  - [GitHub](https://github.com/facebookresearch/dinov2)
  - description: Visual encoder used to extract image features for SphAgent training/evaluation.

- ConvNeXt/ConvNeXt V2
  - [GitHub](https://github.com/facebookresearch/ConvNeXt)
  - description: Convolutional visual backbone used to extract image features in the SPHINX Agent setup.

- CogAgent
  - [GitHub](https://github.com/THUDM/CogAgent)
  - description: Open-source GUI-focused LVLM used as a comparison baseline on ScreenSpot in the paper.

- Qwen2-VL
  - [GitHub](https://github.com/QwenLM/Qwen2-VL)
  - description: Vision-language model used as a comparison baseline on the ScreenSpot evaluation.

<!-- paper_id: b25af982bfeac66fd317e0aa76e3719dc36c6c50 -->

## 83. LLM-Based Agent Society Investigation: Collaboration and Confrontation in Avalon Gameplay - EMNLP - 2024 - citation_count 58 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_58_Main_LLM-Based_Agent_Society_Investigation_Collaboration_and_Confrontation_in_Avalon_Gameplay.pdf
- Link: https://aclanthology.org/2024.emnlp-main.7/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_58_Main_LLM-Based_Agent_Society_Investigation_Collaboration_and_Confrontation_in_Avalon_Gameplay.pdf
- Token Usage: input 17557, output 3112, total 20669

### GitHub & Websites

- LLM-Game-Agent · [GitHub](https://github.com/3DAgentWorld/LLM-Game-Agent)
  - description: Official code release for this EMNLP 2024 paper; contains the multi-agent Avalon framework (memory, analysis, planning, action, response, and experience learning), prompts, and the Python game program used to run all experiments and analyses.

- OpenAI GPT‑3.5 Turbo (Chat Completions API) · [Website](https://platform.openai.com/docs/models/gpt-3-5-turbo) · [Doc](https://platform.openai.com/docs/api-reference/chat)
  - description: The authors use gpt-3.5-turbo-16k as the agent backbone and as the baseline model for all main experiments; these pages provide the official model description and API needed to reproduce calls (temperatures, etc.).

- Llama 2 7B Chat (Llama-2-7b-chat-hf) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
  - description: Used in the paper’s appendix for broader validation of the framework; this is the official model family page (Meta) and the specific Hugging Face model repository employed (Llama-2-7b-chat-hf).

<!-- paper_id: ff406e2ab8fdcce6b051cad1ead794c928440f77 -->

## 84. Progressive Multimodal Reasoning via Active Retrieval - ACL - 2025 - citation_count 24 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ACL_24_Long_Progressive_Multimodal_Reasoning_via_Active_Retrieval.pdf
- Link: https://aclanthology.org/2025.acl-long.180/
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ACL_24_Long_Progressive_Multimodal_Reasoning_via_Active_Retrieval.pdf
- Token Usage: input 33781, output 5931, total 39712

### GitHub & Websites

- Wikipedia Dumps (English, Chinese) · [Website](https://dumps.wikimedia.org/enwiki/) · [Website](https://dumps.wikimedia.org/zhwiki/)
  - description: Source of the general reasoning knowledge base; the paper cleans and chunks English/Chinese Wikipedia dumps for retrieval.

- WikiExtractor · [GitHub](https://github.com/attardi/wikiextractor)
  - description: Used to extract and clean text from Wikipedia dumps when building the retrieval corpus.

- COIG (Chinese Open Instruction Generalist) · [GitHub](https://github.com/BAAI-Open/COIG) · [Website](https://huggingface.co/datasets/BAAI/COIG)
  - description: Included as part of the general reasoning knowledge base for retrieval.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Text-only math dataset used to build the mathematics-specific retrieval corpus.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: Competition math dataset providing step-by-step solutions; used in the text-only math retrieval corpus.

- MathVista · [GitHub](https://github.com/lupantech/MathVista) · [Website](https://mathvista.github.io)
  - description: Visual math benchmark used both as an evaluation benchmark and as part of the multimodal retrieval corpus.

- MathVerse · [GitHub](https://github.com/OpenGVLab/MathVerse) · [Website](https://mathverse-cuhk.github.io)
  - description: Multimodal math dataset used to construct the multimodal retrieval knowledge base.

- MathVision · [GitHub](https://github.com/OpenGVLab/MathVision)
  - description: Multimodal math dataset included in the hybrid-modal retrieval corpus.

- WE-MATH · [Website](https://we-math.github.io)
  - description: Multimodal math benchmark used for evaluation and as part of the retrieval corpus (testmini split usage described).

- GAOKAO-MM · [GitHub](https://github.com/FudanNLP/GAOKAO-MM) · [Website](https://gaokao-mm.github.io)
  - description: Chinese human-level multimodal benchmark used for cross-domain evaluation and to build a domain-specific retrieval base.

- Contriever · [GitHub](https://github.com/facebookresearch/contriever) · [Website](https://huggingface.co/facebook/mcontriever-msmarco)
  - description: Dense text retriever used to retrieve top-K documents from the text-only corpus.

- CLIP (ViT-L/14@336px) · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Contrastive vision-language model used for cross-modal retrieval over the hybrid-modal corpus.

- FAISS · [GitHub](https://github.com/facebookresearch/faiss) · [Codewiki](https://codewiki.google/github.com/facebookresearch/faiss) · [Website](https://faiss.ai)
  - description: Vector indexing/search library employed to index embeddings and retrieve nearest neighbors efficiently.

- Jina-CLIP-v1 · [GitHub](https://github.com/jina-ai/jina-clip) · [Website](https://huggingface.co/jinaai/jina-clip-v1)
  - description: Alternative multimodal retriever used in ablations for cross-modal retrieval.

- BGE M3 Embedding · [GitHub](https://github.com/FlagOpen/FlagEmbedding) · [Website](https://huggingface.co/BAAI/bge-m3)
  - description: Multilingual multi-function embeddings used to compute solution representations in the diversity analysis.

- scikit-learn (DBSCAN) · [Website](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
  - description: DBSCAN clustering used to visualize and analyze the diversity of sampled solutions.

- DeepSpeed (ZeRO Stage 3) · [GitHub](https://github.com/microsoft/DeepSpeed) · [Website](https://www.deepspeed.ai)
  - description: Training system used to scale PRM fine-tuning (ZeRO-3) efficiently.

- FlashAttention-2 · [GitHub](https://github.com/Dao-AILab/flash-attention) · [Codewiki](https://codewiki.google/github.com/Dao-AILab/flash-attention)
  - description: Accelerated attention kernel employed during PRM training for efficiency.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm)
  - description: Suggested in limitations as an efficient serving/decoding stack; relevant for reproducing inference at scale.

- TagLM-13b-v2.0 (InsTag) · [Website](https://huggingface.co/OFA-Sys/TagLM-13b-v2.0)
  - description: Open-world tagger used to annotate knowledge concepts for filtering consistency in the retrieval pipeline.

- LLaVA-OneVision · [GitHub](https://github.com/LLaVA-VL/LLaVA-OneVision)
  - description: Open-source MLLM backbone used in experiments and for generating pseudo-answers for corpus construction.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://qwenlm.github.io)
  - description: Open-source MLLM backbone used in experiments with AR-MCTS.

- InternVL2 · [GitHub](https://github.com/OpenGVLab/InternVL)
  - description: Open-source MLLM backbone (InternVL2-8B) evaluated with AR-MCTS.

- LLaVA-NeXT · [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT)
  - description: Open-source MLLM baseline (LLaMA3-LLaVA-NeXT-8B) used in additional evaluations.

- GPT-4o · [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed-source MLLM used as a backbone to evaluate AR-MCTS performance.

- GPT-4V · [Website](https://openai.com/research/gpt-4) · [Doc](https://platform.openai.com/docs/guides/vision)
  - description: Closed-source multimodal model used as an additional backbone in evaluation.

- Qwen2 (LLM family) · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://qwenlm.github.io)
  - description: Text LLM backbones for PRM and open-source MLLMs used in experiments.

- InternLM2/2.5 · [GitHub](https://github.com/InternLM/InternLM)
  - description: Referenced LLM family with long-context/chat variants; relevant as alternative backbones for reproducing PRM/MLLM experiments.

- Llama 3 · [Website](https://ai.meta.com/llama/) · [GitHub](https://github.com/meta-llama/llama3)
  - description: Open LLM family referenced as a backbone in related baselines (LLaMA3-LLaVA-NeXT-8B).

<!-- paper_id: f05cf0438b5dc19bc4d32ca6cd85d1525c936de6 -->

## 85. VideoWebArena: Evaluating Long Context Multimodal Agents with Video Understanding Web Tasks - ICLR - 2025 - citation_count 22 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICLR_22_Poster_VideoWebArena_Evaluating_Long_Context_Multimodal_Agents_with_Video_Understanding_Web_Tasks.pdf
- Link: https://openreview.net/pdf/0665a611105a080d57247da2317726273a86a222.pdf
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ICLR_22_Poster_VideoWebArena_Evaluating_Long_Context_Multimodal_Agents_with_Video_Understanding_Web_Tasks.pdf
- Token Usage: input 28321, output 2925, total 31246

### GitHub & Websites

- VideoWebArena · [GitHub](https://github.com/ljang0/videowebarena) · [Website](https://www.youtube.com/@webarenawarrior) · [Doc](https://drive.google.com/file/d/17DwmsM7KzBWyz1BN1aq7NHDvgcTIrCgx/view?usp=drive_link)
  - description: Official codebase and video dataset for the benchmark introduced in the paper; includes environment, tasks, evaluators, and 74 tutorial videos (YouTube/Drive) needed to reproduce experiments.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev/)
  - description: The paper hosts and reuses these realistic web environments and evaluators; VideoWebArena maps its skill-retention tasks to WebArena templates and uses its evaluation utilities.

- VisualWebArena · [GitHub](https://github.com/web-arena-x/visualwebarena)
  - description: Provides the multimodal web environments and Set-of-Marks observation interface that VideoWebArena builds upon; also supplies evaluators reused by this paper.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Doc](https://playwright.dev/python/docs/intro)
  - description: Browser automation framework used by the benchmark to execute agent actions (each action maps to Playwright Python code in the environment).

- Whisper (OpenAI) · [GitHub](https://github.com/openai/whisper) · [Codewiki](https://codewiki.google/github.com/openai/whisper)
  - description: Speech-to-text system used to transcribe video audio for the “video frames in-context” and “video summary” agent baselines.

- GPT-4o · [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed-source multimodal model used as a baseline backbone for frame-based and summary agents.

- Gemini 1.5 Pro · [Website](https://ai.google.dev/gemini-api) · [Doc](https://ai.google.dev/gemini-api/docs)
  - description: Google’s multimodal long-context model used as the “video in-context” baseline; processes full video and audio.

- Phi-3.5 Vision · [GitHub](https://github.com/microsoft/Phi-3CookBook) · [Doc](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
  - description: Open multimodal model evaluated as an additional baseline (frame-based setting) in the benchmark.

<!-- paper_id: 90462ced637ddc6c9e3b6033a447c287efe2928d -->

## 86. A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges - ACL - 2025 - citation_count 36 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_36_Findings_A_Survey_of_Mathematical_Reasoning_in_the_Era_of_Multimodal_Large_Language_Model_Benchmark,_Method_&_Challenges.pdf
- Link: https://aclanthology.org/2025.findings-acl.614/
- Tags: multiagent, tool, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_36_Findings_A_Survey_of_Mathematical_Reasoning_in_the_Era_of_Multimodal_Large_Language_Model_Benchmark,_Method_&_Challenges.pdf
- Token Usage: input 47112, output 4914, total 52026

### GitHub & Websites

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Grade School Math dataset used throughout the survey as a core benchmark for text-only mathematical reasoning and robustness analyses.

- MATH (Hendrycks et al.) · [GitHub](https://github.com/hendrycks/math)
  - description: Competition-level math benchmark repeatedly referenced for evaluating LLM and MLLM mathematical reasoning capabilities.

- MathVista · [GitHub](https://github.com/lupantech/MathVista)
  - description: Visual math benchmark cited for multimodal evaluation; used in method and benchmark discussions to test diagram/figure-based reasoning.

- MathVerse · [GitHub](https://github.com/OpenGVLab/MathVerse)
  - description: Multimodal visual math benchmark the survey uses to illustrate step-wise generative evaluation and diagram understanding.

- MMMU · [GitHub](https://github.com/MMMU-Benchmark/MMMU) · [Website](https://mmmu-benchmark.github.io)
  - description: Massive multi-discipline multimodal benchmark (with a math track) used to assess broad MLLM reasoning in expert AGI settings.

- ToRA (Tool-integrated Reasoning Agent) · [GitHub](https://github.com/microsoft/ToRA)
  - description: Tool-augmented math agent referenced under the “LLM as Planner” paradigm; plans natural-language and program/tool steps to solve math problems.

- Chameleon · [GitHub](https://github.com/lupantech/Chameleon)
  - description: Plug-and-play compositional reasoning framework cited as a planner-style system that assembles tools for complex (including multimodal) math tasks.

- Visual Sketchpad · [GitHub](https://github.com/allenai/visual-sketchpad)
  - description: Sketch-based visual chain-of-thought system referenced as a planner-style MLLM approach enabling intermediate sketches for math reasoning.

- DeepSeekMath · [GitHub](https://github.com/deepseek-ai/DeepSeek-Math)
  - description: Open math-specialized LLM used in the survey’s landscape of Math-LLMs and as a strong open-source baseline for math reasoning.

- Llemma · [GitHub](https://github.com/EleutherAI/llemma)
  - description: Open mathematics-focused language models cited among Math-LLMs and baselines for symbolic and theorem-oriented reasoning tasks.

- OpenCompass · [GitHub](https://github.com/open-compass/opencompass) · [Website](https://opencompass.org.cn)
  - description: Evaluation framework referenced for generative/discriminative scoring (e.g., CircularEval) in benchmarking LLM/MLLM mathematical reasoning.

- GAIR-NLP/Abel · [GitHub](https://github.com/GAIR-NLP/abel)
  - description: Generative AI for Math project explicitly linked in the references; included as a related open-source implementation practitioners can inspect.

- Qwen2.5-Math · [GitHub](https://github.com/QwenLM/Qwen2.5-Math) · [Website](https://qwenlm.github.io)
  - description: Math-specialized models cited in the Math-LLM progress timeline; commonly used as strong baselines and for self-improvement instruction tuning.

- MistralAI Mathstral · [Website](https://mistral.ai/news/mathstral/)
  - description: Math-focused model referenced in the Math-LLM landscape; official page provides access and guidance for evaluating against survey benchmarks.

- Moonshot k0-math · [Website](https://www.moonshot.cn/)
  - description: Math-oriented model/platform listed among recent Math-LLMs; official site serves as the entry point for usage and evaluation.

- Khanmigo (Khan Academy) · [Website](https://www.khanacademy.org/khan-labs/ai)
  - description: Educational AI platform mentioned in training data sources; relevant for obtaining math instruction-style content used in pretraining/finetuning corpora.

- Duolingo Math · [Website](https://www.duolingo.com/math)
  - description: Referenced as a real-world educational platform contributing math-related instructional data types discussed in the survey’s training data section.

- Squirrel AI Learning · [Website](https://www.squirrelai.com)
  - description: Education platform referenced in affiliations and as part of real-world educational context; useful for practitioners aligning MLLMs to tutoring scenarios.

<!-- paper_id: 9272146b77e6aa6756984e54ab4edebb2f96a7d6 -->

## 87. Offline Training of Language Model Agents with Functions as Learnable Weights - ICML - 2024 - citation_count 29 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_ICML_29_Poster_Offline_Training_of_Language_Model_Agents_with_Functions_as_Learnable_Weights.pdf
- Link: https://openreview.net/pdf/7fc4174c8876ce9cd87db0e2a33d814d09583fb1.pdf
- Tags: science, agent, biology, protein-function, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_ICML_29_Poster_Offline_Training_of_Language_Model_Agents_with_Functions_as_Learnable_Weights.pdf
- Token Usage: input 27198, output 5418, total 32616

### GitHub & Websites

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://www.autogen.org) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent LLM framework into which the authors integrated their AgentOptimizer; use this to reproduce the paper’s agent-training within a maintained library.

- MATH (Measuring Mathematical Problem Solving) · [GitHub](https://github.com/hendrycks/math) · [Website](https://people.eecs.berkeley.edu/~hendrycks/MATH.html)
  - description: Dataset used for mathematical reasoning experiments; the paper samples training/testing problems from MATH.

- TabMWP (Tabular Math Word Problems) · [Website](https://allenai.org/data/tabmwp)
  - description: Dataset used to evaluate tabular processing; the paper subsamples training/testing examples from TabMWP.

- GAIA: A Benchmark for General AI Assistants · [GitHub](https://github.com/GAIA-benchmark/GAIA) · [Website](https://gaia-benchmark.github.io/)
  - description: Benchmark used for real-world question answering; the paper uses its public subset for training/testing.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting agent baseline; the authors train a ReAct agent with their AgentOptimizer.

- OpenAI API (GPT-4-1106-preview, GPT-3.5-turbo-1106) · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Proprietary LLMs used as both the agent backbone and the AgentOptimizer in main experiments.

- Bing Web Search API (v7) · [Doc](https://learn.microsoft.com/azure/cognitive-services/bing-web-search/overview)
  - description: External tool used in GAIA experiments via learned functions to perform web search.

- SymPy · [GitHub](https://github.com/sympy/sympy) · [Website](https://www.sympy.org/en/index.html)
  - description: Python CAS used in generated math functions (e.g., evaluate_expression, solve equations) during MATH experiments.

- Beautiful Soup (bs4) · [Website](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
  - description: HTML parsing library used by learned functions to scrape Wikipedia tables in GAIA tasks.

- PyMuPDF (fitz) · [GitHub](https://github.com/pymupdf/PyMuPDF) · [Doc](https://pymupdf.readthedocs.io/)
  - description: PDF text extraction dependency used by learned functions in GAIA experiments.

- Lizard · [GitHub](https://github.com/terryyin/lizard)
  - description: Static analysis tool used by the paper to compute cyclomatic complexity of generated functions.

- Code Llama · [GitHub](https://github.com/facebookresearch/codellama) · [Website](https://ai.meta.com/research/publications/code-llama/)
  - description: Open-source code model used in supplementary evaluations as an alternative agent backbone.

- Mistral/Mixtral (Mistral 7B, Mixtral-8x7B) · [Website](https://mistral.ai/news/mixtral-of-experts/) 
  - description: Open models used in supplementary experiments to show agent training improvements with non-OpenAI LLMs.

<!-- paper_id: 90ee7c1d50793f7f0da731a1c07d6d2fc47324ac -->

## 88. Multi-Agent Collaboration via Cross-Team Orchestration - ACL - 2025 - citation_count 22 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ACL_22_Findings_Multi-Agent_Collaboration_via_Cross-Team_Orchestration.pdf
- Tags: multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ACL_22_Findings_Multi-Agent_Collaboration_via_Cross-Team_Orchestration.pdf
- Token Usage: input 21069, output 4109, total 25178

### GitHub & Websites

- Croto (Cross-Team Orchestration) · [GitHub](https://github.com/OpenBMB/ChatDev/tree/macnet)
  - description: Official code and data release for the paper; implements the cross-team orchestration framework (greedy aggregation, hierarchical partitioning, pruning) on top of ChatDev for software and story-generation experiments.

- ChatDev · [GitHub](https://github.com/OpenBMB/ChatDev)
  - description: LLM-powered agent collaborative software development framework used as a single-team baseline and underlying execution environment; the Croto implementation is released as a branch within this repository.

- SRDD (Software Repository-Level Development Dataset) · [GitHub](https://github.com/OpenBMB/ChatDev)
  - description: Dataset used for software generation experiments (15 tasks); provided with the ChatDev project and cited as the task source in the paper.

- ROCStories (Story Cloze Test) · [Website](https://www.cs.rochester.edu/nlp/rocstories/)
  - description: Commonsense short-story dataset used for the paper’s story-generation experiments (10 tasks) to test generalization beyond software.

- GPT-Engineer · [GitHub](https://github.com/AntonOsika/gpt-engineer) · [Codewiki](https://codewiki.google/github.com/AntonOsika/gpt-engineer)
  - description: Single-agent software generation system used as a comparison baseline in the experiments.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Website](https://metagpt.ai)
  - description: Multi-agent collaborative framework with role assignment and SOPs; used as a comparison baseline.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Multi-agent framework for assembling expert agents in structured topologies; used as a comparison baseline.

- GPTSwarm
  - description: Graph-like multi-agent system that formulates LLM agents as computational graphs; used as a comparison baseline in the paper.

- OpenAI GPT-3.5-Turbo · [Doc](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  - description: Foundation model used to run all teams/agents in the experiments; necessary to reproduce the results.

<!-- paper_id: d65265a88d7d5dc4833d7d0864deb6a6744b1cad -->

## 89. MatPlotAgent: Method and Evaluation for LLM-Based Agentic Scientific Data Visualization - ACL - 2024 - citation_count 60 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2024_ACL_60_Findings_MatPlotAgent_Method_and_Evaluation_for_LLM-Based_Agentic_Scientific_Data_Visualization.pdf
- Link: https://aclanthology.org/2024.findings-acl.701/
- Tags: science, agent, biology, protein-function, agent-coding
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2024_ACL_60_Findings_MatPlotAgent_Method_and_Evaluation_for_LLM-Based_Agentic_Scientific_Data_Visualization.pdf
- Token Usage: input 18519, output 3860, total 22379

### GitHub & Websites

- MatPlotAgent / MatPlotBench · [GitHub](https://github.com/thunlp/MatPlotAgent)
  - description: Official release from the paper containing the MatPlotAgent framework, MatPlotBench benchmark (100 cases), prompts, and evaluation scripts needed to reproduce results.

- Matplotlib · [Website](https://matplotlib.org) · [Doc](https://matplotlib.org/stable/gallery/index.html)
  - description: Core plotting library used by the agent; the Matplotlib Gallery served as a primary source of original examples for constructing MatPlotBench.

- OriginLab GraphGallery · [Website](https://www.originlab.com/graphgallery)
  - description: Source of advanced visualization examples (e.g., Sankey, sunburst, chord) used to design part of MatPlotBench; ground-truth figures were derived from these examples when needed.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Doc](https://docs.vllm.ai/en/latest/)
  - description: High-throughput LLM inference engine used to run the open-source code LLM baselines in the experiments.

- Qwen-Agent Code Interpreter Benchmark · [GitHub](https://github.com/QwenLM/Qwen-Agent/tree/main/benchmark)
  - description: External benchmark whose visualization subsets were used to further evaluate MatPlotAgent and analyze the effect of visual feedback.

- OpenAI GPT-4 / GPT-4V / GPT-3.5 · [Website](https://openai.com/product/gpt-4) · [Doc](https://platform.openai.com/docs)
  - description: Proprietary code LLMs used as agents and GPT-4V used as the automatic evaluator (scoring visualizations 0–100) and as a visual agent in MatPlotAgent.

- Google Gemini Pro Vision · [Website](https://ai.google.dev/gemini-api) · [Doc](https://ai.google.dev/gemini-api/docs)
  - description: Alternative multimodal model used as the visual agent to provide visual feedback in MatPlotAgent.

- Magicoder · [GitHub](https://github.com/ise-uiuc/Magicoder)
  - description: Open-source code LLM baseline (Magicoder-S-DS-6.7B) evaluated within MatPlotAgent and in direct decoding comparisons.

- DeepSeek-Coder · [GitHub](https://github.com/deepseek-ai/deepseek-coder) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/deepseek-coder)
  - description: Open-source code LLM baselines (6.7B/33B instruct) evaluated as code agents in the experiments.

- Code Llama · [GitHub](https://github.com/facebookresearch/codellama)
  - description: Open-source code LLM baseline (CodeLlama-34B-Instruct) included in the experimental comparisons.

- WizardCoder / WizardLM · [GitHub](https://github.com/nlpxucan/WizardLM)
  - description: Open-source code LLM baseline (WizardCoder-Python-33B-V1.1) used for comparisons with and without MatPlotAgent.

- SciPy · [GitHub](https://github.com/scipy/scipy) · [Codewiki](https://codewiki.google/github.com/scipy/scipy) · [Doc](https://docs.scipy.org/doc/scipy/reference/stats.html)
  - description: Used to compute Pearson correlation and p-values to validate that GPT-4V automatic scores correlate with human evaluations.

- HoloViews · [GitHub](https://github.com/holoviz/holoviews) · [Website](https://holoviews.org)
  - description: Visualization library referenced in MatPlotBench queries (e.g., chord diagram with Bokeh backend), which the agent may use to satisfy task requirements.

- Bokeh · [GitHub](https://github.com/bokeh/bokeh) · [Codewiki](https://codewiki.google/github.com/bokeh/bokeh) · [Website](https://bokeh.org) · [Doc](https://docs.bokeh.org/en/latest/)
  - description: Visualization backend specified in some MatPlotBench tasks (e.g., HoloViews+Bokeh chord diagrams) and thus relevant for reproducing those plots.

<!-- paper_id: 8410e4416df27c6f72df3b691edded84a4e86f16 -->

## 90. DS-Agent: Automated Data Science by Empowering Large Language Models with Case-Based Reasoning - ICML - 2024 - citation_count 68 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_ICML_68_Poster_DS-Agent_Automated_Data_Science_by_Empowering_Large_Language_Models_with_Case-Based_Reasoning.pdf
- Link: https://openreview.net/pdf/ca6f6e07e47b268198c00cca4b58903bd015c219.pdf
- Tags: science, agent, biology, protein-function, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_ICML_68_Poster_DS-Agent_Automated_Data_Science_by_Empowering_Large_Language_Models_with_Case-Based_Reasoning.pdf
- Token Usage: input 46592, output 4695, total 51287

### GitHub & Websites

- DS-Agent · [GitHub](https://github.com/guosyjlu/DS-Agent)
  - description: Official code release for the paper; includes the DS-Agent framework, datasets/splits, prompts, case banks, and scripts to reproduce both development and deployment experiments.

- Kaggle
  · [Website](https://www.kaggle.com) · [Doc](https://www.kaggle.com/docs)
  - description: Primary source of human insight cases (competition reports and notebooks) and many datasets used in both stages; DS-Agent retrieves and reuses Kaggle knowledge within its CBR pipeline.

- ResearchAgent · [GitHub](https://github.com/snap-stanford/ResearchAgent)
  - description: Baseline agent compared against DS-Agent in the development stage experiments.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai) · [Doc](https://vllm.readthedocs.io)
  - description: Serving framework used to run the open-source LLM (Mixtral-8x7B-Instruct) efficiently in the deployment stage.

- Mixtral‑8x7B‑Instruct · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Open-source instruction-tuned LLM used as one of the backbones for DS-Agent in the deployment stage.

- OpenAI GPT‑3.5 / GPT‑4 · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Closed-source LLMs used as core backbones for DS-Agent in both stages; main results reported with GPT‑3.5 and GPT‑4.

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers)
  - description: Library used in generated training code (e.g., BERT/RoBERTa/DeBERTa models) within DS-Agent’s experiments.

- scikit‑learn · [GitHub](https://github.com/scikit-learn/scikit-learn) · [Codewiki](https://codewiki.google/github.com/scikit-learn/scikit-learn) · [Website](https://scikit-learn.org/stable/)
  - description: ML toolkit commonly invoked by DS-Agent and baselines for tabular tasks (e.g., logistic regression, preprocessing).

- AutoGluon · [GitHub](https://github.com/autogluon/autogluon) · [Website](https://auto.gluon.ai)
  - description: AutoML system used for empirical comparison in Appendix C.1 on tabular tasks.

- H2O AutoML · [GitHub](https://github.com/h2oai/h2o-3) · [Doc](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
  - description: Related AutoML system discussed in the paper’s comparison of DS-Agent vs. AutoML approaches.

- ETDataset (ETT) · [GitHub](https://github.com/zhouhaoyi/ETDataset)
  - description: Research time-series forecasting dataset family (e.g., ETTm2) used in development/deployment tasks.

- UCR/UEA Time Series Classification Repository · [Website](https://www.timeseriesclassification.com)
  - description: Source for several “Research Dataset” time-series classification tasks used in experiments (e.g., UWaveGestureLibrary, Heartbeat, Self-Regulation SCP1).

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Technique that inspires DS-Agent’s Debugger component to reflect on execution feedback and iteratively fix code.

- llm‑embedder · [Website](https://huggingface.co/maidalun1020/llm-embedder)
  - description: Pretrained embedding model used as the retriever encoder for case similarity in DS-Agent’s CBR pipeline.

<!-- paper_id: 850dadb26cafc16539074d22745783c0e0bbd01f -->

## 91. Multimodal Procedural Planning via Dual Text-Image Prompting - EMNLP - 2024 - citation_count 51 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2024_EMNLP_51_Findings_Multimodal_Procedural_Planning_via_Dual_Text-Image_Prompting.pdf
- Link: http://arxiv.org/pdf/2305.01795
- Tags: planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2024_EMNLP_51_Findings_Multimodal_Procedural_Planning_via_Dual_Text-Image_Prompting.pdf
- Token Usage: input 32133, output 4561, total 36694

### GitHub & Websites

- WIKIPLAN (dataset)
  - description: Authors’ curated multimodal procedural planning testbed collected from WikiHow; used as a main evaluation benchmark. The paper notes “Our code and data are provided in supplemental materials” and “We plan to release the raw data,” indicating no public repo link in the paper.

- RECIPEPLAN (dataset)
  - description: Authors’ repurposed benchmark built from the RECIPEQA test split (Visual Ordering) for multimodal procedural planning; used for zero-shot evaluation alongside WIKIPLAN.

- WikiHow
  - [Website](https://www.wikihow.com)
  - description: Source site the authors crawled to construct the WIKIPLAN dataset (titles, step texts, and step images).

- RECIPEQA
  - [Website](https://hucvl.github.io/recipeqa/)
  - description: Public dataset of multimodal cooking recipes; the authors repurpose its test set to build RECIPEPLAN.

- Stable Diffusion
  - [GitHub](https://github.com/CompVis/stable-diffusion) · [Website](https://stability.ai)
  - description: Text-to-image diffusion model used to generate image plans; serves as the backbone T2I model in TIP and in baseline comparisons.

- DALL·E 2
  - [Website](https://openai.com/dall-e-2) · [Doc](https://platform.openai.com/docs/guides/images)
  - description: OpenAI text-to-image model used as a baseline for image plan generation conditioned on text references.

- InstructGPT / Text-Davinci-002/003 (OpenAI)
  - [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs/api-reference/completions)
  - description: LLMs used to produce the vanilla text plan and to perform revisions in TIP; also used in baselines that pair LLM-generated text with Stable Diffusion images.

- BLIP (Bootstrapping Language-Image Pre-training)
  - [GitHub](https://github.com/salesforce/BLIP)
  - description: Image captioning model used in the Image-to-Text Bridge (I2T-B) to verbalize generated images; also used as a captioning baseline (ImageRef + BLIP-Caption).

- OFA (One-For-All)
  - [GitHub](https://github.com/OFA-Sys/OFA) · [Doc](https://huggingface.co/OFA-Sys/ofa-base)
  - description: Vision-language model used as an alternative captioner in baselines (ImageRef + OFA-Caption).

- CLIP
  - [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Vision-language model used to compute alignment and to support CLIPScore-based evaluation of image plans.

- CLIPScore
  - [GitHub](https://github.com/jmhessel/clipscore)
  - description: Reference-free image captioning/vision-language metric; used to automatically evaluate visual plan quality and alignment.

- Sentence-BERT (S-BERT)
  - [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net)
  - description: Sentence embedding toolkit used to compute Text-S, Cap-S, and ALL-S scores for semantic similarity of text and image plans to references.

- Fréchet Inception Distance (FID)
  - [GitHub](https://github.com/mseitzer/pytorch-fid)
  - description: Standard generative image quality metric used to evaluate the realism of generated image plans.

- METEOR
  - [Website](https://www.cs.cmu.edu/~alavie/METEOR/)
  - description: Text evaluation metric used to measure overlap between generated text plans and references.

- ROUGE
  - [GitHub](https://github.com/pltrdy/rouge)
  - description: Text summarization metric (ROUGE-L) used to evaluate similarity between generated and reference text plans.

- Word Mover’s Distance (WMD)
  - [GitHub](https://github.com/RaRe-Technologies/gensim)
  - description: Distance metric (implemented in Gensim) used to quantify semantic distance between generated text plans and reference texts.

<!-- paper_id: b908824639d18f11883abcab21efeb22e315ab9c -->

## 92. Plan-and-Act: Improving Planning of Agents for Long-Horizon Tasks - ICML - 2025 - citation_count 49 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_ICML_49_Poster_Plan-and-Act_Improving_Planning_of_Agents_for_Long-Horizon_Tasks.pdf
- Tags: planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_ICML_49_Poster_Plan-and-Act_Improving_Planning_of_Agents_for_Long-Horizon_Tasks.pdf
- Token Usage: input 46455, output 4933, total 51388

### GitHub & Websites

- WebArena / WebArena-Lite · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Primary web-navigation benchmark and environment used for training, ablations, and main evaluation; WebArena-Lite provides the 165 human-verified tasks and executor prompt format used by the paper.

- WebVoyager
  - description: Real-world web benchmark on which the authors report text-only SOTA; used to test generalization beyond simulator-based tasks.

- OpenWebVoyager
  - description: Iterative real-world exploration variant used as a comparison point in the paper’s results section.

- WebRL · [GitHub](https://github.com/THUDM/WebRL)
  - description: Framework used to generate/score trajectories in the synthetic data pipeline; the authors use WebRL-Llama-3.1-70B as the actor and ORM-Llama-3.1-8B as the outcome-supervised reward model for filtering successful trajectories.

- AutoWebGLM · [GitHub](https://github.com/THUDM/AutoWebGLM)
  - description: Baseline LLM web agent referenced for comparison on WebArena.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Prompting style used for baseline executors without an explicit planner; forms the “No Planner / ReAct-style” training baseline.

- Self-Instruct · [GitHub](https://github.com/yizhongw/self-instruct)
  - description: Instruction-generation approach that inspires the paper’s Alpaca-style synthetic query generation used in their data pipeline.

- Stanford Alpaca · [GitHub](https://github.com/tatsu-lab/stanford_alpaca) · [Codewiki](https://codewiki.google/github.com/tatsu-lab/stanford_alpaca)
  - description: Provides the Alpaca-style data generation recipe the authors adapt to expand user queries and plans.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai/)
  - description: Inference engine used for running the fine-tuned PLANNER and EXECUTOR during evaluation.

- torchtune · [GitHub](https://github.com/pytorch/torchtune)
  - description: Training framework specified in the paper’s hyperparameters; used to fine-tune the PLANNER and EXECUTOR.

- LoRA · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient finetuning method used for dynamic replanning experiments due to compute constraints.

- DeepSeek-R1 (Reasoning) · [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-R1)
  - description: Teacher model used to generate Chain-of-Thought traces for both PLANNER and EXECUTOR in the CoT experiments.

- Llama (Meta) · [Website](https://llama.meta.com)
  - description: Base LLM family; the paper fine-tunes Llama-3.3-70B-Instruct for both PLANNER and EXECUTOR and also reports results with smaller Llama variants.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web)
  - description: Popular web-agent dataset cited as related work; relevant for practitioners seeking additional training/evaluation data aligned with the paper’s domain.

<!-- paper_id: 81d0a2c6001e2b9a8770d36737ec4022436a9e4c -->

## 93. ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention - NeurIPS - 2024 - citation_count 47 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_NeurIPS_47_Poster_ProSST_Protein_Language_Modeling_with_Quantized_Structure_and_Disentangled_Attention.pdf
- Link: https://openreview.net/pdf/6c75eed7fafd81f83f87f954b10f95768b59f37b.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_NeurIPS_47_Poster_ProSST_Protein_Language_Modeling_with_Quantized_Structure_and_Disentangled_Attention.pdf
- Token Usage: input 25205, output 5055, total 30260

### GitHub & Websites

- ProSST · [GitHub](https://github.com/ai4protein/ProSST)
  - description: Official codebase and pre-trained models released by the paper; implements the structure quantizer, disentangled attention Transformer, training and evaluation pipelines.

- ProteinGym · [GitHub](https://github.com/OATML-Markslab/ProteinGym)
  - description: Benchmark and evaluation scripts used for zero-shot mutation effect prediction; the paper computes Spearman/NDCG/Top-Recall with the provided scripts.

- AlphaFold Protein Structure Database (AFDB) · [Website](https://alphafold.ebi.ac.uk) · [Doc](https://cluster.foldseek.com)
  - description: Primary source of 18.8M protein structures used for ProSST pre-training; the paper downloaded the 90% reduced clustered release from cluster.foldseek.com.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk)
  - description: Structure prediction system used to generate wild-type structures for ProteinGym and suggested for obtaining structures for sequence-only datasets.

- ESMFold · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/resources?action=fold)
  - description: Alternative structure prediction model suggested by the paper for sequence-only cases; also used in the sequence-only analysis.

- CATH (CATH43-S40) · [Website](https://www.cathdb.info) · [Doc](http://download.cathdb.info/cath/releases/all-releases/v4_3_0/non-redundant-data-sets/)
  - description: Manually curated structure dataset used to train the GVP autoencoder and to build the local-structure codebook (k-means clustering on residue-local embeddings).

- SaProt · [GitHub](https://github.com/westlake-repl/SaProt)
  - description: Structure-aware PLM baseline and data resource; the paper follows SaProt’s downstream splits and AFDB retrieval procedure and compares against its models.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com)
  - description: Fast structure search and quantized structure tokens used in prior work; ProSST compares against Foldseek-based tokenization in ablations and references foldseek clusters for AFDB.

- DSSP · [GitHub](https://github.com/cmbi/dssp) · [Website](https://swift.cmbi.nl/structure/dssp/)
  - description: Physics-based secondary structure assignment used as an alternative discretization baseline in ProSST’s ablations.

- GVP (Geometric Vector Perceptron) · [GitHub](https://github.com/drorlab/gvp-pytorch)
  - description: Structure encoder backbone used in ProSST’s quantization module to embed residue-level local structures before k-means clustering.

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion)
  - description: Referenced for Brownian motion on rotation manifolds; used to define the denoising objective for training the structure autoencoder.

- DeepLoc · [Website](http://www.cbs.dtu.dk/services/DeepLoc/)
  - description: Subcellular localization dataset/task used for supervised fine-tuning evaluation (accuracy metric).

- TAPE · [GitHub](https://github.com/songlab-cal/tape) · [Doc](https://tape.readthedocs.io)
  - description: Source of the Metal Ion Binding dataset and splits used in supervised fine-tuning comparisons.

- DeepFRI · [GitHub](https://github.com/flatironinstitute/DeepFRI)
  - description: Provider of GO annotation tasks (MF/BP/CC) used to evaluate ProSST in multi-label function prediction.

<!-- paper_id: f28c68f517715dd238478c20c03700e2e6a1b172 -->

## 94. AlphaFold Meets Flow Matching for Generating Protein Ensembles - ICML - 2024 - citation_count 161 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_161_Oral_AlphaFold_Meets_Flow_Matching_for_Generating_Protein_Ensembles.pdf
- Link: https://openreview.net/pdf/29f29a3b22e2038d3f465822db02f5d303a5d273.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_161_Oral_AlphaFold_Meets_Flow_Matching_for_Generating_Protein_Ensembles.pdf
- Token Usage: input 34491, output 4094, total 38585

### GitHub & Websites

- AlphaFLOW (this paper) · [GitHub](https://github.com/bjing2016/alphaflow)
  - description: Official code release for AlphaFLOW/ESMFLOW, including training and sampling under the custom flow-matching framework described in the paper.

- OpenFold · [GitHub](https://github.com/aqlaboratory/openfold) · [Website](https://aqlaboratory.github.io/openfold/)
  - description: Open-source reimplementation of AlphaFold used for architecture and training pipeline; authors fine-tune all weights via OpenFold and use it to run baselines.

- OpenProteinSet · [Website](https://openprotein.ai/openproteinset)
  - description: Large-scale MSA/training data resource used to provide MSAs for training; cited as the source of training MSAs.

- ATLAS: Protein flexibility from atomistic MD · [Website](https://atlas.protein-dynamics.org)
  - description: Dataset of all-atom MD simulation ensembles; used for additional fine-tuning and all MD-based evaluations in the paper.

- AlphaFold · [GitHub](https://github.com/deepmind/alphafold)
  - description: Original AlphaFold code and pretrained weights; authors initialize from the CASP14 DeepMind weights and compare against AlphaFold with MSA subsampling.

- ESMFold (ESM) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/esmfold)
  - description: Protein language model and folding pipeline used as the second backbone (ESMFLOW) that the authors adapt to flow matching and fine-tune.

- ColabFold · [GitHub](https://github.com/sokrypton/ColabFold) · [Website](https://colabfold.mmseqs.com)
  - description: Used to generate MSAs at inference time (the “ColabFold MMSeqs2 pipeline”) for both PDB and ATLAS evaluations.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Sequence search/cluster engine underlying the ColabFold MSA pipeline and used for PDB clustering in dataset prep.

- MDTraj · [GitHub](https://github.com/mdtraj/mdtraj) · [Doc](http://mdtraj.org/latest/)
  - description: Library used for ensemble analysis (e.g., RMSD/RMSF, PCA projections, SASA via Shrake–Rupley) in MD evaluations.

- ProDy · [GitHub](https://github.com/prody/ProDy) · [Website](http://prody.csb.pitt.edu)
  - description: Toolkit used to run normal mode analysis (GNM/ANM) baselines for comparison against AlphaFLOW-MD.

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org)
  - description: Source of experimental protein structures used for training (with date cutoffs) and for constructing the heterogeneous PDB test set.

- SIFTS (Structure Integration with Function, Taxonomy and Sequences) · [Website](https://www.ebi.ac.uk/pdbe/docs/sifts/)
  - description: Mapping resource used to associate PDB chains with UniProt segments when constructing the PDB test set.

- ECOD: Evolutionary Classification of Protein Domains · [Website](https://prodata.swmed.edu/ecod/)
  - description: Domain classification used to select structurally diverse proteins in the ATLAS dataset.

- UniProt · [Website](https://www.uniprot.org)
  - description: Reference sequences used to define targets and align PDB chains for evaluation.

- ClustalW · [Website](http://www.clustal.org/clustal2/)
  - description: Multiple sequence alignment tool used to align PDB sequences with UniProt references for PCA-based analyses.

<!-- paper_id: 8136c9a5915cee9bf332e0969719dd4884f7c673 -->

## 95. Diffusion Language Models Are Versatile Protein Learners - ICML - 2024 - citation_count 80 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICML_80_Poster_Diffusion_Language_Models_Are_Versatile_Protein_Learners.pdf
- Link: https://openreview.net/pdf/38c7073abf10f2aa8147ec5cb085074bbbe3e8ed.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICML_80_Poster_Diffusion_Language_Models_Are_Versatile_Protein_Learners.pdf
- Token Usage: input 39504, output 4831, total 44335

### GitHub & Websites

- ESM (ESM-2, ESMFold, ESM-IF) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com)
  - description: Used throughout for baselines and evaluation—ESM-2 as a masked-LM baseline and initialization checkpoint, ESMFold to predict structures/pLDDT for generated sequences, and the ESM inverse-folding (GVP-Transformer) encoder as the structure expert for DPLM’s adapter-based inverse folding.

- UniRef50 · [Website](https://www.uniprot.org/uniref/)
  - description: Primary pretraining dataset; the authors train DPLM on ~45M UniRef50 protein sequences (~14B tokens).

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Used to assess structural novelty of generated sequences by comparing against known PDB structures (via TM-score).

- CATH (v4.2, v4.3) · [Website](https://www.cathdb.info/)
  - description: Benchmarks for structure-conditioned inverse folding; the paper reports amino acid recovery and structural self-consistency on CATH 4.2/4.3.

- OmegaFold · [GitHub](https://github.com/HeliXonProtein/OmegaFold)
  - description: Employed to fold motif-scaffolding generations for success-rate evaluation (pLDDT and motif RMSD).

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion) · [Website](https://rfdiffusion.github.io/)
  - description: Structure-design baseline compared against DPLM in motif-scaffolding experiments.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Strong inverse-folding baseline cited and compared in CATH experiments.

- GVP (Geometric Vector Perceptron) · [GitHub](https://github.com/drorlab/gvp-pytorch)
  - description: Core geometric module underlying the GVP-Transformer encoder used as the structural expert in DPLM’s adapter-tuned inverse folding.

- TAPE Benchmark · [GitHub](https://github.com/songlab-cal/tape)
  - description: Source for secondary-structure prediction (SSP) data and evaluation protocol used to train the classifier providing discrete guidance for controllable generation.

- DeepLoc 1.0 · [Website](https://services.healthtech.dtu.dk/service.php?DeepLoc-1.0)
  - description: Dataset/task used in downstream protein localization prediction benchmarking.

- TM-score · [Website](https://zhanggroup.org/TM-score/)
  - description: Tool used to quantify structural similarity; the paper reports pdb-TM and inner-TM metrics when evaluating novelty and diversity.

<!-- paper_id: 9ec09a9534fe41d3a5231693050150e0bbec8515 -->

## 96. LLM and Simulation as Bilevel Optimizers: A New Paradigm to Advance Physical Scientific Discovery - ICML - 2024 - citation_count 55 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_ICML_55_Poster_LLM_and_Simulation_as_Bilevel_Optimizers_A_New_Paradigm_to_Advance_Physical_Scientific_Discovery.pdf
- Link: https://openreview.net/pdf/8e1aee641dee0dc2d14eb58b4c9d0f25392e7047.pdf
- Tags: science, agent, biology, protein-function, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_ICML_55_Poster_LLM_and_Simulation_as_Bilevel_Optimizers_A_New_Paradigm_to_Advance_Physical_Scientific_Discovery.pdf
- Token Usage: input 29246, output 4702, total 33948

### GitHub & Websites

- NVIDIA Warp · [GitHub](https://github.com/NVIDIA/warp) · [Doc](https://nvidia.github.io/warp/)
  - description: GPU simulation framework used as the differentiable MPM simulator in the inner-level optimization to compute gradients and feedback.

- PyTorch · [GitHub](https://github.com/pytorch/pytorch) · [Website](https://pytorch.org) · [Doc](https://pytorch.org/docs/stable/)
  - description: Deep learning library providing autograd and optimization; the inner-level differentiable optimization and physics-code templates are implemented in PyTorch.

- RDKit · [GitHub](https://github.com/rdkit/rdkit) · [Website](https://www.rdkit.org) · [Doc](https://www.rdkit.org/docs/)
  - description: Cheminformatics toolkit used for molecule 3D conformation generation (ETKDG) and MMFF94 optimization in the molecular design tasks.

- Uni-Mol · [GitHub](https://github.com/dptech-corp/Uni-Mol)
  - description: Pretrained 3D molecular representation model used to compute quantum mechanical property values; the paper uses a UniMol model fine-tuned on QM9.

- QM9 Dataset · [Website](https://quantum-machine.org/datasets/) · [Doc](https://figshare.com/articles/dataset/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/1057643)
  - description: Quantum chemistry dataset providing structures and properties; used for UniMol fine-tuning and for normalizing targets in molecular design experiments.

- OpenAI GPT-4 Turbo · [Website](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4) · [Doc](https://platform.openai.com/docs/guides/text-generation)
  - description: Main LLM backbone (gpt-4-turbo-preview) driving the outer-level search/generation of hypotheses and code.

- Anthropic Claude 3 Sonnet · [Website](https://www.anthropic.com/news/claude-3-family) · [Doc](https://docs.anthropic.com/claude/docs)
  - description: Alternative LLM evaluated in ablations for the outer-level optimizer.

- Mixtral-8x7B · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Open-source LLM evaluated in ablations as another backbone for the outer-level search.

- Eureka · [GitHub](https://github.com/eureka-research/eureka) · [Website](https://eureka-research.github.io/)
  - description: Baseline method for LLM-driven reward design/program synthesis; compared against SGA in experiments.

- ChemGE (Population-based de novo molecule generation) · [GitHub](https://github.com/tsudalab/ChemGE)
  - description: Population-based molecular design baseline used for comparison on molecular optimization tasks.

- SRBench · [GitHub](https://github.com/cavalab/srbench)
  - description: Benchmark suite for symbolic regression; used to evaluate and compare against symbolic regression baselines in the constitutive-law task reformulation.

- Stanford 3D Scanning Repository (Stanford Bunny) · [Website](http://graphics.stanford.edu/data/3Dscanrep/)
  - description: Source of the Stanford Bunny mesh used in a figure illustrating the simulation pipeline.

<!-- paper_id: e51dff31f56847c16de3a2f2682d16109537b96e -->

## 97. DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines - ICLR - 2024 - citation_count 431 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-sdk/2024_ICLR_431_DSPy_Compiling_Declarative_Language_Model_Calls_into_Self-Improving_Pipelines.pdf
- Tags: agent-sdk
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-sdk/2024_ICLR_431_DSPy_Compiling_Declarative_Language_Model_Calls_into_Self-Improving_Pipelines.pdf
- Token Usage: input 31519, output 4526, total 36045

### GitHub & Websites

- DSPy · [GitHub](https://github.com/stanfordnlp/dspy) · [Codewiki](https://codewiki.google/github.com/stanfordnlp/dspy) · [Website](https://dspy.ai)
  - description: Official code and docs for the programming model introduced in the paper; provides signatures, modules (Predict, ChainOfThought, ReAct, etc.), teleprompters, and the compiler used in all experiments.

- GSM8K (Grade School Math 8K) · [GitHub](https://github.com/openai/grade-school-math) · [Doc](https://huggingface.co/datasets/gsm8k)
  - description: Dataset of math word problems used in the paper’s Case Study 1 to evaluate CoT, reflection, and DSPy compilation strategies.

- HotPotQA · [GitHub](https://github.com/hotpotqa/hotpot) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used in Case Study 2; the paper evaluates in the fullwiki setting using the official Wikipedia 2017 abstracts index.

- ColBERT / ColBERTv2 · [GitHub](https://github.com/stanford-futuredata/ColBERT) · [Website](https://colbert.ai)
  - description: Late-interaction neural retriever used as the default search retriever in DSPy (built-in dspy.Retrieve) and for HotPotQA experiments; the paper mentions using a ColBERTv2 retrieval server.

- Pyserini · [GitHub](https://github.com/castorini/pyserini) · [Website](https://pyserini.io)
  - description: BM25/Anserini-based retrieval toolkit that DSPy supports as a built-in retriever option.

- Pinecone · [Website](https://www.pinecone.io) · [Doc](https://docs.pinecone.io)
  - description: Managed vector database referenced as a supported retriever backend in DSPy.

- Llama 2 (Llama2-13b-chat) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
  - description: Open model used as one of the primary LMs in both case studies; DSPy compiles programs for this model to achieve strong quality without hand-crafted prompts.

- OpenAI GPT-3.5 · [Website](https://openai.com/product) · [Doc](https://platform.openai.com/docs)
  - description: Proprietary LM used as another primary model in experiments and in compilation teacher/student settings.

- FLAN-T5 (flan-t5-large) · [Doc](https://huggingface.co/google/flan-t5-large)
  - description: Smaller open LM used in the paper’s finetuning experiments via DSPy’s BootstrapFinetune teleprompter.

- T5-Large · [Doc](https://huggingface.co/t5-large)
  - description: 770M-parameter model referenced as a target for finetuning to build efficient compiled programs.

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers/index)
  - description: Core modeling library used for hosting/fine-tuning LMs in DSPy experiments (listed in Appendix F).

- Hugging Face Datasets · [GitHub](https://github.com/huggingface/datasets) · [Codewiki](https://codewiki.google/github.com/huggingface/datasets) · [Doc](https://huggingface.co/docs/datasets)
  - description: Dataset handling library used during model fine-tuning (per Appendix F).

- PEFT (Parameter-Efficient Fine-Tuning) · [GitHub](https://github.com/huggingface/peft) · [Codewiki](https://codewiki.google/github.com/huggingface/peft)
  - description: Library used for efficient fine-tuning of LMs as part of DSPy’s finetuning teleprompter setup (Appendix F).

- TRL (Transformer Reinforcement Learning) · [GitHub](https://github.com/huggingface/trl) · [Codewiki](https://codewiki.google/github.com/huggingface/trl)
  - description: Training utilities referenced among the fine-tuning stack used in experiments (Appendix F).

- Text Generation Inference (TGI) · [GitHub](https://github.com/huggingface/text-generation-inference) · [Doc](https://huggingface.co/docs/text-generation-inference)
  - description: High-performance inference server used to host LMs for DSPy (listed as a requirement in Appendix F).

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Doc](https://python.langchain.com)
  - description: Agent/pipeline toolkit cited as a popular baseline framework that relies on prompt templates; contrasted with DSPy’s modular, compiler-driven approach.

- Semantic Kernel · [GitHub](https://github.com/microsoft/semantic-kernel) · [Codewiki](https://codewiki.google/github.com/microsoft/semantic-kernel) · [Doc](https://learn.microsoft.com/semantic-kernel/)
  - description: Microsoft’s orchestration framework mentioned as a related toolkit that connects LMs and tools via prompt templates; relevant for practitioners comparing SDKs.

- LlamaIndex · [GitHub](https://github.com/jerryjliu/llama_index) · [Website](https://www.llamaindex.ai/)
  - description: Retrieval/agent library discussed as a popular framework relying on prompt engineering; serves as a comparison point to DSPy’s abstractions.

- LMQL (Language Model Query Language) · [GitHub](https://github.com/eth-sri/lmql) · [Website](https://lmql.ai/)
  - description: Related query language for constrained decoding that the paper notes could be used to implement specific advanced DSPy modules.

<!-- paper_id: 2069aaaa281eb13bcd9330fc4d43f24f6b436a53 -->

## 98. Protein Conformation Generation via Force-Guided SE(3) Diffusion Models - ICML - 2024 - citation_count 43 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_43_Poster_Protein_Conformation_Generation_via_Force-Guided_SE(3)_Diffusion_Models.pdf
- Link: https://openreview.net/pdf/e517c97a7e9d4f13dca2fb7971865e8d5fb0194a.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_43_Poster_Protein_Conformation_Generation_via_Force-Guided_SE(3)_Diffusion_Models.pdf
- Token Usage: input 32678, output 5236, total 37914

### GitHub & Websites

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org/) · [Doc](https://www.wwpdb.org/documentation)
  - description: Primary dataset used to train and validate CONFDIFF; the paper samples single-chain experimental structures from PDB to learn the conformation distribution.

- OpenMM · [GitHub](https://github.com/openmm/openmm) · [Website](https://openmm.org/) · [Doc](https://openmm.org/documentation.html)
  - description: Molecular simulation toolkit used to compute potential energies and forces, perform protonation/solvation, and run restrained minimization for energy/force guidance.

- FASPR (Fast and Accurate Side-chain Packing) · [Website](https://doi.org/10.1093/bioinformatics/btaa234)
  - description: Open-source side-chain packing tool used to add side-chain atoms to backbone-only samples before OpenMM energy/force evaluation.

- Amber ff14SB force field (with GBn2 implicit solvent) · [Website](https://ambermd.org/)
  - description: Force field and solvent model employed within OpenMM to evaluate energies/forces and run restrained minimization of generated structures.

- PyEMMA · [GitHub](https://github.com/markovmodel/PyEMMA) · [Website](https://www.pyemma.org/)
  - description: Used to featurize protein structures (pairwise Cα distances) and for preparing inputs to time-lagged independent component analysis (TICA) in evaluation.

- Deeptime · [GitHub](https://github.com/deeptime-ml/deeptime) · [Doc](https://deeptime-ml.github.io/deeptime/)
  - description: Library used to perform TICA and project structures onto slow dynamical components for distributional comparisons (JS distance).

- ESMFold (ESM) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/resources/esmfold)
  - description: Provides pretrained sequence representations concatenated into the conditional score model; also subjected to noise injection during training to enhance diversity.

- AlphaFold · [GitHub](https://github.com/google-deepmind/alphafold) · [Codewiki](https://codewiki.google/github.com/google-deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Referenced folding model; AlphaFold predictions are used for comparisons and to inform related baselines (e.g., STR2STR heating/annealing setup).

- Fast-folding protein MD dataset (Lindorff-Larsen et al.) · [Website](https://www.science.org/doi/10.1126/science.1208351)
  - description: Benchmark of 12 short proteins with long-timescale MD trajectories used to evaluate whether generated ensembles match Boltzmann-distributed conformations (JS metrics, contacts, diversity).

- BPTI metastable states MD dataset (Shaw et al., 2010) · [Website](https://www.science.org/doi/10.1126/science.1187409)
  - description: MD trajectories and five kinetic cluster references used to assess recovery of metastable states; cluster PDBs are provided in the paper’s supplementary materials.

- SARS‑CoV‑2 MD simulations (DESRES) · [Website](https://www.deshawresearch.com/resources_sarscov2.html)
  - description: Public MD datasets for Spike RBD and Mpro referenced in the paper’s SARS‑CoV‑2 case study section for distributional evaluation against real trajectories.

- EIGENFOLD
  - description: Baseline diffusion model compared against CONFDIFF for conformation generation and distribution recovery (fully conditioned on sequence representations).

- STR2STR
  - description: Baseline score-based framework used for comparison; unconditional model with heating/annealing used to assess diversity/quality trade-offs against CONFDIFF.

<!-- paper_id: 2516bb58657965236cab56e71a98b9fa7ffc886d -->

## 99. InstructProtein: Aligning Human and Protein Language via Knowledge Instruction - ACL - 2024 - citation_count 35 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ACL_35_Long_InstructProtein_Aligning_Human_and_Protein_Language_via_Knowledge_Instruction.pdf
- Link: https://aclanthology.org/2024.acl-long.62/
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ACL_35_Long_InstructProtein_Aligning_Human_and_Protein_Language_via_Knowledge_Instruction.pdf
- Token Usage: input 29640, output 6068, total 35708

### GitHub & Websites

- InstructProtein · [GitHub](https://github.com/HICAI-ZJU/InstructProtein)
  - description: Official code release for the paper; contains training/tuning scripts and resources to align human and protein language and reproduce the experiments.

- UniProtKB/Swiss-Prot · [Website](https://www.uniprot.org/)
  - description: Curated protein knowledgebase used to build the protein knowledge graph (proteins, relations, annotations) that drives instruction generation.

- UniRef100 · [Website](https://www.uniprot.org/help/uniref)
  - description: Protein sequence clusters used as the protein-language pretraining corpus.

- PubMed · [Website](https://pubmed.ncbi.nlm.nih.gov/)
  - description: Source of biomedical text sentences used as the natural-language pretraining corpus.

- InterPro · [Website](https://www.ebi.ac.uk/interpro/)
  - description: Database used to retrieve domain/family relationships for knowledge causal modeling (KCM) in the instruction generation pipeline.

- Gene Ontology (GO) · [Website](https://geneontology.org/)
  - description: Knowledgebase providing BP/MF/CC terms used both for KCM relations and as downstream evaluation labels.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2)
  - description: Tool used to compute sequence identity and cluster/filter proteins (e.g., 70% identity) and to define sequence similarity for debiased sampling.

- ColabFold · [GitHub](https://github.com/sokrypton/ColabFold) · [Website](https://colabfold.mmseqs.com/)
  - description: AlphaFold2-based pipeline used to predict structures (pLDDT) of generated sequences to assess foldability.

- AlphaFold · [GitHub](https://github.com/google-deepmind/alphafold) · [Codewiki](https://codewiki.google/github.com/google-deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Structure prediction method underpinning ColabFold; cited and used for evaluating generated proteins’ structures.

- ESM2 · [GitHub](https://github.com/facebookresearch/esm)
  - description: Protein LM used as a feature extractor to embed generated sequences for clustering/visualization in design evaluation.

- SCOPe · [Website](https://scop.berkeley.edu/)
  - description: Structural classification resource used to define structure-based design instructions (all-α, all-β, α/β).

- DiffDock · [GitHub](https://github.com/gcorso/DiffDock)
  - description: Docking method used to place ligands (e.g., heme) into designed proteins for function-based design evaluation.

- Smina · [GitHub](https://github.com/mwojcikowski/smina)
  - description: Docking/scoring tool used to estimate binding affinity of designed proteins to ligands.

- AutoDock Vina · [GitHub](https://github.com/ccsb-scripps/AutoDock-Vina)
  - description: Docking engine on which Smina is based; cited alongside Smina in binding affinity evaluation.

- HH-suite (HHblits) · [GitHub](https://github.com/soedinglab/hh-suite)
  - description: Used to search for homologs of generated sequences to assess novelty against sequence databases.

- Uniclust30 · [Website](https://uniclust.mmseqs.com/)
  - description: Database queried by HHblits to measure similarity/novelty of generated sequences.

- Protein Data Bank (wwPDB) · [Website](https://www.wwpdb.org/)
  - description: Source of proteins/labels for the Metal Ion Binding (MIB) held-out evaluation dataset.

- DeepLoc (Subcellular Localization) · [Website](https://services.healthtech.dtu.dk/service.php?DeepLoc-1.0)
  - description: Dataset used for subcellular localization prediction tasks (binary and 10-class) in sequence understanding evaluation.

- TAPE Benchmark · [GitHub](https://github.com/songlab-cal/tape)
  - description: Provides the contact map prediction task used to evaluate structural understanding of model representations.

- OPT · [GitHub](https://github.com/facebookresearch/metaseq)
  - description: Open pre-trained transformer used as the base architecture for InstructProtein and as a comparison baseline.

- LLaMA · [GitHub](https://github.com/facebookresearch/llama)
  - description: General LLM baseline used for comparison on sequence understanding tasks.

- Stanford Alpaca · [GitHub](https://github.com/tatsu-lab/stanford_alpaca) · [Codewiki](https://codewiki.google/github.com/tatsu-lab/stanford_alpaca)
  - description: Instruction-tuned LLaMA baseline included in comparisons.

- Galactica · [GitHub](https://github.com/paperswithcode/galai) · [Website](https://galactica.org/)
  - description: Scientific LLM baseline evaluated against InstructProtein on understanding tasks.

- Mol-Instructions · [GitHub](https://github.com/zjunlp/Mol-Instructions)
  - description: Biomolecular instruction dataset/model used as a baseline; paper notes its limited template diversity.

- OpenAI ChatGPT / GPT‑4 · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: General LLM used to translate KG triples (with KCM) into instruction-output pairs; also used as closed-source baselines.

- Claude‑2 (Anthropic) · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/claude)
  - description: Closed-source LLM baseline evaluated on sequence understanding tasks.

<!-- paper_id: f131b342e3aede46d24afc9b9055a94cceb0936a -->

## 100. Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents - NIPS - 2023 - citation_count 409 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2023_NIPS_409_Describe,_Explain,_Plan_and_Select_Interactive_Planning_with_Large_Language_Models_Enables_Open-World_Multi-Task_Agents.pdf
- Link: http://arxiv.org/pdf/2302.01560
- Tags: planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2023_NIPS_409_Describe,_Explain,_Plan_and_Select_Interactive_Planning_with_Large_Language_Models_Enables_Open-World_Multi-Task_Agents.pdf
- Token Usage: input 44362, output 5577, total 49939

### GitHub & Websites

- MC-Planner (DEPS) · [GitHub](https://github.com/CraftJarvis/MC-Planner)
  - description: Official implementation of the paper’s “Describe, Explain, Plan and Select” interactive planner, including prompts, selector, and Minecraft integrations used in all experiments.

- MineDojo · [GitHub](https://github.com/MineDojo/MineDojo) · [Website](https://minedojo.org)
  - description: Open-ended Minecraft research platform used as one of the evaluation environments (v1.11.2) and task providers; also supplies programmatic tasks referenced in the paper.

- MineRL · [GitHub](https://github.com/minerllabs/minerl) · [Website](https://minerl.io)
  - description: Minecraft RL platform and dataset used as another evaluation environment (v1.16.5) for testing DEPS with different controllers.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
  - description: Text-to-embodied benchmark used to evaluate DEPS beyond Minecraft; the paper runs ALFWorld tasks and compares planning baselines there.

- CLIPort · [GitHub](https://github.com/cliport/cliport) · [Website](https://cliport.github.io)
  - description: Robot manipulation policy used as the controller in the Tabletop Manipulation experiments; DEPS provides plans that CLIPort executes.

- Microsoft Malmo Platform · [GitHub](https://github.com/Microsoft/malmo) · [Website](https://www.microsoft.com/en-us/research/project/project-malmo/)
  - description: Minecraft experimentation platform underpinning many Minecraft research stacks cited by the paper; relevant for reproducing environment backends.

- TextWorld · [GitHub](https://github.com/microsoft/TextWorld)
  - description: Text-based environment underlying ALFWorld; used indirectly when the paper evaluates DEPS on ALFWorld tasks.

- OpenAI API (GPT-3/3.5/4, Codex) · [Doc](https://platform.openai.com/docs)
  - description: LLMs accessed via the OpenAI API (text-davinci-003, code-davinci-002, gpt-3.5-turbo, GPT‑4) are the core planner/explainer models used throughout the paper and baselines.

- CLIP · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Vision-language model used by prior work and as a comparator/option for the selector and scene description in DEPS ablations.

- MineCLIP · [GitHub](https://github.com/MineDojo/MineCLIP)
  - description: Video–language model released with MineDojo; used as a baseline selector in the paper’s ablations for goal selection.

- Sentence-Transformers (SBERT) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net)
  - description: Used by the paper’s goal parser to map free-form LLM plans to predefined controller skills via semantic similarity.

- Language Models as Zero-Shot Planners (baseline) · [GitHub](https://github.com/huangwl18/language-planner)
  - description: Baseline LLM planning approach reproduced in the paper for comparisons on Minecraft, ALFWorld, and Tabletop tasks.

- Code as Policies (baseline) · [Website](https://code-as-policies.github.io)
  - description: Baseline that programs robot skills with code-generating LLMs; the paper adapts its prompt style and compares against it on the same tasks.

<!-- paper_id: ccb1ccc4deacc4fb18000f0e1ce24329548963ae -->

## 101. Proteina: Scaling Flow-based Protein Structure Generative Models - ICLR - 2025 - citation_count 45 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICLR_45_Oral_Proteina_Scaling_Flow-based_Protein_Structure_Generative_Models.pdf
- Link: https://openreview.net/pdf/4b7a263bc7986c92ded70f173e7809c3153445a2.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICLR_45_Oral_Proteina_Scaling_Flow-based_Protein_Structure_Generative_Models.pdf
- Token Usage: input 63280, output 6624, total 69904

### GitHub & Websites

- Proteína · [GitHub](https://github.com/NVIDIA-Digital-Bio/proteina) · [Website](https://research.nvidia.com/labs/genair/proteina/)
  - description: Official codebase and project page for the paper’s flow-based protein backbone generator, including training, sampling, guidance, metrics, and instructions to reproduce results.

- AlphaFold Protein Structure Database (AFDB) · [Website](https://alphafold.ebi.ac.uk/) · [Doc](https://alphafold.ebi.ac.uk/download)
  - description: Primary structure source used to curate both DFS and the new D21M datasets (filtered AFDB subsets); also serves as the reference distribution for evaluation.

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Natural protein structures used for evaluation (novelty, FPSD/fJSD reference) and for LoRA fine-tuning on designable PDB chains.

- CATH Protein Structure Classification · [Website](https://www.cathdb.info/)
  - description: Structural fold hierarchy (C/A/T) used to label training data and enable hierarchical fold-class conditioning and guidance.

- SIFTS (Structure Integration with Function, Taxonomy and Sequences) · [Website](https://www.ebi.ac.uk/pdbe/docs/sifts/)
  - description: Mapping resource used to connect PDB chains to UniProt and CATH annotations when preparing labeled datasets.

- UniProt · [Website](https://www.uniprot.org/)
  - description: Protein identifiers leveraged via SIFTS to obtain domain and CATH mappings during data curation.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com/)
  - description: Used for structure-based clustering (DFS) and for evaluation (easy-cluster for diversity; easy-search for novelty/TM-score against PDB/AFDB).

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://www.mmseqs.com/)
  - description: Used for sequence-based clustering to balance datasets (e.g., D21M clustering and AFDB preprocessing).

- FoldComp · [GitHub](https://github.com/steineggerlab/foldcomp)
  - description: Compression/indexing tool used to store and stream very large AFDB subsets efficiently for training.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Inverse folding model used to design sequences for generated backbones as part of the designability pipeline.

- ESMFold (Meta ESM) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/)
  - description: Used to predict structures from ProteinMPNN-designed sequences to compute scRMSD and determine designability.

- Biotite (P-SEA) · [GitHub](https://github.com/biotite-dev/biotite) · [Website](https://www.biotite-python.org/)
  - description: Library used to compute secondary structure content (P-SEA) of generated proteins for the reported α/β/coil statistics.

- ESM3 (Meta) · [GitHub](https://github.com/facebookresearch/esm) · [Doc](https://huggingface.co/facebook/esm3)
  - description: Baseline large protein language model capable of structure prediction; evaluated under the paper’s unconditional benchmarks.

- RFDiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion)
  - description: Backbone diffusion baseline; authors ran its official implementation for comparative evaluation.

- Genie2 · [GitHub](https://github.com/aqlaboratory/genie2)
  - description: Backbone diffusion baseline trained on AFDB subsets; evaluated both at full temperature and with reduced noise according to the paper’s protocol.

- FrameDiff (SE(3) diffusion) · [GitHub](https://github.com/jasonkyuyim/se3_diffusion)
  - description: Backbone diffusion baseline; sampled with its public repository and defaults as part of the benchmark.

- Chroma · [GitHub](https://github.com/generative-biology/chroma)
  - description: Backbone diffusion baseline and the only compared method supporting fold-class–conditional generation; used for conditional comparisons and re-classification analysis.

- FoldFlow · [GitHub](https://github.com/joeybose/foldflow)
  - description: Flow-matching baseline (base/OT/stochastic variants); evaluated using the authors’ public repo with settings matching the paper.

<!-- paper_id: f271a65d845eeb0c824717c656e5fbc6e5f384be -->

## 102. A Comprehensive Survey of Scientific Large Language Models and Their Applications in Scientific Discovery - EMNLP - 2024 - citation_count 75 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_75_Main_A_Comprehensive_Survey_of_Scientific_Large_Language_Models_and_Their_Applications_in_Scientific_Discovery.pdf
- Tags: agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_75_Main_A_Comprehensive_Survey_of_Scientific_Large_Language_Models_and_Their_Applications_in_Scientific_Discovery.pdf
- Token Usage: input 55389, output 8631, total 64020

### GitHub & Websites

- Awesome-Scientific-Language-Models · [GitHub](https://github.com/yuzhimanhua/Awesome-Scientific-Language-Models)
  - description: Official repository released by this survey; it curates 260+ scientific LLMs, datasets, and benchmarks across fields and modalities that the paper analyzes, useful for reproducing and extending the survey.

- S2ORC (Semantic Scholar Open Research Corpus) · [Website](https://allenai.org/data/s2orc)
  - description: Large corpus of scientific papers cited as a core pre-training source for general science LLMs and downstream benchmarks in the survey.

- AMiner · [Website](https://www.aminer.cn/)
  - description: Bibliographic database used in the survey as a major source of scientific papers and metadata for pre-training and graph-aware modeling.

- Microsoft Academic Graph (MAG) · [Website](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/)
  - description: Academic graph referenced as a key metadata source for graph-augmented scientific LLMs.

- Semantic Scholar · [Website](https://www.semanticscholar.org/)
  - description: Paper corpus and citation graph repeatedly used across surveyed models for pre-training and evaluation.

- SciDocs · [GitHub](https://github.com/allenai/scidocs)
  - description: Benchmark suite for scientific document representations; used in the survey to evaluate link prediction, recommendation, and retrieval for graph-aware LLMs.

- SciRepEval · [GitHub](https://github.com/allenai/SciRepEval)
  - description: Multi-format benchmark for scientific document representations, cited as a key evaluation resource for graph-aware scientific LLMs.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Math word problem dataset highlighted as a dominant benchmark for training and evaluating math LLMs.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: High-difficulty mathematics problem set used extensively in the survey for evaluating mathematical reasoning.

- MathVista · [GitHub](https://github.com/lupantech/MathVista) · [Website](https://mathvista.github.io/)
  - description: Visual mathematics benchmark for geometry and multimodal reasoning; used to evaluate geometry-capable LLMs.

- Geometry3K · [Website](https://geometry3k.github.io/)
  - description: Geometry QA dataset with diagrams referenced as a primary pre-training/evaluation source for geometry LLMs.

- GeoQA · [GitHub](https://github.com/chenjiaqi/GeoQA)
  - description: Multimodal geometric QA dataset used for training and evaluating geometry reasoning models.

- WikiTableQuestions · [Website](https://nlp.stanford.edu/software/sempre/wikitable/)
  - description: Core table QA dataset used for pre-training/evaluating table-focused LLMs in the survey.

- WikiSQL · [GitHub](https://github.com/salesforce/WikiSQL)
  - description: Table semantic parsing/QA dataset widely used to pre-train or evaluate table LLMs.

- WDC Web Table Corpus · [Website](http://webdatacommons.org/webtables/)
  - description: Large corpus of web tables referenced as pre-training data for tabular LLMs.

- VizNet · [Website](https://viznet.mit.edu/)
  - description: Massive table corpus used in TABBIE and related table LLM pre-training cited in the survey.

- Materials Project · [Website](https://materialsproject.org/)
  - description: Materials science database used for materials and crystal modeling tasks (e.g., CrystalLLM) surveyed in the paper.

- PubChem · [Website](https://pubchem.ncbi.nlm.nih.gov/)
  - description: Core chemistry database for instruction tuning and text–molecule tasks (e.g., Mol-Instructions), repeatedly used by surveyed chemistry LLMs.

- ChEBI · [Website](https://www.ebi.ac.uk/chebi/)
  - description: Chemical Entities of Biological Interest ontology/dataset used for molecule graph and text–molecule benchmarks (e.g., ChEBI-20) in the survey.

- ZINC · [Website](https://zinc15.docking.org/)
  - description: Large molecule database commonly used for pre-training and evaluation of molecular language models.

- ChEMBL · [Website](https://www.ebi.ac.uk/chembl/)
  - description: Bioactivity database referenced for molecular representation learning, reaction prediction, and downstream evaluation.

- MoleculeNet · [Website](https://moleculenet.org/)
  - description: Benchmark suite for molecular property prediction repeatedly used to evaluate molecular LLMs.

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: 3D structural data source used by models like Uni-Mol and others surveyed for structure-aware molecular tasks.

- UniRef (UniProt Reference Clusters) · [Website](https://www.uniprot.org/help/uniref)
  - description: Protein sequence clusters used for pre-training protein LLMs (e.g., ESM, ProtTrans) surveyed in the paper.

- Swiss-Prot (UniProtKB/Swiss-Prot) · [Website](https://www.uniprot.org/)
  - description: Curated protein sequences used for pre-training/evaluation of protein LLMs and text–protein contrastive models (e.g., ProtST).

- GRCh38 (Human reference genome) · [Website](https://www.ncbi.nlm.nih.gov/grc/human)
  - description: Human genome reference cited as a core resource for DNA language models (e.g., DNABERT, DNABERT‑2).

- 1000 Genomes Project · [Website](https://www.internationalgenome.org/)
  - description: Human genetic variation data used by several genomic LLMs for pre-training and evaluation.

- RNAcentral · [Website](https://rnacentral.org/)
  - description: Aggregated non‑coding RNA sequences used for RNA-focused LLMs (e.g., RNABERT, RNA-FM).

- PubMed · [Website](https://pubmed.ncbi.nlm.nih.gov/)
  - description: Biomedical literature database extensively used as pre-training text for biomedical LLMs surveyed.

- PubMed Central (PMC) · [Website](https://www.ncbi.nlm.nih.gov/pmc/)
  - description: Full-text biomedical articles used as pre-training corpora for multiple biomedical models (e.g., BioBERT, PMC-LLaMA).

- UMLS (Unified Medical Language System) · [Website](https://www.nlm.nih.gov/research/umls/)
  - description: Biomedical knowledge base used for knowledge-enhanced pre-training and entity linking benchmarks in the survey.

- MIMIC-III · [Website](https://physionet.org/content/mimiciii/1.4/)
  - description: De-identified EHR database used to pre-train/evaluate clinical LLMs (e.g., ClinicalBERT, Clinical-Longformer).

- MIMIC-IV · [Website](https://physionet.org/content/mimiciv/2.2/)
  - description: Latest EHR dataset referenced for clinical LLM training and tasks such as diagnosis group prediction (DRG-LLaMA).

- MIMIC-CXR · [Website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
  - description: Large-scale chest X‑ray images with reports; foundational vision–language pre-training/evaluation dataset in the biomedical section.

- ROCO · [Website](https://www.inf-cv.uni-jena.de/en/roco)
  - description: Radiology Objects in COntext figure–caption dataset used for biomedical vision–language pre-training and evaluation.

- MedICaT · [GitHub](https://github.com/allenai/medicat)
  - description: Biomedical figures, captions, and references dataset; used for vision–language pre-training and VQA tasks in the survey.

- CheXpert · [Website](https://stanfordmlgroup.github.io/competitions/chexpert/)
  - description: Chest X‑ray benchmark used widely to evaluate classification and report generation models in the biomedical section.

- PadChest · [Website](https://bimcv.cipf.es/bimcv-projects/padchest/)
  - description: Annotated chest X‑ray dataset referenced as an evaluation set for biomedical vision–language models.

- SLAKE · [Website](https://www.med-vqa.com/slake/)
  - description: Semantically‑Labeled medical VQA dataset used to benchmark VQA capabilities of biomedical LLMs.

- OpenStreetMap · [Website](https://www.openstreetmap.org/)
  - description: Geospatial POI/graph data used to pre-train geoscience language models (e.g., SpaBERT, GeoLM) highlighted in the survey.

- ERA5 (ECMWF Reanalysis) · [Website](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
  - description: Climate time-series dataset foundational to weather forecasting models (e.g., FourCastNet, Pangu‑Weather) discussed in the survey.

- CMIP6 · [Website](https://pcmdi.llnl.gov/CMIP6/)
  - description: Coupled climate model intercomparison dataset used for climate pre-training/evaluation (e.g., ClimaX) covered by the survey.

- OceanBench · [Website](https://oceanbench.org/)
  - description: Ocean science benchmark cited for evaluating ocean-domain LLMs such as OceanGPT.

- BERT · [GitHub](https://github.com/google-research/bert) · [Codewiki](https://codewiki.google/github.com/google-research/bert)
  - description: Foundational masked language model architecture that underpins many encoder-style scientific LLMs surveyed (e.g., SciBERT, BioBERT).

- GPT-2 · [GitHub](https://github.com/openai/gpt-2)
  - description: Decoder-only next-token predictor referenced as the base paradigm for many generative scientific LLMs (e.g., BioGPT).

- CLIP · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Contrastive text–image pre-training framework repeatedly used/extended in the biomedical and geoscience vision–language models.

- LLaVA · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA)
  - description: Vision–language assistant paradigm referenced for adapting general LLMs to medical and math images (e.g., LLaVA‑Med, G‑LLaVA).

- TAPAS · [GitHub](https://github.com/google-research/tapas)
  - description: BERT-based table modeling framework cited as an early approach to tabular pre-training and QA in mathematics.

- SPECTER · [GitHub](https://github.com/allenai/specter)
  - description: Citation-informed Transformer producing paper embeddings; used as a representative contrastive/document-representation model in the survey.

- OAG-BERT · [GitHub](https://github.com/THUDM/OAG-BERT)
  - description: Graph-aware academic LLM pre-trained with text and metadata, highlighted as a Type 1.B example.

- SciBERT · [GitHub](https://github.com/allenai/scibert)
  - description: Scientific-domain BERT variant trained on Semantic Scholar; baseline encoder model for many tasks covered by the survey.

- BioBERT · [GitHub](https://github.com/dmis-lab/biobert)
  - description: Biomedical BERT used as a staple baseline/encoder across numerous biomedical tasks listed in the survey.

- ClinicalBERT · [GitHub](https://github.com/EmilyAlsentzer/clinicalBERT)
  - description: EHR-adapted BERT referenced for clinical tasks (NER, NLI, prediction) in the survey.

- BioGPT · [GitHub](https://github.com/microsoft/BioGPT)
  - description: GPT-2–style biomedical generative model featured among decoder-only biomedical LLMs.

- BiomedCLIP · [GitHub](https://github.com/microsoft/BiomedCLIP)
  - description: Large-scale biomedical image–text contrastive pre-training resource used for VQA, retrieval, and classification benchmarks in the survey.

- LLaVA‑Med · [GitHub](https://github.com/microsoft/LLaVA-Med)
  - description: Medical adaptation of LLaVA cited as a representative vision–language assistant in the biomedical section.

- GLoRIA · [GitHub](https://github.com/marshuang80/gloria)
  - description: Medical global‑local contrastive pre-training approach used for image–text alignment tasks referenced in the survey.

- CheXzero · [GitHub](https://github.com/rajpurkarlab/CheXzero)
  - description: Zero-shot chest X‑ray classification method via image–text pre-training, included among biomedical VLP baselines.

- DNABERT · [GitHub](https://github.com/jerryji1993/DNABERT)
  - description: BERT-style DNA language model highlighted as a Type 1.D example for genomic sequence modeling.

- ChemBERTa · [GitHub](https://github.com/seyonechithrananda/chemberta)
  - description: RoBERTa-style molecular language model on SMILES data included among encoder-only molecular LLMs.

<!-- paper_id: 3dd5ad34012164c4ec9c571a12cc6a7561683dea -->

## 103. Str2Str: A Score-based Framework for Zero-shot Protein Conformation Sampling - ICLR - 2024 - citation_count 42 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-optimization/2024_ICLR_42_Poster_Str2Str_A_Score-based_Framework_for_Zero-shot_Protein_Conformation_Sampling.pdf
- Link: https://openreview.net/pdf/8af65d24f25a4b2e78702b1efb981126bd3bf160.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-optimization/2024_ICLR_42_Poster_Str2Str_A_Score-based_Framework_for_Zero-shot_Protein_Conformation_Sampling.pdf
- Token Usage: input 35285, output 5867, total 41152

### GitHub & Websites

- Str2Str · [GitHub](https://github.com/lujiarui/Str2Str)
  - description: Official implementation of the paper’s score-based structure-to-structure framework for zero-shot protein conformation sampling; used for all experiments and released by the authors.

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org)
  - description: Source of all training structures (mmCIF) used to train STR2STR in an amortized manner without simulation data.

- ESMFold · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/) · [Doc](https://esmatlas.com/resources/esmfold)
  - description: Single-sequence structure predictor used to initialize the starting conformation for STR2STR sampling.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold)
  - description: Used for baselines (MSA subsampling protocol) and as an alternative initializer; authors ran AF2 pipelines with specific MSA settings.

- ColabFold · [GitHub](https://github.com/sokrypton/ColabFold) · [Website](https://colabfold.mmseqs.com)
  - description: Practical AF2 implementation with accelerated MSA search; used by the authors to run AF2 inference during experiments.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Fast MSA/search toolkit referenced in relation to AF2/ColabFold pipelines used in experiments.

- JackHMMER (HMMER) · [GitHub](https://github.com/EddyRivasLab/hmmer) · [Website](http://hmmer.org/)
  - description: MSA search tool used in the AF2 baseline pipeline to build alignments (UniRef90, MGnify, BFD).

- HHblits (HH-suite) · [GitHub](https://github.com/soedinglab/hh-suite) · [Website](https://wwwuser.gwdg.de/~compbiol/data/hhsuite/)
  - description: Profile HMM-based MSA tool used in the AF2 baseline pipeline for sequence alignment construction.

- UniRef90 · [Website](https://www.uniprot.org/help/uniref)
  - description: Protein sequence database used to generate MSAs for AF2 baseline inference.

- MGnify · [Website](https://www.ebi.ac.uk/metagenomics/)
  - description: Metagenomic sequence resource used as an MSA database in AF2 baseline runs.

- BFD (Big Fantastic Database) · [Website](https://bfd.mmseqs.com)
  - description: Large-scale protein sequence database used for AF2 MSA construction in baseline experiments.

- OmegaFold · [GitHub](https://github.com/HeliXonProtein/OmegaFold)
  - description: Sequence embedding/structure prediction model whose embeddings condition EigenFold; included as it is integral to the EigenFold baseline referenced.

- EigenFold
  - description: Diffusion-based sequence-to-ensemble baseline evaluated by the authors; they ran the released model to compare against STR2STR.

- idpGAN · [GitHub](https://github.com/feiglab/idpGAN)
  - description: GAN-based baseline for sequence-conditioned conformation ensembles; authors ran the official code for comparisons.

- FASPR (Fast and Accurate Side-Chain Packing) · [Website](https://zhanggroup.org/FASPR/)
  - description: Side-chain packing tool used by the paper to place side chains on sampled backbones during reconstruction.

- OpenMM · [GitHub](https://github.com/openmm/openmm) · [Website](https://openmm.org) · [Doc](https://openmm.org/documentation/)
  - description: Molecular simulation toolkit used for energy minimization and potential energy evaluation of sampled conformations.

- DSSP · [GitHub](https://github.com/cmbi/dssp) · [Website](https://swift.cmbi.umcn.nl/gv/dssp/)
  - description: Secondary structure assignment tool used to annotate training structures for coil-ratio filtering in ablation studies.

- deeptime · [GitHub](https://github.com/deeptime-ml/deeptime) · [Doc](https://deeptime-ml.github.io/deeptime/)
  - description: Library used to fit Time-lagged Independent Component Analysis (TICA) for fidelity metrics and visualization.

- TM-score · [Website](https://zhanggroup.org/TM-score/)
  - description: Official program used to compute TM-score (and paired RMSD-based diversity metrics) for ensemble evaluation.

<!-- paper_id: 122a1743b8481874cbab7238ccd6fd7bbea6ebc0 -->

## 104. DPLM-2: A Multimodal Diffusion Protein Language Model - ICLR - 2025 - citation_count 44 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2025_ICLR_44_Poster_DPLM-2_A_Multimodal_Diffusion_Protein_Language_Model.pdf
- Link: https://openreview.net/pdf/13c6bc9b904922e7352e690eae7cad8a2d4526f7.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2025_ICLR_44_Poster_DPLM-2_A_Multimodal_Diffusion_Protein_Language_Model.pdf
- Token Usage: input 38204, output 5339, total 43543

### GitHub & Websites

- DPLM-2 · [Website](https://bytedance.github.io/dplm/dplm-2)
  - description: Official project page for the paper’s multimodal diffusion protein language model, referenced as the central resource for the work; hosts project information, demos/checkpoints and links for reproducing results.

- DPLM (sequence-based pretraining) · [Website](https://bytedance.github.io/dplm/)
  - description: The sequence-only Diffusion Protein Language Model used to warm-start DPLM-2; provides pre-trained LMs that the paper fine-tunes for multimodal learning.

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org) · [Doc](https://www.rcsb.org/pages/help)
  - description: Experimental protein structures used to train and evaluate DPLM-2 (20K clustered PDB structures).

- AlphaFold Protein Structure Database (AFDB) · [Website](https://alphafold.ebi.ac.uk) · [Doc](https://alphafold.ebi.ac.uk/download)
  - description: High-quality synthetic structures (AFDB SwissProt split; and AFDB representatives in ablations) used to train and augment DPLM-2’s structure data.

- UniRef50 · [Website](https://www.uniprot.org/uniref/UniRef50)
  - description: Large-scale evolutionary sequence database (45M) behind the pre-trained DPLM models used for warm-up in DPLM-2.

- CAMEO · [Website](https://www.cameo3d.org)
  - description: Benchmark used for folding and inverse folding evaluation (CAMEO 2022).

- CATH 4.2 · [Website](http://www.cathdb.info)
  - description: Dataset used for additional inverse-folding benchmarking in the appendix.

- ESMFold (ESM) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com)
  - description: Structure predictor used extensively for evaluation (pLDDT, refolding for scTM/scRMSD) and baselines; DPLM-2 compares against ESMFold and uses its outputs.

- ESM3-Open · [Website](https://www.evolutionaryscale.ai/esm3)
  - description: Multimodal generative baseline compared throughout (sequence→structure generation), evaluated against DPLM-2 on co-generation and conditional tasks.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Inverse-folding model used as a comparison point and for some evaluation pipelines (e.g., sequence prediction from generated structures).

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion)
  - description: Structure-based generative baseline used for benchmarking unconditional structure generation and motif-scaffolding.

- EvoDiff · [GitHub](https://github.com/salesforce/evodiff)
  - description: Sequence-generation baseline used for unconditional sequence generation comparisons and motif-scaffolding sequence-based evaluation.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com)
  - description: Fast structure search/clustering used to quantify diversity (#clusters) and to cluster structural motifs in the tokenizer analysis.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Sequence clustering/search toolkit used in extended evaluations for sequence diversity and novelty analyses.

- ProGen2 · [GitHub](https://github.com/salesforce/progen)
  - description: Used to compute perplexity as a measure of sequence naturalness in extended unconditional sequence-generation analysis.

- AlphaFold2 · [GitHub](https://github.com/google-deepmind/alphafold) · [Codewiki](https://codewiki.google/github.com/google-deepmind/alphafold)
  - description: Source of the Invariant Point Attention (IPA)/Evoformer-style modules and FAPE loss used in DPLM-2’s structure de-tokenizer and tokenizer training.

- GVP (Geometric Vector Perceptrons) · [GitHub](https://github.com/drorlab/gvp-pytorch)
  - description: GVP-based encoder components used to extract invariant backbone geometric features for the structure tokenizer.

- Lookup-Free Quantizer (LFQ, MAGVIT-v2) · [GitHub](https://github.com/google-research/magvit-v2)
  - description: Discrete tokenizer used by DPLM-2 to convert encoded structural features into structure tokens; the paper reports LFQ substantially outperforms VQ-VAE for structure tokenization.

- SaProt · [GitHub](https://github.com/DeepGraphLearning/SaProt)
  - description: Provides datasets and benchmarks for downstream predictive tasks where DPLM-2 representations are evaluated against sequence-only and structure-aware baselines.

<!-- paper_id: 0456bec726800502f72bf63fde1a637d1807a62b -->

## 105. LLM Agents Making Agent Tools - ACL - 2025 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_16_Long_LLM_Agents_Making_Agent_Tools.pdf
- Link: https://aclanthology.org/2025.acl-long.1266/
- Tags: multiagent, tool, science, agent, biology, protein-function, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_16_Long_LLM_Agents_Making_Agent_Tools.pdf
- Token Usage: input 43596, output 2463, total 46059

### GitHub & Websites

- TOOLMAKER (includes TM-BENCH) · [GitHub](https://github.com/KatherLab/ToolMaker)
  - description: Official repository released with the paper; contains the TOOLMAKER agent framework, the TM-BENCH benchmark tasks, unit tests, and scripts to reproduce all experiments.

- OpenHands · [GitHub](https://github.com/All-Hands-AI/OpenHands)
  - description: Software-engineering agent used as the main comparison baseline; adapted by the authors to generate install scripts and tool functions for TM-BENCH tasks.

- CONCH · [GitHub](https://github.com/mahmoodlab/CONCH)
  - description: Pathology foundation model repository used as a target codebase for the conch_extract_features task in TM-BENCH.

- MUSK · [GitHub](https://github.com/lilab-stanford/MUSK)
  - description: Vision-language oncology foundation model; the vision component is wrapped by TOOLMAKER in the musk_extract_features task.

- PathFinderCRC · [GitHub](https://github.com/LiangJunhao-THU/PathFinderCRC)
  - description: Repository used to verify biomarkers from WSI probability maps; serves as the target project for the pathfinder_verify_biomarker task.

- STAMP · [GitHub](https://github.com/KatherLab/STAMP)
  - description: End-to-end weakly supervised computational pathology pipeline; TOOLMAKER builds tools for feature extraction and model training (stamp_extract_features, stamp_train_classification_model).

- UNI · [GitHub](https://github.com/mahmoodlab/UNI)
  - description: Pathology foundation model repository used as the target codebase for the uni_extract_features task.

- MedSAM · [GitHub](https://github.com/bowang-lab/MedSAM)
  - description: Medical segmentation adaptation of SAM; used as the repository for the medsam_inference segmentation task.

- nnU-Net · [GitHub](https://github.com/MIC-DKFZ/nnUNet)
  - description: Self-configuring biomedical segmentation framework; used in the nnunet_train_model training task.

- Cytopus · [GitHub](https://github.com/wallet-maker/cytopus)
  - description: Knowledge base for cell-type gene programs; used to generate Spectra-ready JSON in the cytopus_db task.

- ESM (Evolutionary Scale Modeling) · [GitHub](https://github.com/facebookresearch/esm)
  - description: Protein language models and tools; the esm_fold_predict task uses the esm2_t33_650M_UR50D model for sequence representations and contact maps.

- RETFound_MAE · [GitHub](https://github.com/rmaphoh/RETFound_MAE)
  - description: Retinal imaging foundation model; target repository for extracting feature vectors in the retfound_feature_vector task.

- MedSSS · [GitHub](https://github.com/pixas/MedSSS)
  - description: Medical small language model with slow-thinking policy; used to generate responses in the medsss_generate task.

- ModernBERT · [GitHub](https://github.com/AnswerDotAI/ModernBERT)
  - description: Modern bidirectional encoder architecture; used for masked sentence prediction in the modernbert_predict_masked task.

- FlowMap · [GitHub](https://github.com/dcharatan/flowmap)
  - description: Gradient-descent-based camera pose/intrinsic/depth estimation; repository used for the flowmap_overfit_scene task to overfit camera extrinsics.

- TabPFN · [GitHub](https://github.com/PriorLabs/TabPFN)
  - description: Tabular foundation model and training code; used to train and evaluate predictors in the tabpfn_predict task.

- OpenAI Developer Platform · [Doc](https://platform.openai.com/docs)
  - description: API documentation referenced for function calling and structured outputs; the paper uses OpenAI models (gpt-4o, o1-mini, o3-mini) to run TOOLMAKER and ablations.

<!-- paper_id: 100a2ded17ff3136acc2fbb893f7714c4b837e95 -->

## 106. Pre-training Sequence, Structure, and Surface Features for Comprehensive Protein Representation Learning - ICLR - 2024 - citation_count 17 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICLR_17_Poster_Pre-training_Sequence,_Structure,_and_Surface_Features_for_Comprehensive_Protein_Representation_Learning.pdf
- Link: https://openreview.net/pdf/d85441e5f968c9ae883e09ec5a06e9f9eda773c2.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICLR_17_Poster_Pre-training_Sequence,_Structure,_and_Surface_Features_for_Comprehensive_Protein_Representation_Learning.pdf
- Token Usage: input 16703, output 4265, total 20968

### GitHub & Websites

- TorchDrug · [GitHub](https://github.com/DeepGraphLearning/torchdrug) · [Website](https://torchdrug.ai) · [Doc](https://torchdrug.ai/docs)
  - description: Framework used to train and evaluate downstream tasks and includes official implementations of GearNet; the paper fine-tunes models and reports metrics within this framework.

- GearNet (Geometric Structure Pretraining) · [GitHub](https://github.com/DeepGraphLearning/torchdrug) · [Doc](https://torchdrug.ai/docs/tutorials/protein_function.html)
  - description: Protein structure encoder architecture (including GearNet-Edge-IEConv) used for structure pre-training and downstream evaluation; implemented and run via TorchDrug.

- ESM-1b (Evolutionary Scale Modeling) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com) · [Doc](https://esmatlas.com/docs)
  - description: Pre-trained protein language model used as the sequence encoder in the series-fusion setup for pre-training and downstream tasks.

- AlphaFold Protein Structure Database (v2) · [Website](https://alphafold.ebi.ac.uk) · [Doc](https://alphafold.ebi.ac.uk/download)
  - description: Source of predicted protein structures used for structure pre-training across 20 species and Swiss-Prot entries.

- UniProtKB/Swiss-Prot · [Website](https://www.uniprot.org)
  - description: Curated protein sequence database; Swiss-Prot subset is included in the AlphaFold DB data used for structure pre-training.

- MSMS (Molecular Surface) · [Website](https://ccsb.scripps.edu/msms)
  - description: Triangulation software used to generate molecular surface meshes; the paper computes SDF training targets from MSMS meshes.

- DeepSDF · [GitHub](https://github.com/facebookresearch/DeepSDF)
  - description: Reference framework for signed distance function sampling and training procedure; the paper follows DeepSDF’s SDF sampling and clamping strategy for ProteinINR.

- KPConv (Kernel Point Convolution) · [GitHub](https://github.com/HuguesTHOMAS/KPConv)
  - description: Point cloud convolution implementation used to downsample and process protein surface points within the ProteinINR point encoder pipeline.

- dMaSIF · [GitHub](https://github.com/LPDI-EPFL/dMaSIF)
  - description: Protein surface learning approach referenced for building end-to-end “chemical color” features from atom categories and distances; informs the surface feature design used in ProteinINR.

- MaSIF · [GitHub](https://github.com/LPDI-EPFL/masif)
  - description: Foundational molecular surface fingerprinting method; cited as prior work on mesh-based surface features that motivates the paper’s INR-based surface modeling.

<!-- paper_id: 4dd3e332c79bb06cad6efec8951787f9d727c6e1 -->

## 107. ProtT3: Protein-to-Text Generation for Text-based Protein Understanding - ACL - 2024 - citation_count 27 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ACL_27_Long_ProtT3_Protein-to-Text_Generation_for_Text-based_Protein_Understanding.pdf
- Link: https://aclanthology.org/2024.acl-long.324/
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ACL_27_Long_ProtT3_Protein-to-Text_Generation_for_Text-based_Protein_Understanding.pdf
- Token Usage: input 23582, output 4231, total 27813

### GitHub & Websites

- ProtT3 · [GitHub](https://github.com/acharkq/ProtT3)
  - description: Official code release for the paper, including training/evaluation scripts, dataset processing, and pretrained checkpoints for the proposed protein-to-text generation framework.

- ESM-2 · [GitHub](https://github.com/facebookresearch/esm) · [Doc](https://esm.tools/)
  - description: Protein language model used as the frozen protein encoder in ProtT3; the paper uses ESM-2 (150M) and its smaller variants in ablations.

- Galactica · [Website](https://galactica.org/) · [Doc](https://huggingface.co/facebook/galactica-1.3b)
  - description: Scientific language model serving as ProtT3’s base LM (1.3B) for text generation; fine-tuned via LoRA and also used as a baseline.

- BLIP-2 / Q-Former (LAVIS) · [GitHub](https://github.com/salesforce/LAVIS) · [Doc](https://huggingface.co/docs/transformers/model_doc/blip-2)
  - description: Source of the Q-Former cross-modal projector architecture and training objectives (contrast/match/caption) that ProtT3 adapts to bridge PLM and LM.

- PubMedBERT-Abstract · [Doc](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
  - description: Biomedical BERT used to initialize the Q-Former weights in ProtT3’s cross-modal projector.

- LoRA · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient fine-tuning method applied to the base language model in ProtT3’s stage-2 training.

- OpenDelta · [GitHub](https://github.com/thunlp/OpenDelta)
  - description: Library the authors used to implement LoRA adapters for fine-tuning the base LM.

- FlashAttention-2 · [GitHub](https://github.com/Dao-AILab/flash-attention) · [Codewiki](https://codewiki.google/github.com/Dao-AILab/flash-attention)
  - description: Kernel used by the authors to speed up ESM-2 and Galactica during training/inference.

- Swiss-Prot (UniProtKB/Swiss-Prot) · [Website](https://www.uniprot.org/) · [Doc](https://www.uniprot.org/help/uniprotkb)
  - description: Curated protein sequence database used to build the protein–text captioning and retrieval benchmark splits in the paper.

- ProteinKG25 (from OntoProtein) · [GitHub](https://github.com/zjunlp/OntoProtein) · [Website](https://zjunlp.github.io/project/OntoProtein/)
  - description: Knowledge-graph–derived protein dataset the authors convert into free-text descriptions for captioning and retrieval; used jointly with Swiss-Prot in stage-1 training.

- Gene Ontology · [Website](https://geneontology.org/)
  - description: Source ontology underlying ProteinKG25; cited as the origin of triples transformed into text for the ProteinKG25 benchmark.

- PDB-QA · [Website](https://www.rcsb.org) · [Doc](https://data.rcsb.org/)
  - description: Protein single-turn QA dataset (from ProteinChat) evaluated in the paper; authors retrieved 1D sequences via the RCSB PDB web API for this benchmark.

- ProtST · [GitHub](https://github.com/DeepGraphLearning/ProtST)
  - description: Baseline protein–text retrieval method re-run by the authors with released code and original hyperparameters for comparison.

- ProteinCLAP (components used in reimplementation) — SciBERT · [GitHub](https://github.com/allenai/scibert) · [Website](https://allenai.org/scibert)
  - description: Text encoder used by the authors when re-implementing ProteinCLAP (original code unavailable) for retrieval baselines.

- ProteinCLAP (components used in reimplementation) — ProtTrans / ProtBERT · [GitHub](https://github.com/agemagician/ProtTrans) · [Doc](https://huggingface.co/Rostlab/prot_bert)
  - description: Protein encoder (ProtBERT) used by the authors in their ProteinCLAP reimplementation baseline.

- Phi-1.5 · [Doc](https://huggingface.co/microsoft/phi-1_5)
  - description: Alternative base language model used in ablations for stage-2 protein captioning to study the effect of the textual LM.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold)
  - description: Used by the authors only to render protein structure figures in qualitative examples (not required for training/evaluation of ProtT3).

<!-- paper_id: 2d4f4a2b830de825c2afe3cf66f3e4d4da504657 -->

## 108. Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System - ACL - 2025 - citation_count 17 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ACL_17_Long_Many_Heads_Are_Better_Than_One_Improved_Scientific_Idea_Generation_by_A_LLM-Based_Multi-Agent_System.pdf
- Link: https://aclanthology.org/2025.acl-long.1368/
- Tags: multiagent, tool, science, agent, biology, protein-function, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ACL_17_Long_Many_Heads_Are_Better_Than_One_Improved_Scientific_Idea_Generation_by_A_LLM-Based_Multi-Agent_System.pdf
- Token Usage: input 42778, output 4419, total 47197

### GitHub & Websites

- Virtual Scientists (VIRSCI) · [GitHub](https://github.com/open-sciencelab/Virtual-Scientists)
  - description: Official code release for this paper; implements the LLM-based multi-agent system (team formation, inter/intra-team discussion, novelty assessment, abstract generation) and evaluation pipeline described in the work.

- AgentScope · [GitHub](https://github.com/modelscope/agentscope) · [Website](https://modelscope.cn/agentscope)
  - description: Multi-agent framework used to implement VIRSCI; the paper specifically uses AgentScope’s KnowledgeBank module to store and retrieve scientist profiles during collaboration.

- FAISS · [GitHub](https://github.com/facebookresearch/faiss) · [Codewiki](https://codewiki.google/github.com/facebookresearch/faiss) · [Website](https://faiss.ai)
  - description: Vector similarity search library used to build the past and contemporary paper databases for retrieval and novelty evaluation.

- Ollama · [GitHub](https://github.com/ollama/ollama) · [Codewiki](https://codewiki.google/github.com/ollama/ollama) · [Website](https://ollama.com)
  - description: Local model runner used to host open-weight LLaMA 3.1 models in the experiments.

- Llama 3 / 3.1 (Meta) · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: Open-weight LLMs (8B and 70B) used as agent backbones in several experimental settings.

- OpenAI GPT‑4o · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed-weight LLM accessed via API; used as an agent model and as the LLM-based reviewer for baseline comparison and scoring.

- mxbai-embed-large-v1 (Mixedbread AI) · [Website](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
  - description: Embedding model used to embed papers and author profiles for retrieval and metric computation.

- AMiner Computer Science Dataset · [Website](https://www.aminer.cn/aminernetwork)
  - description: Source dataset for building the Computer Science research ecosystem (authors, papers, co-authorship graph) used to initialize agents and construct past/contemporary paper databases.

- Open Academic Graph 3.1 · [Website](https://open.aminer.cn/open/article?id=65bf053091c938e5025a31e2)
  - description: Cross-domain dataset used to build the broader research ecosystem and to run main experiments beyond computer science.

- Open Academic Graph 3.2 · [Website](https://open.aminer.cn/open/article?id=67aaf63af4cbd12984b6a5f0)
  - description: Newer OAG snapshot used in Appendix A for temporal isolation experiments to test effects of potential data leakage.

- Semantic Scholar API · [Website](https://www.semanticscholar.org/product/api) · [Doc](https://api.semanticscholar.org/)
  - description: Paper retrieval API used in comparison settings to align baselines and to obtain abstracts/citations for evaluation when replacing other sources.

- PubMed · [Website](https://pubmed.ncbi.nlm.nih.gov/)
  - description: Literature database/API referenced in baseline settings (HypoGen) and discussed when harmonizing retrieval sources for fair comparison.

- NeurIPS 2024 Reviewer Guidelines · [Website](https://neurips.cc/Conferences/2024/ReviewerGuidelines)
  - description: Review rubric the paper adopts for LLM-based scoring to fairly compare against AI Scientist.

- AI Scientist · [GitHub](https://github.com/SakanaAI/AI-Scientist) · [Website](https://sakana.ai/ai-scientist/)
  - description: Baseline single-agent system compared against VIRSCI; the paper aligns settings and evaluates against its LLM-review metric.

<!-- paper_id: 2e2b5d3589f31cdc5ca0bcd918b22794eb4fe5e4 -->

## 109. MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration - ACL - 2025 - citation_count 19 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/multiagent/2025_ACL_19_Findings_MAM_Modular_Multi-Agent_Framework_for_Multi-Modal_Medical_Diagnosis_via_Role-Specialized_Collaboration.pdf
- Link: https://aclanthology.org/2025.findings-acl.1298/
- Tags: multiagent, tool, science, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/multiagent/2025_ACL_19_Findings_MAM_Modular_Multi-Agent_Framework_for_Multi-Modal_Medical_Diagnosis_via_Role-Specialized_Collaboration.pdf
- Token Usage: input 19565, output 5693, total 25258

### GitHub & Websites

- MAM (Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis) · [GitHub](https://github.com/yczhou001/MAM)
  - description: Official code release of the paper; implements the role-specialized multi-agent pipeline (General Practitioner, Specialist Team, Radiologist, Medical Assistant, Director), prompts, retrieval, and evaluation setup.

- MedQA (USMLE-style medical QA dataset) · [GitHub](https://github.com/jind11/MedQA)
  - description: Text QA benchmark used for evaluation of MAM and baselines.

- PubMedQA · [GitHub](https://github.com/pubmedqa/pubmedqa)
  - description: Biomedical research QA dataset used for text-based evaluation.

- PathVQA · [GitHub](https://github.com/UCSD-AI4H/PathVQA)
  - description: Pathology visual question answering dataset used for image-based evaluation.

- PMC-VQA · [Doc](https://arxiv.org/abs/2305.10415)
  - description: Medical visual question answering dataset constructed from PubMed Central articles; used for image-based VQA evaluation.

- Brain Tumor Classification (MRI) · [Website](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
  - description: MRI image dataset for brain tumor classification; used as an image-based diagnostic benchmark.

- DeepLesion · [Website](https://nihcc.app.box.com/v/DeepLesion) · [Doc](https://nihcc.github.io/DeepLesion/)
  - description: NIH Clinical Center large-scale radiology lesion dataset used for image-based evaluation.

- NIH ChestX-ray14 (ChestX-ray8) · [Website](https://nihcc.app.box.com/v/ChestXray-NIHCC) · [Doc](https://nihcc.github.io/)
  - description: Hospital-scale chest X-ray dataset used for image-based classification evaluation.

- Heart Sounds CHSC2011 (PASCAL Classifying Heart Sounds Challenge) · [Website](http://www.peterjbentley.com/heartchallenge/)
  - description: Clinical heart sound recordings used for audio-based evaluation (Heartbeat dataset).

- Sound-Dr (Respiratory sound dataset) · [Doc](https://arxiv.org/abs/2201.04581)
  - description: Respiratory audio dataset used for audio-based evaluation.

- MedVidQA · [Doc](https://arxiv.org/abs/2201.12888)
  - description: Medical instructional video QA dataset; used for video-based evaluation (converted to yes/no questions as described).

- LLaVA · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io/)
  - description: Vision–language model baseline compared against MAM on image and video tasks.

- LLaVA-Med · [GitHub](https://github.com/microsoft/LLaVA-Med)
  - description: Biomedical/clinical adaptation of LLaVA used as a medical LVLM baseline in image-based comparisons.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL)
  - description: Vision–language model baseline used for image and video comparisons.

- HuatuoGPT-Vision · [GitHub](https://github.com/FreedomIntelligence/HuatuoGPT-Vision)
  - description: Medical LVLM baseline used for image-based evaluation and as one of the specialized models leveraged in the study.

- Qwen-Audio · [GitHub](https://github.com/QwenLM/Qwen-Audio)
  - description: Audio-language model used as a baseline and leveraged for audio tasks within the study.

- VideoLLaMA2 · [GitHub](https://github.com/DAMO-NLP-SG/VideoLLaMA2)
  - description: Video-LLM baseline compared to MAM on MedVidQA and used as a specialized video model.

- LLaVA-NeXT (Video) · [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) · [Website](https://llava-vl.github.io/blog/2024-llava-next-video/)
  - description: Video-capable LLaVA baseline used for video-based evaluation.

- Medichat-Llama3-8B · [Website](https://huggingface.co/sethuiyer/Medichat-Llama3-8B)
  - description: Medical LLaMA-3 model used for text tasks in the empirical study/baselines.

- Google Custom Search API · [Doc](https://developers.google.com/custom-search/v1/overview)
  - description: Web retrieval service used by the Medical Assistant agent to fetch external medical knowledge when no hospital database is available.

<!-- paper_id: 999a2dbcf42a6510d429d4568908370a321717f4 -->

## 110. Retrieved Sequence Augmentation for Protein Representation Learning - EMNLP - 2024 - citation_count 13 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_EMNLP_13_Main_Retrieved_Sequence_Augmentation_for_Protein_Representation_Learning.pdf
- Link: https://aclanthology.org/2024.emnlp-main.104/
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_EMNLP_13_Main_Retrieved_Sequence_Augmentation_for_Protein_Representation_Learning.pdf
- Token Usage: input 33311, output 4933, total 38244

### GitHub & Websites

- Retrieved Sequence Augmentation (RSA) official release
  - description: The paper states “Code and data are available at this repo,” and the appendix references scripts under RSA-code; this is the official implementation, datasets, and retrieval index for reproducing RSA.

- ESM (ESM-1b/ESM-2, MSA Transformer, ESMFold) · [GitHub](https://github.com/facebookresearch/esm)
  - description: Used as the pretrained encoder for dense retrieval embeddings (ESM‑1b), as a baseline (ESM‑1b/ESM‑2, MSA Transformer), and as a folding model augmented by RSA (ESMFold).

- FAISS · [GitHub](https://github.com/facebookresearch/faiss) · [Codewiki](https://codewiki.google/github.com/facebookresearch/faiss) · [Website](https://faiss.ai)
  - description: Library used to build the approximate nearest-neighbor index for fast dense retrieval of protein sequences.

- HH-suite3 (HHblits, hhalign) · [GitHub](https://github.com/soedinglab/hh-suite)
  - description: Toolkit used to build MSAs and compute E-values; provides HHblits and hhalign commands used for MSA baselines and analyses.

- HMMER/JackHMMER · [GitHub](https://github.com/EddyRivasLab/hmmer) · [Website](http://hmmer.org/)
  - description: JackHMMER is used in the paper’s Accelerated MSA pipeline to align top retrieved sequences.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Structure prediction model augmented with RSA (AlphaFold‑RSA) and compared against standard AlphaFold variants.

- ColabFold · [GitHub](https://github.com/sokrypton/ColabFold) · [Website](https://colabfold.mmseqs.com/)
  - description: Used to generate MSAs in some comparisons/analyses (e.g., for AlphaFold experiments and MSA coverage analysis).

- ProtTrans/ProtBERT · [GitHub](https://github.com/agemagician/ProtTrans)
  - description: ProtBERT serves as a pretrained backbone combined with RSA for downstream tasks.

- TAPE Benchmark · [GitHub](https://github.com/songlab-cal/tape)
  - description: The paper follows TAPE splits for four tasks (secondary structure, contact prediction, remote homology, stability).

- PEER Benchmark · [GitHub](https://github.com/DeepGraphLearning/PEER_Benchmark)
  - description: Provides datasets/splits for subcellular localization and protein–protein interaction tasks used in evaluation.

- Pfam · [Website](https://pfam.xfam.org/)
  - description: Source of the 44M-sequence database (Pfam-A) used to build the dense retrieval index and also for MSA baselines.

- UniProtKB · [Website](https://www.uniprot.org/)
  - description: Protein knowledgebase referenced for coverage context; many retrieved proteins are mapped to UniProt IDs in demos.

- SCOPe · [Website](https://scop.berkeley.edu/)
  - description: Used to evaluate retrieval accuracy for structural homology (Fold/Superfamily/Family levels).

- ProteinNet · [GitHub](https://github.com/aqlaboratory/proteinnet) · [Website](https://www.proteinnet.org/)
  - description: Dataset used for contact prediction and structure-related evaluations; RSA structural similarity analyses also reference ProteinNet.

- NetSurfP‑2.0 · [Website](https://services.healthtech.dtu.dk/services/NetSurfP-2.0/)
  - description: Dataset/benchmark for secondary structure prediction used in evaluation and speed comparisons.

- DeepLoc · [Website](https://services.healthtech.dtu.dk/services/DeepLoc-1.0/)
  - description: Dataset for subcellular localization prediction used among the downstream tasks.

- CASP14 (Critical Assessment of protein Structure Prediction) · [Website](https://predictioncenter.org/casp14/)
  - description: Benchmark targets used for structure prediction experiments, including CASP14-FM for de novo-like evaluation.

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Source of de novo proteins used to assess RSA on proteins with few or no homologs.

- InterPro · [Website](https://www.ebi.ac.uk/interpro/)
  - description: Database queried by the RSA-augmented LLM agent (ProteinChat) for functional annotations in tool-based demos.

- PubMed · [Website](https://pubmed.ncbi.nlm.nih.gov/)
  - description: Literature database used by the RSA-enabled LLM agent to fetch scientific context in protein understanding demos.

<!-- paper_id: 77e5d0a68afcffb27191572590deced60feb9d5d -->

## 111. Robust Optimization in Protein Fitness Landscapes Using Reinforcement Learning in Latent Space - ICML - 2024 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-optimization/2024_ICML_14_Spotlight_Robust_Optimization_in_Protein_Fitness_Landscapes_Using_Reinforcement_Learning_in_Latent_Space.pdf
- Link: https://openreview.net/pdf/a6ef2163b0c8e9627e5c526fc4d73fb2e9eec207.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-optimization/2024_ICML_14_Spotlight_Robust_Optimization_in_Protein_Fitness_Landscapes_Using_Reinforcement_Learning_in_Latent_Space.pdf
- Token Usage: input 20871, output 5735, total 26606

### GitHub & Websites

- LatProtRL · [GitHub](https://github.com/haewonc/LatProtRL)
  - description: Official code release for the paper; implements the Variant Encoder-Decoder (VED), latent-space PPO policy, frontier buffer, constrained decoding, and experiment scripts for GFP and AAV optimization.

- ESM-2 (Evolutionary Scale Modeling) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com)
  - description: Pretrained protein language model used as the encoder/decoder backbone to obtain CLS embeddings and fine-tuned initial transformer layers for sequence reconstruction in the VED.

- Stable Baselines · [GitHub](https://github.com/hill-a/stable-baselines) · [Doc](https://stable-baselines.readthedocs.io/)
  - description: RL library providing the PPO implementation used to train the policy in LatProtRL.

- pycma (CMA-ES) · [GitHub](https://github.com/CMA-ES/pycma) · [Doc](https://cma-es.github.io/)
  - description: Covariance Matrix Adaptation Evolution Strategy library used to run the CMAES baseline (and CMAES with VED encoding) for comparison.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Used in the paper’s in vitro validation to predict structures and pLDDT for selected GFP variants prior to wet-lab testing.

- GFP local fitness landscape (Sarkisyan et al., 2016) · [Website](https://www.nature.com/articles/nature17995)
  - description: Deep mutational scanning dataset of GFP (54,025 sequences) used as D* and to evaluate optimization; the paper trains VED and uses oracles/predictors on this dataset.

- AAV capsid diversification dataset (Bryant et al., 2021) · [Website](https://www.nature.com/articles/s41587-021-00881-8)
  - description: AAV functional segment dataset (44,156 sequences) used as D* for AAV tasks; the paper evaluates optimization on this dataset and uses associated oracles/predictors.

<!-- paper_id: a9589c7a5062134012d56bf3ea6aedd1d76bfb1d -->

## 112. ESM All-Atom: Multi-scale Protein Language Model for Unified Molecular Modeling - ICML - 2024 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICML_14_Poster_ESM_All-Atom_Multi-scale_Protein_Language_Model_for_Unified_Molecular_Modeling.pdf
- Link: https://openreview.net/pdf/a701e753389a226ace8e293f2f5d53cabc38f484.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICML_14_Poster_ESM_All-Atom_Multi-scale_Protein_Language_Model_for_Unified_Molecular_Modeling.pdf
- Token Usage: input 30089, output 5268, total 35357

### GitHub & Websites

- ESM All-Atom (ESM-AA) · [GitHub](https://github.com/zhengkangjie/ESM-AA)
  - description: Official code release for the paper’s multi-scale protein language model; used to reproduce pretraining on mixed protein/molecule data and all downstream evaluations.

- ESM (ESM-2) · [GitHub](https://github.com/facebookresearch/esm)
  - description: Protein language model family used as initialization (ESM-2 35M checkpoint) for ESM-AA and as a baseline protein encoder in comparisons.

- Uni-Mol · [GitHub](https://github.com/dptech-corp/Uni-Mol) · [Doc](https://openreview.net/forum?id=6K2RM6wVqKu)
  - description: 3D molecular representation learning framework; its released dataset (19M molecules, 209M conformations) is used for ESM-AA pretraining and it serves as a molecule-encoder baseline.

- AlphaFold Protein Structure Database (AlphaFold DB) · [Website](https://alphafold.ebi.ac.uk/) · [Doc](https://alphafold.ebi.ac.uk/download)
  - description: Source of 8M high-confidence predicted protein structures used to build code-switch sequences and pairwise-distance supervision during ESM-AA pretraining.

- RDKit · [GitHub](https://github.com/rdkit/rdkit) · [Website](https://www.rdkit.org/)
  - description: Cheminformatics toolkit used to generate 3D conformations from SMILES for fine-tuning; also provides ETKDG and MMFF94 used in the molecule dataset’s conformation generation.

- TAPE · [GitHub](https://github.com/songlab-cal/tape)
  - description: Benchmark and evaluation protocol followed for secondary structure prediction and contact prediction to assess protein understanding.

- ProteinNet · [GitHub](https://github.com/aqlaboratory/proteinnet)
  - description: Dataset/splits used for the unsupervised contact prediction evaluation following ESM/TAPE protocols.

- NetSurfP-2.0 dataset · [Website](https://services.healthtech.dtu.dk/service.php?NetSurfP-2.0)
  - description: Training/validation data source for secondary structure prediction; test sets include CB513, CASP12, and TS115 as described in the paper.

- MoleculeNet · [Website](http://moleculenet.org/) · [GitHub](https://github.com/deepchem/deepchem)
  - description: Standard suite of molecular property benchmarks (QM7/8/9, HIV, MUV, BACE, BBBP, TOX21, PCBA, SIDER) used to evaluate ESM-AA’s molecular performance.

- DUD-E (Directory of Useful Decoys, Enhanced) · [Website](http://dude.docking.org/)
  - description: Virtual screening benchmark employed in the paper’s zero-shot evaluation appendix.

- PDBbind 2019 · [Website](http://www.pdbbind.org.cn/)
  - description: Pocket–ligand complex dataset used for contrastive pretraining in the virtual screening experiments discussed in the appendix.

- XGBoost · [GitHub](https://github.com/dmlc/xgboost) · [Codewiki](https://codewiki.google/github.com/dmlc/xgboost)
  - description: Gradient-boosting baseline used in the ProSmith-style fusion framework to assess compatibility of protein and molecule representations.

<!-- paper_id: bbfc82109297ca06cd8c163c3aeaa1b4d1a23fb9 -->

## 113. SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning - NeurIPS - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_NeurIPS_15_Poster_SiriuS_Self-improving_Multi-agent_Systems_via_Bootstrapped_Reasoning.pdf
- Link: https://openreview.net/pdf/856c35f60eccff17ac8726de9ca5f7fbf9bcf3ee.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_NeurIPS_15_Poster_SiriuS_Self-improving_Multi-agent_Systems_via_Bootstrapped_Reasoning.pdf
- Token Usage: input 27545, output 5570, total 33115

### GitHub & Websites

- SIRIUS (Self-improving Multi-agent Systems via Bootstrapped Reasoning)
  - description: Official code release mentioned in the paper (“Code is available here.”). It contains the implementation of the SIRIUS framework, experience library construction/augmentation, and training/evaluation for the problem-solving, actor–critic, and competitive negotiation settings.

- NEGOTIATIONARENA Platform
  - description: Platform used to run the competitive scenarios (Resource Exchange, Ultimatum, Seller–Buyer) for evaluating negotiation behaviors of agents in this paper.

- PubMedQA
  - [GitHub](https://github.com/pubmedqa/pubmedqa)
  - description: Biomedical yes/no QA dataset used as a main benchmark (with the original train/test split) in both the Problem-Solving and Actor–Critic settings.

- MMLU (Hendrycks Test)
  - [GitHub](https://github.com/hendrycks/test)
  - description: Source dataset used to construct the College Physics and College Chemistry benchmarks evaluated in this paper.

- TheoremQA
  - [GitHub](https://github.com/wenhuchen/TheoremQA)
  - description: Theorem-driven QA dataset included in the composition of the College Physics benchmark used for evaluation.

- GPQA (Graduate-Level Google-Proof Q&A)
  - description: Graduate-level QA dataset included in building the College Physics/Chemistry benchmarks evaluated by the paper.

- DSPy
  - [GitHub](https://github.com/stanfordnlp/dspy) · [Codewiki](https://codewiki.google/github.com/stanfordnlp/dspy) · [Website](https://dspy.ai) · [Doc](https://dspy.ai/docs)
  - description: Prompt/program optimizer baseline (MIPROv2) used for comparison against SIRIUS.

- TextGrad
  - [GitHub](https://github.com/stanfordmlgroup/textgrad)
  - description: Gradient-based natural-language optimization toolkit used as a multi-agent prompt-optimization baseline in the experiments.

- STaR (Self-Taught Reasoner)
  - description: Bootstrapped reasoning baseline used for comparison with SIRIUS on reasoning tasks.

- CoMM (Collaborative Multi-Agent, Multi-Reasoning-Path Prompting)
  - description: Training-free collaborative multi-agent prompting baseline used for comparison.

- OpenAI Fine-tuning API
  - [Doc](https://platform.openai.com/docs/guides/fine-tuning)
  - description: Fine-tuning infrastructure used by the authors to train agent policies (GPT-3.5-turbo and GPT-4o-mini) in SIRIUS.

- OpenAI GPT-3.5-turbo
  - [Doc](https://platform.openai.com/docs/models/gpt-3-5)
  - description: Backbone model used for sampling, agent interactions, and supervised fine-tuning in the experiments.

- OpenAI GPT-4o-mini
  - [Doc](https://platform.openai.com/docs/models/gpt-4o-mini)
  - description: Backbone model used for sampling, agent interactions, and supervised fine-tuning in the experiments.

- Llama-3.2-3B-Instruct
  - [Website](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
  - description: Open-source backbone evaluated with SIRIUS in the problem-solving tasks.

<!-- paper_id: 76f93c2b16af5e838e1631cd9e00bb253b2fd1ea -->

## 114. From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models - ICML - 2025 - citation_count 19 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2025_ICML_19_Spotlightposter_From_Mechanistic_Interpretability_to_Mechanistic_Biology_Training,_Evaluating,_and_Interpreting_Sparse_Autoencoders_on_Protein_Language_Models.pdf
- Link: https://openreview.net/pdf/9f17c03cb8b900c8987d0efd2edda7945a8224c9.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2025_ICML_19_Spotlightposter_From_Mechanistic_Interpretability_to_Mechanistic_Biology_Training,_Evaluating,_and_Interpreting_Sparse_Autoencoders_on_Protein_Language_Models.pdf
- Token Usage: input 17625, output 5784, total 23409

### GitHub & Websites

- InterProt · [GitHub](https://github.com/etowahadams/interprot) · [Website](https://interprot.com)
  - description: Official codebase and interactive feature visualizer released by the paper; used to train/evaluate TopK SAEs on ESM-2 activations and to visualize latent activations on sequences/structures for interpretation and the human rater study.

- InterProt-ESM2-SAEs (pretrained weights) · [Website](https://huggingface.co/liambai/InterProt-ESM2-SAEs)
  - description: Hugging Face repository hosting the pretrained SAE checkpoints released by the authors for reproducing analyses and probing.

- ESM (ESM-2, 650M) · [GitHub](https://github.com/facebookresearch/esm)
  - description: Base protein language model whose residual stream activations the authors train SAEs on and probe against.

- ESM-2 650M checkpoint · [Website](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
  - description: The specific ESM-2 model variant used throughout the paper for extracting layer activations.

- UniRef50 · [Website](https://www.uniprot.org/uniref/UniRef50)
  - description: Training corpus for SAEs; authors sample 1M sequences ≤1022 aa from UniRef50 to train TopK SAEs.

- UniProtKB/Swiss-Prot · [Website](https://www.uniprot.org/help/swiss-prot)
  - description: Curated protein set used to compute embeddings for the human interpretability study and to evaluate family-specific latent classification.

- InterPro · [Website](https://www.ebi.ac.uk/interpro/)
  - description: Protein family annotations used to label sequences and determine family-specificity of SAE latents.

- TAPE · [GitHub](https://github.com/songlab-cal/tape)
  - description: Source of the secondary structure dataset and splits used for residue-level classification probes.

- DeepLoc 1.0 · [Website](https://services.healthtech.dtu.dk/service.php?DeepLoc-1.0)
  - description: Dataset/provider for subcellular localization; the paper trains protein-level probes with DeepLoc’s homology-controlled splits.

- Meltome Atlas · [Website](https://meltomeatlas.proteomics.wzw.tum.de/)
  - description: Dataset of protein melting temperatures used for the thermostability regression task (with the FLIP train/test split).

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com/)
  - description: Tool used to cluster Swiss-Prot sequences at 30% identity for evaluation and to reduce redundancy.

- NCBI BLAST+ · [GitHub](https://github.com/ncbi/blast) · [Website](https://blast.ncbi.nlm.nih.gov/Blast.cgi)
  - description: Used in the steering analysis to find homologs of top-activating sequences before multiple sequence alignment.

- Clustal Omega · [Website](http://www.clustal.org/omega/)
  - description: Multiple sequence alignment tool used to align sequences within the same InterPro family for the steering experiments.

- scikit-learn · [GitHub](https://github.com/scikit-learn/scikit-learn) · [Codewiki](https://codewiki.google/github.com/scikit-learn/scikit-learn) · [Website](https://scikit-learn.org/)
  - description: Library used for logistic regression and ridge regression probes (protein-level tasks) with hyperparameter search.

- PyTorch · [GitHub](https://github.com/pytorch/pytorch) · [Website](https://pytorch.org/)
  - description: Framework used to implement the residue-level linear classifier for secondary structure prediction.

- InterPLM · [Website](https://interplm.ai)
  - description: Concurrent, closely related project that also trains SAEs on ESM-2; referenced as an alternative visualizer and analysis resource for interpretable pLM features.

<!-- paper_id: 2ac42be0b5fb0b61f4be220bbd65322bed59ecbf -->

## 115. A Survey of LLM-based Agents in Medicine: How far are we from Baymax? - ACL - 2025 - citation_count 52 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/medical-clinical/2025_ACL_52_Findings_A_Survey_of_LLM-based_Agents_in_Medicine_How_far_are_we_from_Baymax.pdf
- Tags: agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/medical-clinical/2025_ACL_52_Findings_A_Survey_of_LLM-based_Agents_in_Medicine_How_far_are_we_from_Baymax.pdf
- Token Usage: input 16835, output 5398, total 22233

### GitHub & Websites

- MIMIC-III · [Website](https://physionet.org/content/mimiciii/1.4/)
  - description: Large, de-identified ICU EHR dataset used in the survey’s cited works for tasks like mortality prediction and readmission analysis; recommended foundation for developing/evaluating medical LLM agents.

- MIMIC-IV · [Website](https://physionet.org/content/mimiciv/2.2/)
  - description: Successor to MIMIC-III with expanded ICU and hospital data; referenced as a key dataset supporting clinical analytics by LLM agents.

- eICU Collaborative Research Database · [Website](https://physionet.org/content/eicu-crd/2.0/)
  - description: Multi-center critical care database cited as a pivotal source for building and testing agentic clinical analytics.

- i2b2/n2c2 Clinical NLP Corpora · [Website](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
  - description: Widely-used clinical text datasets from the i2b2/n2c2 challenges; referenced as core resources for training/evaluating medical language agents on de-identified clinical notes.

- MedNLI · [GitHub](https://github.com/jgc128/mednli)
  - description: Natural language inference dataset in the clinical domain cited as a key benchmark for evaluating medical reasoning and text understanding of LLM agents.

- MedQA (USMLE) · [GitHub](https://github.com/jind11/MedQA)
  - description: USMLE-style multiple-choice QA benchmark used in the survey’s referenced systems to assess factual medical knowledge and diagnostic reasoning.

- MedMCQA · [GitHub](https://github.com/medmcqa/medmcqa)
  - description: Large-scale multi-subject medical MCQ benchmark cited for evaluating broad medical knowledge of LLM-based agents.

- PubMedQA · [GitHub](https://github.com/qiaojin/PubMedQA)
  - description: Biomedical research question answering dataset referenced for testing literature-grounded medical reasoning.

- MMLU (Hendrycks Test) · [GitHub](https://github.com/hendrycks/test)
  - description: General-purpose knowledge benchmark with a medical subset; cited for measuring cross-domain factual knowledge of LLM agents.

- HL7 FHIR · [Doc](https://www.hl7.org/fhir/overview.html)
  - description: Interoperability standard highlighted in the survey as essential documentation for integrating agents with EHR systems safely and reliably.

- HIPAA Privacy Rule · [Website](https://www.hhs.gov/hipaa/for-professionals/privacy/index.html)
  - description: Regulatory guidance cited as required reading for compliant deployment of medical LLM agents handling protected health information.

- FDA AI/ML in SaMD · [Website](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device)
  - description: Official FDA resource referenced for governance of AI/ML-based clinical software; relevant for deploying agent systems as medical devices.

- GDPR (EU Data Protection) · [Website](https://commission.europa.eu/law/law-topic/data-protection/data-protection-eu_en)
  - description: EU data protection framework cited for privacy and security requirements when training or deploying LLM agents with EU health data.

- OpenAI CLIP · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Vision-language model mentioned in the survey’s perception subsystem for processing medical images as part of multimodal agent pipelines.

- HaluEval · [GitHub](https://github.com/RUCAIBox/HaluEval)
  - description: Hallucination evaluation benchmark cited for assessing and mitigating hallucinations in medical LLM agents.

- BERTScore · [GitHub](https://github.com/Tiiiger/bert_score)
  - description: Semantic similarity metric referenced for evaluating generated clinical text (e.g., reports, summaries) produced by LLM agents.

<!-- paper_id: a7c6ab6ce4aef57bb1c046627a0fdf3c53901cb4 -->

## 116. SurfPro: Functional Protein Design Based on Continuous Surface - ICML - 2024 - citation_count 10 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_10_Poster_SurfPro_Functional_Protein_Design_Based_on_Continuous_Surface.pdf
- Link: https://openreview.net/pdf/e20d4e2dc0e1eec6b208ec275b34023987e7f976.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_10_Poster_SurfPro_Functional_Protein_Design_Based_on_Continuous_Surface.pdf
- Token Usage: input 20697, output 5896, total 26593

### GitHub & Websites

- MSMS (Michel Sanner’s Molecular Surface) · [Website](https://ccsb.scripps.edu/msms/)
  - description: Tool used by SurfPro to compute raw molecular surfaces as point clouds for each protein prior to smoothing and compression.

- CATH 4.2 processed dataset (NeurIPS’19 Graph Protein Design) · [GitHub](https://github.com/jingraham/neurips19-graph-protein-design)
  - description: Repository providing the curated CATH 4.2 inverse folding dataset and splits originally used by Ingraham et al.; SurfPro trains/evaluates on these splits (following Jing et al. 2020).

- CATH (Class, Architecture, Topology, Homologous superfamily) · [Website](https://www.cathdb.info/)
  - description: Source protein structure classification database underlying the CATH 4.2 benchmark used for inverse folding evaluation.

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Global repository of macromolecular structures; SurfPro pretrains on 179,278 <surface, sequence> pairs constructed from the PDB.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Structure prediction system used for the pAE interaction-based binding evaluation; SurfPro follows Bennett et al. to compute AF2 pAE interaction scores.

- ESMFold (part of ESM) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/)
  - description: Fast structure predictor used to predict designed binder structures before computing AlphaFold2 pAE interaction scores.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Inverse folding baseline used for comparison and for generating candidate binders in the MaSIF comparison.

- PiFold
  - description: State-of-the-art inverse folding baseline; the paper uses the authors’ released GitHub implementation for fair comparison.

- LM-DESIGN
  - description: Structure-informed language model baseline; the paper uses the authors’ released GitHub implementation and settings for comparison.

- MaSIF · [GitHub](https://github.com/LPDI-EPFL/masif)
  - description: Surface-based geometric deep learning framework; used by the authors for a head-to-head binder design comparison by ranking ProteinMPNN-generated candidates.

- ESP (Enzyme–Substrate Prediction) score (Kroll et al., 2023)
  - description: Official codebase used to compute ESP scores for enzyme–substrate binding evaluation in the enzyme design task.

<!-- paper_id: cb73832f442484075c4b908a9c67ad7293b6b25d -->

## 117. De novo protein design using geometric vector field networks - ICLR - 2024 - citation_count 17 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICLR_17_Spotlight_De_novo_protein_design_using_geometric_vector_field_networks.pdf
- Link: https://openreview.net/pdf/88687a41b251ea3e5018907ebe06860c15e31ff3.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICLR_17_Spotlight_De_novo_protein_design_using_geometric_vector_field_networks.pdf
- Token Usage: input 26646, output 4129, total 30775

### GitHub & Websites

- Vector Field Network (VFN) · [GitHub](https://github.com/aim-uofa/VFN)
  - description: Official code release for this paper; includes VFN layers and the VFN-Diff, VFN-IF, and VFN-IFE implementations used to reproduce the diffusion and inverse folding experiments.

- FrameDiff (SE(3) Diffusion for protein backbones) · [GitHub](https://github.com/jasonkyuyim/se3_diffusion)
  - description: Baseline diffusion framework the authors build upon; VFN-Diff replaces FrameDiff’s IPA point-attention with VFN while keeping other components.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Inverse folding model used during evaluation to design sequences for generated backbones and as a pipeline baseline.

- ESM / ESMFold · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/)
  - description: Protein language model family (including ESMFold) used to fold designed sequences and, via LoRA, as the external knowledge model in VFN-IFE to refine VFN-IF predictions.

- AlphaFold2 (incl. IPA) · [GitHub](https://github.com/deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Source of the IPA frame-based encoder idea and local frame construction referenced by the paper; also used for visualization in the appendix.

- PiFold
  - description: State-of-the-art inverse folding baseline and framework; VFN-IF retains PiFold’s decoder/global context attention while replacing its GNN layer with VFN for comparisons.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com/)
  - description: Structure search tool used to compute pdbTM and assess novelty of generated proteins by comparing to the PDB database.

- MaxCluster · [Website](http://www.sbg.bio.ic.ac.uk/~maxcluster/)
  - description: Clustering tool used to compute the diversity metric of generated backbones.

- CATH Database (v4.2) · [Website](https://www.cathdb.info/)
  - description: Primary dataset and split used for training/evaluating inverse folding (sequence recovery and structure recovery) in the paper.

- TS50 and TS500 benchmark sets · [GitHub](https://github.com/drorlab/gvp-pytorch)
  - description: Standard inverse folding test sets employed for evaluation; commonly provided via the GVP PyTorch repository referenced by prior work.

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Source of structures used to construct training data for VFN-Diff and to scale VFN-IF+; also the reference database for novelty evaluation (pdbTM).

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion) · [Website](https://rfdiffusion.github.io/)
  - description: Related diffusion-based protein design system discussed in the paper; not directly compared under identical settings but relevant for extending this line of work.

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient fine-tuning method used to adapt the ESM model in VFN-IFE.

<!-- paper_id: ba712d7a1bb988644c952589059625c623a57120 -->

## 118. Evaluating Representation Learning on the Protein Structure Universe - ICLR - 2024 - citation_count 20 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICLR_20_Poster_Evaluating_Representation_Learning_on_the_Protein_Structure_Universe.pdf
- Link: https://openreview.net/pdf/69d3702ce7faefe484e8143abfb3242c2e6f99da.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICLR_20_Poster_Evaluating_Representation_Learning_on_the_Protein_Structure_Universe.pdf
- Token Usage: input 39679, output 6313, total 45992

### GitHub & Websites

- ProteinWorkshop · [GitHub](https://github.com/a-r-j/ProteinWorkshop) · [Website](https://proteins.sh) · [Doc](https://proteins.sh)
  - description: Official codebase and documentation for the benchmark introduced in the paper; includes dataloaders (AFDB/ESM Atlas), models, featurization, tasks, and CLI for pretraining/finetuning.

- ProteinWorkshop Preprocessed Datasets (Zenodo) · [Website](https://zenodo.org/record/8282470)
  - description: Official Zenodo record hosting the processed datasets used in the paper’s experiments.

- ProteinWorkshop Pretrained Weights (Zenodo) · [Website](https://zenodo.org/record/8287754)
  - description: Official Zenodo record with pretrained model checkpoints released by the authors.

- AlphaFold Protein Structure Database (AlphaFoldDB) · [Website](https://alphafold.ebi.ac.uk)
  - description: Large collection of predicted protein structures; used as the primary pretraining corpus (2.27M representative structures) and for tasks like pLDDT prediction.

- FoldSeek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com)
  - description: Fast structural search/clustering tool used to cluster AlphaFoldDB and select the 2.27M representative structures for pretraining.

- FoldComp · [GitHub](https://github.com/steineggerlab/foldcomp)
  - description: Compression/indexing library for large protein structure sets; the benchmark’s storage‑efficient AFDB/ESM Atlas dataloaders are built on FoldComp.

- ESMFold / ESM Atlas · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com)
  - description: ESMFold code and the ESM Atlas dataset of predicted structures; the benchmark provides loaders for ESM Atlas and evaluates ESM-2-650M as a baseline.

- MGnify · [Website](https://www.ebi.ac.uk/metagenomics/)
  - description: Source of the sequence collection used to generate ESM Atlas structures (MGnify 2023 release referenced for ESM Atlas).

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Primary source of experimental protein structures; many benchmark datasets are curated directly from PDB.

- MMTF (Macromolecular Transmission Format) · [Website](http://mmtf.rcsb.org)
  - description: Efficient structure format supported by the benchmark for working with PDB data at scale.

- CATH (4.2, 40% NR) · [Website](http://cathdb.info)
  - description: Structural classification resource; used for inverse folding (Ingraham et al. dataset) and as an additional pretraining dataset.

- ASTRAL · [Website](http://scop.berkeley.edu/astral/)
  - description: Compendium of protein domains; included as an additional source of experimental structures for pretraining/analysis.

- MaSIF-Site (PPI site prediction dataset) · [GitHub](https://github.com/LPDI-EPFL/masif)
  - description: Dataset and code from Gainza et al. for PPI site prediction; the paper adopts this dataset (with a proximity-based relabeling) for node‑level evaluation.

- ccPDB (Metal-binding dataset source) · [Website](https://academic.oup.com/database/article/doi/10.1093/database/bay142/5298333#130010908)
  - description: Source database referenced for metal-binding annotations; the benchmark includes a zinc-binding dataset derived from related resources for residue‑level prediction.

- PTM Site Prediction Dataset (Yan et al. 2023) · [Website](https://zenodo.org/record/7655709)
  - description: AlphaFold2-predicted structure set with PTM metadata; used for multilabel residue‑level PTM site prediction.

- SCOP 1.75 Fold Classification Dataset (Hou et al. 2017) · [Website](https://academic.oup.com/bioinformatics/article/34/8/1295/4708302)
  - description: Fold classification benchmark (Fold/Superfamily/Family splits) used for graph‑level evaluation.

- Structure-based GO Function Prediction Dataset (Gligorijević et al. 2021) · [Website](https://www.nature.com/articles/s41467-021-23303-9)
  - description: PDB‑derived dataset for Gene Ontology prediction (BP/CC/MF); adopted in the benchmark with the original 30% sequence‑identity cutoff.

- Reaction Class Prediction (IEConv Proteins) · [GitHub](https://github.com/phermosilla/IEConv_proteins)
  - description: Dataset and code for enzyme reaction class prediction (EC-based); used for graph‑level classification with 50% sequence identity split.

- Therapeutics Data Commons (Antibody Developability) · [Website](https://tdcommons.ai/single_pred_tasks/develop/#sabdab-chen-et-al)
  - description: Curated developability dataset (from SabDab) used for binary graph‑level classification of antibody developability.

- SabDab (Structural Antibody Database) · [Website](https://opig.stats.ox.ac.uk/webapps/sabdab/)
  - description: Source of antibody structures underlying the TDC developability task used in the benchmark.

- ESM (ESM-2) · [GitHub](https://github.com/facebookresearch/esm)
  - description: Protein language models used as a baseline; paper evaluates ESM‑2‑650M augmented with structural features.

- SchNetPack (SchNet) · [GitHub](https://github.com/atomistic-machine-learning/schnetpack)
  - description: Reference implementation of SchNet; SchNet is one of the invariant GNN baselines reimplemented within ProteinWorkshop.

- EGNN · [GitHub](https://github.com/vgsatorras/egnn)
  - description: E(n)-equivariant GNN baseline evaluated across tasks in the benchmark.

- e3nn (Tensor Field Networks) · [GitHub](https://github.com/e3nn/e3nn)
  - description: Library for SO(3)/SE(3) equivariant neural networks; used for TFN and spherical tensor operations in the benchmark.

- MACE · [GitHub](https://github.com/ACEsuit/mace)
  - description: Higher‑order equivariant GNN; included as an evaluated architecture class in the benchmark.

- TorchDrug · [GitHub](https://github.com/DeepGraphLearning/torchdrug) · [Website](https://torchdrug.ai)
  - description: ML platform that includes GearNet; cited as a dependency and used for protein‑specific architectures in the benchmark.

- Graphein · [GitHub](https://github.com/a-r-j/graphein)
  - description: Library for geometric deep learning on biomolecular structures; listed as a dependency for dataset/graph construction.

- PyTorch Geometric · [GitHub](https://github.com/pyg-team/pytorch_geometric) · [Codewiki](https://codewiki.google/github.com/pyg-team/pytorch_geometric)
  - description: Graph learning framework used to implement GNN architectures in the benchmark.

- PyTorch Lightning · [GitHub](https://github.com/Lightning-AI/lightning)
  - description: Training framework used to organize experiments and runs.

- Hydra · [GitHub](https://github.com/facebookresearch/hydra)
  - description: Configuration framework used for experiment management and hyperparameter sweeps.

- PyTorch · [GitHub](https://github.com/pytorch/pytorch)
  - description: Core deep learning framework used throughout the benchmark.

- pNeRF · [GitHub](https://github.com/aqlaboratory/pnerf)
  - description: Tool for reconstructing Cartesian coordinates from torsions; used in torsional denoising tasks to rebuild structures before feature computation.

<!-- paper_id: 7f8b959347116a3fa14c7edcb067c39cd60781e1 -->

## 119. ProteinBench: A Holistic Evaluation of Protein Foundation Models - ICLR - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/benchmarks-evaluation/2025_ICLR_15_Poster_ProteinBench_A_Holistic_Evaluation_of_Protein_Foundation_Models.pdf
- Link: https://openreview.net/pdf/3544ce00631e6c89a866f5f9ff2ba3d59149c0d7.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/benchmarks-evaluation/2025_ICLR_15_Poster_ProteinBench_A_Holistic_Evaluation_of_Protein_Foundation_Models.pdf
- Token Usage: input 45474, output 7026, total 52500

### GitHub & Websites

- ProteinBench · [Website](https://proteinbench.github.io/)
  - description: Official project page for the paper’s benchmark; hosts the public leaderboard, evaluation datasets, and code framework referenced for reproducing results.

- Protein Frame Flow (FrameFlow) · [GitHub](https://github.com/microsoft/protein-frame-flow) · [Doc](https://github.com/microsoft/protein-frame-flow/tree/main/motif_scaffolding)
  - description: SE(3) flow-matching backbone generator; used as a backbone design and motif-scaffolding baseline with its released motif-scaffolding evaluation scripts.

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion)
  - description: Structure-based protein backbone and motif-scaffolding generator; used as a strong backbone-design baseline and to generate de novo backbones for inverse-folding evaluation.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Inverse folding model used to design sequences for generated backbones and to assess designability (scTM/scRMSD) across tasks.

- ESM (ESMFold, ESM-IF1, ESM3) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esm.ai)
  - description: FAIR’s protein modeling suite; used for structure prediction (ESMFold), inverse folding (ESM-IF1), and evaluation/generation with ESM3 open models.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold)
  - description: Folding model used as the structural oracle for scTM/scRMSD/pLDDT in several tasks and as a base reference in conformation prediction.

- ColabFold (AlphaFold2 inference and MSA pipeline) · [GitHub](https://github.com/sokrypton/ColabFold) · [Website](https://colabfold.com)
  - description: Accelerated AlphaFold2 inference and standardized MSA querying; used for AF2 predictions and MSA generation throughout evaluations.

- OpenFold · [GitHub](https://github.com/aqlaboratory/openfold)
  - description: Open reimplementation of AlphaFold2; used as an MSA-based folding baseline and backbone for conformation sampling (e.g., MSA-subsampling).

- RoseTTAFold2 · [GitHub](https://github.com/RosettaCommons/RoseTTAFold2)
  - description: MSA-based folding model; included as a single-state folding baseline.

- EigenFold · [GitHub](https://github.com/bjing2016/eigenfold)
  - description: Diffusion-based generative folding model; benchmarked for folding and multiple-state prediction.

- Str2Str · [GitHub](https://github.com/DeepGraphLearning/Str2Str)
  - description: Score-based structure-perturbation framework for conformation sampling; used for multi-state and distribution prediction comparisons.

- AlphaFlow / ESMFlow · [GitHub](https://github.com/bjing2016/alphaflow)
  - description: Flow/diffusion models extending folding to ensemble generation; used for multiple-state and distribution prediction with PDB- and MD-fine-tuned checkpoints.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com)
  - description: Fast structure search/clustering; used to compute novelty (Max TM to PDB) and diversity (pairwise TM, clustering) across design tasks.

- TM-score / TM-align · [Website](https://zhanggroup.org/TM-score/)
  - description: Structural similarity metrics used widely in the benchmark (TM-score for accuracy, TMalign for structural alignment).

- lDDT (Swiss-Model) · [Doc](https://swissmodel.expasy.org/lddt/downloads/)
  - description: Local superposition-free structural quality metric; used to assess folding/local quality.

- RCSB PDB · [Website](https://www.rcsb.org/)
  - description: Primary structural database; used as the search space for novelty metrics and as the reference source for native structures.

- CAMEO · [Website](https://www.cameo3d.org/)
  - description: Continuous structure prediction benchmark; CAMEO2022 targets are used to evaluate folding models.

- CASP15 · [Website](https://predictioncenter.org/casp15/)
  - description: Community benchmark for structure prediction; used as a source of recent native structures for inverse folding evaluation.

- ATLAS (protein MD dataset) · [Website](https://www.dsimb.inserm.fr/ATLAS/index.html)
  - description: Dataset of atomistic MD simulations; used for distribution prediction benchmarking and MD-based fine-tuning comparisons.

- UniRef50 · [Website](https://www.uniprot.org/uniref/UniRef50)
  - description: Protein sequence clustering dataset; referenced as a training/evaluation source for sequence-generation baselines.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Fast sequence search/clustering; used for antibody dataset clustering (CDR-H3) and involved in ColabFold MSA generation.

- SAbDab (Structural Antibody Database) · [Website](https://opig.stats.ox.ac.uk/webapps/newsabdab/)
  - description: Source of antibody–antigen complex structures; used to construct unified training data for antibody-design baselines.

- IgFold · [GitHub](https://github.com/Graylab/IgFold)
  - description: Antibody structure prediction tool; used to predict CDR-H3 structures in scRMSD and pipeline steps for antibody evaluation.

- AntiBERTy · [GitHub](https://github.com/Graylab/AntiBERTy)
  - description: Antibody language model; used to score sequence naturalness (SeqNat) in antibody evaluations.

- Biopython · [GitHub](https://github.com/biopython/biopython) · [Website](https://biopython.org/)
  - description: Bioinformatics toolkit; used for sequence alignment (PairwiseAligner) and peptide bond break evaluations.

- MEAN / dyMEAN · [GitHub](https://github.com/THUNLP-MT/MEAN)
  - description: Antibody design baselines; retrained with unified data/settings, and the repo hosts the referenced rabd_summary.jsonl.

- DiffAb · [GitHub](https://github.com/luost26/diffab)
  - description: Diffusion-based antigen-specific antibody design; used as a baseline (with multiple samples) in the antibody benchmark.

- Rosetta / RosettaAntibodyDesign (RAbD) · [GitHub](https://github.com/RosettaCommons/rosetta) · [Website](https://www.rosettacommons.org/)
  - description: Suite used for side-chain packing, energy minimization, binding energy (InterfaceAnalyzer) and as the source of the RAbD dataset reference in antibody evaluation.

<!-- paper_id: 7c6729bd4b2b3bdf013482b499c7401ab6de44ef -->

## 120. Proteus: Exploring Protein Structure Generation for Enhanced Designability and Efficiency - ICML - 2024 - citation_count 26 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_26_Poster_Proteus_Exploring_Protein_Structure_Generation_for_Enhanced_Designability_and_Efficiency.pdf
- Link: https://openreview.net/pdf/c0e3630d2c44de576cfbb58426802f4a8b71c9db.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_ICML_26_Poster_Proteus_Exploring_Protein_Structure_Generation_for_Enhanced_Designability_and_Efficiency.pdf
- Token Usage: input 23538, output 5567, total 29105

### GitHub & Websites

- Proteus · [GitHub](https://github.com/Wangchentong/Proteus)
  - description: Official code release for the paper; implements the Proteus diffusion architecture, training, sampling, and evaluation pipelines described in the work.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Inverse folding model used to design sequences for generated backbones when computing designability metrics (8 sequences per backbone).

- ESMFold (ESM) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com)
  - description: Structure prediction used for self-consistency evaluation; the best ESMFold prediction per designed sequence/backbone is used to compute scTM/scRMSD.

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion)
  - description: Strong comparison baseline for backbone generation; the paper matches/exceeds its designability and efficiency without pretraining.

- Genie · [GitHub](https://github.com/aqlaboratory/genie)
  - description: Baseline model (equivariant diffusion of oriented residue clouds) used for comparisons in designability, diversity, and efficiency.

- Chroma / ChromaDesign
  - description: Baseline generative model and its inverse folding component; ChromaDesign is used to design sequences for Chroma backbones for fair evaluation as reported in its paper.

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org)
  - description: Primary dataset source; the authors curate 50,773 single-chain training samples from PDB (with date cutoffs and filters).

- UniProt · [Website](https://www.uniprot.org)
  - description: Used to map protein chains to UniProt IDs to remove redundancy and select highest-resolution structures for chains with ≥80% sequence overlap.

- DSSP · [GitHub](https://github.com/PDB-REDO/dssp) · [Website](https://swift.cmbi.nl/gv/dssp/)
  - description: Secondary structure assignment tool used to filter training data (exclude proteins with >50% loop content).

- MaxCluster · [Website](http://www.sbg.bio.ic.ac.uk/phyre2/html/wiki/index.php/MaxCluster)
  - description: Tool used to cluster generated backbones (TM-score cutoff 0.6) for the diversity metric, counting only designable structures.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold)
  - description: Architectural inspiration (IPA, triangle modules, template features) for Proteus’ multi-track and triangle mechanisms; referenced extensively in the method design.

<!-- paper_id: 001dc5a75429ba493c4e6601a3e0da160567a9a2 -->

## 121. Structure Language Models for Protein Conformation Generation - ICLR - 2025 - citation_count 17 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICLR_17_Poster_Structure_Language_Models_for_Protein_Conformation_Generation.pdf
- Link: https://openreview.net/pdf/063242f3610e3ab2e08fd2073e307b10d917dffb.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICLR_17_Poster_Structure_Language_Models_for_Protein_Conformation_Generation.pdf
- Token Usage: input 40870, output 5276, total 46146

### GitHub & Websites

- ESMDiff (Structure Language Models for Protein Conformation Generation) · [GitHub](https://github.com/lujiarui/esmdiff)
  - description: Official code release for this paper; includes training and inference pipelines for SLMs and ESMDiff, data processing, sampling strategies, and evaluation used in the experiments.

- ESM (ESM3/ESMFold) · [GitHub](https://github.com/evolutionaryscale/esm) · [Website](https://esm.ai)
  - description: Foundation protein language models; the paper fine-tunes ESM3-1.4B and uses its dVAE structure tokenizer and sequence encoder, and uses ESMFold in baselines/initialization.

- AlphaFold2 · [GitHub](https://github.com/google-deepmind/alphafold) · [Codewiki](https://codewiki.google/github.com/google-deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk)
  - description: Structure prediction system; used via its official repository (v2.3.2) for the MSA-subsampling baseline and for AlphaFlow/ESMFlow configurations.

- ColabFold · [GitHub](https://github.com/sokrypton/ColabFold) · [Website](https://colabfold.mmseqs.com)
  - description: Fast AlphaFold2/ESMFold inference with online MSA search; used to retrieve MSAs for the MSA-based baselines.

- OmegaFold · [GitHub](https://github.com/HeliXonProtein/OmegaFold)
  - description: Single-sequence structure predictor; used to generate embeddings for the EigenFold baseline pipeline.

- idpGAN · [GitHub](https://github.com/feiglab/idpGAN)
  - description: GAN-based generator for intrinsically disordered protein ensembles; used as an open-source baseline on the IDP benchmark.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Sequence search and clustering toolkit; used to cluster/filter sequences when curating the IDP test set and to avoid data leakage.

- deeptime · [GitHub](https://github.com/deeptime-ml/deeptime) · [Doc](https://deeptime-ml.github.io/latest/)
  - description: Library for dynamical modeling and TICA; used to compute TICA projections for distributional evaluation (JS divergence on TIC components).

- TM-score · [Website](https://zhanggroup.org/TM-score/)
  - description: Official binary for TM-score and RMSD calculations; used to compute TM-ens and RMSD-ens metrics with structural alignment.

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org/)
  - description: Primary structural dataset; used as the training data source (cutoff May 1, 2020) for the structure language models and baselines.

- Protein Ensemble Database (PED) · [Website](https://proteinensemble.org/)
  - description: Repository of experimentally validated protein conformational ensembles; used to build the IDP benchmark set and evaluate ensemble statistics.

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers)
  - description: Transformer implementations; used to implement S-T5 and S-GPT structure language models (encoder-decoder and decoder-only variants).

<!-- paper_id: 86637c5e96d397b6e81211027712167c7cfbf04f -->

## 122. Diffusion on language model encodings for protein sequence generation - ICML - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICML_11_Poster_Diffusion_on_language_model_encodings_for_protein_sequence_generation.pdf
- Link: https://openreview.net/pdf/247bdc9c2a584f53f7defdfab8bd5e01da3316bc.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICML_11_Poster_Diffusion_on_language_model_encodings_for_protein_sequence_generation.pdf
- Token Usage: input 47279, output 6058, total 53337

### GitHub & Websites

- DiMA (official code release)
  - description: Official implementation of the paper’s latent diffusion framework for protein sequence generation; the PDF states “Code is released at GitHub,” and this repo contains training/inference scripts, configs, and checkpoints used across ESM, CHEAP, and SaProt encoders.

- ESM (ESM-2 and ESMFold) · [GitHub](https://github.com/facebookresearch/esm)
  - description: Protein language models and ESMFold structure predictor used as encoders (ESM-2 family), for pseudo‑perplexity scoring, and to predict 3D structures (pLDDT) throughout evaluation.

- ESM Cambrian (ESM-C) · [Website](https://evolutionaryscale.ai/blog/esm-cambrian)
  - description: The ESM-C 300M encoder variant is evaluated in the paper as an alternative representation space for DiMA.

- UniProt/SwissProt · [Website](https://www.uniprot.org/)
  - description: Curated protein sequence database used to build the SwissProt training/evaluation set (after filtering and clustering).

- AlphaFold Database (AFDB v4) · [Website](https://alphafold.ebi.ac.uk/)
  - description: Source for the AFDBv4-90 dataset (high-confidence AF2 predictions) used for large-scale training and evaluation; also used for structural references.

- CATH S40 non-redundant dataset · [Website](https://cathdb.info/)
  - description: Structure set used to fine-tune and evaluate fold-conditioned generation with CHEAP encodings.

- RFdiffusion (benchmark and baseline) · [GitHub](https://github.com/RosettaCommons/RFdiffusion) · [Website](https://rfdiffusion.github.io/)
  - description: Structure-generation method providing the functional motif scaffolding benchmark tasks and a fold-conditioned baseline compared against DiMA.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com/)
  - description: Fast structure search tool used to find nearest structural neighbors and for distributional/structural comparisons (e.g., TM-score references).

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com/)
  - description: Used for clustering sequences (e.g., computing CD0.5/CD0.95, co‑clustering with datasets) to assess diversity and distribution coverage.

- InterProScan · [GitHub](https://github.com/ebi-pf-team/interproscan) · [Website](https://www.ebi.ac.uk/interpro/)
  - description: Functional annotation pipeline used to assess biological relevance and family membership (e.g., SUPERFAMILY, MobiDB).

- NCBI BLAST+ · [GitHub](https://github.com/ncbi/blast) · [Website](https://blast.ncbi.nlm.nih.gov/Blast.cgi)
  - description: Sequence similarity searches used for nearest-neighbor identity, novelty analysis, and alignment-based metrics.

- DSSP · [GitHub](https://github.com/cmbi/dssp) · [Website](https://www.cmbi.umcn.nl/dssp.html)
  - description: Secondary-structure assignment tool used on predicted structures to compare secondary-structure content with natural proteins.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Structure-to-sequence model used for structure-only evaluation and co-design metrics (e.g., scPerplexity, scRMSD) and to assess RFDiffusion outputs.

- EvoDiff · [GitHub](https://github.com/microsoft/evodiff)
  - description: Discrete diffusion baseline used for comparison and for conditional family-specific generation baselines.

- ProGen2 · [GitHub](https://github.com/salesforce/progen)
  - description: Autoregressive protein LM baseline; also used to compute perplexity for quality evaluation and as a controlled generation baseline.

- ProGen2 fine-tuning helper · [GitHub](https://github.com/hugohrban/ProGen2-finetuning)
  - description: Third-party training scripts referenced for fine-tuning ProGen2 on family-conditioned data for the controllable generation experiments.

- ProLLaMA · [Website](https://huggingface.co/GreatCaptainNemo/ProLLaMA)
  - description: Large protein LM baseline; fine-tuned with LoRA for family-conditioned generation and compared against DiMA.

- nanoGPT · [GitHub](https://github.com/karpathy/nanoGPT) · [Codewiki](https://codewiki.google/github.com/karpathy/nanoGPT)
  - description: Lightweight autoregressive baseline trained from scratch for unconditional generation comparisons.

- Fold-conditioned generation targets (TIM barrel 6WVS, NTF2 1GY6) · [Website](https://www.rcsb.org/)
  - description: PDB structures used as canonical fold targets to evaluate DiMA’s fold-conditioned sequence generation.

<!-- paper_id: 9e81434435fe1cbb032006334b089c70e863732b -->

## 123. Sequence-Augmented SE(3)-Flow Matching For Conditional Protein Generation - NeurIPS - 2024 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_NeurIPS_12_Poster_Sequence-Augmented_SE(3)-Flow_Matching_For_Conditional_Protein_Generation.pdf
- Link: https://openreview.net/pdf/503e86547852b43509aa82eecef8210d45232c5b.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2024_NeurIPS_12_Poster_Sequence-Augmented_SE(3)-Flow_Matching_For_Conditional_Protein_Generation.pdf
- Token Usage: input 33316, output 4769, total 38085

### GitHub & Websites

- FoldFlow / FOLDFLOW-2 · [GitHub](https://github.com/DreamFold/FoldFlow)
  - description: Official code release for the FoldFlow family including FOLDFLOW-2; used to train/evaluate the sequence-conditioned SE(3) flow matching model, run baselines/ablations, and reproduce figures and metrics in the paper.

- Protein Data Bank (PDB) · [Website](https://www.rcsb.org)
  - description: Primary source of experimental protein structures; used for training, evaluation splits, and novelty/diversity comparisons.

- AlphaFold Protein Structure Database (AFDB) · [Website](https://alphafold.ebi.ac.uk)
  - description: Repository of AlphaFold2-predicted structures used to curate a large synthetic training set (filtered on pLDDT, etc.) to augment PDB data.

- UniProt/SwissProt · [Website](https://www.uniprot.org) · [Doc](https://www.uniprot.org/help/swiss-prot)
  - description: Curated protein sequence database; SwissProt entries underpin the AFDB synthetic structures filtered for training.

- ESM (ESM2, ESMFold) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/models?tab=folding)
  - description: Protein language models and folding model used in FOLDFLOW-2; ESM2-650M provides sequence embeddings (frozen) and ESMFold is used for refolding in designability/self-consistency evaluation.

- AlphaFold2 · [GitHub](https://github.com/deepmind/alphafold) · [Website](https://alphafold.ebi.ac.uk)
  - description: Provides the Invariant Point Attention (IPA) architecture used in encoders/decoders and is the model generating AFDB structures that are filtered into the synthetic dataset.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Inverse folding model used in the evaluation pipeline to design sequences for generated backbones before refolding with ESMFold for self-consistency metrics.

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion)
  - description: Strong baseline for protein backbone generation and motif scaffolding; the paper compares FOLDFLOW-2 against this model across designability, novelty, diversity, and conditional tasks.

- FrameDiff (SE(3) diffusion for protein backbones) · [GitHub](https://github.com/jasonkyuyim/se3_diffusion)
  - description: Baseline method for unconditional protein backbone generation; used for comparisons on designability, novelty, and diversity.

- FrameFlow (SE(3) flow matching for proteins)
  - description: Baseline flow model referenced for conditional generation; the paper states “FrameFlow does not have public code for motif-scaffolding and thus cannot be evaluated on the VHH benchmark.”

- EigenFold · [GitHub](https://github.com/microsoft/eigenfold)
  - description: Non–MD-finetuned baseline for conformation ensemble generation; used for zero-shot equilibrium conformation sampling comparisons.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://www.mmseqs.com/)
  - description: Tool used to recluster the PDB dataset at 50% sequence identity during data preparation.

- SAbDab (Structural Antibody Database) · [Website](https://opig.stats.ox.ac.uk/webapps/newsabdab/)
  - description: Source of VHH/nanobody structures and sequences used to build the CDR motif-scaffolding benchmark and fine-tuning dataset.

<!-- paper_id: bc6be1ae39aa0a5d18efb827312119a9094fc4f2 -->

## 124. Self-supervised Pocket Pretraining via Protein Fragment-Surroundings Alignment - ICLR - 2024 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICLR_12_Poster_Self-supervised_Pocket_Pretraining_via_Protein_Fragment-Surroundings_Alignment.pdf
- Link: https://openreview.net/pdf/fcb728a7e3b8f50b0b315d0afbd69ca490308c3e.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-language-models/2024_ICLR_12_Poster_Self-supervised_Pocket_Pretraining_via_Protein_Fragment-Surroundings_Alignment.pdf
- Token Usage: input 24269, output 4901, total 29170

### GitHub & Websites

- ProFSA · [GitHub](https://github.com/bowen-gao/ProFSA)
  - description: Official code and data release for the paper; contains the pocket encoder, training pipeline, and the large pseudo ligand–pocket dataset used to pretrain/evaluate ProFSA.

- Uni-Mol · [GitHub](https://github.com/dptech-corp/Uni-Mol) · [Website](https://openreview.net/forum?id=6K2RM6wVqKu)
  - description: Universal 3D molecular representation framework; ProFSA uses the official pretrained molecular encoder and architecture from Uni-Mol and adopts Uni-Mol's pocket druggability dataset and pocket construction settings.

- Protein Data Bank (RCSB PDB) · [Website](https://www.rcsb.org)
  - description: Source of high-resolution protein structures; ProFSA constructs 5.5M pseudo ligand–pocket complexes from non-redundant PDB entries clustered at 70% sequence identity.

- PDBbind · [Website](http://www.pdbbind.org.cn/)
  - description: Standard benchmark of protein–ligand complexes with binding affinities; used for distribution alignment during pseudo-pair sampling and for downstream LBA evaluation (v2019 splits via ATOM3D).

- BioLiP/BioLiP2 · [Website](https://zhanggroup.org/BioLiP)
  - description: Database of biologically relevant ligand–protein interactions; used to assess pocket representation separation (t-SNE visualization of ligand-specific pockets) and to discuss data scarcity.

- ATOM3D · [GitHub](https://github.com/drorlab/atom3d) · [Website](https://atom3d.ai)
  - description: Tasks, standardized splits, and loaders for 3D molecular learning; ProFSA follows ATOM3D’s preprocessing and strict 30%/60% sequence-identity splits for the PDBbind LBA task.

- Fpocket · [GitHub](https://github.com/Discngine/fpocket)
  - description: Open-source pocket detection and scoring tool; its scores (Fpocket score, druggability, SASA, hydrophobicity) define the Uni-Mol druggability dataset on which ProFSA is evaluated.

- MMseqs2 (easy-cluster) · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Fast sequence clustering used to derive non-redundant PDB chains at 70% identity for ProFSA’s large-scale pseudo-pair construction.

- FreeSASA · [GitHub](https://github.com/mittinatten/freesasa) · [Website](https://freesasa.github.io)
  - description: Solvent accessible surface area library; used to compute relative buried surface area (rBSA) for distributional alignment and analysis of pseudo vs. real complexes.

- TOUGH-M1 dataset · [Website](http://cssb.biology.lsu.edu/tough/)
  - description: Large-scale benchmark of similar/dissimilar pocket pairs; used to evaluate ProFSA on pocket matching (AUC-ROC in zero-shot and finetuned settings).

- Kahraman binding-site dataset
  - description: Classic dataset of non-homologous proteins binding common ligands; ProFSA evaluates pocket matching on the reduced set (excluding PO4 pockets) following DeeplyTough’s protocol.

- TM-align · [Website](https://zhanggroup.org/TM-align/)
  - description: Protein structure alignment baseline; included as a traditional method comparator for pocket matching.

- NCBI BLAST · [Website](https://blast.ncbi.nlm.nih.gov/Blast.cgi)
  - description: Sequence alignment utility used in the paper to verify zero sequence similarity in a qualitative pocket-matching case study.

- AlphaFold · [GitHub](https://github.com/deepmind/alphafold)
  - description: High-accuracy protein structure prediction; cited as a scalable upstream structure source to extend ProFSA’s pseudo-pair construction beyond PDB.

- ESMFold (ESM) · [GitHub](https://github.com/facebookresearch/esm)
  - description: Language model-based protein structure prediction; proposed by the authors as another source of predicted structures to expand ProFSA’s pretraining corpus.

<!-- paper_id: 95bcd9faba8374f7b8145b0be93c94e9be7b344e -->

## 125. P(all-atom) Is Unlocking New Path For Protein Design - ICML - 2025 - citation_count 17 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICML_17_Spotlightposter_P(all-atom)_Is_Unlocking_New_Path_For_Protein_Design.pdf
- Link: https://openreview.net/pdf/5665b4b423ab9ef2c6c00d97d55feffda4dae4ec.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_ICML_17_Spotlightposter_P(all-atom)_Is_Unlocking_New_Path_For_Protein_Design.pdf
- Token Usage: input 35121, output 5315, total 40436

### GitHub & Websites

- Pallatom · [GitHub](https://github.com/levinthal/Pallatom)
  - description: Official code release for the paper’s all-atom protein generative model, including the atom14 representation, training/inference code, and evaluation used throughout the experiments.

- RCSB Protein Data Bank (PDB) · [Website](https://www.rcsb.org) · [Doc](https://pdb101.rcsb.org)
  - description: Primary structural dataset used for training/curation; the paper selects high-resolution monomers and applies additional quality filters from this database.

- PISCES sequence culling server · [Website](https://dunbrack3.fccc.edu/PISCES.php)
  - description: Used to obtain a nonredundant PDB list (sequence identity filtering) for building the training set.

- AlphaFold Protein Structure Database (AFDB) · [Website](https://alphafold.ebi.ac.uk) · [Doc](https://alphafold.ebi.ac.uk/download#alphafolddb-clusters)
  - description: Augmented training data source; the paper uses AFDB (and the redundancy-reduced AFDB cluster sets) with additional pLDDT/packing/secondary-structure filters.

- DSSP · [GitHub](https://github.com/cmbi/dssp) · [Website](https://www.cmbi.nl/dssp)
  - description: Used to compute secondary structure and solvent-accessible surface area (SASA) for dataset filtering and analysis.

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com)
  - description: Employed for structural clustering/diversity evaluation and for redundancy removal via the easy-cluster pipeline; also used for novelty comparisons to PDB.

- MMseqs2 · [GitHub](https://github.com/soedinglab/MMseqs2) · [Website](https://mmseqs.com)
  - description: Used to cluster sequences and quantify sequence diversity (DIV-seq) of designable samples.

- ESMFold · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/resources/proteinfold)
  - description: Structure prediction model used to fold designed sequences for designability evaluation (pLDDT, RMSD) and for benchmarking.

- ProteinMPNN · [GitHub](https://github.com/dauparas/ProteinMPNN)
  - description: Fixed‑backbone sequence design baseline used in evaluations (DES-bb) and for comparison against Pallatom’s sequence head.

- RFdiffusion · [GitHub](https://github.com/RosettaCommons/RFdiffusion) · [Website](https://rfdiffusion.github.io)
  - description: State-of-the-art backbone generative baseline; evaluated with the paper’s PMPNN-1 protocol for comparison.

- Protpardelle · [Website](https://www.pnas.org/doi/10.1073/pnas.2311500121)
  - description: All‑atom diffusion baseline referenced and evaluated; the paper compares Pallatom’s designability and diversity to this method.

- Multiflow (protein co-design) · [Website](https://openreview.net/forum?id=kQwSbv0BR4)
  - description: Backbone+sequence co-design baseline evaluated in the paper; used for comparative benchmarking under the CO-DESIGN and PMPNN modes.

- CarbonNovo · [Website](https://openreview.net/forum?id=FSxTEvuFa7)
  - description: Unified energy-based co-design baseline; included in extended comparisons on designability, diversity, and novelty.

- JAX · [GitHub](https://github.com/google/jax) · [Codewiki](https://codewiki.google/github.com/google/jax) · [Website](https://jax.readthedocs.io)
  - description: The implementation leverages JAX (noted in runtime analysis) for efficient sampling via JIT-compilation.

<!-- paper_id: ea6c2cd4100bb1b5db0246fa45b92d90c8a888b2 -->

## 126. "Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents - ACL - 2025 - citation_count 13 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_13_Findings_Nuclear_Deployed!_Analyzing_Catastrophic_Risks_in_Decision-making_of_Autonomous_LLM_Agents.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_13_Findings_Nuclear_Deployed!_Analyzing_Catastrophic_Risks_in_Decision-making_of_Autonomous_LLM_Agents.pdf
- Token Usage: input 90635, output 4016, total 94651

### GitHub & Websites

- “Nuclear Deployed!” official resources
  - description: The paper states “we will release the code for reproducibility in an upon-request manner,” and the PDF header includes “Project Page” and “Code” links, but no public URLs are provided. Use these materials (upon request) to reproduce the three-stage simulation framework and prompts used in the study.

- LiveBench · [GitHub](https://github.com/LiveBench/LiveBench) · [Website](https://livebench.ai)
  - description: Uncontaminated LLM benchmark used to obtain the “Reasoning Average” scores that the paper correlates with catastrophic-behavior and deception rates.

- OpenAI API and models (GPT‑4o, GPT‑4o‑mini, o1, o1‑mini, o3‑mini) · [Website](https://platform.openai.com) · [Doc](https://platform.openai.com/docs) · [Doc](https://cdn.openai.com/o3-mini-system-card.pdf)
  - description: Closed-source models used as Mauto and as the fixed Mstate (GPT‑4o‑mini) in simulations; authors specify API model versions, decoding settings, and the o1/o3 “reasoning_effort” parameter (system card for o3‑mini linked).

- OpenAI o1 system card · [Website](https://openai.com/research) · [Doc](https://arxiv.org/abs/2412.16720)
  - description: Cited to characterize the o1 “reasoning” family whose behavior is analyzed as particularly risky in the paper.

- OpenAI Model Specification — Follow the chain of command · [Doc](https://cdn.openai.com/spec/model-spec-2024-05-08.html#follow-the-chain-of-command)
  - description: Documentation cited in the paper’s discussion of instruction hierarchy and command-following relevant to the violation experiments.

- Anthropic Claude 3.5 Sonnet · [Website](https://www.anthropic.com/claude) · [Doc](https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf)
  - description: Closed-source baseline model evaluated (notably refused to act in War scenarios); model card addendum is cited.

- Qwen2.5 Instruct (7B/32B/72B) · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io) · [Doc](https://huggingface.co/Qwen)
  - description: Open-source models used as Mauto baselines in all scenarios; specific Instruct checkpoints (7B/32B/72B) are evaluated.

- QwQ‑32B‑Preview (Qwen Team) · [GitHub](https://github.com/QwenLM/QwQ) · [Website](https://huggingface.co/qwen/QwQ-32B-Preview)
  - description: Open-source “reasoning” model variant evaluated (marked as o1‑like in the paper) to study catastrophic and deceptive behaviors.

- Meta Llama 3.x Instruct (Llama3.3‑70B‑Instruct) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open-source baseline model evaluated as Mauto in all scenarios; used to compare safety behavior with other open/closed models.

<!-- paper_id: afee11ed18bc996704bdb16ce288f26195d43daa -->

## 127. Scaling Unlocks Broader Generation and Deeper Functional Understanding of Proteins - NeurIPS - 2025 - citation_count 17 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_NeurIPS_17_Spotlight_Scaling_Unlocks_Broader_Generation_and_Deeper_Functional_Understanding_of_Proteins.pdf
- Link: https://openreview.net/pdf/e39feef56ff28883049bd1ddc3b506a53fcaba90.pdf
- Tags: science, agent, biology, protein-function, agent-bio
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-bio/protein-design/2025_NeurIPS_17_Spotlight_Scaling_Unlocks_Broader_Generation_and_Deeper_Functional_Understanding_of_Proteins.pdf
- Token Usage: input 49320, output 6839, total 56159

### GitHub & Websites

- ProGen3 · [GitHub](https://github.com/Profluent-AI/progen3)
  - description: Official code release and model weights for the ProGen3 sparse protein language models introduced in the paper; used for generation, GLM infilling, and alignment experiments.

- Profluent Protein Atlas v1 (PPA-1)
  - description: The 3.4B-sequence curated pretraining dataset constructed by the authors; used to train all ProGen3 models (the paper notes the dataset itself is not released).

- ESMFold (ESM) · [GitHub](https://github.com/facebookresearch/esm) · [Website](https://esmatlas.com/)
  - description: Structure prediction model used to predict structures/secondary structure and to score/generated sequences in several analyses (e.g., Figs. 2f, 3a, 4f).

- Foldseek · [GitHub](https://github.com/steineggerlab/foldseek) · [Website](https://foldseek.com)
  - description: Used with easy-cluster to structurally cluster protein domains when creating train/val/test splits for the Megascale stability alignment experiments.

- PyRosetta · [Website](https://www.pyrosetta.org) · [Doc](https://www.pyrosetta.org/documentation)
  - description: Used for structure minimization and relaxation (MinMover, FastRelax) and to compute Rosetta energies when evaluating in silico stability of generated sequences.

- DIAMOND · [GitHub](https://github.com/bbuchfink/diamond)
  - description: High-throughput protein aligner used to hierarchically cluster proteins at 90/50/30% identity in building PPA-1.

- Prodigal-GV · [GitHub](https://github.com/apcamargo/prodigal-gv)
  - description: Gene-calling tool (via geNomad) used for protein-coding prediction from genomic/metagenomic inputs when constructing PPA-1.

- Prodigal · [GitHub](https://github.com/hyattpd/Prodigal)
  - description: Prokaryotic gene prediction tool cited/used alongside Prodigal-GV during dataset construction.

- Minimap2 · [GitHub](https://github.com/lh3/minimap2)
  - description: Used in the sequencing validation pipeline to map nanopore reads to reference amplicons for clone verification.

- SAMtools/BCFtools · [GitHub](https://github.com/samtools/samtools) · [Website](http://www.htslib.org/)
  - description: Used to generate consensus sequences from read alignments during sequence verification.

- MUMmer (DNADiff) · [GitHub](https://github.com/mummer4/mummer) · [Website](http://mummer.sourceforge.net/)
  - description: DNADiff from MUMmer used to call variants between consensus and reference sequences in the verification pipeline.

- BBTools (BBDuk, BBSeal) · [Website](https://sourceforge.net/projects/bbmap/)
  - description: Employed for read filtering (BBDuk) and demultiplexing by unique k-mers (BBSeal) in the sequencing validation workflow.

- Jellyfish · [GitHub](https://github.com/gmarcais/Jellyfish)
  - description: K-mer counting used to pre-validate amplicon pools for sufficient sequence diversity before long-read sequencing.

- DNA Chisel · [GitHub](https://github.com/Edinburgh-Genome-Foundry/DnaChisel)
  - description: Used to codon-optimize gene fragments for E. coli in constructing GFP11 fusion libraries for split-GFP assays.

- FlashAttention-2 · [GitHub](https://github.com/Dao-AILab/flash-attention) · [Codewiki](https://codewiki.google/github.com/Dao-AILab/flash-attention)
  - description: Optimized attention kernels used during ProGen3 model training for efficiency.

- Megablocks (Mixture-of-Experts) · [GitHub](https://github.com/stanford-futuredata/megablocks)
  - description: Library leveraged to implement efficient sparse MoE layers used in ProGen3.

- MosaicML Composer · [GitHub](https://github.com/mosaicml/composer)
  - description: Training orchestration framework used by the authors to run distributed pretraining.

- MosaicML Streaming · [GitHub](https://github.com/mosaicml/streaming)
  - description: Data streaming/dataloading library employed for efficient large-scale pretraining.

- PyTorch · [GitHub](https://github.com/pytorch/pytorch) · [Website](https://pytorch.org/)
  - description: Deep learning framework used to implement and train all ProGen3 models.

- ProteinGym · [GitHub](https://github.com/OATML-Markslab/ProteinGym) · [Website](https://proteingym.org)
  - description: Large-scale benchmark datasets used for zero-shot and supervised fitness prediction evaluations and alignment experiments.

- IMG/M (Integrated Microbial Genomes & Microbiomes) · [Website](https://img.jgi.doe.gov/)
  - description: Source of metagenomic protein sequences used to build the PPA-1 pretraining corpus.

- European Nucleotide Archive (ENA) · [Website](https://www.ebi.ac.uk/ena)
  - description: Source of metagenomic data for PPA-1 construction.

- NCBI GenBank · [Website](https://www.ncbi.nlm.nih.gov/genbank/)
  - description: Source of genomic protein sequences included in PPA-1.

- NCBI non-redundant protein database (NCBI-nr) · [Website](https://www.ncbi.nlm.nih.gov/refseq/about/nonredundantproteins/)
  - description: Additional genomic protein source integrated into PPA-1.

- UniRef90 (UniProt Reference Clusters) · [Website](https://www.uniprot.org/help/uniref)
  - description: Clustered protein dataset incorporated as a genomic source in PPA-1.

- MetaEuk · [GitHub](https://github.com/soedinglab/metaeuk)
  - description: Eukaryotic gene discovery/annotation resource cited as one of the metagenomic/eukaryotic-focused sources used when assembling PPA-1.

- MMETSP (Marine Microbial Eukaryote Transcriptome Sequencing Project) · [Website](https://www.imicrobe.us/#/projects/104)
  - description: Eukaryotic sequence resource listed among additional databases contributing proteins to PPA-1.

<!-- paper_id: cf7297714fa88a9ace1535f6cd0d1f81c4535903 -->

## 128. ResearchTown: Simulator of Human Research Community - ICML - 2025 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICML_18_Poster_ResearchTown_Simulator_of_Human_Research_Community.pdf
- Link: https://openreview.net/pdf/17f00ca8bf036f1be17d8f405f4e63cc025c1f64.pdf
- Tags: multiagent, tool, science, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICML_18_Poster_ResearchTown_Simulator_of_Human_Research_Community.pdf
- Token Usage: input 45768, output 3762, total 49530

### GitHub & Websites

- RESEARCHTOWN · [GitHub](https://github.com/ulab-uiuc/research-town)
  - description: Official implementation of the paper’s multi-agent TextGNN simulator for research communities; primary codebase to reproduce all methods and experiments.

- RESEARCHBENCH · [Website](https://huggingface.co/datasets/ulab-ai/research-bench)
  - description: Official benchmark released with the paper containing 1,000 paper-writing and 200 review-writing tasks used for evaluation and ablations.

- OpenAI GPT-4o-mini · [Website](https://platform.openai.com/docs/models#gpt-4o-mini) · [Doc](https://platform.openai.com/docs/models/gpt-4o-mini)
  - description: Main LLM backbone for agent functions in RESEARCHTOWN; all generations use the 2024-07-18 model via the OpenAI API.

- OpenAI text-embedding-3-large · [Website](https://openai.com/index/new-embedding-models-and-api-updates/) · [Doc](https://platform.openai.com/docs/guides/embeddings)
  - description: Primary embedding model for similarity-based evaluation (paper and review scoring) across the benchmark.

- Voyage AI voyage-3 Embeddings · [GitHub](https://github.com/voyage-ai/voyageai-python) · [Website](https://www.voyageai.com) · [Doc](https://docs.voyageai.com)
  - description: Alternative embedding model used for ablations and reviewer ranking; accessed via the voyageai Python client.

- Qwen2.5-7B-Instruct · [Website](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  - description: Open-source LLM used as an alternative generator in ablations to assess model robustness.

- DeepSeek-V3 · [GitHub](https://github.com/deepseek-ai/DeepSeek-V3) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-V3) · [Website](https://www.deepseek.com)
  - description: Additional LLM used for generation ablations; evaluated alongside GPT-4o-mini and Qwen to compare simulation quality.

- Together Inference API · [Website](https://www.together.ai/inference) · [Doc](https://docs.together.ai/docs/inference-overview)
  - description: Inference service the paper used to run Qwen-2.5-7B-Instruct-Turbo and DeepSeek-V3-0324.

- OpenAI API · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs/overview)
  - description: API used to access GPT-4o-mini and text-embedding-3-large throughout the simulation and evaluation pipeline.

- arxiv Python package · [GitHub](https://github.com/lukasschwab/arxiv.py) · [Doc](https://pypi.org/project/arxiv/)
  - description: Used to fetch paper metadata (titles, abstracts, IDs) when constructing RESEARCHBENCH.

- Semantic Scholar Python package · [GitHub](https://github.com/danielnsilva/semanticscholar) · [Doc](https://pypi.org/project/semanticscholar/)
  - description: Used to resolve author IDs and collect author publication profiles for building the community graph.

- openreview-py · [GitHub](https://github.com/openreview/openreview-py) · [Doc](https://openreview-py.readthedocs.io/en/latest/)
  - description: Library used to retrieve public ICLR 2024 reviews for REVIEWBENCH construction.

- LiteLLM · [GitHub](https://github.com/BerriAI/litellm) · [Doc](https://docs.litellm.ai)
  - description: Wrapper used to call OpenAI embeddings (text-embedding-3-large) during the embedding-based evaluation.

- voyageai Python client · [GitHub](https://github.com/voyage-ai/voyageai-python) · [Doc](https://docs.voyageai.com)
  - description: Client library the paper used to compute voyage-3 embeddings for similarity metrics and reviewer retrieval.

<!-- paper_id: 2b37107e99568d7ece09c888d7e67b2d369ccc45 -->

## 129. MobileUse: A GUI Agent with Hierarchical Reflection for Autonomous Mobile Operation - NeurIPS - 2025 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NeurIPS_12_Poster_MobileUse_A_GUI_Agent_with_Hierarchical_Reflection_for_Autonomous_Mobile_Operation.pdf
- Link: https://openreview.net/pdf/0dfcd45aa867bf09cef726711a929960f019ea27.pdf
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NeurIPS_12_Poster_MobileUse_A_GUI_Agent_with_Hierarchical_Reflection_for_Autonomous_Mobile_Operation.pdf
- Token Usage: input 26516, output 3798, total 30314

### GitHub & Websites

- MobileUse Toolkit · [GitHub](https://github.com/MadeAgents/mobile-use)
  - description: Official code release from the paper; provides the hierarchical-reflection mobile GUI agent and an out-of-the-box toolkit with WebUI for operating physical Android devices via ADB.

- AndroidWorld · [GitHub](https://github.com/google-deepmind/android_world) · [Website](https://openreview.net/forum?id=il5yUQsrjC)
  - description: Dynamic Android benchmark used for evaluation; the paper reports SOTA on this environment and follows its standardized initialization and automated evaluation.

- AndroidLab · [GitHub](https://github.com/THUDM/AndroidLab) · [Website](https://arxiv.org/abs/2410.24024)
  - description: Android benchmark with 138 tasks and fine-grained metrics; used to assess MobileUse with the reported success rates.

- Qwen2.5-VL · [GitHub](https://github.com/QwenLM/Qwen2.5-VL) · [Website](https://qwenlm.github.io/)
  - description: Open-source multimodal LLM family; the paper uses Qwen2.5-VL-72B/32B/7B Instruct as the backbone model for MobileUse.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai/) · [Doc](https://docs.vllm.ai/en/latest/)
  - description: High-throughput LLM serving engine used to deploy the multimodal model for MobileUse experiments.

- Android Debug Bridge (ADB) · [Doc](https://developer.android.com/tools/adb) · [Website](https://developer.android.com/tools/releases/platform-tools)
  - description: Android SDK Platform-Tools used by the MobileUse Toolkit to connect to and control physical devices.

- Gradio · [GitHub](https://github.com/gradio-app/gradio) · [Codewiki](https://codewiki.google/github.com/gradio-app/gradio) · [Website](https://www.gradio.app/) · [Doc](https://www.gradio.app/docs)
  - description: Web UI framework used by the MobileUse Toolkit to provide a visual interface for issuing commands and monitoring agent execution.

<!-- paper_id: c53d56df4b9286de3d4704517da07294124b0b72 -->

## 130. AdaptAgent: Adapting Multimodal Web Agents with Few-Shot Learning from Human Demonstrations - ACL - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_14_Long_AdaptAgent_Adapting_Multimodal_Web_Agents_with_Few-Shot_Learning_from_Human_Demonstrations.pdf
- Link: https://aclanthology.org/2025.acl-long.1008/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_14_Long_AdaptAgent_Adapting_Multimodal_Web_Agents_with_Few-Shot_Learning_from_Human_Demonstrations.pdf
- Token Usage: input 24038, output 4866, total 28904

### GitHub & Websites

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Large-scale web-agent dataset used for training/meta-learning and evaluation (cross-task, cross-website, cross-domain); the paper also amends the cross-task split for proper evaluation.

- VisualWebArena · [Website](https://visualwebarena.github.io/)
  - description: Visual web-agent benchmark and live environment used to evaluate task success both with human trajectories and in a live browser setting.

- SeeAct · [GitHub](https://github.com/OSU-NLP-Group/SeeAct)
  - description: Prompting-based multimodal web agent (GPT-4V/4o) baseline; AdaptAgent augments its prompt with 1-shot multimodal in-context demonstrations to obtain the proprietary-model results.

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent) · [Doc](https://huggingface.co/THUDM/cogagent-chat-hf)
  - description: Open-weights GUI-focused MLLM used as the open-source backbone; the paper meta-trains (FOMAML) and few-shot adapts CogAgent, and loads the THUDM/cogagent-chat-hf checkpoint.

- CogVLM · [GitHub](https://github.com/THUDM/CogVLM)
  - description: Visual-language backbone family behind CogAgent; cited as the underlying pretrained model family for GUI understanding.

- OpenAI GPT-4o · [Website](https://openai.com/index/gpt-4o) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Proprietary multimodal LLM used with SeeAct prompting; AdaptAgent provides multimodal in-context demonstrations to GPT-4o for few-shot adaptation.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Website](https://playwright.dev)
  - description: Browser automation framework used to execute predicted actions on webpages in the agent loop (mentioned as an execution tool).

- pynput · [GitHub](https://github.com/moses-palmer/pynput) · [Doc](https://pynput.readthedocs.io/en/latest/)
  - description: Python library to control mouse/keyboard for GUI automation; used as an alternative tool to execute actions.

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Website](https://huggingface.co)
  - description: Library and model hub used to load and fine-tune the THUDM/cogagent-chat-hf checkpoint during meta-learning and adaptation.

- Set-of-Mark Prompting (SoM) · [Doc](https://arxiv.org/abs/2310.11441)
  - description: Visual grounding augmentation used in the paper’s SeeAct* baseline by overlaying marks in the image input.

<!-- paper_id: 0fcfea3dd30ac11064cb37e80b3dd447d711e723 -->

## 131. Agent-Oriented Planning in Multi-Agent Systems - ICLR - 2025 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_18_Poster_Agent-Oriented_Planning_in_Multi-Agent_Systems.pdf
- Link: https://openreview.net/pdf/82bac503b8ff6352c48f82986a318160a941b2a4.pdf
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_18_Poster_Agent-Oriented_Planning_in_Multi-Agent_Systems.pdf
- Token Usage: input 22812, output 3844, total 26656

### GitHub & Websites

- Agent-Oriented-Planning (AOP) · [GitHub](https://github.com/lalaliat/Agent-Oriented-Planning)
  - description: Official code release of the paper; implements the AOP framework (meta-agent, reward model, detector, representative works, prompts) used in all experiments.

- HUSKY (and Husky-QA dataset) · [GitHub](https://github.com/allenai/husky)
  - description: Open-source language agent framework used as a strong multi-agent baseline; its Husky-QA dataset (train/test splits) is the primary benchmark used to train/evaluate AOP.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Baseline method combining reasoning and acting; compared against AOP in the experiments.

- DROP · [GitHub](https://github.com/allenai/drop) · [Website](https://allennlp.org/drop)
  - description: Reading comprehension dataset requiring discrete reasoning; used as an additional evaluation benchmark (decontextualized subset) for AOP.

- IIRC · [GitHub](https://github.com/allenai/iirc) · [Website](https://allenai.org/data/iirc)
  - description: Incomplete Information Reading Comprehension dataset; used as another evaluation benchmark (decontextualized subset) for AOP.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: Mathematical problem-solving dataset; used in Appendix experiments to evaluate AOP with multiple specialized math/code agents.

- Sentence-Transformers (all-MiniLM-L6-v2) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Doc](https://www.sbert.net/) · [Model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - description: Embedding model used as the backbone of AOP’s reward model to encode sub-tasks and agent descriptions.

- Bing Web Search API · [Doc](https://learn.microsoft.com/bing/search-apis/bing-web-search/overview)
  - description: External tool used by the Search Agent to retrieve up-to-date information for sub-tasks.

- OpenAI GPT-4o · [Website](https://openai.com/index/gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: LLM used to power the meta-agent and all agents (code, math, search, commonsense), and to assist with automatic scoring/evaluation.

- Qwen2-Math-7B-Instruct · [GitHub](https://github.com/QwenLM/Qwen2) · [Model](https://huggingface.co/Qwen/Qwen2-Math-7B-Instruct)
  - description: Task-specific math LLM used in Appendix experiments as an expert Math Agent to extend AOP.

- DeepSeek-Coder-V2 · [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-Coder) · [Model](https://huggingface.co/deepseek-ai/deepseek-coder-v2-instruct)
  - description: Code-focused LLM used in Appendix experiments as an expert Code Agent within AOP.

- Llama 3.2 3B Instruct · [Website](https://llama.meta.com/) · [Model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
  - description: Additional LLM used in Appendix experiments to form multiple agents with the same expertise when testing AOP on MATH.

<!-- paper_id: 6d8ce667aaaa611e6a9a002f9fbf7513be95492b -->

## 132. Tools Fail: Detecting Silent Errors in Faulty Tools - EMNLP - 2024 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_14_Main_Tools_Fail_Detecting_Silent_Errors_in_Faulty_Tools.pdf
- Link: https://aclanthology.org/2024.emnlp-main.790/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_14_Main_Tools_Fail_Detecting_Silent_Errors_in_Faulty_Tools.pdf
- Token Usage: input 23556, output 3457, total 27013

### GitHub & Websites

- Tools Fail · [GitHub](https://github.com/jiminsun/tools-fail)
  - description: Official code and data release for this paper; includes the calculator experiments, ALFRED-based tool-error detection datasets, prompts, and scripts to reproduce the detection results.

- ALFRED (A Benchmark for Interpreting Grounded Instructions for Everyday Tasks) · [GitHub](https://github.com/askforalfred/alfred) · [Website](https://askforalfred.com)
  - description: Embodied instruction-following benchmark used to create the multimodal tool-error detection datasets (object detector and action planner) and collect agent trajectories for evaluation.

- AI2-THOR · [GitHub](https://github.com/allenai/ai2thor) · [Website](https://ai2thor.allenai.org) · [Doc](https://ai2thor.allenai.org/ithor/)
  - description: Interactive 3D household simulator underlying ALFRED; the paper’s embodied experiments and limitations reference AI2-THOR’s affordances when assessing planner/detector failures.

- FILM: Following Instructions in Language with Modular Methods · [GitHub](https://github.com/soyeonm/FILM)
  - description: Modular ALFRED agent architecture whose component tools (finetuned Mask R-CNN detector and Fast Marching Method planner) are the basis for the paper’s multimodal tool-error evaluation.

- OpenAI GPT-4o · [Website](https://openai.com/index/gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed-source vision-language model used as a tool evaluator in the ALFRED experiments to Accept/Reject planner and detector outputs.

- Google Gemini 1.5 Pro · [Website](https://ai.google.dev/gemini) · [Doc](https://ai.google.dev/docs/gemini_api_overview)
  - description: Closed-source vision-language model used alongside GPT-4o to evaluate tool outputs in the ALFRED setting.

- OpenAI GPT-3.5/GPT-4 · [Website](https://openai.com/index/introducing-chatgpt-and-whisper-apis/) · [Doc](https://platform.openai.com/docs/models)
  - description: Language models used in the calculator experiments to study overtrust and error detection with correct vs. broken tools.

- Cohere Command-R / Command-R+ · [Website](https://cohere.com/command) · [Doc](https://docs.cohere.com/docs/command-r)
  - description: Language models evaluated in the calculator setting for detecting silent tool errors and assessing the impact of in-context interventions.

<!-- paper_id: 491e129c10bca2f9e50c9c4859b3cb825d217d37 -->

## 133. Self-Challenging Language Model Agents - NeurIPS - 2025 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NeurIPS_16_Poster_Self-Challenging_Language_Model_Agents.pdf
- Link: https://openreview.net/pdf/d7856a8a4e991dab71814a3f39f28fe17b6b93bd.pdf
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NeurIPS_16_Poster_Self-Challenging_Language_Model_Agents.pdf
- Token Usage: input 31972, output 3297, total 35269

### GitHub & Websites

- No official SCA release
  - description: The paper explicitly states there is no open release of code or data. Quote: “Open access to data and code — Answer: [No] … Justification: N/A” and “No new asset is released.”

- M3ToolEval · [Website](https://arxiv.org/abs/2402.01030)
  - description: Multi-turn tool-use benchmark with Calculation and Web Browsing environments and built-in verifiers; the paper generates synthetic CaT tasks in these environments and evaluates SCA on their official test sets.

- Tau-Bench · [Website](https://arxiv.org/abs/2406.12045)
  - description: Customer-service tool-agent benchmark (Retail and Airline) with simulated users and database-backed verifiers; the paper uses these environments to synthesize CaT tasks, train with RL/distillation, and evaluate success rates.

- Llama 3.1 (Llama‑3.1‑8B/70B Instruct) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com) · [Doc](https://llama.meta.com/docs)
  - description: Base and teacher models used throughout—Llama‑3.1‑8B‑Instruct as the executor/challenger and Llama‑3.1‑70B‑Instruct for distillation demonstrations.

- OpenAI GPT‑4o · [Website](https://platform.openai.com/docs/models#gpt-4o)
  - description: Used to simulate the user in Tau‑Bench during evaluation (and referenced as the proprietary model baseline).

<!-- paper_id: f04229adec1cf2cd780d680f8f97e57be08d28b0 -->

## 134. Flow: Modularized Agentic Workflow Automation - ICLR - 2025 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_18_Poster_Flow_Modularized_Agentic_Workflow_Automation.pdf
- Link: https://openreview.net/pdf/883b7c240d2be5b635b144abc546885546d7fa50.pdf
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_18_Poster_Flow_Modularized_Agentic_Workflow_Automation.pdf
- Token Usage: input 23608, output 3104, total 26712

### GitHub & Websites

- Flow (official) · [GitHub](https://github.com/tmllab/2025_ICLR_FLOW)
  - description: Official implementation of Flow, the modularized agentic workflow automation framework proposed in the paper; used to reproduce experiments and extend the AOV-based dynamic workflow system.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent LLM framework used as a baseline; the paper compares Flow against AutoGen across coding and document-generation tasks.

- CAMEL · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel) · [Website](https://camel-ai.org/)
  - description: Multi-agent framework emphasizing role-playing agents; used as a comparison baseline in the experiments.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT)
  - description: Multi-agent collaborative coding framework following SOP-style workflows; evaluated as a baseline against Flow.

- OpenAI GPT-4o-mini and GPT-3.5-Turbo · [Website](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) · [Doc](https://platform.openai.com/docs)
  - description: LLM backends used to power agents in all experiments (primary dependency for reproducing results).

<!-- paper_id: e3e217f36ee7cd41367115e70fcd946ea0799120 -->

## 135. EMOS: Embodiment-aware Heterogeneous Multi-robot Operating System with LLM Agents - ICLR - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_11_Poster_EMOS_Embodiment-aware_Heterogeneous_Multi-robot_Operating_System_with_LLM_Agents.pdf
- Link: https://openreview.net/pdf/4bb1e48853d7189b88d1e952850f6d577f85b849.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_11_Poster_EMOS_Embodiment-aware_Heterogeneous_Multi-robot_Operating_System_with_LLM_Agents.pdf
- Token Usage: input 24641, output 3414, total 28055

### GitHub & Websites

- EMOS / Habitat-MAS Project · [Website](https://emos-project.github.io/)
  - description: Official project page for the paper, hosting resources for the EMOS multi-agent system and the Habitat-MAS benchmark used for evaluation.

- Habitat (Habitat-Sim/Lab) · [GitHub](https://github.com/facebookresearch/habitat-lab) · [Website](https://aihabitat.org/) · [Doc](https://aihabitat.org/docs/intro.html)
  - description: Simulator and task framework the benchmark is built on; the paper runs all HMRS tasks in Habitat and uses its PDDL integration and navigation stack.

- Matterport3D Dataset (MP3D) · [Website](https://niessner.github.io/Matterport/)
  - description: Real multi-floor indoor scans used as environments in Habitat-MAS to evaluate navigation and rearrangement across floors.

- Habitat Synthetic Scenes Dataset (HSSD-200) · [Website](https://aihabitat.org/datasets/hssd-200/)
  - description: Synthetic indoor scenes used alongside MP3D to diversify Habitat-MAS evaluation environments.

- Hydra (3D Scene Graph) · [GitHub](https://github.com/MIT-SPARK/Hydra) · [Website](https://mit-spark.github.io/Hydra/)
  - description: Referenced as the multi-layer scene representation pipeline (L1–L3) informing the textual scene context; cited for deployable multi-robot perception.

- Hydra-Multi · [GitHub](https://github.com/MIT-SPARK/Hydra-Multi)
  - description: Multi-robot extension of Hydra used as a reference for obtaining region connectivity, semantic mesh, and agent/object state layers in real deployments.

- Recast Navigation · [GitHub](https://github.com/recastnavigation/recastnavigation)
  - description: Used to build the L4 navmesh for path planning in Habitat-MAS scenes.

- PyBullet · [GitHub](https://github.com/bulletphysics/bullet3) · [Website](http://pybullet.org/)
  - description: Physics engine integrated with Habitat; the paper disables PyBullet-based physics during benchmark runs but notes it can be re-enabled for extended tasks.

<!-- paper_id: faffd56c4024f552335a76639526d5d60e7a278b -->

## 136. GAM-Agent: Game-Theoretic and Uncertainty-Aware Collaboration for Complex Visual Reasoning - NeurIPS - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NeurIPS_14_Poster_GAM-Agent_Game-Theoretic_and_Uncertainty-Aware_Collaboration_for_Complex_Visual_Reasoning.pdf
- Link: https://openreview.net/pdf/55c4606303cc86c81ca99b5d5743cfe40d7fb140.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NeurIPS_14_Poster_GAM-Agent_Game-Theoretic_and_Uncertainty-Aware_Collaboration_for_Complex_Visual_Reasoning.pdf
- Token Usage: input 54778, output 4058, total 58836

### GitHub & Websites

- Qwen2.5-VL · [GitHub](https://github.com/QwenLM/Qwen2.5-VL) · [Website](https://qwenlm.github.io)
  - description: Used as a primary VLM backbone (7B and 72B variants) for all experiments; GAM-Agent is layered on top of Qwen2.5-VL to evaluate improvements on MMMU, MMBench, MVBench, and V*Bench.

- InternVL3 · [GitHub](https://github.com/OpenGVLab/InternVL) · [Website](https://internvl.github.io)
  - description: Employed as another main VLM backbone (14B and 78B variants); the paper implements GAM-Agent over InternVL3 and reports gains across all benchmarks.

- InternVideo2 / InternVideo2.5 · [GitHub](https://github.com/OpenGVLab/InternVideo2)
  - description: Referenced in the video understanding setup for MVBench as a video-specialized model option when token logprobs are available for uncertainty estimation.

- GPT‑4o · [Website](https://openai.com/index/gpt-4o/) · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Used as a strong closed-source baseline (GPT‑4o‑0513) to test the generality of GAM-Agent; accessed via API with semantic-marker uncertainty since token probabilities are unavailable.

- OpenRouter API · [Website](https://openrouter.ai) · [Doc](https://openrouter.ai/docs)
  - description: The paper accesses GPT‑4o‑0513 through OpenRouter; this API provider is the basis for running closed-source models in the experiments.

- MMMU (Massive Multi-discipline Multimodal Understanding) · [GitHub](https://github.com/MMMU-Benchmark/MMMU) · [Website](https://mmmu-benchmark.github.io)
  - description: One of the core evaluation benchmarks for complex multimodal reasoning; the paper reports accuracy improvements of GAM-Agent on this dataset.

- MMBench · [GitHub](https://github.com/open-compass/MMBench) · [Website](https://mmbench.opencompass.org.cn)
  - description: Primary image understanding benchmark (TEST_V11/TEST_EN variants) used for main results, ablations, and hyperparameter analyses.

- MVBench · [GitHub](https://github.com/OpenGVLab/MVBench) · [Website](https://mvbench.github.io)
  - description: Video temporal reasoning benchmark; the paper evaluates GAM-Agent’s debate framework on multi-frame/video inputs here.

- V*Bench
  - description: Guided visual search benchmark used to assess fine-grained grounding and spatial/attribute reasoning (AR/SR); the paper compares GAM-Agent to other multi-agent methods and reports overall scores.

<!-- paper_id: 79fb57ab1a137acaf6bb61dd3473459786052510 -->

## 137. Assessing and Verifying Task Utility in LLM-Powered Applications - EMNLP - 2024 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_15_Main_Assessing_and_Verifying_Task_Utility_in_LLM-Powered_Applications.pdf
- Link: https://aclanthology.org/2024.emnlp-main.1219/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_15_Main_Assessing_and_Verifying_Task_Utility_in_LLM-Powered_Applications.pdf
- Token Usage: input 25938, output 4503, total 30441

### GitHub & Websites

- AgentEval · [GitHub](https://github.com/Narabzad/AgentEval/)
  - description: Official code, data, prompts, and logs released by this paper for reproducing the AgentEval framework and experiments on MATH and ALFWorld.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent LLM framework used to implement AgentEval; also provides the AutoGen 2-agent and 3-agent baseline systems evaluated in the paper.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Website](https://www.langchain.com/) · [Doc](https://python.langchain.com/docs/)
  - description: LLM application toolkit; the paper uses the LangChain ReAct agent as one of the baseline solutions.

- ReAct (Reason+Act) · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Original ReAct prompting approach for reasoning-and-acting; serves as a comparison baseline (via LangChain ReAct) in the experiments.

- MATH dataset · [GitHub](https://github.com/hendrycks/math)
  - description: Benchmark of 12,500 competition math problems; the paper evaluates on 120 Level-5 problems to assess AgentEval and baseline solvers.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
  - description: Language-based interactive household tasks/environment; used as the second evaluation domain for AgentEval, following Wu et al. (2023).

- TextWorld · [GitHub](https://github.com/microsoft/TextWorld) · [Doc](https://microsoft.github.io/TextWorld/)
  - description: Text-based game engine underlying ALFWorld; cited as the environment foundation relevant for reproducing ALFWorld-based experiments.

- ALFRED · [GitHub](https://github.com/askforalfred/alfred) · [Website](https://askforalfred.github.io/)
  - description: Vision-language instruction dataset/environment extended by ALFWorld; referenced as part of the ALFWorld stack used in evaluation.

- Sentence-Transformers · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Doc](https://www.sbert.net/)
  - description: Library for SBERT embeddings; the paper uses a paraphrase-tuned model to compute cosine similarity and merge semantically similar criteria.

- Paraphrase similarity model (Sentence-BERT) · [Doc](https://bit.ly/3UgsYOp)
  - description: Pretrained paraphrase model used to measure semantic similarity between criteria descriptions for deduplication in VerifierAgent.

- Azure OpenAI Service (GPT-4-0613) · [Website](https://learn.microsoft.com/azure/ai-services/openai/) · [Doc](https://platform.openai.com/docs/models/gpt-4)
  - description: All experiments use GPT-4 (version 0613) accessed via Azure OpenAI; required model endpoint to reproduce AgentEval and baseline outputs.

<!-- paper_id: 180f51787e4205de6ee4c0ae722529cc5e06c712 -->

## 138. SciVerse: Unveiling the Knowledge Comprehension and Visual Reasoning of LMMs on Multi-modal Scientific Problems - ACL - 2025 - citation_count 10 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ACL_10_Findings_SciVerse_Unveiling_the_Knowledge_Comprehension_and_Visual_Reasoning_of_LMMs_on_Multi-modal_Scientific_Problems.pdf
- Link: https://aclanthology.org/2025.findings-acl.1010/
- Tags: multiagent, tool, science, agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ACL_10_Findings_SciVerse_Unveiling_the_Knowledge_Comprehension_and_Visual_Reasoning_of_LMMs_on_Multi-modal_Scientific_Problems.pdf
- Token Usage: input 27322, output 5174, total 32496

### GitHub & Websites

- SCIVERSE · [Website](https://sciverse-cuhk.github.io)
  - description: Official project page for the paper’s benchmark; hosts the multi-modal scientific evaluation dataset (five versions per problem) and materials for the proposed scientific CoT evaluation strategy.

- SceMQA · [Website](https://arxiv.org/abs/2402.05138)
  - description: College-entrance-level multimodal scientific QA benchmark; SCIVERSE curates and filters problems from SceMQA during dataset construction.

- MMMU · [GitHub](https://github.com/MMMU-Benchmark/MMMU) · [Website](https://mmmu-benchmark.github.io)
  - description: Massive multi-discipline multimodal understanding benchmark; SCIVERSE sources problems from MMMU as part of its curated pool.

- CMMMU · [Website](https://arxiv.org/abs/2401.20847)
  - description: Chinese multi-discipline multimodal benchmark; SCIVERSE includes selected problems from CMMMU in its curated dataset.

- GPT-4o · [Website](https://openai.com/index/hello-gpt-4o/)
  - description: Closed-source multimodal model used twice in SCIVERSE: (1) as one of the evaluated LMMs; (2) as the automatic judge for the scientific CoT step categorization and step-wise evaluation.

- GPT-4V · [Doc](https://openai.com/research/gpt-4v-system-card)
  - description: Closed-source baseline evaluated on SCIVERSE’s five versions to compare scientific knowledge and visual reasoning.

- Gemini 1.5 Pro · [Website](https://deepmind.google/technologies/gemini/) · [Doc](https://ai.google.dev/gemini-api/docs/models/gemini)
  - description: Closed-source multimodal model evaluated as a baseline across all SCIVERSE versions.

- Claude 3.5 Sonnet · [Website](https://www.anthropic.com/news/claude-3-5-sonnet)
  - description: Closed-source multimodal model evaluated on SCIVERSE; also used during annotation review to help check formatting/consistency.

- MiniGPT-v2 · [GitHub](https://github.com/Vision-CAIR/MiniGPT-4) · [Codewiki](https://codewiki.google/github.com/Vision-CAIR/MiniGPT-4) · [Website](https://minigpt-4.github.io)
  - description: Open-source multimodal model baseline; assessed on SCIVERSE for open-source comparisons.

- LLaVA-1.5 · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io)
  - description: Open-source LLaVA baseline version evaluated across SCIVERSE’s five problem versions.

- LLaVA-NeXT · [Website](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)
  - description: Stronger LLaVA variant evaluated as an open-source baseline in SCIVERSE.

- LLaVA-OneVision · [GitHub](https://github.com/LLaVA-VL/LLaVA-NeXT) · [Doc](https://arxiv.org/abs/2408.03326)
  - description: Recent LLaVA model emphasizing easy visual task transfer; evaluated on SCIVERSE including vision-rich/vision-only settings.

- ShareGPT4V · [GitHub](https://github.com/InternLM/ShareGPT4V) · [Website](https://sharegpt4v.github.io)
  - description: Open-source large multimodal model; included among SCIVERSE evaluation baselines.

- SPHINX series (e.g., SPHINX-Tiny, SPHINX-MoE, SPHINX-Plus) · [Website](https://arxiv.org/abs/2402.05935)
  - description: Family of open-source LMMs assessed as baselines on SCIVERSE to benchmark data/parameter scaling effects.

- InternLM-XComposer-2 · [GitHub](https://github.com/InternLM/InternLM-XComposer)
  - description: Open-source vision-language model for free-form composition/comprehension; evaluated on SCIVERSE.

- InternVL (1.5/2) · [GitHub](https://github.com/OpenGVLab/InternVL) · [Website](https://internvl.github.io)
  - description: Open-source multimodal suite; InternVL-1.5 and InternVL-2 are evaluated across SCIVERSE’s versions.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://qwenlm.github.io)
  - description: Open-source vision-language model evaluated on SCIVERSE; results reported for knowledge and CoT metrics.

- Qwen2.5-VL · [GitHub](https://github.com/QwenLM/Qwen2.5-VL)
  - description: Newer Qwen VL model cited and evaluated in SCIVERSE for comparison with other open-source LMMs.

<!-- paper_id: ef0a8ce38208ab7eefaeb56c435821dd9f1e40ec -->

## 139. Voting or Consensus? Decision-Making in Multi-Agent Debate - ACL - 2025 - citation_count 19 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ACL_19_Findings_Voting_or_Consensus_Decision-Making_in_Multi-Agent_Debate.pdf
- Link: https://aclanthology.org/2025.findings-acl.606/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ACL_19_Findings_Voting_or_Consensus_Decision-Making_in_Multi-Agent_Debate.pdf
- Token Usage: input 29000, output 4893, total 33893

### GitHub & Websites

- Decision Protocols (Voting or Consensus?) · [GitHub](https://github.com/lkaesberg/decision-protocols)
  - description: Official code and data release for this paper; contains implementations of the seven decision protocols, AAD/CI methods, and experiment scripts to reproduce results.

- MALLM (Multi‑Agent Large Language Models) · [GitHub](https://github.com/Multi-Agent-LLMs/mallm)
  - description: Framework used by the authors to run multi‑agent debates (personas, response generators, discussion paradigms, decision protocols) in their experiments.

- context-plus · [GitHub](https://github.com/Multi-Agent-LLMs/context-plus)
  - description: RAG utility used in the “challenge” ablation to retrieve additional context from Wikipedia.

- Meta Llama 3 8B Instruct · [Website](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) · [Doc](https://llama.meta.com/docs/)
  - description: Base LLM used to instantiate all agents in most experiments.

- Meta Llama 3 70B Instruct · [Website](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) · [Doc](https://llama.meta.com/docs/)
  - description: Larger LLM used for additional evaluations to compare scaling effects.

- Sentence-Transformers (SBERT) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Doc](https://www.sbert.net/)
  - description: Used to compute cosine similarity between agents’ final answers for measuring answer diversity.

- Qwen2‑7B · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://huggingface.co/Qwen/Qwen2-7B)
  - description: External model used to generate intentionally incorrect/irrelevant answers in challenge experiments.

- MMLU (Hendrycks Test) · [GitHub](https://github.com/hendrycks/test) · [Website](https://huggingface.co/datasets/hendrycks_test)
  - description: Knowledge benchmark used for evaluation; authors sample subsets for experiments.

- MMLU‑Pro · [GitHub](https://github.com/TIGER-Lab/MMLU-Pro) · [Website](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
  - description: Harder professional-level extension of MMLU included in the knowledge-task evaluations.

- GPQA · [GitHub](https://github.com/idavidrein/gpqa) · [Website](https://gpqa.github.io/)
  - description: Graduate‑level, Google‑proof multiple-choice benchmark used as a knowledge task.

- StrategyQA · [Website](https://allenai.org/data/strategyqa) · [Doc](https://huggingface.co/datasets/strategyqa)
  - description: Multi-step reasoning benchmark used extensively for scaling and diversity experiments.

- SQuAD 2.0 · [GitHub](https://github.com/rajpurkar/SQuAD-explorer) · [Website](https://rajpurkar.github.io/SQuAD-explorer/) · [Doc](https://huggingface.co/datasets/squad_v2)
  - description: Reading comprehension dataset with answerable/unanswerable questions used to analyze decision protocols on edge cases.

<!-- paper_id: b420b06e94902664150a85ab89ec329641ba666d -->

## 140. TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets - NeurIPS - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3world-models/2025_NeurIPS_14_Poster_TwinMarket_A_Scalable_Behavioral_and_Social_Simulation_for_Financial_Markets.pdf
- Link: https://openreview.net/pdf/cefb01d52e462ade9c6e1daafffb65ed49bcaa23.pdf
- Tags: multiagent, tool, science, world-models
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3world-models/2025_NeurIPS_14_Poster_TwinMarket_A_Scalable_Behavioral_and_Social_Simulation_for_Financial_Markets.pdf
- Token Usage: input 44796, output 2567, total 47363

### GitHub & Websites

- TwinMarket · [Website](https://freedomintelligence.github.io/TwinMarket)
  - description: Official project page for the paper’s LLM-based multi-agent market simulator; the paper states code is provided in the supplement and will be publicly released upon acceptance.

- Xueqiu
  · [Website](https://xueqiu.com/)
  - description: Retail-investor platform whose user profiles and 11,965 transaction records were used to initialize agent personas and behavioral biases.

- Guba Stock Forum (Eastmoney)
  · [Website](https://guba.eastmoney.com/)
  - description: Historical trading data (2017–2024) used to inform/train the stock recommendation component and model popularity-driven exposure.

- CSMAR (China Stock Market & Accounting Research database)
  · [Website](https://data.csmar.com/)
  - description: Provider of fundamental and market data (Jan–Dec 2023) for SSE 50 constituents used to ground asset prices/ratios in the simulator.

- Sina News
  · [Website](https://www.sina.com.cn/)
  - description: News source used to populate the world knowledge database that agents read during simulation.

- 10jqka (iFinD/Hexun Tonghuashun)
  · [Website](https://www.10jqka.com.cn/)
  - description: Additional financial news source integrated into the world knowledge database for agent perception.

- CNINFO (巨潮资讯网)
  · [Website](http://www.cninfo.com.cn/new/index.jsp)
  - description: Official company announcements feed used as structured information input to agents.

- Shanghai Stock Exchange — SSE 50 Index
  · [Website](https://www.sse.com.cn/market/sseindex/overview/)
  - description: Official reference for the SSE 50 constituents; TwinMarket focuses on these blue-chip A-share stocks and also builds 10 aggregated sector indices from them.

- OpenAI GPT-4o
  · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Primary LLM backbone powering agents in the main simulations; accessed via API for reasoning, posting, and BDI updates.

- Google Gemini 1.5 Flash
  · [Doc](https://ai.google.dev/gemini-api/docs/models/gemini)
  - description: Alternative LLM backbone used in reproducibility/stability experiments to show TwinMarket’s robustness across models.

<!-- paper_id: e21626e7d2ccf270e0d26f360082791f7d88a045 -->

## 141. AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems - NeurIPS - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_NeurIPS_14_Poster_AgentNet_Decentralized_Evolutionary_Coordination_for_LLM-based_Multi-Agent_Systems.pdf
- Link: https://openreview.net/pdf/38bdf6e7191adba7391b1fde1ad37e27887b2bac.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_NeurIPS_14_Poster_AgentNet_Decentralized_Evolutionary_Coordination_for_LLM-based_Multi-Agent_Systems.pdf
- Token Usage: input 23784, output 5901, total 29685

### GitHub & Websites

- AgentNet (official release) 
  - description: The paper’s proposed decentralized multi-agent framework. The authors state “The code will be made public upon acceptance,” but no URL is provided in the PDF.

- MATH (Measuring Mathematical Problem Solving) · [GitHub](https://github.com/hendrycks/math)
  - description: Math word problem dataset used to evaluate AgentNet and baselines on mathematics tasks; the paper constructs specific train/test splits from MATH.

- BIG-bench Hard (BBH) · [GitHub](https://github.com/suzgunmirac/BIG-bench-hard)
  - description: Logical reasoning benchmark used for training/validation and analysis (e.g., router ablations, heterogeneity, scalability) in the paper.

- API-Bank · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
  - description: Function-calling/tool-use benchmark used to test API/tool-augmented planning; the paper samples training and test tasks from API-Bank and annotates categories/difficulties.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting prompting framework; AgentNet’s agents adopt ReAct-style reasoning/acting and ReAct is also used as a single-agent baseline.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Website](https://metagpt.ai/)
  - description: Centralized multi-agent software engineering framework used as a multi-agent comparison baseline.

- GPTSwarm · [GitHub](https://github.com/VITA-Group/GPTSwarm)
  - description: Agents-as-graphs framework with optimizable collaboration patterns; used as a multi-agent comparison baseline.

- AFLOW
  - description: Automated agentic workflow generation framework; used as a multi-agent comparison baseline in the experiments.

- MorphAgent
  - description: Multi-agent framework with self-evolving agent profiles; used as a multi-agent comparison baseline.

- BAAI/bge-large-en-v1.5 · [Website](https://huggingface.co/BAAI/bge-large-en-v1.5)
  - description: Sentence embedding model used by the authors to compute similarity for RAG memory retrieval in AgentNet.

- OpenAI GPT-4o-mini · [Doc](https://platform.openai.com/docs/models#gpt-4o-mini)
  - description: One of the backbone LLMs used in experiments across tasks.

- DeepSeek-V3 · [Doc](https://api-docs.deepseek.com)
  - description: Another backbone LLM used in the evaluations reported in the paper.

- Qwen-turbo · [Doc](https://qwen.readthedocs.io/en/latest/)
  - description: Backbone LLM from the Qwen family used in the paper’s comparisons.

- AgentScope · [GitHub](https://github.com/modelscope/agentscope)
  - description: Modular multi-agent platform cited in related work; included as a relevant toolkit practitioners may inspect alongside the paper’s decentralized approach.

<!-- paper_id: 8de70eaa909fef2adb6d8e15bd7178b7d0fd9749 -->

## 142. MegaAgent: A Large-Scale Autonomous LLM-based Multi-Agent System Without Predefined SOPs - ACL - 2025 - citation_count 10 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_10_Findings_MegaAgent_A_Large-Scale_Autonomous_LLM-based_Multi-Agent_System_Without_Predefined_SOPs.pdf
- Link: https://aclanthology.org/2025.findings-acl.259/
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_10_Findings_MegaAgent_A_Large-Scale_Autonomous_LLM-based_Multi-Agent_System_Without_Predefined_SOPs.pdf
- Token Usage: input 43632, output 3536, total 47168

### GitHub & Websites

- MegaAgent · [GitHub](https://github.com/Xtra-Computing/MegaAgent)
  - description: Official code release of the paper’s large-scale autonomous LLM-based multi-agent system used in all experiments (Gobang development, national policy simulation, and benchmarks).

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework used as a baseline in both Gobang and national policy experiments.

- MetaGPT · [GitHub](https://github.com/metagpt-dev/MetaGPT)
  - description: Meta-programming multi-agent framework evaluated as a baseline; authors report results with MetaGPT v0.8.1.

- CAMEL · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel) · [Website](https://www.camel-ai.org) · [Doc](https://camel-ai.github.io/camel/)
  - description: Communicative agents framework used as a baseline across tasks (benchmarks, Gobang, policy simulation).

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse) · [Doc](https://openbmb.github.io/agentverse/)
  - description: Multi-agent collaboration toolkit used as a baseline for Gobang and national policy generation.

- Chroma (Vector Database) · [GitHub](https://github.com/chroma-core/chroma) · [Codewiki](https://codewiki.google/github.com/chroma-core/chroma) · [Website](https://www.trychroma.com) · [Doc](https://docs.trychroma.com)
  - description: Vector database used in MegaAgent for long-term memory storage and retrieval across agents.

- MBPP (Mostly Basic Python Problems) · [GitHub](https://github.com/google-research/google-research/tree/master/mbpp)
  - description: Programming benchmark used to evaluate MegaAgent’s foundational performance.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code generation benchmark used for baseline comparison and MegaAgent evaluation.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: Mathematical reasoning dataset used to assess MegaAgent on standard benchmarks.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Grade-school math word problem benchmark used to evaluate reasoning capability.

- QMSum · [GitHub](https://github.com/Yale-LILY/QMSum)
  - description: Meeting summarization dataset used to source non-policy negative samples in the LLM-as-a-Judge validation of policy reasonableness.

- MT-Bench Human Judgments · [Doc](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
  - description: Human-judged multi-turn conversation dataset used to construct additional non-policy negative samples for validation.

- USA.gov (Policy Sources) · [Website](https://www.usa.gov)
  - description: Official U.S. government portal used to collect authentic national policies for the policy validation dataset.

- GOV.UK (Policy Sources) · [Website](https://www.gov.uk)
  - description: U.K. government policy portal used to gather real policy documents for validation.

- World Health Organization (Policy Sources) · [Website](https://www.who.int)
  - description: WHO portal used to source public health policies included in the validation dataset.

<!-- paper_id: d5b847411bf80ccaf1314476f2dbb8e2623fe0c9 -->

## 143. MasRouter: Learning to Route LLMs for Multi-Agent Systems - ACL - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_15_Long_MasRouter_Learning_to_Route_LLMs_for_Multi-Agent_Systems.pdf
- Link: https://aclanthology.org/2025.acl-long.757/
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_15_Long_MasRouter_Learning_to_Route_LLMs_for_Multi-Agent_Systems.pdf
- Token Usage: input 30049, output 6023, total 36072

### GitHub & Websites

- MasRouter · [GitHub](https://github.com/yanweiyue/masrouter)
  - description: Official code release of the paper; implements the cascaded controller (collaboration determiner, role allocator, LLM router) and experiments across benchmarks.

- MMLU · [GitHub](https://github.com/hendrycks/test) · [Website](https://people.eecs.berkeley.edu/~hendrycks/) · [Doc](https://arxiv.org/abs/2009.03300)
  - description: Benchmark used for evaluation; the paper reports accuracy and cost on MMLU.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math) · [Doc](https://arxiv.org/abs/2110.14168)
  - description: Math word-problem dataset used for evaluation and ablations.

- MATH · [GitHub](https://github.com/hendrycks/math) · [Doc](https://arxiv.org/abs/2103.03874)
  - description: Mathematical reasoning benchmark; the paper samples problems and reports performance/cost.

- HumanEval · [GitHub](https://github.com/openai/human-eval) · [Doc](https://arxiv.org/abs/2107.03374)
  - description: Code-generation benchmark used for performance and cost comparisons.

- MBPP · [GitHub](https://github.com/google-research/google-research/tree/master/mbpp) · [Doc](https://arxiv.org/abs/2108.07732)
  - description: Python coding problems benchmark; used to compare against MAS and routing baselines.

- Sentence-BERT · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net/) · [Doc](https://arxiv.org/abs/1908.10084)
  - description: Encoder used in the collaboration determiner to extract query semantics.

- MiniLM (UniLM) · [GitHub](https://github.com/microsoft/unilm) · [Codewiki](https://codewiki.google/github.com/microsoft/unilm) · [Doc](https://arxiv.org/abs/2002.10957)
  - description: Lightweight encoder listed as an alternative module encoder in the collaboration determiner.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion) · [Doc](https://arxiv.org/abs/2303.11366)
  - description: Included in the paper’s collaboration-modes repository; used as one of the MAS reasoning modes.

- RouteLLM · [GitHub](https://github.com/lm-sys/RouteLLM) · [Website](https://lmsys.org) · [Doc](https://arxiv.org/abs/2406.18665)
  - description: LLM routing baseline compared against MasRouter.

- FrugalGPT · [GitHub](https://github.com/stanford-futuredata/FrugalGPT) · [Doc](https://arxiv.org/abs/2305.05176)
  - description: Cost-aware LLM routing baseline included in comparisons.

- GraphRouter · [GitHub](https://github.com/snap-stanford/GraphRouter) · [Website](https://snap.stanford.edu) · [Doc](https://arxiv.org/abs/2410.03834)
  - description: Routing framework referenced; the paper follows GraphRouter to build LLM profiles and compares with its PromptLLM baseline.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://microsoft.github.io/autogen/) · [Doc](https://arxiv.org/abs/2308.08155)
  - description: Multi-agent conversation framework cited as a representative MAS platform.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Doc](https://arxiv.org/abs/2308.00352)
  - description: Multi-agent collaborative framework cited in related work for MAS.

- AgentGPT · [GitHub](https://github.com/reworkd/AgentGPT)
  - description: Cited agent framework; example of LLM-based agents discussed in the introduction.

- Auto-GPT · [GitHub](https://github.com/Significant-Gravitas/Auto-GPT)
  - description: Cited autonomous agent project; representative baseline framework in the agent ecosystem.

- OpenAI GPT-4o mini · [Website](https://openai.com/index/gpt-4o-mini/) · [Doc](https://platform.openai.com/docs/models#gpt-4o-mini)
  - description: One of the LLM backbones in the routing pool used throughout experiments.

- Anthropic Claude 3.5 Haiku · [Website](https://www.anthropic.com) · [Doc](https://docs.anthropic.com/claude/docs)
  - description: LLM backbone evaluated in the pool for routing and performance-cost analysis.

- Google Gemini 1.5 Flash · [Website](https://deepmind.google/technologies/gemini/) · [Doc](https://ai.google.dev/gemini-api)
  - description: LLM backbone included in the candidate pool and used in baseline comparisons.

- Llama 3.1 70B Instruct · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: Open LLM backbone used as part of the routing pool.

- DeepSeek-V3 · [Website](https://www.deepseek.com/) · [Doc](https://api-docs.deepseek.com)
  - description: Unseen LLM added to test MasRouter’s inductive generalization; selection distribution and gains reported.

<!-- paper_id: 086830da14fbf4fccfb3197d97b7b01adf7014a6 -->

## 144. ToolPlanner: A Tool Augmented LLM for Multi Granularity Instructions with Path Planning and Feedback - EMNLP - 2024 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_EMNLP_16_Main_ToolPlanner_A_Tool_Augmented_LLM_for_Multi_Granularity_Instructions_with_Path_Planning_and_Feedback.pdf
- Link: https://aclanthology.org/2024.emnlp-main.1018/
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_EMNLP_16_Main_ToolPlanner_A_Tool_Augmented_LLM_for_Multi_Granularity_Instructions_with_Path_Planning_and_Feedback.pdf
- Token Usage: input 33253, output 3954, total 37207

### GitHub & Websites

- ToolPlanner / MGToolBench · [GitHub](https://github.com/XiaoMi/toolplanner)
  - description: Official code and data release for the paper; includes the ToolPlanner two-stage RL framework, prompts, training scripts, evaluation, and the MGToolBench multi-granularity instruction dataset constructed from ToolBench.

- ToolBench (ToolLLM) · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Public tool-use dataset and framework the paper builds upon; the authors use the G3 split as seed data, the official 100-task test set, DFSDT tree decoding, and ToolEval for Win Rate evaluation; also provides ToolLLaMA baseline code.

- Meta LLaMA · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Backbone model used for ToolPlanner (LLaMA-7B); repository contains inference/fine-tuning code and model access instructions.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Chain-based reasoning-and-acting method used to implement the chain decoding baseline (CoT@N) for comparisons in the paper.

- Sentence-Transformers (Sentence-BERT) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Doc](https://www.sbert.net/)
  - description: Dense retriever baseline used in the ablation comparing tag extraction vs. retrieval for selecting candidate tools/APIs.

- Self-Instruct · [GitHub](https://github.com/yizhongw/self-instruct)
  - description: Instruction-generation procedure the authors follow to create multi-granularity prompts (statement/category/tool/API) for MGToolBench.

- RapidAPI Hub · [Website](https://rapidapi.com/hub)
  - description: API marketplace from which ToolBench collected its 16k+ real-world APIs; relevant source if extending the tool pool used by ToolPlanner/ToolBench.

<!-- paper_id: ace1a82d97d024c26e588cef084dcb322f157811 -->

## 145. Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models - ICLR - 2025 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_18_Poster_Hypothetical_Minds_Scaffolding_Theory_of_Mind_for_Multi-Agent_Tasks_with_Large_Language_Models.pdf
- Link: https://openreview.net/pdf/b85a375d8c5672bb041b204393e3cab2f1552cac.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_18_Poster_Hypothetical_Minds_Scaffolding_Theory_of_Mind_for_Multi-Agent_Tasks_with_Large_Language_Models.pdf
- Token Usage: input 35595, output 4403, total 39998

### GitHub & Websites

- Hypothetical Minds · [GitHub](https://github.com/locross93/Hypothetical-Minds/)
  - description: Official code release for the paper; contains the LLM-agent implementation, prompts, ToM module, subgoal/action planner, and experiment scripts to reproduce results.

- Melting Pot 2.0 · [GitHub](https://github.com/google-deepmind/meltingpot) · [Doc](https://meltingpot.readthedocs.io/en/latest/)
  - description: Multi-agent evaluation benchmark used for all environments in the paper (Running With Scissors, Running With Scissors Arena, Prisoner’s Dilemma, Collaborative Cooking Asymmetric).

- Melting Pot Contest 2023 Starter Code · [GitHub](https://github.com/rstrivedi/Melting-Pot-Contest-2023)
  - description: PPO starter pipeline for Melting Pot that the authors used to implement and train their PPO baseline.

- Ray RLlib · [GitHub](https://github.com/ray-project/ray) · [Codewiki](https://codewiki.google/github.com/ray-project/ray) · [Doc](https://docs.ray.io/en/latest/rllib/index.html)
  - description: RL library used to train the PPO baseline agents reported in the paper.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct) · [Website](https://react-lm.github.io/)
  - description: Reasoning-and-acting agent framework used as a baseline; prompts/logic adapted to the paper’s embodied multi-agent settings.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Self-reflection/evaluation framework used as a baseline; also integrated for self-reflection in the Collaborative Cooking experiments.

- OpenAI API (GPT-4, GPT-3.5) · [GitHub](https://github.com/openai/openai-python) · [Codewiki](https://codewiki.google/github.com/openai/openai-python) · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Commercial LLMs used as the backbone in most experiments (model names given in the paper), accessed via the OpenAI API/SDK.

- Meta Llama 3 (Meta-Llama-3-70B-Instruct) · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)
  - description: Open-weight LLM used in ablations to replace GPT-4/3.5 and evaluate model sensitivity to the base LLM.

- Generative Agents · [GitHub](https://github.com/joonspk-research/generative_agents) · [Codewiki](https://codewiki.google/github.com/joonspk-research/generative_agents) · [Website](https://generativeagents.dev/)
  - description: Prior cognitive-agent architecture that Hypothetical Minds builds upon (modules for perception, memory, and planning), extended here with a new Theory-of-Mind module.

<!-- paper_id: b2469523d558a9f3efb9eb4b9184b09925dc7dbc -->

## 146. Ponder & Press: Advancing Visual GUI Agent towards General Computer Control - ACL - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_11_Findings_Ponder_&_Press_Advancing_Visual_GUI_Agent_towards_General_Computer_Control.pdf
- Link: https://aclanthology.org/2025.findings-acl.76/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_11_Findings_Ponder_&_Press_Advancing_Visual_GUI_Agent_towards_General_Computer_Control.pdf
- Token Usage: input 18013, output 5789, total 23802

### GitHub & Websites

- Qwen2-VL / Qwen2-VL-Instruct · [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://qwenlm.github.io) · [Doc](https://github.com/QwenLM/Qwen2-VL#readme)
  - description: Base multimodal model used as the Visual Element Locator; the paper fine-tunes Qwen2‑VL‑Instruct with LoRA on GUI data to predict on-screen coordinates.

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)
  - description: Parameter-efficient fine-tuning method applied to both the visual encoder and language model layers when adapting Qwen2‑VL to GUI grounding.

- ScreenSpot (GUI grounding benchmark)
  - description: Benchmark used to evaluate GUI element localization across mobile/desktop/web; the paper reports state-of-the-art results of its Locator on ScreenSpot.

- SeeClick (GUI grounding dataset/model)
  - description: Source of the 1M GUI grounding training set; the paper fine-tunes its Locator on a 2.5% subset and also compares against the SeeClick model as a baseline.

- Multimodal-Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web)
  - description: Web agent benchmark used for offline evaluation (element accuracy and step success rate) under Cross-Task/Website/Domain splits.

- OmniACT · [Website](https://arxiv.org/abs/2402.17553)
  - description: Desktop and web GUI agent benchmark; the paper evaluates sequence/action scores and discusses issues in the official evaluation script.

- OSWorld · [GitHub](https://github.com/xlang-ai/OSWorld) · [Website](https://os-world.github.io)
  - description: Real computer environment and execution-based benchmark; used for online evaluation where the agent formats pyautogui code to act.

- AndroidWorld · [GitHub](https://github.com/microsoft/AndroidWorld) · [Website](https://arxiv.org/abs/2405.14573)
  - description: Interactive Android environment benchmark; the paper evaluates a vision-only agent and reports success rates vs. M3A.

- OmniParser · [GitHub](https://github.com/microsoft/OmniParser)
  - description: Commercial GUI locator baseline (GPT‑4V + Grounding DINO) compared against in grounding and agent tasks.

- CogAgent · [GitHub](https://github.com/THUDM/CogAgent)
  - description: GUI-specialized multimodal model used as a grounding/agent baseline in comparisons.

- Grounding DINO · [GitHub](https://github.com/IDEA-Research/GroundingDINO)
  - description: Open-set detector used by the OmniParser baseline referenced in the paper.

- GPT‑4o · [Website](https://openai.com/index/gpt-4o) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Commercial MLLM used as an instruction interpreter in the Ponder stage and as a single-stage baseline.

- Claude 3.5 Sonnet · [Website](https://www.anthropic.com/news/claude-3-5-sonnet) · [Doc](https://docs.anthropic.com/en/docs/models/claude-3-5)
  - description: Commercial MLLM used as an instruction interpreter in the Ponder stage; shown to provide stronger element descriptions and actions.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai)
  - description: Inference serving stack used to deploy the Qwen2‑VL‑7B Locator for latency measurements.

- CLIP · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Used to compute embeddings and filter icons by cosine similarity when constructing the paper’s OOD variant of ScreenSpot (ScreenSpot‑P&P‑OOD).

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui) · [Doc](https://pyautogui.readthedocs.io)
  - description: Automation library used to format and execute action triplets into runnable code in the OSWorld environment.

<!-- paper_id: 52d0be520845f45cac46ddcf3764ac84ff04625c -->

## 147. MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems - ICML - 2025 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICML_16_Poster_MAS-GPT_Training_LLMs_to_Build_LLM-based_Multi-Agent_Systems.pdf
- Link: https://openreview.net/pdf/1f479b93fec4660d50da88780197574d2cb64155.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICML_16_Poster_MAS-GPT_Training_LLMs_to_Build_LLM-based_Multi-Agent_Systems.pdf
- Token Usage: input 30318, output 4943, total 35261

### GitHub & Websites

- MAS-GPT · [GitHub](https://github.com/rui-ye/MAS-GPT)
  - description: Official code release of the paper, providing the MAS generation model, training scripts, and executable MAS examples used in all experiments.

- Qwen2.5-Coder-32B-Instruct · [Website](https://qwenlm.github.io/) · [Doc](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
  - description: The open-source LLM used as the base model for supervised fine-tuning to create MAS-GPT.

- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory) · [Codewiki](https://codewiki.google/github.com/hiyouga/LLaMA-Factory) · [Doc](https://llamafactory.readthedocs.io/)
  - description: Training framework used to fine-tune the base LLM; the paper cites using standard SFT via this tool.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://microsoft.github.io/autogen/)
  - description: Multi-agent framework whose evaluation prompts are reused in the paper’s LLM-based answer extraction and judging.

- Llama 3 (Llama‑3‑70B‑Instruct) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: One of the MAS-driving LLMs used during dataset construction and testing for baseline and MAS execution.

- Qwen2.5‑72B‑Instruct · [Website](https://qwenlm.github.io/) · [Doc](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
  - description: MAS-driving LLM used in testing to demonstrate MAS-GPT’s compatibility across backbones.

- GPT‑4o mini (2024‑07‑18) · [Website](https://platform.openai.com/docs/models#gpt-4o-mini)
  - description: Proprietary MAS-driving model used in experiments to evaluate cross-backbone performance.

- OpenAI o1‑preview · [Website](https://openai.com/index/introducing-openai-o1-preview/)
  - description: Advanced reasoning model used as the MAS-driving LLM to test whether MAS-GPT can further boost strong reasoners.

- DeepSeek‑R1 · [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) · [Codewiki](https://codewiki.google/github.com/deepseek-ai/DeepSeek-R1) · [Website](https://www.deepseek.com/)
  - description: Open-source reasoning model used as a MAS-driving LLM in AIME‑2024 evaluations.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT) · [Website](https://metagpt.site/)
  - description: Manually designed multi-agent framework cited as a representative baseline MAS.

- ChatDev · [GitHub](https://github.com/OpenBMB/ChatDev)
  - description: Multi-agent software development system used as a task-specific baseline in comparisons.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Multi-agent collaboration framework used as a dynamic MAS baseline in experiments.

- Self-Refine · [GitHub](https://github.com/madaan/self-refine)
  - description: Iterative self-feedback method re-implemented as one of the baseline MAS designs in the MAS pool.

- MATH · [GitHub](https://github.com/hendrycks/math) · [Doc](https://huggingface.co/datasets/hendrycks/competition_math)
  - description: Training and testing benchmark for mathematical reasoning used to build queries and evaluate MAS-GPT.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math) · [Doc](https://huggingface.co/datasets/gsm8k)
  - description: Grade-school math dataset used for training and evaluation.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code generation benchmark; correctness evaluated via test cases.

- HumanEval+ (EvalPlus) · [GitHub](https://github.com/evalplus/evalplus) · [Website](https://evalplus.github.io/)
  - description: Enhanced HumanEval with additional test cases; used for coding evaluation.

- MMLU · [GitHub](https://github.com/hendrycks/test)
  - description: General knowledge benchmark used both in training (subset) and testing.

- GPQA · [Doc](https://huggingface.co/datasets/Idavidrein/gpqa)
  - description: Graduate-level, Google-proof QA benchmark used for out-of-domain evaluation.

- SciBench · [GitHub](https://github.com/JetRunner/SciBench)
  - description: College-level scientific problem-solving benchmark used for out-of-domain testing.

- AIME‑2024 · [Doc](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)
  - description: Challenging math benchmark used to test MAS-GPT’s ability to augment strong reasoning LLMs.

- MBPP · [GitHub](https://github.com/google-research/google-research/tree/master/mbpp) · [Doc](https://huggingface.co/datasets/mbpp)
  - description: Python program synthesis dataset used in training and evaluation.

- SciQ · [Website](https://allenai.org/data/sciq) · [Doc](https://huggingface.co/datasets/sciq)
  - description: Multiple-choice science QA dataset used for training.

- AQUA-RAT (AQuA) · [GitHub](https://github.com/deepmind/AQuA)
  - description: Algebraic word-problem dataset with rationales used as part of the training query pool.

<!-- paper_id: ad07189387af46e5d5636c0e8d3a7bf81124cc7e -->

## 148. CODESIM: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging - NAACL - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_NAACL_15_Findings_CODESIM_Multi-Agent_Code_Generation_and_Problem_Solving_through_Simulation-Driven_Planning_and_Debugging.pdf
- Link: https://aclanthology.org/2025.findings-naacl.285/
- Tags: multiagent, tool, science, agent-coding
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_NAACL_15_Findings_CODESIM_Multi-Agent_Code_Generation_and_Problem_Solving_through_Simulation-Driven_Planning_and_Debugging.pdf
- Token Usage: input 24371, output 4360, total 28731

### GitHub & Websites

- CODESIM · [GitHub](https://github.com/KagNLP/codesim.github.io) · [Website](https://kagnlp.github.io/codesim.github.io/)
  - description: Official release of the paper’s multi‑agent, simulation-driven planning/coding/debugging framework; the site hosts the open-source code and project page referenced in the paper.

- MapCoder · [GitHub](https://github.com/KagNLP/MapCoder)
  - description: Prior multi-agent baseline used for comparison; the authors state they collected all datasets/evaluation setup from this repository for fair comparison.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Program synthesis benchmark on which CODESIM is evaluated (pass@1), including base tasks and sample I/O used for simulation and testing.

- EvalPlus · [GitHub](https://github.com/evalplus/evalplus) · [Website](https://evalplus.github.io/)
  - description: Extended unit tests for HumanEval/MBPP used in the paper’s evaluation (EvalPlus results reported and used to augment testing).

- MBPP · [GitHub](https://github.com/google-research/google-research/tree/master/mbpp)
  - description: Basic Python programming benchmark used to evaluate CODESIM (including MBPP and MBPP-ET variants reported).

- APPS · [GitHub](https://github.com/hendrycks/apps)
  - description: Competitive programming dataset used for contest-level evaluation of CODESIM.

- CodeContests · [GitHub](https://github.com/deepmind/code_contests)
  - description: DeepMind CodeContests dataset used as a competitive programming benchmark in the paper.

- xCodeEval · [GitHub](https://github.com/microsoft/xCodeEval)
  - description: Multilingual code benchmark used in the paper’s ablation to assess performance across programming languages.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Strong baseline method compared against CODESIM for basic programming tasks and as a seed method in the second-pass debugging analysis.

- LDB (Debug Like a Human) · [GitHub](https://github.com/TIGER-AI-Lab/LDB)
  - description: External LLM-based debugger used by the authors as a second pass to further improve CODESIM outputs; results reported with and without LDB.

<!-- paper_id: 62079734b1c062d294f508cac7cc27e46806f126 -->

## 149. Re-Invoke: Tool Invocation Rewriting for Zero-Shot Tool Retrieval - EMNLP - 2024 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_EMNLP_18_Findings_Re-Invoke_Tool_Invocation_Rewriting_for_Zero-Shot_Tool_Retrieval.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.270/
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2024_EMNLP_18_Findings_Re-Invoke_Tool_Invocation_Rewriting_for_Zero-Shot_Tool_Retrieval.pdf
- Token Usage: input 25663, output 5208, total 30871

### GitHub & Websites

- ToolBench / ToolLLM (includes ToolLLaMA + DFSDT agent) · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Benchmark dataset and agent framework for tool-augmented LLMs; the paper evaluates retrieval on ToolBench subsets and uses ToolLLaMA with DFSDT as the agent for end-to-end tests.

- ToolE (MetaTool Benchmark) · [Website](https://arxiv.org/abs/2310.03128)
  - description: Benchmark that provides single-tool and multi-tool query–tool pairs; used as a primary evaluation dataset for retrieval experiments in the paper.

- scikit-learn · [Website](https://scikit-learn.org/) · [Doc](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
  - description: Library whose nDCG@k implementation is used to evaluate retrieval performance.

- TensorFlow Ranking · [Website](https://www.tensorflow.org/ranking) · [Doc](https://www.tensorflow.org/ranking/api_docs/python/tfr/keras/metrics/RecallMetric)
  - description: Used for recall@k evaluation via the RecallMetric API cited in the paper.

- Vertex AI Text Embeddings (textembedding-gecko) · [Doc](https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-text-embeddings)
  - description: The dense retriever backbone used in experiments to encode tool docs and queries/intents.

- Vertex AI Text Models (text-bison) · [Doc](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text)
  - description: LLM used to generate synthetic queries (query generator) and extract intents (intent extractor), and to create hypothetical documents for the HyDE baseline.

- OpenAI GPT-3.5 Turbo · [Doc](https://platform.openai.com/docs/models/gpt-3-5-turbo)
  - description: Alternative backbone LLM used to replicate Re-Invoke results.

- Mistral-7B-Instruct-v0.3 · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  - description: Open-source backbone LLM variant evaluated as an alternative for Re-Invoke’s query generator and intent extractor.

- Gemini API · [Doc](https://ai.google.dev/gemini-api)
  - description: Mentioned as a compatible LLM option for the query generator/intent extractor components of Re-Invoke.

<!-- paper_id: 078bedf41f7ff2850bd52f39e8c0b3b239f709d5 -->

## 150. CodeScientist: End-to-End Semi-Automated Scientific Discovery with Code-based Experimentation - ACL - 2025 - citation_count 19 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_ACL_19_Findings_CodeScientist_End-to-End_Semi-Automated_Scientific_Discovery_with_Code-based_Experimentation.pdf
- Link: https://aclanthology.org/2025.findings-acl.692/
- Tags: multiagent, tool, science, agent-coding
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-coding/2025_ACL_19_Findings_CodeScientist_End-to-End_Semi-Automated_Scientific_Discovery_with_Code-based_Experimentation.pdf
- Token Usage: input 66858, output 5675, total 72533

### GitHub & Websites

- CodeScientist · [GitHub](https://github.com/allenai/codescientist)
  - description: Official open-source implementation of the CodeScientist system introduced in the paper; includes the end-to-end pipeline, codeblock library, instrumented execution sandbox, prompts, and example experiments used for the reported results.

- TextWorldExpress · [GitHub](https://github.com/allenai/textworld-express) · [Doc](https://pypi.org/project/textworld-express/)
  - description: High-performance text-game simulator used for many experiments (e.g., CookingWorld state prediction, action prediction); the paper’s code imports the TextWorldExpress API to generate environments and episodes.

- DiscoveryWorld · [GitHub](https://github.com/allenai/discoveryworld)
  - description: Virtual environment for automated scientific discovery; used in the “Graph Agent for Discovery” experiments via DiscoveryWorldAPI to evaluate knowledge-graph-augmented agents.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld)
  - description: Text-based science environment referenced for several experiment ideas and baselines (e.g., affordance/agent evaluations), indicating relevance for extending the paper’s evaluations.

- ReAct (Synergizing Reasoning and Acting) · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Baseline/agent template repeatedly used as a codeblock and comparison point (e.g., modified ReAct agents, goal tracking, spatial memory) within the system’s experiments.

- ConceptNet 5 · [GitHub](https://github.com/commonsense/conceptnet5) · [Website](https://conceptnet.io) · [Doc](https://github.com/commonsense/conceptnet5/wiki)
  - description: Commonsense knowledge graph referenced and used in codeblocks (e.g., knowledge-graph memory/lookup) to augment agents and analyses.

- WordNet · [Website](https://wordnet.princeton.edu) · [Doc](https://wordnet.princeton.edu/documentation)
  - description: Lexical knowledge base used in a WordNet-based agent baseline; relevant for reproducing knowledge-driven agent variants and metrics.

- OpenAI API (GPT-4o-mini, etc.) · [Doc](https://platform.openai.com/docs)
  - description: Primary LLM used by the experiment builder and experiments (e.g., gpt-4o-mini for predictions, LLM-as-a-judge scoring); the sandbox proxies these API calls and enforces cost limits.

- Anthropic Claude API (Claude 3.5 Sonnet) · [Website](https://www.anthropic.com) · [Doc](https://docs.anthropic.com/claude)
  - description: Base model used to run the CodeScientist system’s internal stages (ideation/planning/builder); relevant for reproducing system behavior.

- Together.ai API · [Website](https://www.together.ai) · [Doc](https://docs.together.ai/docs)
  - description: Additional LLM provider supported by the instrumented sandbox proxy mentioned in the system design; useful for extending experiments to alternative models.

- NetworkX · [GitHub](https://github.com/networkx/networkx) · [Doc](https://networkx.org/documentation/stable/)
  - description: Graph library used in experiments that construct and analyze knowledge graphs (e.g., graph-based agents, text–graph alignment metric).

- Graphviz (DOT) · [Website](https://graphviz.org) · [Doc](https://graphviz.org/documentation/)
  - description: Used for rendering/exporting DOT graphs generated by graph-based agents and analysis utilities in the experiments.

- Matplotlib · [GitHub](https://github.com/matplotlib/matplotlib) · [Website](https://matplotlib.org/stable/)
  - description: Plotting library referenced in codeblocks and used for visualizations (e.g., ROC curves, scatter plots, metric distributions) throughout the experiments.

- scikit-learn · [GitHub](https://github.com/scikit-learn/scikit-learn) · [Codewiki](https://codewiki.google/github.com/scikit-learn/scikit-learn) · [Doc](https://scikit-learn.org/stable/)
  - description: Used for evaluation utilities such as ROC curve and AUC in the state-prediction confidence analysis and related metrics.

- SciPy · [GitHub](https://github.com/scipy/scipy) · [Codewiki](https://codewiki.google/github.com/scipy/scipy) · [Doc](https://docs.scipy.org/doc/scipy/)
  - description: Statistical tools (e.g., Pearson correlation) used to analyze confidence–accuracy relationships and other experiment metrics.

- pandas · [GitHub](https://github.com/pandas-dev/pandas) · [Codewiki](https://codewiki.google/github.com/pandas-dev/pandas) · [Doc](https://pandas.pydata.org/docs/)
  - description: Data handling/analysis in experiments (e.g., summarizing generation methods, computing summary statistics and tables).

- seaborn · [GitHub](https://github.com/mwaskom/seaborn) · [Codewiki](https://codewiki.google/github.com/mwaskom/seaborn) · [Doc](https://seaborn.pydata.org/)
  - description: Statistical data visualization used in plotting distributions and comparative figures in code-generated experiment reports.

<!-- paper_id: 5bbc93573edc73b360f76ae3ba87fe698188ab7c -->

## 151. MMAU: A Holistic Benchmark of Agent Capabilities Across Diverse Domains - NAACL - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_11_Findings_MMAU_A_Holistic_Benchmark_of_Agent_Capabilities_Across_Diverse_Domains.pdf
- Link: https://aclanthology.org/2025.findings-naacl.267/
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_NAACL_11_Findings_MMAU_A_Holistic_Benchmark_of_Agent_Capabilities_Across_Diverse_Domains.pdf
- Token Usage: input 29798, output 5618, total 35416

### GitHub & Websites

- MMAU Benchmark (this paper)
  - description: Official dataset and evaluation scripts for the benchmark; the paper states “The supplementary materials include both datasets and evaluation scripts for MMAU.” Used to run all 20 offline tasks across five domains.

- CodeContests · [GitHub](https://github.com/deepmind/code_contests)
  - description: Competitive programming dataset used to build the Contest-level Coding tasks, including E2E standard, planner-shift, solver-shift, problem parsing, and self-correction.

- DeepMind Mathematics Dataset · [GitHub](https://github.com/deepmind/mathematics_dataset)
  - description: Source of 1,000 math problems across 56 subjects for the Math domain; also used to create the Comprehend+ subset and the planner-/solver-shift tasks.

- Meta Kaggle (Code/Notebooks) · [Website](https://www.kaggle.com/datasets/kaggle/meta-kaggle) · [Doc](https://www.kaggle.com/docs)
  - description: Kaggle resource from which the authors curated notebook-style conversations for the Data Science & Machine Learning coding tasks and associated QA.

- RapidAPI Hub · [Website](https://rapidapi.com/hub) · [Doc](https://docs.rapidapi.com/)
  - description: API catalog used to construct the in-house tool-use dataset; provides real function endpoints, arguments, and returns for single- and multi-step tool-use, DAG-QA, and tool-use self-correction tasks.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm)
  - description: High-throughput LLM inference engine used to host and evaluate open-source models reproducibly in the paper’s experiments.

- Hugging Face Hub · [Website](https://huggingface.co) · [Doc](https://huggingface.co/docs)
  - description: Model hub from which the authors pulled checkpoints for open-source models evaluated on MMAU.

- Command R · [Doc](https://docs.cohere.com/) · [Website](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
  - description: Baseline LLM evaluated in MMAU; the paper used the official tool-use prompt templates from its repository for tool-calling experiments.

- Command R+ · [Website](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
  - description: Stronger variant of Command R; also evaluated and its official tool-use prompts were used for tool-calling tasks.

- Hermes-2-Pro-Mistral-7B · [Website](https://huggingface.co/NousResearch/Hermes-2-Pro-Mistral-7B)
  - description: Open-source chat model evaluated on MMAU; the paper explicitly used its official prompts for tool-use evaluation.

- OpenAI Models (GPT-4 family) · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs/models)
  - description: Commercial API models (e.g., GPT-4o, GPT-4-Turbo) used across domains; also provide dedicated tool-use/function-calling APIs used in the tool-use evaluations.

- Google Gemini API · [Website](https://ai.google.dev) · [Doc](https://ai.google.dev/gemini-api/docs)
  - description: Commercial multimodal models (Gemini-1.5-pro/1.0-pro) evaluated on MMAU; used via their API including tool-use support.

- Anthropic Claude 3 · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/claude)
  - description: Commercial models (Opus/Sonnet/Haiku) evaluated on MMAU; used through Claude’s API, which supports tool-use for the tool-calling experiments.

<!-- paper_id: 9532286123b181ec77512fc65cd61a47694600da -->

## 152. LONGAGENT: Achieving Question Answering for 128k-Token-Long Documents through Multi-Agent Collaboration - EMNLP - 2024 - citation_count 13 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_13_Main_LONGAGENT_Achieving_Question_Answering_for_128k-Token-Long_Documents_through_Multi-Agent_Collaboration.pdf
- Link: https://aclanthology.org/2024.emnlp-main.912/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_13_Main_LONGAGENT_Achieving_Question_Answering_for_128k-Token-Long_Documents_through_Multi-Agent_Collaboration.pdf
- Token Usage: input 18412, output 5396, total 23808

### GitHub & Websites

- Needle-in-a-Haystack PLUS · [GitHub](https://github.com/zuucan/NeedleInAHaystack-PLUS)
  - description: Official benchmark released by the paper for long-document QA with single-needle and multi-needle tasks; used throughout to evaluate LONGAGENT up to 128k tokens and to test reasoning while preventing data leakage.

- Needle-in-a-Haystack (original) · [GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
  - description: The original long-context stress test on which the paper builds; referenced as the base design that PLUS extends to multi-needle and diversified settings.

- LongBench · [GitHub](https://github.com/THUDM/LongBench)
  - description: Public long-context benchmark; the paper evaluates all long-document QA tasks from LongBench (NarrativeQA, Qasper, MuSiQue, HotpotQA, 2WikiMQA).

- InfiniteBench · [Website](https://arxiv.org/abs/2311.07622)
  - description: Long-context benchmark featuring the Fake Book QA task; used by the paper to assess performance on very long inputs (up to ~200k tokens).

- FlagEmbedding (BGE m3) · [GitHub](https://github.com/FlagOpen/FlagEmbedding) · [Doc](https://huggingface.co/BAAI/bge-m3)
  - description: Retriever used for the RAG baselines (top-k and multi-round); BGE m3 embeddings power the retrieval component compared against LONGAGENT.

- IRCoT (Interleaving Retrieval with Chain-of-Thought) · [GitHub](https://github.com/allenai/ir-cot)
  - description: Strong multi-round retrieval baseline implemented for comparison with LONGAGENT on long-document QA.

- Llama 2 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Base open-source LLM (LLaMA2-7B) that the paper fine-tunes to instantiate the leader and member agents in LONGAGENT.

- SQuAD · [GitHub](https://github.com/rajpurkar/SQuAD-explorer) · [Website](https://rajpurkar.github.io/SQuAD-explorer/)
  - description: Dataset used to construct the 25k member-agent training set and to source 100 “needles” for single-needle evaluation in Needle-in-a-Haystack PLUS.

- HotpotQA · [GitHub](https://github.com/hotpotqa/hotpot) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used to build 60 needle pairs for the paper’s multi-needle (two-needle) evaluation setting.

- NarrativeQA · [GitHub](https://github.com/deepmind/narrativeqa)
  - description: LongBench task used in evaluation; measures comprehension over long narratives.

- Qasper · [GitHub](https://github.com/allenai/qasper)
  - description: Scientific paper QA dataset included via LongBench evaluation in the paper.

- MuSiQue · [GitHub](https://github.com/stanfordnlp/musique)
  - description: Multi-hop QA dataset used as part of LongBench evaluation in the paper.

- 2WikiMultihopQA · [GitHub](https://github.com/Alab-NII/2wikimultihop)
  - description: Multi-hop QA dataset evaluated through LongBench tasks in the paper.

- FlashAttention · [GitHub](https://github.com/HazyResearch/flash-attention)
  - description: Optimization referenced in the efficiency study; used to compare latency/memory of full attention vs. LONGAGENT’s O(N) chunking pipeline.

- Claude 2.1 · [Website](https://www.anthropic.com/product)
  - description: Commercial long-context model baseline (200k context) evaluated against LONGAGENT.

- GPT-4 Turbo (GPT-4-0125-preview) · [Doc](https://platform.openai.com/docs/models/gpt-4-turbo)
  - description: Commercial long-context model baseline (128k context) used for comparisons with LONGAGENT.

<!-- paper_id: be5658be10be6300967ca4906a8c862a8f76cf30 -->

## 153. Multi-Agent Collaboration via Evolving Orchestration - NeurIPS - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NeurIPS_11_Poster_Multi-Agent_Collaboration_via_Evolving_Orchestration.pdf
- Link: https://openreview.net/pdf/9727f658d788c52f49f12ae4b230baf4cf0d4007.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_NeurIPS_11_Poster_Multi-Agent_Collaboration_via_Evolving_Orchestration.pdf
- Token Usage: input 29946, output 5862, total 35808

### GitHub & Websites

- ChatDev (Puppeteer branch) · [GitHub](https://github.com/OpenBMB/ChatDev/tree/puppeteer)
  - description: Official release of this paper’s orchestration-based multi-agent system; contains code to reproduce results, the orchestrator policy, agent prompts/tools, and evaluation scripts (SRDD data/metrics are hosted in this repo).

- MMLU-Pro · [GitHub](https://github.com/TIGER-AI-Lab/MMLU-Pro) · [Website](https://mmlu.pro)
  - description: Robust multi-task language understanding benchmark used as a closed-domain evaluation dataset in the paper.

- GSM8K (basis for GSM-Hard split) · [GitHub](https://github.com/openai/grade-school-math)
  - description: Grade-school math word problem dataset from which the widely used GSM-Hard split is derived; used for the paper’s closed-domain math reasoning evaluation.

- CommonGen · [GitHub](https://github.com/INK-USC/CommonGen) · [Website](https://inklab.usc.edu/CommonGen/)
  - description: Commonsense composition benchmark; the paper evaluates on the harder CommonGen-Hard subset to measure open-ended generation quality.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld) · [Website](https://alfworld.github.io/)
  - description: Text-to-embodied environment used in the appendix to demonstrate the framework’s generalization to interactive/embodied tasks.

- AI2-THOR · [GitHub](https://github.com/allenai/ai2thor) · [Doc](https://ai2thor.allenai.org/ithor/documentation/)
  - description: Interactive 3D environment that underlies ALFWorld; relevant dependency for reproducing the embodied experiments shown in the appendix.

- Self-Refine · [GitHub](https://github.com/madaan/self-refine)
  - description: Iterative self-correction baseline compared against in the paper’s experiments.

- Llama-3.1-Nemotron-70B-Reward-HF · [Website](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF)
  - description: Hugging Face model used to initialize the orchestrator policy variant, as noted in the implementation details.

- Wikipedia API · [Doc](https://www.mediawiki.org/wiki/API:Main_page)
  - description: Documentation for the WikiSearch tool integrated into agents for information retrieval.

- Bing Web Search API · [Doc](https://learn.microsoft.com/bing-web-search/)
  - description: Documentation for the BingSearch tool used by agents to retrieve web information.

- arXiv API · [Doc](https://arxiv.org/help/api)
  - description: Official API documentation for arXivSearch, one of the external tools integrated into the agent system.

<!-- paper_id: e1938a3fbd505d9666ed252b191aaf96ffd5f0d7 -->

## 154. Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation - ACL - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_11_Long_Magnet_Multi-turn_Tool-use_Data_Synthesis_and_Distillation_via_Graph_Translation.pdf
- Link: https://aclanthology.org/2025.acl-long.1566/
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_11_Long_Magnet_Multi-turn_Tool-use_Data_Synthesis_and_Distillation_via_Graph_Translation.pdf
- Token Usage: input 21782, output 5930, total 27712

### GitHub & Websites

- Berkeley Function Calling Leaderboard (BFCL‑v3) · [Website](https://bfcl.berkeley.edu)
  - description: Primary evaluation benchmark used in the paper; the authors also leverage BFCL‑v3’s multi‑turn function implementations to build their function pool.

- StableToolBench
  - description: Source of verified tool/function implementations used to construct the paper’s ~5K-function pool and categories for graph construction and synthesis.

- ToolQuery
  - description: Multi-turn, multi-step function-calling benchmark used for evaluation; the paper reports main results and emphasizes that its functions are unseen during training.

- Hugging Face TRL (Transformer Reinforcement Learning) · [GitHub](https://github.com/huggingface/trl) · [Codewiki](https://codewiki.google/github.com/huggingface/trl) · [Doc](https://huggingface.co/docs/trl/index)
  - description: Training toolkit used by the authors for SFT and as the base framework for their mDPO stage implementation.

- Qwen2.5‑Coder (7B/14B Instruct) · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Doc](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
  - description: Base models fine-tuned by the authors (full-parameter SFT and LoRA/mDPO) to produce MAGNET‑7B/14B variants.

- Mixtral‑8x7B‑Instruct‑v0.1 · [Doc](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Additional base model used to test the generalization of the MAGNET data/training pipeline.

- Functionary (Functionary‑Small‑v3.1) · [GitHub](https://github.com/MeetKai/functionary)
  - description: Open-source function-calling model used as a comparison baseline on BFCL‑v3.

- APIGen · [GitHub](https://github.com/SalesforceAIResearch/APIGen)
  - description: Public function-calling dataset pipeline used for comparison in ablations (training Qwen2.5‑Coder with APIGen+ToolAce vs. MAGNET data).

- ToolAce
  - description: Public tool-use dataset referenced and combined with APIGen for comparison in ablation training.

- NexusRaven v2 · [Website](https://nexusflow.ai/blogs/ravenv2)
  - description: Related multi-step tool-use benchmark mentioned in related work; provides additional context for tool-use evaluation beyond the datasets directly used.

<!-- paper_id: 4d4c46badc6e84cca48282bd0486733014f0d0b9 -->

## 155. BRIDGE: Bootstrapping Text to Control Time-Series Generation via Multi-Agent Iterative Optimization and Diffusion Modelling - ICML - 2025 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICML_12_Poster_BRIDGE_Bootstrapping_Text_to_Control_Time-Series_Generation_via_Multi-Agent_Iterative_Optimization_and_Diffusion_Modelling.pdf
- Link: https://openreview.net/pdf/c9eebe9ec79971c3504b2a13413d34dfc66ede75.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICML_12_Poster_BRIDGE_Bootstrapping_Text_to_Control_Time-Series_Generation_via_Multi-Agent_Iterative_Optimization_and_Diffusion_Modelling.pdf
- Token Usage: input 36769, output 5168, total 41937

### GitHub & Websites

- TimeCraft (BRIDGE) · [GitHub](https://github.com/microsoft/TimeCraft)
  - description: Official code release for this paper’s BRIDGE framework, including the multi-agent text-template pipeline and the hybrid prototype+text conditioned diffusion model for text-controlled time-series generation.

- GluonTS · [GitHub](https://github.com/awslabs/gluonts) · [Doc](https://ts.gluon.ai/)
  - description: Probabilistic/neural time-series toolkit the authors use as a dataset provider; many in-domain benchmarks (e.g., Electricity, Solar, Traffic, etc.) are obtained from GluonTS.

- Monash Time Series Forecasting Repository · [Website](https://forecastingdata.org/)
  - description: Public dataset archive used to source multiple benchmark time-series datasets for evaluation in the paper.

- Kaggle Wikipedia Web Traffic Time Series Forecasting · [Website](https://www.kaggle.com/c/web-traffic-time-series-forecasting)
  - description: Competition dataset used as the “Web” out-of-domain benchmark for few-shot evaluation.

- FRED-MD (Federal Reserve Economic Data – Macro Database) · [Website](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
  - description: Macroeconomic monthly dataset used as one of the in-domain benchmarks.

- M4 Competition Datasets · [Website](https://www.mcompetitions.unic.ac.cy/the-m4-competition/)
  - description: Standard forecasting benchmarks used for short-term forecasting evaluations (SMAPE/MASE/OWA) on synthetic vs. real data.

- AEMO (Australian Energy Market Operator) Data Portal · [Website](https://aemo.com.au/)
  - description: Source for the Wind power production time series used by the authors (Wind dataset noted as downloaded from the AEMO online platform).

- Chronos (KernelSynth) · [GitHub](https://github.com/amazon-science/chronos-forecasting)
  - description: Time-series foundation model and synthetic data baselines; the paper compares synthetic-data-trained forecasting models against KernelSynth data from Chronos.

- ReAct (Reason+Act prompting) · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Prompting framework the authors adopt to build a single-agent crawler/reasoner for collecting candidate documents and extracting general-purpose text templates in their multi-agent data-prep stage.

- Llama 3 · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: The paper uses Llama3-8B as the text encoder inside the diffusion model for conditioning with refined textual descriptions.

- TimeGAN · [GitHub](https://github.com/jsyoon0823/TimeGAN)
  - description: Baseline time-series generative model used for comparison in fidelity metrics across datasets.

<!-- paper_id: bd2eb51e00392bd8a148ba2f60b2284e3dc6a93a -->

## 156. Infogent: An Agent-Based Framework for Web Information Aggregation - NAACL - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NAACL_15_Findings_Infogent_An_Agent-Based_Framework_for_Web_Information_Aggregation.pdf
- Link: https://aclanthology.org/2025.findings-naacl.318/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NAACL_15_Findings_Infogent_An_Agent-Based_Framework_for_Web_Information_Aggregation.pdf
- Token Usage: input 16196, output 5519, total 21715

### GitHub & Websites

- INFOGENT · [GitHub](https://github.com/gangiswag/infogent)
  - description: Official code release for the paper’s modular web information aggregation framework (Navigator, Extractor, Aggregator) covering both Direct API-Driven and Interactive Visual Access setups.

- AssistantBench
  - [Website](https://assistantbench.github.io)
  - description: Benchmark of realistic, time-consuming web information-seeking tasks; used to evaluate INFOGENT under the Interactive Visual Access setting and to compare against SPA and SEEACT baselines.

- SeeAct · [GitHub](https://github.com/OSU-NLP-Group/SeeAct) · [Website](https://arxiv.org/abs/2401.01614)
  - description: Web task-completion agent that grounds actions on webpage screenshots; INFOGENT augments SeeAct (adding GO BACK and AGGREGATE actions) to build its Navigator in the Interactive Visual Access setting.

- SPA (See-Plan-Act)
  - [Website](https://assistantbench.github.io) · [Doc](https://arxiv.org/abs/2407.15711)
  - description: Information-seeking web agent introduced with AssistantBench; used as the primary comparison baseline for Interactive Visual Access.

- FanOutQA · [Website](https://fanoutqa.github.io)
  - description: Multi-hop, multi-document QA dataset; used to evaluate INFOGENT in both Direct API-Driven and Interactive Visual Access settings.

- FRAMES
  - [Website](https://arxiv.org/abs/2409.12941)
  - description: “Fact, Fetch, and Reason” evaluation benchmark with diverse reasoning types; used to assess INFOGENT under the Direct API-Driven Access setting.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reason+Act prompting framework for tool-using LLM agents; used to implement the tool-based Navigator for Direct API-Driven Access.

- AutoGPT · [GitHub](https://github.com/Significant-Gravitas/AutoGPT) · [Website](https://agpt.co)
  - description: Autonomous agent scaffold; INFOGENT’s Direct API-Driven Navigator is built upon an AutoGPT-style tool-using agent.

- Playwright · [GitHub](https://github.com/microsoft/playwright) · [Codewiki](https://codewiki.google/github.com/microsoft/playwright) · [Website](https://playwright.dev)
  - description: Browser automation framework; used (via the SEEACT tooling) to simulate real web browsing for INFOGENT’s Interactive Visual Access.

- Google Custom Search JSON API · [Doc](https://developers.google.com/custom-search/v1/overview)
  - description: Search API used for the Navigator’s SEARCH tool to retrieve URLs and snippets in the Direct API-Driven Access setup.

- MindSearch · [GitHub](https://github.com/OpenGVLab/MindSearch) · [Doc](https://arxiv.org/abs/2407.20183)
  - description: Multi-agent search framework modeling information seeking via iterative graph construction; used as a baseline for Direct API-Driven Access comparisons.

<!-- paper_id: 164920e94df411242b904e3d841ca029c3cb05eb -->

## 157. OS Agents: A Survey on MLLM-based Agents for Computer, Phone and Browser Use - ACL - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_15_Long_OS_Agents_A_Survey_on_MLLM-based_Agents_for_Computer,_Phone_and_Browser_Use.pdf
- Link: https://aclanthology.org/2025.acl-long.369/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_15_Long_OS_Agents_A_Survey_on_MLLM-based_Agents_for_Computer,_Phone_and_Browser_Use.pdf
- Token Usage: input 40665, output 4905, total 45570

### GitHub & Websites

- OS Agents Survey · [GitHub](https://github.com/os-agent-survey/os-agent-survey.github.io) · [Website](https://os-agent-survey.github.io/)
  - description: Official project page and continuously-updated repository maintained by the authors with curated papers, benchmarks, products, and resources on OS Agents; cited in the paper as the open-source resource accompanying the survey.

- OSWorld · [GitHub](https://github.com/xlang-ai/OSWorld)
  - description: Real-computer evaluation benchmark and platform (Windows/Linux/macOS) for multimodal OS agents; used in the survey’s evaluation section as a key computer-use benchmark.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev/)
  - description: Realistic online web environment and benchmark for autonomous web agents; referenced as a primary browser-based interactive benchmark.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Large-scale dataset and evaluation suite for generalist web agents; cited in the survey among core browser benchmarks (static and live variants).

- VisualWebArena · [GitHub](https://github.com/jykoh/visualwebarena)
  - description: Multimodal extension of WebArena for visual web tasks; included in the survey’s browser platform benchmarks.

- MiniWoB++ · [GitHub](https://github.com/stanfordnlp/miniwob-plusplus)
  - description: Classic suite of synthetic web UI tasks for training/evaluating web agents (incl. MiniWoB/FormWoB); used throughout the literature summarized by the survey for step/task-level evaluation.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Open-domain web shopping environment and benchmark for grounded language agents; listed in the survey’s browser benchmarks.

- Operator (OpenAI) · [Website](https://operator.chatgpt.com/)
  - description: Commercial product highlighted by the survey; an OS/web agent for real-world task execution used for comparison and context in the products section.

- Computer Use (Anthropic) · [Website](https://www.anthropic.com/news/3-5-models-and-computer-use)
  - description: Official product page for Claude’s Computer Use, referenced as a prominent commercial OS agent capable of controlling a user’s computer.

- Apple Intelligence (Apple) · [Website](https://www.apple.com/apple-intelligence/)
  - description: Apple’s system-level AI features (including phone/computer use) noted in the survey’s product analysis of OS agents.

- Project Mariner (Google DeepMind) · [Website](https://deepmind.google/technologies/project-mariner/)
  - description: DeepMind’s browser agent project referenced among recent commercial/industrial OS agent products.

- ChatGPT Memory (OpenAI) · [Website](https://openai.com/index/memory-and-new-controls-for-chatgpt/)
  - description: Official documentation for OpenAI’s memory feature, cited in the survey’s discussion of personalization and self-evolving OS agents.

<!-- paper_id: 255684830fb1eb3bef0986052cc50b4d455913a3 -->

## 158. Reducing Tool Hallucination via Reliability Alignment - ICML - 2025 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICML_16_Poster_Reducing_Tool_Hallucination_via_Reliability_Alignment.pdf
- Link: https://openreview.net/pdf/a6830971d126495933a67327ca0c64b6008cdaae.pdf
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICML_16_Poster_Reducing_Tool_Hallucination_via_Reliability_Alignment.pdf
- Token Usage: input 21681, output 5202, total 26883

### GitHub & Websites

- ToolHallucination (Relign + RelyToolBench) · [GitHub](https://github.com/X-LANCE/ToolHallucination)
  - description: Official release for this paper. Contains the Relign reliability-alignment implementation, the RelyToolBench benchmark and prompts, and scripts to run evaluation metrics (RePR, Benefit-Cost Utility) and hallucination detection.

- ToolBench / ToolLLaMA · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolbench.github.io)
  - description: Large-scale tool-use dataset and training/evaluation framework used to construct training data (10k samples for SFT/DPO), provide baselines (ToolLLaMA-7B), and supply the APIBench OOD test used in this paper’s generalization study.

- StableToolBench
  - description: The benchmark the authors build upon to create RelyToolBench; also provides the evaluation setup followed in the paper.

- DeepSpeed-Chat · [GitHub](https://github.com/microsoft/DeepSpeedExamples) · [Doc](https://www.deepspeed.ai/tutorials/deepspeed-chat/)
  - description: Training framework used for efficient SFT/DPO fine-tuning of LLMs in this work.

- Llama 3.1 (Meta) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  - description: One of the main base models fine-tuned and evaluated with Relign and RelyToolBench.

- Qwen2.5 (Alibaba) · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io)
  - description: Another base model fine-tuned and evaluated in the experiments (Qwen2.5-7B-Instruct).

- OpenAI GPT-4o · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed-source model used as the LLM evaluator for hallucination detection and as a comparison baseline in the paper.

<!-- paper_id: 17a1e1a7db6d3694874ee32d84c4825a244241ad -->

## 159. Strength Lies in Differences! Improving Strategy Planning for Non-collaborative Dialogues via Diversified User Simulation - EMNLP - 2024 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2024_EMNLP_14_Main_Strength_Lies_in_Differences!_Improving_Strategy_Planning_for_Non-collaborative_Dialogues_via_Diversified_User_Simulation.pdf
- Link: https://aclanthology.org/2024.emnlp-main.26/
- Tags: multiagent, tool, science, planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2024_EMNLP_14_Main_Strength_Lies_in_Differences!_Improving_Strategy_Planning_for_Non-collaborative_Dialogues_via_Diversified_User_Simulation.pdf
- Token Usage: input 21282, output 4774, total 26056

### GitHub & Websites

- CraigslistBargain (CB) · [GitHub](https://github.com/stanfordnlp/craigslistbargains) · [Website](https://stanfordnlp.github.io/craigslistbargains/)
  - description: Price negotiation benchmark used for evaluation; the paper uses its test split to assess agents’ negotiation success, efficiency, and sale-to-list ratio.

- PersuasionForGood (P4G) · [GitHub](https://github.com/facebookresearch/ParlAI) · [Doc](https://parl.ai/docs/tasks.html#persuasionforgood)
  - description: Charity persuasion benchmark used for evaluation; the paper uses the test set, and the dataset is available via the ParlAI task implementation.

- OpenAI GPT-3.5 / GPT-4 · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs/)
  - description: LLMs used as user simulators, for Theory-of-Mind inference, and as a reward/feedback judge during training and evaluation.

- BERT Base Uncased (google-bert/bert-base-uncased) · [Doc](https://huggingface.co/google-bert/bert-base-uncased)
  - description: Pretrained encoder adopted as the trainable external strategy planner within TRIP (and for the PPDPP baseline implementation).

<!-- paper_id: 84d5d280f726ddf94ba85621581c4fe5f243a04f -->

## 160. Do as We Do, Not as You Think: the Conformity of Large Language Models - ICLR - 2025 - citation_count 13 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/2model-mechanism/2025_ICLR_13_Oral_Do_as_We_Do,_Not_as_You_Think_the_Conformity_of_Large_Language_Models.pdf
- Link: https://openreview.net/pdf/cefcfec072e2db0f4b2196da7ea26a99d12a19a7.pdf
- Tags: multiagent, tool, science, model-mechanism
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/2model-mechanism/2025_ICLR_13_Oral_Do_as_We_Do,_Not_as_You_Think_the_Conformity_of_Large_Language_Models.pdf
- Token Usage: input 47654, output 4569, total 52223

### GitHub & Websites

- BenchForm · [GitHub](https://github.com/Zhiyuan-Weng/BenchForm)
  - description: Official repository for the paper’s BENCHFORM benchmark and code, used to reproduce all protocols, prompts, metrics, and experiments studying conformity in multi‑agent LLMs.

- BIG-bench Hard (BBH) · [GitHub](https://github.com/suzgunmirac/BIG-bench-hard)
  - description: Curated subset of challenging BIG-bench tasks; BENCHFORM derives its reasoning-intensive multiple‑choice problems from BBH.

- BIG-bench · [GitHub](https://github.com/google/BIG-bench)
  - description: The original large language model evaluation benchmark from which BBH is drawn; relevant background and task sources referenced in the paper.

- Ollama · [GitHub](https://github.com/ollama/ollama) · [Codewiki](https://codewiki.google/github.com/ollama/ollama) · [Website](https://ollama.com/) · [Doc](https://ollama.com/docs)
  - description: Runtime used by the authors to obtain and run nine open‑source LLM checkpoints; enables local inference for models like Llama, Gemma, and Qwen.

- OpenAI GPT‑4o ·  [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed‑source model evaluated in the study; specific version gpt‑4o‑0513 noted for reproducibility.

- OpenAI GPT‑3.5 · [Doc](https://platform.openai.com/docs/models/gpt-3-5)
  - description: Closed‑source model evaluated as a baseline; the paper uses gpt‑3.5‑turbo‑16k‑0613.

- Llama 3 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/blog/meta-llama-3/)
  - description: Open LLM family from Meta evaluated across BENCHFORM protocols; models fetched via Ollama.

- Llama 3.1 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/blog/meta-llama-3-1/)
  - description: Updated Meta LLMs (including 70B and 405B via API) assessed in the benchmark; highlighted for differing conformity characteristics.

- Gemma 2 · [GitHub](https://github.com/google-deepmind/gemma) · [Website](https://ai.google.dev/gemma) · [Doc](https://ai.google.dev/gemma/docs)
  - description: Google’s open models tested on BENCHFORM to analyze conformity and independence rates.

- Qwen2 · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://qwenlm.github.io/)
  - description: Alibaba’s open LLM series evaluated in the paper, with detailed conformity behaviors reported.

- GLM‑4 / GLM‑4‑Plus · [GitHub](https://github.com/THUDM/GLM-4) · [Website](https://chatglm.cn) · [Doc](https://open.bigmodel.cn/dev/api)
  - description: Model family from Zhipu AI used in additional experiments (appendix) as a comparison point on BENCHFORM.

- MMLU‑Pro · [GitHub](https://github.com/TIGER-AI-Lab/MMLU-Pro) · [Website](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
  - description: A harder, more robust multi‑task benchmark the authors plan to incorporate in future extensions of BENCHFORM.

<!-- paper_id: d6b5452d5dfc0bdc3e571166ebafc1d86a544bd0 -->

## 161. MMRole: A Comprehensive Framework for Developing and Evaluating Multimodal Role-Playing Agents - ICLR - 2025 - citation_count 13 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICLR_13_Poster_MMRole_A_Comprehensive_Framework_for_Developing_and_Evaluating_Multimodal_Role-Playing_Agents.pdf
- Link: https://openreview.net/pdf/6ea2631589425d5cf3de283aed71b80c80b56b62.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICLR_13_Poster_MMRole_A_Comprehensive_Framework_for_Developing_and_Evaluating_Multimodal_Role-Playing_Agents.pdf
- Token Usage: input 32944, output 4187, total 37131

### GitHub & Websites

- MMRole (MMRole-Agent, MMRole-Data, MMRole-Eval) · [GitHub](https://github.com/YanqiDai/MMRole)
  - description: Official release of this paper containing code, dataset, prompts, trained models, and the reward-model based evaluation needed to reproduce MMRole-Agent, MMRole-Data, and MMRole-Eval.

- Microsoft COCO (MS-COCO) · [GitHub](https://github.com/cocodataset/cocoapi) · [Website](https://cocodataset.org)
  - description: Generic image dataset used to provide diverse visual inputs for each character; required if re-generating or extending the dataset pipeline.

- Qwen-VL · [GitHub](https://github.com/QwenLM/Qwen-VL) · [Doc](https://qwenlm.github.io)
  - description: Open-source vision-language model fine-tuned to build MMRole-Agent and also used to train the specialized reward model; serves as a baseline MRPA (Qwen-VL-Chat).

- LLaVA / LLaVA-NeXT · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io)
  - description: Open-source LMMs evaluated as baselines (LLaVA-NeXT-34B and LLaVA-NeXT-Mistral-7B) in the MMRole-Eval benchmark.

- InternVL · [GitHub](https://github.com/OpenGVLab/InternVL) · [Website](https://internvl.github.io)
  - description: Open-source InternVL-Chat-V1.5 is evaluated as an MRPA baseline.

- Yi / Yi-VL · [GitHub](https://github.com/01-ai/Yi) · [Website](https://01.ai)
  - description: Open foundation models from 01.AI; Yi-VL-34B and Yi-VL-6B are included as open-source MRPA baselines.

- OpenAI GPT-4 / GPT-4 Turbo · [Website](https://openai.com/gpt-4) · [Doc](https://platform.openai.com/docs)
  - description: Used extensively to summarize profiles, generate multimodal dialogues, and provide judging trajectories to train the reward model; also evaluated as a closed-source baseline.

- Google Gemini Pro Vision · [Website](https://ai.google.dev/gemini-api) · [Doc](https://ai.google.dev/gemini-api/docs/vision)
  - description: Closed-source LMM evaluated via official API as a baseline in MMRole-Eval.

- Anthropic Claude 3 Opus · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com)
  - description: Closed-source LMM evaluated via official API as a baseline, achieving top performance among >100B parameter models in the paper.

- Qwen-VL-Max (DashScope) · [Website](https://dashscope.aliyun.com) · [Doc](https://help.aliyun.com/zh/dashscope)
  - description: Closed-source API version of Qwen-VL used as a baseline; accessed through Alibaba Cloud’s DashScope.

- Vicuna · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat) · [Website](https://lmsys.org/blog/2023-03-30-vicuna/)
  - description: Referenced for its evaluation methodology; MMRole-Eval’s reward-model scoring is inspired by Vicuna/LLaVA style comparative judgments.

- Wikipedia · [Website](https://www.wikipedia.org)
  - description: Source used to compile English character profiles that seed MMRole-Data.

- Baidu Baike · [Website](https://baike.baidu.com)
  - description: Source used to compile Chinese character profiles for MMRole-Data.

- BrainyQuote · [Website](https://www.brainyquote.com)
  - description: Referenced during manual quality control to verify and refine authentic catchphrases for character profiles.

<!-- paper_id: ea9e065bcbac0c5d3241d444c80fa60994fee2f7 -->

## 162. Tool-Planner: Task Planning with Clusters across Multiple Tools - ICLR - 2025 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_12_Poster_Tool-Planner_Task_Planning_with_Clusters_across_Multiple_Tools.pdf
- Link: https://openreview.net/pdf/60305c49e9c179b0b6d6201aca2de8a489d664a5.pdf
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_12_Poster_Tool-Planner_Task_Planning_with_Clusters_across_Multiple_Tools.pdf
- Token Usage: input 35547, output 4719, total 40266

### GitHub & Websites

- Tool-Planner · [GitHub](https://github.com/OceannTwT/Tool-Planner)
  - description: Official code release of this paper’s framework for toolkit-based task planning and evaluation.

- ToolBench / ToolLLM (includes ToolEval and DFSDT) · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolllm.github.io)
  - description: Benchmark and tooling with 16,464 real-world APIs sourced from RapidAPI; used as a primary dataset and evaluation suite (ToolEval) and provides the DFSDT baseline implementation used in comparisons.

- APIBench (from Gorilla) · [GitHub](https://github.com/gorilla-llm/gorilla) · [Website](https://gorilla.cs.berkeley.edu)
  - description: Dataset and evaluation harness for API/tool-use (covering TorchHub, HuggingFace, TensorFlow); used to test Tool-Planner and report pass/win/hallucination rates.

- RapidAPI Hub · [Website](https://rapidapi.com/)
  - description: Source platform from which ToolBench APIs are collected; used to simulate real-world API calls.

- SimCSE · [GitHub](https://github.com/princeton-nlp/SimCSE) · [Website](https://simcse.github.io)
  - description: Sentence embedding method used to embed tool descriptions and perform clustering into toolkits.

- OpenAI API (GPT-3.5, GPT-4) · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Foundation models used for plan generation and tool reasoning in experiments.

- Anthropic Claude 3 API · [Website](https://www.anthropic.com) · [Doc](https://docs.anthropic.com/claude)
  - description: Additional foundation model evaluated with Tool-Planner.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Baseline method combining reasoning and acting; used for comparison across datasets.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Baseline with self-reflective feedback; compared against Tool-Planner.

- AdaPlanner
  - description: Adaptive planning baseline compared in the paper; refer to the paper for algorithm details (no official URL provided in the PDF).

- DFSDT (ToolLLM search baseline) · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Deep-first search over decision trees for tool use; Tool-Planner improves upon DFSDT by searching at the toolkit level; implementation available in the ToolBench repo.

- ToolChain* (A* search baseline)
  - description: Heuristic A* search baseline included in appendix comparisons; inspect official repo if needed (URL not provided in the PDF).

- Contriever · [GitHub](https://github.com/facebookresearch/contriever)
  - description: Text embedding model used in ablations for alternative clustering similarity comparisons.

- RoBERTa-base · [GitHub](https://github.com/facebookresearch/fairseq) · [Codewiki](https://codewiki.google/github.com/facebookresearch/fairseq) · [Doc](https://huggingface.co/roberta-base)
  - description: Backbone referenced for the supervised SimCSE variant used to compute tool embeddings.

- Llama 2 (13B) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Llama-2-13b-hf)
  - description: Smaller LLM used in additional experiments for planning/behavior analysis within the Tool-Planner framework.

- PyTorch Hub · [Website](https://pytorch.org/hub/)
  - description: One of the API sources evaluated via APIBench (TorchHub subset).

- Hugging Face Hub · [Website](https://huggingface.co/)
  - description: One of the API sources evaluated via APIBench (HuggingFace subset).

- TensorFlow · [Website](https://www.tensorflow.org/)
  - description: One of the API sources evaluated via APIBench (TensorFlow subset).

<!-- paper_id: 2f764021a7bcde086a33c4706269a0bb06b29edc -->

## 163. G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems - NeurIPS - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_NeurIPS_14_Spotlight_G-Memory_Tracing_Hierarchical_Memory_for_Multi-Agent_Systems.pdf
- Link: https://openreview.net/pdf/52f961783a3212459f228b4ec297f523ba2d0c95.pdf
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_NeurIPS_14_Spotlight_G-Memory_Tracing_Hierarchical_Memory_for_Multi-Agent_Systems.pdf
- Token Usage: input 36663, output 4986, total 41649

### GitHub & Websites

- G-Memory · [GitHub](https://github.com/bingreeky/GMemory)
  - description: Official code release for the paper’s hierarchical memory system for MAS, including prompts and integration hooks for AutoGen, DyLAN, and MacNet to reproduce experiments and extend the method.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Doc](https://microsoft.github.io/autogen/)
  - description: Multi-agent conversation framework used as one of the three MAS backbones; G-Memory is plugged into AutoGen to evaluate memory augmentation.

- DyLAN
  - description: Dynamic LLM-Agent Network used as a second MAS backbone in experiments; the paper plugs G-Memory into DyLAN for debate-style multi-agent evaluation.

- MacNet
  - description: Decentralized multi-agent collaboration framework (with edge agents) used as the third MAS backbone; G-Memory is integrated to test memory benefits under random graph topologies.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT)
  - description: MAS framework with SOP-style workflows; its memory variant (MetaGPT-M) is used as a comparison baseline against G-Memory.

- ChatDev · [GitHub](https://github.com/OpenBMB/ChatDev)
  - description: Software-development MAS framework; its memory variant (ChatDev-M) provides inside- and cross-trial memory baselines for comparison.

- Voyager · [GitHub](https://github.com/MineDojo/Voyager)
  - description: Single-agent embodied agent with evolving memory; adapted by the authors as a memory baseline in MAS settings.

- Generative Agents · [GitHub](https://github.com/joonspk-research/generative_agents) · [Codewiki](https://codewiki.google/github.com/joonspk-research/generative_agents)
  - description: Agent framework featuring observational and reflective memory; used as a single-agent memory baseline adapted to MAS for comparison.

- ALFWorld · [Website](https://alfworld.github.io/)
  - description: Text-based embodied environment for household tasks; used as one of the main benchmarks to assess G-Memory on embodied action.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld)
  - description: Text-based embodied science environment; employed as a benchmark to evaluate procedural reasoning with G-Memory.

- AgentBoard (PDDL tasks) · [GitHub](https://github.com/hkust-nlp/AgentBoard)
  - description: Provides the PDDL game tasks used by the paper as the strategic game benchmark for evaluating MAS memory.

- HotpotQA · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset; used for knowledge reasoning experiments with G-Memory.

- FEVER · [Website](https://fever.ai/dataset/fever.html)
  - description: Fact verification dataset; used to measure evidence-based reasoning performance with and without G-Memory.

- Qwen 2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io/)
  - description: Open-source LLM backbone (7B/14B) used for most experiments; models are locally served in the paper’s setup.

- Ollama · [GitHub](https://github.com/ollama/ollama) · [Codewiki](https://codewiki.google/github.com/ollama/ollama)
  - description: Local LLM runner used by the authors to deploy Qwen-2.5 models for reproducing MAS experiments.

- OpenAI API · [Doc](https://platform.openai.com/docs)
  - description: API used to access GPT-4o-mini, the proprietary LLM backbone in several experiments.

- all-MiniLM-L6-v2 (Sentence-Transformers) · [Website](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - description: Embedding model used to implement the similarity function v(·) for coarse-grained query retrieval in G-Memory.

<!-- paper_id: daa9bcf3ffefdfa8c0f25654aa0512a19ff08157 -->

## 164. Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System - ACL - 2025 - citation_count 13 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_13_Findings_Optima_Optimizing_Effectiveness_and_Efficiency_for_LLM-Based_Multi-Agent_System.pdf
- Link: https://aclanthology.org/2025.findings-acl.601/
- Tags: multiagent, tool, science, agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_13_Findings_Optima_Optimizing_Effectiveness_and_Efficiency_for_LLM-Based_Multi-Agent_System.pdf
- Token Usage: input 29464, output 4145, total 33609

### GitHub & Websites

- OPTIMA · [GitHub](https://github.com/thunlp/Optima)
  - description: Official code release of the paper’s framework for training LLM-based multi-agent systems via iterative SFT/DPO and MCTS-inspired data generation; used to reproduce the methods and experiments.

- AutoForm · [GitHub](https://github.com/thunlp/AutoForm)
  - description: Prior THUNLP baseline that uses concise, non–natural-language formats for agent communication; OPTIMA compares against it and adopts its insight for format-diverse initialization.

- Llama 3 · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: Base foundation model (Llama 3 8B / 3.2 3B) used to instantiate agents for all OPTIMA experiments.

- HotpotQA · [GitHub](https://github.com/hotpotqa/hotpot) · [Website](https://hotpotqa.github.io)
  - description: Multi-hop QA dataset; used in the information-exchange setting where contexts are split across two agents.

- 2WikiMultiHopQA · [GitHub](https://github.com/Alab-NII/2WikiMultiHopQA)
  - description: Multi-hop QA dataset; used for information-exchange tasks with split evidence between agents.

- TriviaQA · [GitHub](https://github.com/mandarjoshi90/triviaqa) · [Website](https://nlp.cs.washington.edu/triviaqa/)
  - description: Reading comprehension/QA dataset; used as an information-exchange benchmark in OPTIMA evaluations.

- Children’s Book Test (CBT) · [Doc](https://huggingface.co/datasets/cbt)
  - description: Cloze-style reading comprehension dataset; used as an information-exchange benchmark with randomly assigned contexts.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math) · [Website](https://openai.com/research/grade-school-math)
  - description: Grade-school math word problems; used in the debate setting (solver–critic agents) and for transfer/scaling studies.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: High-school competition math problems; used in the debate setting and as a source model for transfer to GSM8K.

- ARC (AI2 Reasoning Challenge) · [GitHub](https://github.com/allenai/ai2-arc) · [Website](https://allenai.org/data/arc)
  - description: Multiple-choice science questions; the ARC-Challenge subset is used in debate experiments.

- MMLU (Hendrycks Test) · [GitHub](https://github.com/hendrycks/test) · [Doc](https://huggingface.co/datasets/cais/mmlu)
  - description: Massive multitask language understanding benchmark; used in debate evaluations.

- SymPy · [GitHub](https://github.com/sympy/sympy) · [Doc](https://docs.sympy.org)
  - description: Symbolic mathematics library used by the paper for equivalence checking on MATH (answer validation).

<!-- paper_id: 2fc8a207405ac233cde98424c36dd19c8731ce5e -->

## 165. Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning - ICLR - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_15_Poster_Facilitating_Multi-turn_Function_Calling_for_LLMs_via_Compositional_Instruction_Tuning.pdf
- Link: https://openreview.net/pdf/bddc07777716e293010f1c92044e3f7cc25fb4aa.pdf
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_15_Poster_Facilitating_Multi-turn_Function_Calling_for_LLMs_via_Compositional_Instruction_Tuning.pdf
- Token Usage: input 37861, output 3970, total 41831

### GitHub & Websites

- BUTTON (BUTTONInstruct) · [GitHub](https://github.com/PKU-Baichuan-MLSystemLab/BUTTON)
  - description: Official release for the paper, providing the BUTTON pipeline, prompts, and the BUTTONInstruct dataset (8k multi-turn function-calling trajectories) used to train/evaluate models.

- glaive-function-calling-v2 · [Website](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
  - description: Public dataset of function-calling conversations; used by the authors to extract real-world scenarios for bottom-up task construction.

- ToolBench / ToolLLM · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolllm.github.io)
  - description: Large-scale repository of real-world APIs and tool-use data (ToolLLM/ToolBench); used as seed data to extract/expand scenarios during bottom-up instruction construction.

- OpenHermes 2.5 · [Website](https://huggingface.co/datasets/teknium/OpenHermes-2.5)
  - description: General instruction-tuning dataset mixed with BUTTONInstruct (100k samples sampled) to retain broad instruction-following ability during fine-tuning.

- OpenAI GPT-4o · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Frontier LLM used to synthesize BUTTON data at every stage and to simulate agents during top-down trajectory generation.

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers) · [Doc](https://huggingface.co/docs/transformers)
  - description: Training framework used for full-parameter supervised fine-tuning of Llama3 and Qwen2 models.

- Mathpix OCR API · [Website](https://mathpix.com/)
  - description: OCR service referenced for the GTA benchmark’s end-to-end evaluation (MathOCR); authors note subscription requirements and excluded some items accordingly.

- Llama 3 (Meta) · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: Base and instruction models (Llama3-8B/70B) fine-tuned and compared in experiments.

- Qwen2 (Alibaba Cloud) · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://qwenlm.ai)
  - description: Base and instruction models (Qwen2-7B/72B) fine-tuned and compared in experiments.

- GTA: A Benchmark for General Tool Agents
  - description: Evaluation benchmark used for step-by-step and end-to-end tool-calling assessment (perception/operation/logic/creation tool selection and final answer accuracy).

- Tool-Query (AgentBoard)
  - description: Multi-domain tool-use environment (weather, movies, academia) used for evaluation with grounding accuracy, process rate, and success rate metrics.

<!-- paper_id: 1b50041bc6cc3b517b1de0b9d5c76a67f8ae2fe7 -->

## 166. AgentStore: Scalable Integration of Heterogeneous Agents As Specialized Generalist Computer Assistant - ACL - 2025 - citation_count 19 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_19_Findings_AgentStore_Scalable_Integration_of_Heterogeneous_Agents_As_Specialized_Generalist_Computer_Assistant.pdf
- Link: https://aclanthology.org/2025.findings-acl.466/
- Tags: multiagent, tool, science, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ACL_19_Findings_AgentStore_Scalable_Integration_of_Heterogeneous_Agents_As_Specialized_Generalist_Computer_Assistant.pdf
- Token Usage: input 25628, output 4716, total 30344

### GitHub & Websites

- AgentStore (this paper) · Website
  - description: The system introduced by the paper; the authors state “All source code will be made public.” indicating an upcoming release of code, agents, and the new OSWorld-Multi benchmark.

- OSWorld · [GitHub](https://github.com/xlang-ai/OSWorld) · [Website](https://os-world.github.io)
  - description: Real-computer evaluation environment and benchmark used for the main experiments; the paper trains/evaluates AgentStore on OSWorld and also builds OSWorld-Multi on top of it.

- OSWorld-Multi (this paper)
  - description: New multi-agent collaboration benchmark constructed by the authors from OSWorld; used to evaluate routing, decomposition, and execution, to be released with the project.

- InternVL2 · [GitHub](https://github.com/OpenGVLab/InternVL) · [Doc](https://internvl.readthedocs.io/)
  - description: Open-source MLLM suite; InternVL2-8B is the base model for the MetaAgent used throughout the paper’s experiments.

- Meta Llama 3.1 · [Website](https://ai.meta.com/llama/)
  - description: Open-source LLM family used as base for some integrated agents in AgentPool (open-source single-modality agents).

- OpenInterpreter · [GitHub](https://github.com/OpenInterpreter/open-interpreter) · [Website](https://openinterpreter.com)
  - description: CLI-based agent baseline compared/re-implemented in Table 1; serves as a generalist coding/CLI agent reference.

- BERTScore · [GitHub](https://github.com/Tiiiger/bert_score) · [Doc](https://bert-score.readthedocs.io/)
  - description: Similarity metric used in the self-instruct filtering pipeline to curate and deduplicate generated demonstrations for AgentToken training.

- openpyxl · [Doc](https://openpyxl.readthedocs.io/)
  - description: Python spreadsheet library leveraged by SheetAgent to programmatically manipulate Calc/Excel files in tasks and demonstrations.

- python-docx · [GitHub](https://github.com/python-openxml/python-docx) · [Doc](https://python-docx.readthedocs.io/)
  - description: Library used by WriterAgent to edit Word/Writer documents programmatically as part of the CLI-based agents.

- python-pptx · [GitHub](https://github.com/scanny/python-pptx) · [Doc](https://python-pptx.readthedocs.io/)
  - description: Library used by SlideAgent to create and modify slides via code in the LibreOffice Impress domain.

- ImageMagick · [Website](https://imagemagick.org) · [Doc](https://imagemagick.org/script/command-line-tools.php)
  - description: Command-line image processing tool invoked by ImageAgent for operations like contrast adjustment within system-wide tasks.

<!-- paper_id: 01101ec460b47c7a26e5a5ff26c38d1ef5bfcb97 -->

## 167. ReachAgent: Enhancing Mobile Agent via Page Reaching and Operation - NAACL - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NAACL_11_Long_ReachAgent_Enhancing_Mobile_Agent_via_Page_Reaching_and_Operation.pdf
- Link: https://aclanthology.org/2025.naacl-long.244/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_NAACL_11_Long_ReachAgent_Enhancing_Mobile_Agent_via_Page_Reaching_and_Operation.pdf
- Token Usage: input 20384, output 5288, total 25672

### GitHub & Websites

- ReachAgent · [GitHub](https://github.com/XiaoMi/reachagent)
  - description: Official code release of the paper, including the MobileReach dataset, two-stage SFT+RL implementation with action alignment, and evaluation scripts for reproducing results.

- MobileReach (dataset) · [GitHub](https://github.com/XiaoMi/reachagent)
  - description: The paper’s new dataset for page navigation, page reaching, and page operation; used to train and evaluate ReachAgent and released together with the code.

- Appium · [Website](https://appium.io) · [Doc](https://appium.io/docs/en/latest/)
  - description: Mobile UI automation framework the paper uses to send actions to the Android emulator and capture resulting GUI states.

- InternVL · [GitHub](https://github.com/OpenGVLab/InternVL) · [Website](https://opengvlab.github.io/InternVL/)
  - description: Vision-language model used by the authors to generate image captions of GUI pages during dataset construction.

- GPT-4V / GPT-4o · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Closed-source VLMs used both as task generators (to create step-by-step/brief task texts) and as strong few-shot baselines.

- Qwen-VL · [GitHub](https://github.com/QwenLM/Qwen-VL) · [Website](https://qwenlm.github.io)
  - description: Open-source VLM baseline; also fine-tuned on the paper’s page navigation split for comparison.

- Auto-UI (You Only Look at Screens) · [GitHub](https://github.com/amazon-science/auto-ui)
  - description: Benchmark and data derived from AITW; the authors fine-tune/evaluate ReachAgent on its official split for cross-dataset testing.

- Android in the Wild (AITW) · [GitHub](https://github.com/google-research/google-research/tree/master/android_in_the_wild)
  - description: Large-scale Android device-control dataset from which Auto-UI is derived; referenced as a key prior dataset in the paper.

- Rico Dataset · [Website](https://interactionmining.org/rico)
  - description: Public GUI dataset referenced as prior work for UI modeling; relevant background dataset for extending training or pretraining.

- Direct Preference Optimization (DPO) · [GitHub](https://github.com/eric-mitchell/direct-preference-optimization)
  - description: Preference-based RL algorithm adopted by the paper in the second stage to optimize ReachAgent with constructed preference pairs.

- MobileVLM / Mobile3M
  - description: MobileVLM is used as the backbone model; its Mobile3M GUI graph is the source from which the authors sample flows to build MobileReach and construct preference data.

<!-- paper_id: a72ce9262c2eb7edf3a761d2b89cbb113699d9de -->

## 168. LeanAgent: Lifelong Learning for Formal Theorem Proving - ICLR - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_11_Poster_LeanAgent_Lifelong_Learning_for_Formal_Theorem_Proving.pdf
- Link: https://openreview.net/pdf/c4bbf8256507953aff546765903d08e3fe6d16a4.pdf
- Tags: multiagent, tool, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2025_ICLR_11_Poster_LeanAgent_Lifelong_Learning_for_Formal_Theorem_Proving.pdf
- Token Usage: input 35348, output 4290, total 39638

### GitHub & Websites

- LeanAgent · [GitHub](https://github.com/lean-dojo/LeanAgent)
  - description: Official code release for the paper; implements the lifelong learning framework (curriculum construction, dynamic database, progressive retriever training, best-first proof search) and scripts to reproduce experiments and generate PRs.

- LeanDojo · [GitHub](https://github.com/lean-dojo/LeanDojo) · [Website](https://leandojo.org)
  - description: Open-source framework used to parse Lean repos, trace proofs, build datasets (LeanDojo Benchmark 4), and run retrieval-augmented proving; LeanAgent relies on it for data extraction, dataset generation, and the proving pipeline.

- ReProver (LeanDojo) · [GitHub](https://github.com/lean-dojo/LeanDojo)
  - description: Retrieval-augmented baseline prover from LeanDojo; LeanAgent initializes from ReProver’s ByT5 retriever and uses its tactic generator, and compares against ReProver as the main baseline.

- Lean (Lean 4) · [GitHub](https://github.com/leanprover/lean4) · [Website](https://leanprover.github.io)
  - description: Interactive theorem prover used to type-check and verify all automatically generated proofs; LeanAgent switches Lean versions to match each target repository.

- mathlib4 · [GitHub](https://github.com/leanprover-community/mathlib4)
  - description: The community mathematics library for Lean 4; ReProver’s retriever is pre-trained/fine-tuned on mathlib4 and its premises are part of the retrieval corpus used by LeanAgent.

- miniF2F-Lean4 · [GitHub](https://github.com/yangky11/miniF2F-lean4)
  - description: Lean4 port of MiniF2F benchmark repository; used as a new repo appended after the initial curriculum to demonstrate LeanAgent’s ability to extend to fresh repositories and for a Lean4 test-set Pass@1 comparison.

- PFR (Polynomial Freiman–Ruzsa) · [GitHub](https://github.com/teorth/pfr)
  - description: Lean repository formalizing results around the PFR Conjecture; part of the evaluation set where LeanAgent proved new sorry theorems and analyzed generalization across commits.

- Hairy Ball Theorem (Lean) · [GitHub](https://github.com/corent1234/hairy-ball-theorem-lean)
  - description: Lean repository for the Hairy Ball Theorem; included in the sub-curriculum where LeanAgent proved a key step (“HairyBallDiff”).

- Coxeter · [GitHub](https://github.com/NUS-Math-Formalization/coxeter)
  - description: Lean formalization of Coxeter groups; used in evaluation where LeanAgent proved a nontrivial lemma about Coxeter systems.

- Mathematics in Lean Source · [GitHub](https://github.com/avigad/mathematics_in_lean_source)
  - description: Source Lean files accompanying the Mathematics in Lean textbook; a major evaluation repo where LeanAgent showed strong progression and highest accuracy gains.

- FormalBook (“Proofs from THE BOOK”) · [GitHub](https://github.com/Mo271/FormalBook)
  - description: Lean formalizations inspired by Proofs from THE BOOK; part of the sub-curriculum where LeanAgent proved results including Wedderburn’s Little Theorem.

- SciLean · [GitHub](https://github.com/lecopivo/SciLean)
  - description: Scientific computing library in Lean; a large evaluation repo where LeanAgent progressively proved theorems ranging from basic algebra to advanced function spaces.

- Carleson · [GitHub](https://github.com/fpvandoorn/carleson)
  - description: Lean repository related to Carleson’s Theorem; included in the sub-curriculum and evaluation.

- Lean4 PDL · [GitHub](https://github.com/M4lvin/lean4-pdl)
  - description: Propositional Dynamic Logic in Lean4; part of the sub-curriculum/evaluation where LeanAgent attempted/proved sorry theorems.

- PrimeNumberTheoremAnd · [GitHub](https://github.com/AlexKontorovich/PrimeNumberTheoremAnd)
  - description: Repository on the Prime Number Theorem; included in the initial curriculum/evaluation set.

- compfiles · [GitHub](https://github.com/dwrensha/compfiles)
  - description: Catalog of Olympiad-style math problems formalized in Lean; used in the initial curriculum ordering and dataset construction.

- FLT (Fermat’s Last Theorem) · [GitHub](https://github.com/ImperialCollegeLondon/FLT)
  - description: Formalization efforts around Fermat’s Last Theorem; part of the initial curriculum/evaluation set.

- Debate · [GitHub](https://github.com/google-deepmind/debate)
  - description: Google DeepMind repository for debate protocols; included in the initial curriculum list of Lean repositories processed via LeanDojo.

- Lean4Lean · [GitHub](https://github.com/digama0/lean4lean)
  - description: Implementation of the Lean4 kernel in Lean4; part of the initial curriculum/evaluation set.

- Lean Matrix Cookbook · [GitHub](https://github.com/eric-wieser/lean-matrix-cookbook)
  - description: “Matrix Cookbook” lemmas formalized in Lean; included in the initial curriculum.

- Lean Math Workshop · [GitHub](https://github.com/yuma-mizuno/lean-math-workshop)
  - description: Detailed Lean tutorial and exercises; part of the initial curriculum and dataset generation.

- LeanEuclid · [GitHub](https://github.com/loganrjmurphy/LeanEuclid)
  - description: Euclidean geometry in Lean; included in the initial curriculum.

- Foundation (Formalized Formal Logic) · [GitHub](https://github.com/FormalizedFormalLogic/Foundation)
  - description: Formalized results in logic; part of the initial curriculum/evaluation.

- con-nf · [GitHub](https://github.com/leanprover-community/con-nf)
  - description: Formal consistency proof of Quine’s New Foundations; included in the initial curriculum/evaluation.

- Saturn · [GitHub](https://github.com/siddhartha-gadgil/Saturn)
  - description: Experiments with SAT solvers with proofs in Lean 4; part of the initial curriculum.

- Zeta 3 Irrational · [GitHub](https://github.com/ahhwuhu/zeta_3_irrational)
  - description: Proof of ζ(3) irrationality; used in the sub-curriculum.

- Formalisation of Constructable Numbers · [GitHub](https://github.com/Louis-Le-Grand/Formalisation-of-constructable-numbers)
  - description: Formalization of ancient constructible-number problems; part of the sub-curriculum.

- LeanAPAP · [GitHub](https://github.com/YaelDillies/LeanAPAP)
  - description: Lean repo on Kelley–Meka bound on Roth numbers; included in the sub-curriculum/evaluation.

<!-- paper_id: c3b60318038abdee2df8df85ba06475b824d1fd9 -->

## 169. DataNarrative: Automated Data-Driven Storytelling with Visualizations and Texts - EMNLP - 2024 - citation_count 17 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_17_Main_DataNarrative_Automated_Data-Driven_Storytelling_with_Visualizations_and_Texts.pdf
- Link: https://aclanthology.org/2024.emnlp-main.1073/
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_17_Main_DataNarrative_Automated_Data-Driven_Storytelling_with_Visualizations_and_Texts.pdf
- Token Usage: input 29054, output 2903, total 31957

### GitHub & Websites

- DATANARRATIVE (code and dataset release)
  - description: Official resources promised by the paper for reproducing their multi-agent system and dataset; the authors state “We make our code and data story corpus publicly available at here.” and “We plan to make the whole corpus and all the collected metadata publicly available.”

- Pew Research Center
  - [Website](https://www.pewresearch.org)
  - description: Primary source of articles, charts, and text used to build the largest portion of the dataset; the authors crawled 4,532 articles and extracted 22,760 figures and associated metadata.

- Tableau Public Stories
  - [Website](https://public.tableau.com)
  - description: Source of curated, paginated data stories with charts and text; the authors manually collected story pages, charts, and underlying tables for the Tableau split.

- Gapminder
  - [Website](https://www.gapminder.org)
  - description: Source of data stories used in the test set; the authors collected paginated stories with charts and text from Gapminder.

- Our World in Data (OWID)
  - [Website](https://ourworldindata.org)
  - description: External provider for “gold” data tables corresponding to some Gapminder charts, used when original tables were not downloadable.

- ChartQA
  - [GitHub](https://github.com/vis-nlp/ChartQA)
  - description: Benchmark dataset used to validate the chart-to-table extraction step; the authors evaluated Gemini-1.0-pro-vision on 100 ChartQA chart images with gold tables.

- Chart-to-Text
  - [GitHub](https://github.com/vis-nlp/Chart-to-text)
  - description: Benchmark used to obtain human-annotated chart–summary pairs and to form the Pew test split; also used to assess GPT-4-turbo paragraph–table linking accuracy.

- OpenAI GPT-4o
  - [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Used as the main Generator and Evaluator LLM agents for story generation and verification in experiments.

- OpenAI GPT-4-turbo
  - [Doc](https://platform.openai.com/docs/models/gpt-4-turbo)
  - description: Used to automatically link article paragraphs to charts/tables and to paraphrase paragraphs for the Pew training set.

- Google Gemini (1.0 Pro Vision; 1.5 Pro)
  - [Doc](https://ai.google.dev/gemini-api/docs) · [Website](https://ai.google.dev)
  - description: Gemini-1.0-pro-vision was used to extract data tables from chart images; Gemini-1.5-pro served as the LLM judge for automatic pairwise evaluation.

- Meta Llama 3 (8B/70B Instruct)
  - [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Open-source baseline LLMs used as Generator/Evaluator agents to compare the agentic framework against direct prompting.

<!-- paper_id: bf746159ec6008fa8e4d4134c848f8611066d62d -->

## 170. Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate - ICLR - 2025 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_18_Poster_Breaking_Mental_Set_to_Improve_Reasoning_through_Diverse_Multi-Agent_Debate.pdf
- Link: https://openreview.net/pdf/d67d70de207899a21f78262107dd3b5ec2d940b6.pdf
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_18_Poster_Breaking_Mental_Set_to_Improve_Reasoning_through_Diverse_Multi-Agent_Debate.pdf
- Token Usage: input 64962, output 4406, total 69368

### GitHub & Websites

- DMAD (Diverse Multi‑Agent Debate) · [GitHub](https://github.com/MraDonkey/DMAD)
  - description: Official code release of the paper; provides prompts, debate pipelines, and scripts to reproduce DMAD and all baselines across LLM and MLLM benchmarks.

- MATH Dataset · [GitHub](https://github.com/hendrycks/math)
  - description: High‑school competition math benchmark used for LLM evaluation; the paper samples problems per subject to test reasoning methods and debate settings.

- GPQA · [GitHub](https://github.com/idavidrein/GPQA)
  - description: Graduate‑level, Google‑proof multiple‑choice benchmark used in the paper to evaluate LLM reasoning under various prompting/debate strategies.

- ScienceQA · [GitHub](https://github.com/lupantech/ScienceQA) · [Website](https://scienceqa.github.io)
  - description: Multimodal multiple‑choice benchmark used for MLLM evaluation (QCM format) in the paper.

- MM‑Vet · [GitHub](https://github.com/yuweihao/MM-Vet) · [Website](https://mm-vet.github.io)
  - description: Open‑ended multimodal reasoning benchmark; the paper evaluates all MLLM methods here and uses the official GPT‑4–based evaluator provided by MM‑Vet.

- MMLU · [GitHub](https://github.com/hendrycks/test)
  - description: Used in an expanded experiment (abstract algebra subset) to test DMAD on more challenging multi‑hop reasoning.

- LLaVA‑1.6 · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io)
  - description: Open‑source MLLM used as one of the base models (LLaVA‑1.6‑13B) for ScienceQA and discussed on MM‑Vet in the paper.

- Llama 3 (Instruct variants) · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: Open‑source LLMs (70B and 8B Instruct) used as base models in the paper’s LLM experiments.

- GPT‑4o / GPT‑4o‑mini · [Website](https://openai.com/index/hello-gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed‑source LLM/MLLM models evaluated across MATH, GPQA, ScienceQA, and MM‑Vet; also used as MM‑Vet’s evaluator per benchmark protocol.

- Gemini 1.5 Flash · [Website](https://ai.google.dev/gemini-api) · [Doc](https://ai.google.dev/gemini-api/docs/models/gemini)
  - description: Google’s MLLM used as a base model in ScienceQA and MM‑Vet experiments with DMAD and baselines.

- Multi‑Agent Debate (Du et al., 2024) · [GitHub](https://github.com/composable-models/llm_multiagent_debate)
  - description: Canonical MAD baseline implementation; the paper adopts MAD’s default settings to compare fixed‑method debates against DMAD.

- Program of Thoughts Prompting (PoT) · [GitHub](https://github.com/wenhuchen/Program-of-Thoughts)
  - description: Numerical reasoning prompting used as one of the three diverse methods in DMAD for LLMs and as a standalone baseline.

- Step‑Back Prompting · [GitHub](https://github.com/google-research/google-research/tree/master/step-back-prompting)
  - description: Abstraction‑first prompting method employed as a DMAD agent strategy and as a baseline in the LLM experiments.

- Self‑Refine · [GitHub](https://github.com/kaistAI/self-refine)
  - description: Iterative self‑reflection baseline evaluated by the paper on both LLMs and MLLMs, contrasted with DMAD’s multi‑agent approach.

<!-- paper_id: fb9759e2e360c64015f4c4794505a79a768d0282 -->

## 171. mABC: multi-Agent Blockchain-Inspired Collaboration for root cause analysis in micro-services architecture - EMNLP - 2024 - citation_count 19 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_19_Findings_mABC_multi-Agent_Blockchain-Inspired_Collaboration_for_root_cause_analysis_in_micro-services_architecture.pdf
- Link: https://aclanthology.org/2024.findings-emnlp.232/
- Tags: multiagent, tool, science, multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_19_Findings_mABC_multi-Agent_Blockchain-Inspired_Collaboration_for_root_cause_analysis_in_micro-services_architecture.pdf
- Token Usage: input 20061, output 3924, total 23985

### GitHub & Websites

- mABC · [GitHub](https://github.com/knediny/mABC)
  - description: Official repository for the paper’s framework and resources; contains the MABC multi-agent system implementation and the newly released Train-Ticket-based dataset used for experiments and ablations.

- Train-Ticket · [GitHub](https://github.com/FudanSELab/train-ticket)
  - description: Open-source microservices benchmark from Fudan University; used by the paper as the application environment to construct the Train-Ticket dataset and simulate RCA scenarios.

- ChaosBlade · [GitHub](https://github.com/chaosblade-io/chaosblade) · [Website](https://chaosblade.io)
  - description: Chaos engineering toolkit used to inject network, CPU, memory, storage, and code faults in the Train-Ticket system to create realistic failure cases.

- 2020 International AIOps Challenge Dataset · [Website](https://iops.ai)
  - description: Public RCA dataset used as a primary benchmark in the paper to evaluate MABC’s root cause and path accuracy.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Doc](https://python.langchain.com)
  - description: Toolkit the authors used to implement the ReAct baseline for comparison.

- ReAct (Reason+Act) · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Baseline method for reasoning-and-acting LLM agents; the paper evaluates against ReAct (implemented via LangChain).

- Llama 3 8B Instruct · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - description: One of the base LLMs powering MABC in experiments; the paper reports results using this instruction-tuned model.

- OpenAI GPT models (GPT-3.5-Turbo, GPT-4-Turbo) · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Commercial LLMs used as backends for MABC and for the ReAct baseline; the paper reports performance with these model variants.

- Kubernetes · [Website](https://kubernetes.io) · [Doc](https://kubernetes.io/docs/home/)
  - description: Container orchestration platform on which the Train-Ticket microservices system was deployed to generate the dataset and run fault-injection experiments.

<!-- paper_id: 7c33415ee78eaf12dd082ed305638d0c384a279c -->

## 172. OpenCUA: Open Foundations for Computer-Use Agents - NeurIPS - 2025 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NeurIPS_18_Spotlight_OpenCUA_Open_Foundations_for_Computer-Use_Agents.pdf
- Link: https://openreview.net/pdf/eb1bd0238abbc386303352dba1049a4d5d1fec83.pdf
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NeurIPS_18_Spotlight_OpenCUA_Open_Foundations_for_Computer-Use_Agents.pdf
- Token Usage: input 51316, output 4433, total 55749

### GitHub & Websites

- OpenCUA (framework, code, models) · [Website](https://opencua.xlang.ai)
  - description: Official project hub for the paper’s open release, including the end-to-end CUA framework, training code/recipes, pretrained OpenCUA models, and links to all released assets.

- AGENTNET TOOL · [Website](https://opencua.xlang.ai)
  - description: The cross-OS annotation application released by the paper to record human computer-use demonstrations (screen video, keyboard/mouse, AXTree); used to build the AGENTNET dataset.

- AGENTNET (dataset) · [Website](https://opencua.xlang.ai)
  - description: The paper’s large-scale desktop CUA dataset (22.6K trajectories across Windows/macOS/Ubuntu, 140+ apps and 190+ websites) used for training and analysis.

- AGENTNETBENCH (offline benchmark) · [Website](https://opencua.xlang.ai)
  - description: The paper’s offline evaluation set with multi-option gold actions per step; used to approximate online success and accelerate iteration.

- OSWorld / OSWorld-Verified · [GitHub](https://github.com/xlang-ai/OSWorld) · [Website](https://os-world.github.io) · [Doc](https://xlang.ai/blog/osworld-verified)
  - description: Real-computer benchmark and verified task suite used for the paper’s main online evaluation; authors submit to the public OSWorld-Verified evaluation service.

- WindowsAgentArena · [GitHub](https://github.com/microsoft/WindowsAgentArena)
  - description: Microsoft’s Windows-centric benchmark used by the paper to evaluate online agent performance on Windows.

- DuckTrack · [GitHub](https://github.com/TheDuckAI/DuckTrack)
  - description: Open-source mouse/keyboard event capture library the paper builds upon in AGENTNET TOOL for recording human interactions.

- OpenAdapt · [GitHub](https://github.com/OpenAdaptAI/OpenAdapt)
  - description: Process-automation toolkit leveraged by AGENTNET TOOL for input tracking and trajectory processing during annotation.

- OBS Studio · [GitHub](https://github.com/obsproject/obs-studio)
  - description: Open-source screen recorder used in AGENTNET TOOL to capture high-quality screen videos of demonstrations.

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui)
  - description: Cross-platform GUI automation library; the paper defines its action space as a subset of PyAutoGUI actions for agent execution.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL)
  - description: Open-source VLM family used as base models (e.g., Qwen2-VL-7B) for supervised fine-tuning into OpenCUA variants.

- Qwen2.5-VL · [GitHub](https://github.com/QwenLM/Qwen2.5-VL)
  - description: The enhanced high-resolution VLMs (e.g., 32B/72B) that serve as the primary backbones for OpenCUA-32B/72B.

- ShowUI · [GitHub](https://github.com/showlab/ShowUI)
  - description: GUI grounding dataset/model used in the paper’s Stage-1 grounding mix to initialize perception/grounding for the CUA models.

<!-- paper_id: b1995f97c41fc38df8b05e3f9e31f3a52dd56a33 -->

## 173. METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling - ACL - 2025 - citation_count 10 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_10_Long_METAL_A_Multi-Agent_Framework_for_Chart_Generation_with_Test-Time_Scaling.pdf
- Tags: agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ACL_10_Long_METAL_A_Multi-Agent_Framework_for_Chart_Generation_with_Test-Time_Scaling.pdf
- Token Usage: input 19769, output 4543, total 24312

### GitHub & Websites

- METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling · [Website](https://metal-chart-generation.github.io)
  - description: Official project page for the paper; the framework and results described in the paper are hosted here and are the primary entry point for reproducing the work.

- ChartMIMIC
  - description: Dataset/benchmark used in all experiments; provides 1,000 (figure, instruction, code) triplets and the evaluation protocol the paper follows.

- OpenAI GPT-4o · [Website](https://openai.com/index/gpt-4o/) · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Closed-source VLM used as a base model and accessed via the OpenAI API to drive METAL’s agents.

- Llama 3.2 11B · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com/) · [Doc](https://github.com/meta-llama/llama/blob/main/MODEL_CARD.md)
  - description: Open-source base VLM used in experiments; the paper runs four agent instances locally on GPUs.

- Matplotlib · [GitHub](https://github.com/matplotlib/matplotlib) · [Doc](https://matplotlib.org/stable/)
  - description: Visualization library targeted by the chart-to-code generation; all charts are reproduced via Matplotlib code.

- EasyOCR · [GitHub](https://github.com/JaidedAI/EasyOCR) · [Codewiki](https://codewiki.google/github.com/JaidedAI/EasyOCR) · [Doc](https://www.jaided.ai/easyocr/)
  - description: OCR toolkit used in the paper’s multi-criteria verifier to extract text from reference and generated charts.

- OpenCV · [GitHub](https://github.com/opencv/opencv) · [Codewiki](https://codewiki.google/github.com/opencv/opencv) · [Doc](https://docs.opencv.org/)
  - description: Image processing library used in the verifier for HSV color masking, histogram computation, and image handling.

- scikit-image · [GitHub](https://github.com/scikit-image/scikit-image) · [Doc](https://scikit-image.org/)
  - description: Used to compute SSIM in the verifier’s overall similarity metric between reference and generated charts.

- scikit-learn · [GitHub](https://github.com/scikit-learn/scikit-learn) · [Codewiki](https://codewiki.google/github.com/scikit-learn/scikit-learn) · [Doc](https://scikit-learn.org/stable/)
  - description: Provides cosine_similarity for comparing color histograms in the verifier.

- NumPy · [GitHub](https://github.com/numpy/numpy) · [Doc](https://numpy.org/doc/)
  - description: Numerical computing dependency used for vectorization and array operations in the verifier implementation.

<!-- paper_id: 580c5dea65666784d1767969b1ee094ad9af4ddf -->

## 174. Hazards in Daily Life? Enabling Robots to Proactively Detect and Resolve Anomalies - NAACL - 2025 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_12_Long_Hazards_in_Daily_Life_Enabling_Robots_to_Proactively_Detect_and_Resolve_Anomalies.pdf
- Link: https://aclanthology.org/2025.naacl-long.379/
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_12_Long_Hazards_in_Daily_Life_Enabling_Robots_to_Proactively_Detect_and_Resolve_Anomalies.pdf
- Token Usage: input 18803, output 6331, total 25134

### GitHub & Websites

- Genesis (Embodied-AI simulator) · [GitHub](https://github.com/Genesis-EmbodiedAI/Genesis) · [Website](https://genesis-embodied-ai.github.io)
  - description: Physics-based simulator used to build and run all 3D scenes for AnomalyGen; the paper deploys their environments and robot learning in Genesis.

- Objaverse · [GitHub](https://github.com/allenai/objaverse) · [Website](https://objaverse.allenai.org)
  - description: Large-scale 3D asset repository; used to retrieve auxiliary surrounding objects for scene construction via text and VLM filtering.

- PartNet-Mobility (SAPIEN) · [GitHub](https://github.com/haosulab/SAPIEN) · [Website](https://sapien.ucsd.edu) · [Doc](https://sapien.ucsd.edu/downloads/partnet-mobility-dataset)
  - description: Interactive articulated-object dataset/platform; the paper curates its anomalous household asset list from a subset of PartNet-Mobility.

- BLIP-2 (via LAVIS) · [GitHub](https://github.com/salesforce/LAVIS) · [Website](https://salesforce.github.io/LAVIS/)
  - description: Vision-language model used to visually validate retrieved 3D assets and assist asset selection.

- Sentence-Transformers (Sentence-BERT) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net)
  - description: Text-embedding toolkit used for retrieving textually similar Objaverse assets and for computing task-description diversity (Sentence-BERT similarity).

- CLIP · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Used to compute embedding similarity of scene images for evaluating visual diversity.

- Vision Transformer (ViT) · [GitHub](https://github.com/google-research/vision_transformer)
  - description: ImageNet-pretrained ViT embeddings are used to assess scene-image diversity.

- Open Motion Planning Library (OMPL) · [GitHub](https://github.com/ompl/ompl) · [Website](https://ompl.kavrakilab.org)
  - description: Motion-planning library used for action primitives; the paper specifically employs the BIT* planner through OMPL.

- Soft Actor-Critic (SAC, PyTorch) · [GitHub](https://github.com/denisyarats/pytorch_sac)
  - description: Reinforcement learning algorithm used to train manipulation and locomotion subtasks in AnomalyGen.

- OpenAI GPT-4 API · [Website](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)
  - description: LLM used for group brainstorming, anomaly detection, task decomposition, algorithm selection, and reward specification.

- RLBench · [GitHub](https://github.com/stepjam/RLBench) · [Website](https://sites.google.com/view/rlbench)
  - description: Human-designed robotics benchmark used as a comparison baseline for task and scene diversity.

- ManiSkill2 · [GitHub](https://github.com/haosulab/ManiSkill2) · [Website](https://maniskill2.github.io)
  - description: Generalizable manipulation benchmark compared against AnomalyGen for diversity.

- Meta-World · [GitHub](https://github.com/rlworkgroup/metaworld) · [Website](https://meta-world.github.io/)
  - description: Multi-task RL benchmark used as a baseline in diversity comparisons.

- BEHAVIOR / BEHAVIOR-1K · [GitHub](https://github.com/StanfordVL/Behavior) · [Website](https://behavior.stanford.edu/)
  - description: Embodied household-activity benchmark used as a baseline for the diversity evaluation.

<!-- paper_id: 804d2583f3e7dcc29187ee7c9704f46c70e2c7a8 -->

## 175. From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions - ICLR - 2025 - citation_count 19 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_19_Oral_From_Exploration_to_Mastery_Enabling_LLMs_to_Master_Tools_via_Self-Driven_Interactions.pdf
- Link: https://openreview.net/pdf/316eeebf4b438f0f482c02184e1b2316f3800805.pdf
- Tags: multiagent, tool, science, agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ICLR_19_Oral_From_Exploration_to_Mastery_Enabling_LLMs_to_Master_Tools_via_Self-Driven_Interactions.pdf
- Token Usage: input 24951, output 3508, total 28459

### GitHub & Websites

- DRAFT · [GitHub](https://github.com/quchangle1/DRAFT)
  - description: Official code release for the paper’s self-driven documentation refinement framework; used to reproduce the experiments and pipelines for Explorer/Analyzer/Rewriter and the termination/diversity mechanisms.

- ToolBench / ToolLLM (includes DFSDT) · [GitHub](https://github.com/OpenBMB/ToolBench)
  - description: Benchmark and tooling suite of real-world APIs and the DFSDT baseline; the paper evaluates on the I3-Instruction subset and uses DFSDT as a comparison method.

- RestGPT / RestBench · [GitHub](https://github.com/Yifan-Song793/RestGPT)
  - description: Repository containing RestBench (TMDB and Spotify scenarios) used as evaluation datasets; the paper reports CP%/Win% on RestBench-TMDB and RestBench-Spotify.

- RapidAPI · [Website](https://rapidapi.com)
  - description: API marketplace from which ToolBench collected many APIs; cited as a source of the APIs used for tool-learning evaluation.

- BMTools · [GitHub](https://github.com/OpenBMB/BMTools)
  - description: Tool aggregation framework used by ToolBench; referenced as another source of APIs leveraged in the evaluation.

- The Movie Database (TMDB) API · [Doc](https://developer.themoviedb.org/docs)
  - description: Official API documentation for TMDB; provides the movie/TV endpoints used in RestBench-TMDB tasks.

- Spotify Web API · [Doc](https://developer.spotify.com/documentation/web-api)
  - description: Official Spotify Web API docs; provides the music endpoints used in RestBench-Spotify tasks.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting prompting framework used as a baseline in the paper’s comparisons.

- EasyTool
  - description: Baseline that rewrites tool descriptions to be concise; compared against DRAFT in experiments.

- Contriever · [GitHub](https://github.com/facebookresearch/contriever)
  - description: Dense retrieval model used in the paper’s tool-retrieval analysis (NDCG@1/@10) to test whether refined docs improve retrieval.

- OpenAI Embeddings (text-embedding-ada-002) · [Website](https://openai.com/index/new-and-improved-embedding-model/)
  - description: Embedding model used for similarity constraints in exploration and for termination via cosine similarity; referenced directly in the method.

- GPT-4o · [Website](https://platform.openai.com/playground/chat?models=gpt-4o-2024-08-06)
  - description: Closed-source LLM backbone used to run DRAFT and as an evaluated model.

- GPT-4o-mini · [Website](https://platform.openai.com/playground/chat?models=gpt-4o-mini)
  - description: Smaller OpenAI model evaluated with and without DRAFT-refined documentation.

- Llama-3-70B · [Website](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
  - description: Open-source LLM evaluated for cross-model generalization of documentation refined by DRAFT.

<!-- paper_id: 3eb25b4eb27808f9a6c619cdcd764140d691dea8 -->

## 176. MCU: An Evaluation Framework for Open-Ended Game Agents - ICML - 2025 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_16_Spotlightposter_MCU_An_Evaluation_Framework_for_Open-Ended_Game_Agents.pdf
- Link: https://openreview.net/pdf/f293c1c135a7e3ab045d15897631d1f8efad035c.pdf
- Tags: multiagent, tool, science, agent-eval
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/4agent-eval/2025_ICML_16_Spotlightposter_MCU_An_Evaluation_Framework_for_Open-Ended_Game_Agents.pdf
- Token Usage: input 64628, output 4452, total 69080

### GitHub & Websites

- MCU (Minecraft Universe) · [GitHub](https://github.com/CraftJarvis/MCU)
  - description: Official code release for the paper; includes atomic task lists, LLM-based task configuration, AutoEval (VLM judge), scripts, and MCU-Turbo subset for standardized benchmarking.

- MineStudio · [GitHub](https://github.com/CraftJarvis/MineStudio) · [Doc](https://arxiv.org/abs/2412.18293)
  - description: The benchmark’s core runtime; MCU’s task initialization, callbacks (commands/summon/reset/record/rewards), unified agent interface, and simulator verification are implemented on top of MineStudio.

- Minecraft Wiki · [Website](https://minecraft.wiki/)
  - description: Primary data source used to compile/extract high-quality atomic tasks (advancements, mechanics, items, mobs) for MCU.

- Minecraft Controls (official documentation) · [Doc](https://minecraft.fandom.com/wiki/Controls)
  - description: Referenced to define the native mouse/keyboard action space used by MCU’s environment and agents.

- MineDojo · [GitHub](https://github.com/MineDojo/MineDojo) · [Website](https://minedojo.org)
  - description: Prior benchmark from which MCU filters and deduplicates tasks; also provides MineCLIP, which MCU uses as an automatic-evaluation baseline.

- MineCLIP (from MineDojo) · [GitHub](https://github.com/MineDojo/MineDojo)
  - description: CLIP model trained on Minecraft videos; used in MCU as a comparison baseline for automatic trajectory evaluation.

- VPT (Video Pre-Training) · [GitHub](https://github.com/openai/Video-Pre-Training)
  - description: Baseline agent evaluated in MCU (both behavior-cloned and RL-tuned-to-diamond variants); MineStudio provides wrappers to run VPT in the MCU tasks.

- STEVE-1 · [Doc](https://arxiv.org/abs/2312.09327)
  - description: Text-instruction-following Minecraft agent evaluated as a baseline in MCU; supported by MineStudio’s unified agent interface.

- GROOT · [GitHub](https://github.com/CraftJarvis/GROOT)
  - description: Video-instruction-following agent evaluated on MCU; requires a reference video per task and is integrated via MineStudio.

- MiniCPM-V · [GitHub](https://github.com/OpenBMB/MiniCPM-V) · [Codewiki](https://codewiki.google/github.com/OpenBMB/MiniCPM-V)
  - description: Open-source VLM used by MCU’s AutoEval as an alternative to GPT-4o for multi-dimensional video-based scoring.

- Jarvis-VLA · [GitHub](https://github.com/CraftJarvis/JARVIS-VLA)
  - description: Open-source vision-language-action model used in MCU’s AutoEval experiments as another open-access evaluator variant.

<!-- paper_id: dee45635aba5d1df5dbb55620800e7570ed2d6fe -->

## 177. OS-Kairos: Adaptive Interaction for MLLM-Powered GUI Agents - ACL - 2025 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_11_Findings_OS-Kairos_Adaptive_Interaction_for_MLLM-Powered_GUI_Agents.pdf
- Link: https://aclanthology.org/2025.findings-acl.348/
- Tags: multiagent, tool, science, agent-web
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-web/2025_ACL_11_Findings_OS-Kairos_Adaptive_Interaction_for_MLLM-Powered_GUI_Agents.pdf
- Token Usage: input 32178, output 4874, total 37052

### GitHub & Websites

- OS-Kairos · [GitHub](https://github.com/Wuzheng02/OS-Kairos)
  - description: Official codebase and dataset release for the paper’s adaptive GUI agent with confidence scoring; includes training, evaluation, and the collaborative probing toolkit.

- Hugging Face Models · [Website](https://huggingface.co/models)
  - description: Model hub used by the authors to obtain open-source MLLMs (e.g., Qwen2-VL-7B) and related checkpoints for zero-shot and fine-tuning experiments.

- ModelScope · [Website](https://modelscope.cn/home)
  - description: Toolkit used to build the layout-parse/OCR pipeline in the collaborative probing framework (resnet18 and convnext-tiny backbones mentioned).

- LLaMA-Factory · [GitHub](https://github.com/hiyouga/LLaMA-Factory) · [Codewiki](https://codewiki.google/github.com/hiyouga/LLaMA-Factory)
  - description: Training framework the authors used to fine-tune the probed model on datasets for confidence integration.

- XTuner · [GitHub](https://github.com/InternLM/xtuner)
  - description: Fine-tuning toolkit used for InternVL-based models referenced in the experiments.

- Android Studio · [Website](https://developer.android.com/studio)
  - description: Official IDE and device/emulator tooling used to connect real mobile devices and run the interactive probing/execution environment.

- OpenAI GPT-4o · [Website](https://openai.com/index/gpt-4o/)
  - description: Proprietary multimodal model used as the critic/judge in the collaborative confidence probing framework (planning, action supervision, and scoring).

- Qwen2-VL-7B · [GitHub](https://huggingface.co/Qwen/Qwen2-VL-7B) · [Website](https://qwenlm.ai)
  - description: Open-source MLLM baseline used for zero-shot and fine-tuning comparisons in the paper.

- Qwen-VL-MAX
  - description: API model variant used as an alternative critic in ablations and as a baseline; compared for scoring quality and interaction performance.

- GLM-4V-Plus · [Website](https://chatglm.cn)
  - description: API multimodal model evaluated as a baseline in the zero-shot setting.

- OS-Atlas-Pro-7B
  - description: Open-source GUI-adapted MLLM used as the probing model backbone and as a baseline; the paper builds OS-Kairos on top of it for confidence integration.

- Auto-UI
  - description: Open-source GUI agent baseline evaluated in both zero-shot and fine-tuning settings for comparison.

- CogAgent
  - description: Open-source GUI/vision-language agent baseline used for comparative evaluation on benchmarks.

- AITZ (Android in the Zoo)
  - description: Benchmark dataset used for evaluation; the paper reports Type/SR/TSR and interactive metrics on this corpus.

- Meta-GUI
  - description: Task-oriented dialogue + GUI interaction dataset used as an out-of-domain benchmark to test generalization and interactive performance.

<!-- paper_id: 013b17368b4ecd6d019fa0004a4948450b2c437b -->

## 178. Multiverse: Your Language Models Secretly Decide How to Parallelize and Merge Generation - NeurIPS - 2025 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NeurIPS_12_Spotlight_Multiverse_Your_Language_Models_Secretly_Decide_How_to_Parallelize_and_Merge_Generation.pdf
- Link: https://openreview.net/pdf/5f50a250befd3553dd40112c1a440f86b36737da.pdf
- Tags: multiagent, tool, science, applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NeurIPS_12_Spotlight_Multiverse_Your_Language_Models_Secretly_Decide_How_to_Parallelize_and_Merge_Generation.pdf
- Token Usage: input 31645, output 5945, total 37590

### GitHub & Websites

- Multiverse · [Website](https://Multiverse4FM.github.io)
  - description: Official project page for this paper; the authors state they open-sourced the full Multiverse ecosystem here (code, data/Multiverse‑1K, model weights, serving engine, tools, prompts, and training/evaluation recipes).

- SGLang · [GitHub](https://github.com/sgl-project/sglang) · [Codewiki](https://codewiki.google/github.com/sgl-project/sglang) · [Website](https://sgl-project.github.io/sglang/)
  - description: Inference engine the authors extend to build the Multiverse Engine; used for continuous batching and radix attention during serving and evaluation.

- LightEval · [GitHub](https://github.com/huggingface/lighteval)
  - description: Evaluation toolkit used to compute pass@1 and run AIME, MATH500, and GPQA benchmarks in the paper.

- Qwen (Qwen2/2.5) · [GitHub](https://github.com/QwenLM/Qwen) · [Codewiki](https://codewiki.google/github.com/QwenLM/Qwen)
  - description: Base AR model; the authors fine-tune Qwen2.5‑32B‑Instruct to obtain Multiverse‑32B.

- AIME (American Invitational Mathematics Examination) · [Website](https://artofproblemsolving.com/wiki/index.php/American_Invitational_Mathematics_Examination)
  - description: Evaluation benchmark; the paper reports results on AIME24 and AIME25 sourced from AoPS.

- MATH dataset · [GitHub](https://github.com/hendrycks/math)
  - description: Source of the MATH500 evaluation subset used to assess reasoning performance.

- Gemini 2.0 Flash Thinking Mode · [Doc](https://cloud.google.com/vertex-ai/generative-ai/docs/thinking-mode)
  - description: Google Vertex AI “thinking mode” referenced for generating/analyzing long CoT trajectories.

- Gemini 2.5 Pro · [Website](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/)
  - description: Proprietary LLM used in Multiverse Curator (the five-stage, LLM-assisted data curation pipeline) to transform sequential chains into MapReduce structures.

- QWQ‑32B · [Website](https://qwenlm.github.io/blog/qwq-32b/)
  - description: External model whose hidden states are probed in Section 3.2 to test recognition of parallelism.

- DeepSeek‑R1 · [Website](https://arxiv.org/abs/2501.12948)
  - description: External reasoning model used for generating/analyzing long CoT trajectories (e.g., in s1K‑1.1) and for probing experiments.

- PyTorch Fully Sharded Data Parallel (FSDP) · [Doc](https://pytorch.org/docs/stable/fsdp.html)
  - description: Distributed training component; the authors fine-tune Multiverse‑32B with PyTorch FSDP.

<!-- paper_id: 6101d3273f5f69a53c8c5999c4f66aff23a8209a -->

## 179. MuMath-Code: Combining Tool-Use Large Language Models with Multi-perspective Data Augmentation for Mathematical Reasoning - EMNLP - 2024 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_18_Main_MuMath-Code_Combining_Tool-Use_Large_Language_Models_with_Multi-perspective_Data_Augmentation_for_Mathematical_Reasoning.pdf
- Link: https://aclanthology.org/2024.emnlp-main.274/
- Tags: multiagent, tool, mathematics, agent-math
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-math/2024_EMNLP_18_Main_MuMath-Code_Combining_Tool-Use_Large_Language_Models_with_Multi-perspective_Data_Augmentation_for_Mathematical_Reasoning.pdf
- Token Usage: input 18689, output 6209, total 24898

### GitHub & Websites

- MuMath-Code · [GitHub](https://github.com/youweihao-tal/MuMath-Code)
  - description: Official code and data release for this paper; contains MuMath-Code-Data, training/inference scripts (two-stage pipeline), and tool-execution interface for Python.

- MuMath (Multi-perspective Data Augmentation) · [GitHub](https://github.com/youweihao-tal/MuMath)
  - description: Prior dataset and augmentation framework used for Stage-1 training (Dµ, ~750K CoT samples) and as the source of the augmented question set Q referenced throughout the paper.

- Llama 2 · [Website](https://ai.meta.com/llama/)
  - description: Foundation language models (7B/13B/70B) used as base models for full-parameter fine-tuning in both stages.

- Code Llama · [GitHub](https://github.com/facebookresearch/codellama) · [Website](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)
  - description: Code-focused base models (7B/13B/34B/70B) used as alternative foundations (MuMath-Code-CL variants).

- DeepSpeed · [GitHub](https://github.com/microsoft/DeepSpeed) · [Website](https://www.deepspeed.ai/)
  - description: Distributed training framework used to train all models except the 70B variants.

- Megatron-LM · [GitHub](https://github.com/NVIDIA/Megatron-LM)
  - description: Large-scale training framework used for Llama/CodeLlama 70B runs for speed.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Core math word-problem dataset; its training split seeds augmentation and its test split is a main in-domain evaluation.

- MATH · [GitHub](https://github.com/hendrycks/math)
  - description: High-school math competition dataset; training split seeds augmentation and test split is a main in-domain evaluation.

- SVAMP · [GitHub](https://github.com/arkilpatel/SVAMP)
  - description: Out-of-domain math word problem benchmark used for generalization evaluation.

- TabMWP · [GitHub](https://github.com/lupantech/TabMWP)
  - description: Semi-structured (table) math word problems; used for out-of-domain evaluation.

- ASDiv · [GitHub](https://github.com/chaochun/nlu-asdiv-dataset)
  - description: Diverse arithmetic word problems; used for out-of-domain evaluation.

- MAWPS (SingleEQ, SingleOP, AddSub, MultiArith) · [GitHub](https://github.com/MAWPS/MAWPS)
  - description: Repository aggregating classic math word problem sets; subsets are used for out-of-domain evaluation (averaged as MAWPS).

- MMLU (Math subset) · [GitHub](https://github.com/hendrycks/test)
  - description: Multi-task benchmark; the Math categories are used to test transferability to multi-choice math.

- AQuA-RAT · [GitHub](https://github.com/deepmind/AQuA)
  - description: Algebra question answering with rationales; used to test transferability to multi-choice math reasoning.

- ToRA · [GitHub](https://github.com/microsoft/ToRA)
  - description: Tool-integrated reasoning agent with interleaved CoT+code; cited as a closely related baseline and template for interleaving used by MuMath-Code.

- MathCoder · [GitHub](https://github.com/OpenGVLab/MathCoder)
  - description: Interleaved CoT+code training/evaluation for math reasoning; referenced as a contemporary approach combining tool use and augmentation.

- MAmmoTH · [GitHub](https://github.com/TIGER-AI-Lab/MAmmoTH)
  - description: Hybrid instruction-tuned math generalist models; used as a baseline for comparison.

- PAL (Program-Aided Language Models) · [GitHub](https://github.com/reasoning-machines/pal)
  - description: Early tool-use method that delegates computation to Python; referenced as a representative tool-use baseline and for GSM-Hard.

- Program-of-Thoughts (PoT) Prompting · [GitHub](https://github.com/wenhuchen/Program-of-Thoughts)
  - description: Prompting approach that separates computation from reasoning via code; cited as an antecedent to code-nested solutions.

- MetaMath · [GitHub](https://github.com/hkust-nlp/MetaMath)
  - description: Tool-free math data augmentation and finetuning baseline; also used as an alternative Stage-1 checkpoint in two-stage ablations.

- Xwin-Math · [GitHub](https://github.com/Xwin-LM/Xwin-LM)
  - description: Math-optimized checkpoint used as an alternative Stage-1 initialization in ablations; repository documents Xwin-Math-7B-V1.0.

- SymPy · [GitHub](https://github.com/sympy/sympy) · [Doc](https://docs.sympy.org/)
  - description: Symbolic math library frequently used in synthesized code blocks for execution during data creation and model inference.

- NumPy · [GitHub](https://github.com/numpy/numpy) · [Website](https://numpy.org/)
  - description: Numerical computing library used within code-nested solutions.

- SciPy · [GitHub](https://github.com/scipy/scipy) · [Codewiki](https://codewiki.google/github.com/scipy/scipy) · [Website](https://scipy.org/)
  - description: Scientific computing library referenced as part of the tool stack employed in synthesized/executed code.

- CVXPY · [GitHub](https://github.com/cvxpy/cvxpy) · [Doc](https://www.cvxpy.org/)
  - description: Convex optimization library listed among packages used in the generated Python code for certain problems.

<!-- paper_id: 512a5c307fdab29112a0f4af5c94a3436632eda1 -->

## 180. Beyond Natural Language: LLMs Leveraging Alternative Formats for Enhanced Reasoning and Communication - EMNLP - 2024 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3context-engineering/2024_EMNLP_12_Findings_Beyond_Natural_Language_LLMs_Leveraging_Alternative_Formats_for_Enhanced_Reasoning_and_Communication.pdf
- Tags: context-engineering
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3context-engineering/2024_EMNLP_12_Findings_Beyond_Natural_Language_LLMs_Leveraging_Alternative_Formats_for_Enhanced_Reasoning_and_Communication.pdf
- Token Usage: input 19097, output 2713, total 21810

### GitHub & Websites

- AutoForm · [GitHub](https://github.com/thunlp/AutoForm)
  - description: Official code release for this paper; contains prompts and scripts to reproduce the single-LLM reasoning and multi-agent communication experiments.

- BIG-bench · [GitHub](https://github.com/google/BIG-bench)
  - description: Source for three reasoning datasets used in single-LLM experiments (Logic Grid Puzzle, Information Essentiality, Minute Mysteries QA); the paper downloads these tasks from the official repo.

- Coin Flip (skrishna/coin_flip) · [Website](https://huggingface.co/datasets/skrishna/coin_flip)
  - description: Symbolic reasoning dataset used for single-LLM experiments; the paper uses the first 500 test examples.

- AQuA (aqua_rat) · [Website](https://huggingface.co/datasets/aqua_rat)
  - description: Algebraic word problem dataset used as the mathematical reasoning benchmark in single-LLM experiments.

- HotpotQA (via Reflexion data) · [GitHub](https://github.com/noahshinn/reflexion/tree/main/hotpotqa_runs/data) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset used for the multi-agent communication setting; the paper obtains the evaluation split from the Reflexion repository.

- WikiHop (qangaroo/wikihop) · [Website](https://huggingface.co/datasets/qangaroo/wikihop)
  - description: Multi-hop QA dataset acquired via Hugging Face Datasets and used to evaluate multi-agent communication.

- NarrativeQA · [Website](https://huggingface.co/datasets/narrativeqa)
  - description: Reading comprehension dataset used in the multi-agent communication experiments; the paper filters examples to those sourced from Project Gutenberg and within context-length limits.

- Project Gutenberg · [Website](https://www.gutenberg.org/)
  - description: Public-domain book source referenced for filtering NarrativeQA examples (the paper keeps ebooks starting with “Project Gutenberg’s”).

- OpenAI API (GPT-3.5, GPT-4) · [Doc](https://platform.openai.com/docs)
  - description: Model API used to run gpt-3.5-turbo-1106 and gpt-4-1106-preview across all experiments.

- Google Gemini API (Gemini Pro) · [Doc](https://ai.google.dev/gemini-api/docs)
  - description: API used to run Gemini Pro 1.0 in the single-LLM reasoning experiments.

- KQML (Knowledge Query and Manipulation Language) · [Website](http://www.cs.umbc.edu/kqml/)
  - description: Classic agent communication language; the paper analyzes alignment with LLM-devised formats and prompts agents to communicate in a KQML-like structure for comparison.

- FIPA ACL (Agent Communication Language) · [Doc](https://www.fipa.org/specs/fipa00061/SC00061G.html)
  - description: Standardized agent communication message structure; referenced for comparison with the structured formats LLMs autonomously adopt under AutoForm.

<!-- paper_id: bef263083bf5ea965c37b152bc5f0b43aaf74824 -->

## 181. Can Large Language Models Grasp Legal Theories? Enhance Legal Reasoning with Insights from Multi-Agent Collaboration - EMNLP - 2024 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_11_Findings_Can_Large_Language_Models_Grasp_Legal_Theories_Enhance_Legal_Reasoning_with_Insights_from_Multi-Agent_Collaboration.pdf
- Tags: multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_11_Findings_Can_Large_Language_Models_Grasp_Legal_Theories_Enhance_Legal_Reasoning_with_Insights_from_Multi-Agent_Collaboration.pdf
- Token Usage: input 21014, output 4539, total 25553

### GitHub & Websites

- MALR (Multi-Agent framework for improving complex Legal Reasoning) · [GitHub](https://github.com/yuanwk99/MALR)
  - description: Official code release for the paper’s multi-agent, non-parametric framework (auto-planner, sub-task agents, adaptive rule-insights, reasoning modules) used to reproduce experiments and ablations.

- CAIL2018 (China AI and Law Challenge 2018) · [GitHub](https://github.com/thunlp/CAIL2018) · [Website](http://cail.cipsc.org.cn)
  - description: Large-scale Chinese legal judgment dataset used in this paper (400 sampled cases) for confusing charge prediction; legal rules and charges are matched from this dataset.

- CJO dataset
  - description: A Chinese legal dataset constructed to mitigate data leakage (sourced from the same origin as CAIL2018); the paper samples 100 cases from CJO for evaluation of confusing charge prediction.

- CAIL-I (Innocent cases dataset)
  - description: Dataset of 462 innocent cases paired with the most similar charges from An et al. (2022); used to test whether models adhere to legal rules under presumption of innocence.

- Sentence-Transformers (SBERT) · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Doc](https://www.sbert.net/)
  - description: Library used to compute semantic similarity between legal rules (via Sentence-BERT) for retrieving similar rules and transferring rule-insights across datasets.

- OpenAI GPT-3.5/GPT-4 API · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Base LLMs used for all primary experiments (zero-shot, few-shot, and within MALR), with temperature set to 0.

- Qwen2 (Open-source LLM family) · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://qwenlm.ai)
  - description: Open-source models used to evaluate MALR across sizes (1.5B–72B) and analyze scaling behavior and effectiveness on smaller LLMs.

- Farui-200B · [Website](https://tongyi.aliyun.com/farui)
  - description: A legal-domain fine-tuned LLM (based on Qwen) used as an external knowledgeable expert for providing knowledge feedback during the “ask” step in MALR.

- Lawformer · [GitHub](https://github.com/thunlp/LAWformer)
  - description: Chinese legal-domain pre-trained model used in the paper’s additional comparisons on general charge prediction versus confusing charge prediction.

- legal-xlm-roberta-base (Hugging Face model) · [Website](https://huggingface.co/joelito/legal-xlm-roberta-base)
  - description: Multilingual legal-domain model referenced in the paper’s additional experiments for baseline comparisons on the CAIL2018 dataset.

<!-- paper_id: 6a4e9df8f0f06713282d1c4d63dc053de450d1e7 -->

## 182. Inducing Programmatic Skills for Agentic Tasks - COLM - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_COLM_14_Inducing_Programmatic_Skills_for_Agentic_Tasks.pdf
- Tags: agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_COLM_14_Inducing_Programmatic_Skills_for_Agentic_Tasks.pdf
- Token Usage: input 22447, output 4011, total 26458

### GitHub & Websites

- Agent Skill Induction (ASI) · [GitHub](https://github.com/zorazrw/agent-skill-induction)
  - description: Official code release for this paper; implements programmatic skill induction, verification, and agent execution over BrowserGym/WebArena to reproduce results.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Realistic web environment and benchmark used for all main evaluations; provides sandbox sites and program-based evaluators the paper relies on.

- BrowserGym · [GitHub](https://github.com/ServiceNow/browsergym)
  - description: Web agent research framework used to run agents; supplies the accessibility-tree observations and default action space used by all methods.

- Agent Workflow Memory (AWM) · [Website](https://arxiv.org/abs/2409.07429)
  - description: Adaptive agent baseline that induces textual skills; used for comparison against ASI throughout experiments and ablations.

- Anthropic Claude 3.5 Sonnet · [Website](https://www.anthropic.com) · [Doc](https://docs.anthropic.com/claude)
  - description: LLM backbone for the agent policy, the neural evaluator, and the induction module in all experiments.

- Google Maps · [Website](https://maps.google.com)
  - description: Real-world website used in the cross-website generalization experiments (WebArena OpenStreetMap → Google Maps).

- Target · [Website](https://www.target.com)
  - description: Real-world shopping website used for cross-website evaluation (WebArena OneStopMarket → Target).

- Reddit · [Website](https://www.reddit.com)
  - description: Real-world social forum used for cross-website evaluation (WebArena PostMill → Reddit).

- OpenStreetMap · [Website](https://www.openstreetmap.org)
  - description: Map website used as the sandboxed counterpart within WebArena domains evaluated in the paper.

<!-- paper_id: eda5889e6ebcdd761512d1b544c4adeccb9a1981 -->

## 183. Triad: A Framework Leveraging a Multi-Role LLM-based Agent to Solve Knowledge Base Question Answering - EMNLP - 2024 - citation_count 24 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2024_EMNLP_24_Main_Triad_A_Framework_Leveraging_a_Multi-Role_LLM-based_Agent_to_Solve_Knowledge_Base_Question_Answering.pdf
- Tags: agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2024_EMNLP_24_Main_Triad_A_Framework_Leveraging_a_Multi-Role_LLM-based_Agent_to_Solve_Knowledge_Base_Question_Answering.pdf
- Token Usage: input 15082, output 4048, total 19130

### GitHub & Websites

- Triad · [GitHub](https://github.com/ZJU-DCDLab/Triad)
  - description: Official code and data release for this paper; implements the multi‑role LLM agent framework (G-/D-/A-Agent), prompts, and scripts used in all experiments.

- DBpedia · [GitHub](https://github.com/dbpedia) · [Website](https://www.dbpedia.org)
  - description: Open knowledge base extracted from Wikipedia used as a primary KB in experiments (DBpedia-04/DBpedia-10); Triad indexes DBpedia and queries it via SPARQL.

- YAGO 4 · [GitHub](https://github.com/yago-naga/yago-4) · [Website](https://yago-knowledge.org)
  - description: Reason-able knowledge base used as a second KB (YAGO-4) in experiments; Triad indexes and queries it as part of the KBQA pipeline.

- LC-QuAD 1.0 · [GitHub](https://github.com/AskNowQA/LC-QuAD) · [Website](http://lc-quad.sda.tech/)
  - description: Benchmark dataset of natural-language questions with SPARQL over DBpedia; used to evaluate Triad.

- QALD-9 · [Website](http://qald.aksw.org/)
  - description: Ninth challenge dataset for Question Answering over Linked Data (DBpedia); used as a benchmark to evaluate Triad.

- YAGO-QA (from KGQAn) · [GitHub](https://github.com/qcri/KGQAn)
  - description: Dataset of questions with annotated SPARQL over YAGO introduced by Omar et al.; used as a benchmark, and the KGQAn repo hosts resources associated with the dataset and baseline system.

- KGQAn · [GitHub](https://github.com/qcri/KGQAn)
  - description: Universal QA platform for knowledge graphs used as a comparison baseline and source for the YAGO-QA benchmark referenced in the paper.

- EDGQA
  - description: Baseline system based on entity description graphs for complex KBQA; included for performance comparison in the paper.

- gAnswer
  - description: Baseline KBQA system that answers natural language questions via subgraph matching; included for performance comparison in the paper.

- OpenAI API Models (GPT‑3.5 Turbo, GPT‑4) · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs/models)
  - description: Commercial LLMs accessed via the OpenAI API to power Triad’s agents and the pure-LLM baselines; the paper evaluates Triad with both GPT‑3.5 and GPT‑4.

- Virtuoso Open-Source Edition · [Website](https://virtuoso.openlinksw.com/) · [Doc](https://vos.openlinksw.com/owiki/Wiki.jsp?page=VOS)
  - description: SPARQL endpoint used to load KB triples (DBpedia/YAGO) and execute candidate and final SPARQL queries in Triad.

- Elasticsearch · [Website](https://www.elastic.co/elasticsearch) · [Doc](https://www.elastic.co/guide/en/elasticsearch/reference/7.5/index.html)
  - description: Full-text search engine used to index entity and relation surface forms and perform text matching during the D‑Agent’s URI linking phase.

<!-- paper_id: 04847af01a0ba0beadb98d36a3059b7cac701f1a -->

## 184. Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation - ACL - 2025 - citation_count 10 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_10_Long_Divide-Then-Aggregate_An_Efficient_Tool_Learning_Method_via_Parallel_Tool_Invocation.pdf
- Tags: agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_10_Long_Divide-Then-Aggregate_An_Efficient_Tool_Learning_Method_via_Parallel_Tool_Invocation.pdf
- Token Usage: input 28055, output 5033, total 33088

### GitHub & Websites

- DTA-Llama / DTA-Tool (official release) · [Website](https://corn0205.github.io/)
  - description: Official project page for this paper; hosts the released code, DTA-Tool dataset (DAG-style parallel tool-invocation), and model weights needed to reproduce DTA-Llama.

- ToolBench / ToolLLM · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolbench.github.io/)
  - description: Comprehensive multi-tool learning dataset and toolkit used to construct DTA-Tool (training data transformed from ToolBench) and as the source of APIs; also provides DFSDT search and baseline implementations referenced in the method.

- StableToolBench · [Website](https://arxiv.org/abs/2403.07714)
  - description: Evaluation benchmark the paper uses for all experiments; includes a caching system and an API simulator to stabilize real-time tool evaluation.

- LLMCompiler · [GitHub](https://github.com/mit-han-lab/llm-compiler)
  - description: Non-training-based system for parallel function calling; used as an open-source baseline for comparison with DTA-Llama.

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io/)
  - description: Open-source LLM family with function-calling and parallel tool invocation; Qwen2.5-7B-Instruct is evaluated as a strong open-source baseline.

- Meta Llama (Llama 2 / Llama 3) · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: Backbone models fine-tuned to obtain DTA-Llama; the paper reports results for Llama2-7B, Llama2-13B, and Llama3-8B.

- OpenAI GPT models and Function Calling · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs/guides/function-calling)
  - description: GPT‑3.5/4 with function-calling are used as closed-source baselines; GPT‑4‑turbo is also used to convert serial tool paths to DAGs during data construction.

- ReAct (Reason+Act) · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Widely used tool-using/agent prompting framework; serves as one of the baseline paradigms compared against DTA-Llama.

<!-- paper_id: 96073c37cc32c0efc143762346286325f9cb7c5b -->

## 185. Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs - ICLR - 2025 - citation_count 25 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_25_Poster_Motion-Agent_A_Conversational_Framework_for_Human_Motion_Generation_with_LLMs.pdf
- Tags: agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_25_Poster_Motion-Agent_A_Conversational_Framework_for_Human_Motion_Generation_with_LLMs.pdf
- Token Usage: input 24235, output 4806, total 29041

### GitHub & Websites

- Motion-Agent · [Website](https://knoxzhao.github.io/Motion-Agent)
  - description: Official project page for the paper; hosts demos/supplementary material for the Motion-Agent conversational framework and MotionLLM used throughout the work.

- HumanML3D · [GitHub](https://github.com/EricGuo5513/HumanML3D)
  - description: Dataset of 3D human motions with text descriptions; main training and evaluation set for MotionLLM and Motion-Agent.

- KIT Motion-Language (KIT-ML) · [Website](https://motion-database.humanoids.kit.edu/)
  - description: Motion–language dataset used for additional experiments and ablations in the paper.

- AMASS · [Website](https://amass.is.tue.mpg.de/)
  - description: Large motion capture archive; cited as the source from which HumanML3D sequences are derived.

- HumanAct12 · [GitHub](https://github.com/EricGuo5513/action-to-motion)
  - description: Action-conditioned motion dataset contributing sequences to HumanML3D as noted in the experimental setup.

- Text-to-Motion (T2M) · [GitHub](https://github.com/EricGuo5513/text-to-motion)
  - description: Baseline and evaluation toolkit whose pretrained encoders/metrics (R-precision, MM-Dist, FID, Diversity) are used to evaluate MotionLLM.

- MotionGPT · [GitHub](https://github.com/OpenMotionLab/MotionGPT)
  - description: Bidirectional motion–text model used as a comparison baseline; also substituted into Motion-Agent for an ablation of the agent component.

- MotionChain · [GitHub](https://github.com/OpenMotionLab/MotionChain)
  - description: Conversational motion controller baseline compared in functionality (multi-turn editing/reasoning) against Motion-Agent.

- MoMask · [GitHub](https://github.com/OpenMotionLab/MoMask)
  - description: Token-based motion generation baseline compared in quantitative and qualitative results; discussed regarding its need for manual/estimated length at inference.

- MDM (Motion Diffusion Model) · [GitHub](https://github.com/GuyTevet/motion-diffusion-model)
  - description: Diffusion-based text-to-motion baseline included in quantitative comparisons.

- GPT-4 · [Website](https://openai.com/research/gpt-4) · [Doc](https://platform.openai.com/docs/models/gpt-4)
  - description: Used as the conversational coordinator/planner in Motion-Agent to decompose user instructions and call MotionLLM without additional training.

- Gemma 2 · [Website](https://ai.google.dev/gemma) · [Doc](https://ai.google.dev/gemma/docs)
  - description: Open-source LLM backbone (Gemma2-2B-IT and others) fine-tuned via LoRA to build MotionLLM for text–motion translation.

- Llama · [Website](https://ai.meta.com/llama/)
  - description: Alternative LLM backbone evaluated in ablations for the conversational component and MotionLLM variants.

- Mixtral (Mistral AI) · [Website](https://mistral.ai/)
  - description: Another LLM used in ablations to replace GPT-4 for conversation/planning in Motion-Agent.

- NLG-Eval · [GitHub](https://github.com/Maluuba/nlg-eval)
  - description: Toolkit referenced as “NLPEval” in the appendix; used to compute linguistic captioning metrics (BLEU, ROUGE, CIDEr, BERTScore) for motion-to-text evaluation.

<!-- paper_id: 660af93b31f0774b8e00267cb15731224ead9c46 -->

## 186. AgentsCourt: Building Judicial Decision-Making Agents with Court Debate Simulation and Legal Knowledge Augmentation - EMNLP - 2024 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_15_Findings_AgentsCourt_Building_Judicial_Decision-Making_Agents_with_Court_Debate_Simulation_and_Legal_Knowledge_Augmentation.pdf
- Link: http://arxiv.org/pdf/2403.02959
- Tags: multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2024_EMNLP_15_Findings_AgentsCourt_Building_Judicial_Decision-Making_Agents_with_Court_Debate_Simulation_and_Legal_Knowledge_Augmentation.pdf
- Token Usage: input 22118, output 2982, total 25100

### GitHub & Websites

- AgentsCourt / SimuCourt · [GitHub](https://github.com/Zhitao-He/SimuCourt)
  - description: Official release of the paper, containing the multi-agent AgentsCourt framework and the SimuCourt benchmark; includes code for court debate simulation, retrieval, and evaluation, plus the constructed Legal-KB resources used in experiments.

- China Judgements Online
  - [Website](https://wenshu.court.gov.cn/)
  - description: Official source of Chinese court judgments; used to collect 420 cases for SimuCourt and millions of precedents for Legal-KB.

- National Laws and Regulations Database of China
  - [Website](https://flk.npc.gov.cn/)
  - description: Authoritative repository of Chinese laws, regulations, and judicial interpretations; used to build the laws/regulations portion of Legal-KB.

- Chinese Legal Resources Knowledge Database (CNKI Legal)
  - [Website](https://lawnew.cnki.net/)
  - description: Source of highly cited legal journal articles (2010–2023); used to add expert analyses to Legal-KB.

- Pyserini (BM25) · [GitHub](https://github.com/castorini/pyserini) · [Website](https://pyserini.io/)
  - description: IR toolkit used to implement BM25 rough retrieval over Legal-KB; the assistant agent first retrieves top candidates with BM25 before re-ranking.

- BGE (FlagEmbedding) · [GitHub](https://github.com/FlagOpen/FlagEmbedding) · [Website](https://huggingface.co/BAAI/bge-large-zh)
  - description: Sentence-embedding model (BGE-Large) used to encode and re-rank BM25 candidates to select the optimal precedent and extract related legal articles.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting baseline framework; used as a comparison baseline in the experiments.

- AutoGPT · [GitHub](https://github.com/Significant-Gravitas/AutoGPT) · [Website](https://auto-gpt.ai/)
  - description: Popular autonomous agent framework; adopted as a baseline system for comparison.

- LaWGPT · [GitHub](https://github.com/pengxiao-song/LaWGPT)
  - description: Chinese legal large language model baseline evaluated against AgentsCourt; trained/fine-tuned on legal corpora and instructions.

- OpenAI GPT-3.5/GPT-4
  - [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs/)
  - description: Proprietary LLMs used as backbones/vanilla models in experiments (gpt-3.5-turbo-1106 for agents; gpt-4-1106-preview as a stronger vanilla and as an evaluator for civil/administrative judgments).

<!-- paper_id: 79d72cf13cc86dd46498384b32c8ac77898a23f9 -->

## 187. ReAct: Synergizing Reasoning and Acting in Language Models - ICLR - 2023 - citation_count 4496 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2023_ICLR_4496_ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf
- Tags: prompt-engineering
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2023_ICLR_4496_ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf
- Token Usage: input 34457, output 3765, total 38222

### GitHub & Websites

- ReAct (project page with code) · [Website](https://react-lm.github.io/)
  - description: Official project page for the paper; hosts the released prompts/code used for ReAct prompting, demos, and links referenced in the paper.

- ReAct (GPT‑3 prompting code, anonymous release) · [Website](https://anonymous.4open.science/r/ReAct-2268/)
  - description: Code the authors provided for reproducing GPT‑3 ReAct prompting experiments, cited in the paper’s reproducibility section.

- HotpotQA · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop question answering dataset used to evaluate ReAct’s reasoning + acting with a Wikipedia API.

- FEVER · [Website](https://fever.ai/)
  - description: Fact verification dataset (SUPPORTS/REFUTES/NEI) on which ReAct interacts with Wikipedia to ground its reasoning.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld) · [Website](https://alfworld.github.io/)
  - description: Text-based household environment used to evaluate ReAct for long-horizon interactive decision making across six task types.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Realistic online shopping website environment with 1.18M products and 12k instructions; used to test ReAct’s web navigation and compared against IL/IL+RL baselines.

- Wikipedia API (MediaWiki API) · [Doc](https://www.mediawiki.org/wiki/API:Main_page)
  - description: External knowledge source ReAct interacts with via search[entity] and lookup[string] actions to retrieve evidence during HotpotQA and FEVER.

<!-- paper_id: 99832586d55f540f603637e458a292406a0ed75d -->

## 188. Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model - ACL - 2025 - citation_count 13 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_13_Long_Can_a_Single_Model_Master_Both_Multi-turn_Conversations_and_Tool_Use_CoALM_A_Unified_Conversational_Agentic_Language_Model.pdf
- Tags: agent-tools
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-tools/2025_ACL_13_Long_Can_a_Single_Model_Master_Both_Multi-turn_Conversations_and_Tool_Use_CoALM_A_Unified_Conversational_Agentic_Language_Model.pdf
- Token Usage: input 24936, output 6392, total 31328

### GitHub & Websites

- CoALM (Conversational Agentic Language Model) · [Website](https://emrecanacikgoz.github.io/CoALM/)
  - description: Official project page for this paper; the authors state they release code, model weights, datasets (including CoALM-IT), intermediate checkpoints, and training configurations for reproducing their unified conversational-agent models.

- Oumi · [GitHub](https://github.com/oumi-ai/oumi) · [Website](https://oumi.ai)
  - description: Open, end-to-end training platform used by the authors to fine-tune and scale CoALM models; enables reproducible training pipelines.

- Llama 3.x Instruct (bases used for CoALM) · [Website](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) · [Website](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) · [Website](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
  - description: Base checkpoints fine-tuned to obtain CoALM 8B, 70B, and 405B; required to reproduce model training.

- MultiWOZ 2.4 · [GitHub](https://github.com/smartyfh/MultiWOZ2.4)
  - description: TOD benchmark used to evaluate Success and JGA; the paper runs zero-shot evaluations on its 2.4 test set.

- API-Bank · [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/API-Bank)
  - description: Function-calling benchmark used for Level-1/Level-2 evaluations; authors follow official evaluation setups (via baseline repos) when reporting results.

- Berkeley Function Calling Leaderboard (BFCL V3) · [Website](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) · [Website](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)
  - description: Multi-turn function-calling benchmark used for AST accuracy, executable accuracy, live and relevance metrics; authors use the benchmark-provided handlers for fair comparison.

- SNIPS NLU (repurposed for DST) · [GitHub](https://github.com/snipsco/nlu-benchmark)
  - description: Single-turn dataset transformed by the authors into dialogue state tracking format for CoALM-IT training.

- Schema-Guided Dialogue (SGD) · [GitHub](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
  - description: Multi-turn dialogue corpus used to generate the paper’s CRA (Conversational ReAct API) data with GPT-4o, forming a key part of CoALM-IT.

- ToolACE · [Website](https://huggingface.co/Team-ACE/ToolACE-8B) · [Website](https://huggingface.co/datasets/Team-ACE/ToolACE)
  - description: Baseline function-calling model and its dataset; the paper uses ToolACE data in CoALM-IT and compares against the ToolACE model on API-Bank and BFCL.

- Hammer 2.0 · [GitHub](https://github.com/MadeAgents/Hammer) · [Website](https://huggingface.co/MadeAgents/Hammer2.0-7b)
  - description: Baseline function-calling system; the paper incorporates Hammer-style data for CoALM-IT training and evaluates against Hammer on API-Bank and BFCL.

- XLAM Function-Calling 60K (Salesforce) · [Website](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
  - description: Dataset used by Hammer training (noted in the paper’s baseline reproduction notes); useful reference for practitioners examining function-calling data composition.

- LDST (LLM-driven Dialogue State Tracking) · [GitHub](https://github.com/WoodScene/LDST)
  - description: TOD baseline referenced in the experiments; provides checkpoints and code for LLM-based state tracking.

- FncTOD-Llama-13b · [Website](https://huggingface.co/Zekunli/FncTOD-Llama-13b)
  - description: Zero-shot TOD baseline (function-calling style DST) used for comparison on MultiWOZ; authors evaluate against it.

- NC-Latent-TOD · [Website](https://huggingface.co/Brendan/nc-latent-tod-step-2-final) · [Website](https://huggingface.co/Brendan/tod-zero-bqag3oyb-32000)
  - description: Unsupervised end-to-end TOD baseline; authors compare zero-shot JGA/Success on MultiWOZ using these shared checkpoints.

- CodeActAgent (Mistral-7B) · [Website](https://huggingface.co/xingyaoww/CodeActAgent-Mistral-7b-v0.1)
  - description: Function-calling/code-action baseline; included in comparisons on TOD/LA evaluations.

- Granite-20B-Code-Instruct-8k · [Website](https://huggingface.co/ibm-granite/granite-20b-code-instruct-8k)
  - description: Function-calling baseline model evaluated by the authors on API-Bank/BFCL.

- Mistral-7B-Instruct-v0.3 · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  - description: Open-source baseline LLM used in comparisons.

- EleutherAI LM Evaluation Harness · [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
  - description: Framework the authors used to assess general reasoning/knowledge (MMLU, ARC, GSM8K, HellaSwag) for catastrophic-forgetting analysis.

- bitsandbytes · [GitHub](https://github.com/TimDettmers/bitsandbytes)
  - description: Library used for QLoRA (nf4) during CoALM 405B fine-tuning, as described in the training details.

<!-- paper_id: 3076cfba160fc4d64eec459e2f99b307b72cb12a -->

## 189. Ask-before-Plan: Proactive Language Agents for Real-World Planning - EMNLP - 2024 - citation_count 29 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2024_EMNLP_29_Findings_Ask-before-Plan_Proactive_Language_Agents_for_Real-World_Planning.pdf
- Tags: planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2024_EMNLP_29_Findings_Ask-before-Plan_Proactive_Language_Agents_for_Real-World_Planning.pdf
- Token Usage: input 32562, output 4053, total 36615

### GitHub & Websites

- Ask-before-Plan · [GitHub](https://github.com/magicgh/Ask-before-Plan)
  - description: Official repository for the paper; releases the Ask-before-Plan dataset, CEP multi-agent code, prompts, and environment modifications used to reproduce results.

- TravelPlanner · [GitHub](https://github.com/OSU-NLP-Group/TravelPlanner) · [Website](https://osu-nlp-group.github.io/TravelPlanner/)
  - description: Real-world travel planning benchmark and simulator on which Ask-before-Plan is built; the authors adapt its tools/environment and evaluation for their new dataset and tasks.

- ToolBench / ToolLLM (ToolLLaMA) · [GitHub](https://github.com/OpenBMB/ToolBench) · [Website](https://toolbench.github.io/) · [Doc](https://huggingface.co/ToolBench/ToolLLaMA-2-7b-v2)
  - description: Toolkit and dataset for tool-use LLMs; the paper fine-tunes ToolLLaMA (ToolLLM) and uses it as a baseline for the static execution/tool-learning subtask.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting prompting framework used by the paper as a baseline for dynamic tool interaction and dynamic planning.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Self-reflection framework used as a dynamic execution/planning baseline; the paper’s memory recollection mechanism is compared against it.

- Mistral-7B-Instruct-v0.2 · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - description: Open-source instruction-tuned LLM used as one of the base models for fine-tuning CEP and for multiple baselines.

- Meta Llama 3 8B Instruct · [Website](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - description: Open-source instruction-tuned LLM used as the stronger base model for CEP; achieves the best results in several subtasks.

- OpenAI GPT-3.5 / GPT-4 Turbo · [Doc](https://platform.openai.com/docs/models)
  - description: Proprietary API models used for simulated dialogues (GPT-3.5), several prompting baselines, and GPT-4-based question quality evaluation.

<!-- paper_id: 367e43d1561fce27c919e2d370e42399a40846bd -->

## 190. Revealing the Barriers of Language Agents in Planning - NAACL - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_NAACL_14_Long_Revealing_the_Barriers_of_Language_Agents_in_Planning.pdf
- Tags: planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_NAACL_14_Long_Revealing_the_Barriers_of_Language_Agents_in_Planning.pdf
- Token Usage: input 21740, output 4366, total 26106

### GitHub & Websites

- Revealing the Barriers of Language Agents in Planning (paper resources)
  - description: The authors state “Resources are available on the GitHub.” These materials should include code/scripts for their attribution analysis, memory-updating setups, and prompts used in experiments.

- TravelPlanner Benchmark · [GitHub](https://github.com/OSU-NLP-Group/TravelPlanner) · [Website](https://travelplanner-bench.github.io)
  - description: Real-world travel planning benchmark used as a primary evaluation bed; the paper uses its “sole-planning” mode, training split (45) and validation split (180), and its evaluation protocol and hard/commonsense constraints.

- PlanBench (BlocksWorld) · [GitHub](https://github.com/karthikvalmeekam/planbench)
  - description: Planning benchmark suite providing the BlocksWorld tasks, domain description, and solver-generated ground-truth plans/feedback; used for training (100), validation (500), and attribution analyses of action/constraint tokens.

- OpenAI Platform (o1, GPT‑4o, GPT‑4o‑mini) · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs/guides/fine-tuning)
  - description: Closed-source models and official fine-tuning pipeline used for parametric memory updating and inference; the paper reports results for o1, GPT‑4o, and GPT‑4o‑mini and fine-tunes via the official scripts.

- Meta Llama 3.1 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com/)
  - description: Open-source LLM family used at 8B/70B/405B scales for baseline and analysis; the 405B variant is run via Vertex AI and others are locally fine-tuned/evaluated.

- Qwen2 · [GitHub](https://github.com/QwenLM/Qwen2) · [Website](https://qwenlm.github.io/)
  - description: Open-source LLM family (7B/72B) used as baselines and for episodic/parametric memory-updating experiments and attribution analyses.

- Google Vertex AI · [Website](https://cloud.google.com/vertex-ai) · [Doc](https://cloud.google.com/vertex-ai/docs)
  - description: Managed inference platform used to run Llama3.1‑405B for experiments while attribution scores are computed locally.

<!-- paper_id: 744647fde296303c738f58d3a4093cf54546ec1a -->

## 191. Towards Hierarchical Multi-Agent Workflows for Zero-Shot Prompt Optimization - ICLR - 2025 - citation_count 14 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_14_Workshop_Towards_Hierarchical_Multi-Agent_Workflows_for_Zero-Shot_Prompt_Optimization.pdf
- Tags: multi-agent
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3multi-agent/2025_ICLR_14_Workshop_Towards_Hierarchical_Multi-Agent_Workflows_for_Zero-Shot_Prompt_Optimization.pdf
- Token Usage: input 21127, output 4926, total 26053

### GitHub & Websites

- HMAW (Hierarchical Multi-Agent Workflow) · [Website](https://liuyvchi.github.io/HMAW_project/)
  - description: Official project page announced in the paper; central resource for materials to reproduce or extend the HMAW workflow (prompts, examples, and any future code/demos).

- Mixtral-8x7B-v0.1 · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)
  - description: Open-source LLM used as the main base model for all primary experiments; the model card provides downloads and usage details for local inference.

- OpenAI GPT-3.5 / GPT-4o · [Doc](https://platform.openai.com/docs)
  - description: Models used as LLM agents in ablations and as the automatic evaluator; API documentation for running evaluations and reproducing comparisons.

- GSM8K (Grade School Math 8K) · [GitHub](https://github.com/openai/grade-school-math) · [Website](https://huggingface.co/datasets/openai/gsm8k)
  - description: Math reasoning benchmark on which the paper reports test accuracy; dataset repo and dataset card for downloading and evaluation.

- Project CodeNet · [GitHub](https://github.com/IBM/Project_CodeNet) · [Website](https://developer.ibm.com/exchanges/data/all/project-codenet/)
  - description: Large-scale code dataset used for the code readability task (the paper follows prior work to sample a Python subset); official dataset repository and information page.

- ATLAS (Principled Instructions Are All You Need for Questioning LLaMA/GPT) · [Website](https://arxiv.org/abs/2312.16171)
  - description: Prompt-evaluation benchmark used in the paper to assess generalization across domains; paper/project page describing the dataset and setup.

- FED (Unsupervised Evaluation of Interactive Dialog with DialoGPT) · [Website](https://arxiv.org/abs/2006.12719)
  - description: Dialog evaluation dataset with human-human and human-system conversations used for subjective preference comparisons in the paper.

- Automatic Prompt Engineer (APE) · [GitHub](https://github.com/keirp/automatic_prompt_engineer)
  - description: Baseline method compared in the experiments; code for generating and selecting prompts based on LLM feedback.

<!-- paper_id: 3d70dab237a3b555f98afc846d7c599163348fbc -->

## 192. Scaling Autonomous Agents via Automatic Reward Modeling And Planning - ICLR - 2025 - citation_count 10 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_ICLR_10_Poster_Scaling_Autonomous_Agents_via_Automatic_Reward_Modeling_And_Planning.pdf
- Tags: planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_ICLR_10_Poster_Scaling_Autonomous_Agents_via_Automatic_Reward_Modeling_And_Planning.pdf
- Token Usage: input 31888, output 4619, total 36507

### GitHub & Websites

- ARMAP (Automatic Reward Modeling And Planning) · [Website](https://armap-agent.github.io)
  - description: Official project page for the paper; authors note “We will release all the code, model, and data for easy reproduction upon acceptance,” and the page hosts the project resources and updates.

- VILA (Visual Language Model) · [GitHub](https://github.com/Efficient-Large-Model/VILA) · [Website](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b)
  - description: Backbone used to build the reward model; the paper fine-tunes a small VILA model (e.g., VILA1.5-3B) to score trajectories.

- LLaVA · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA)
  - description: Alternative VLM baseline used in ablations for reward modeling (compared against VILA-13B).

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Web-shopping simulation environment and benchmark; used as one of the main evaluation environments.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld) · [Website](https://allenai.org/data/scienceworld)
  - description: Text-based science experiment environment; used for seen/unseen split evaluations and for generating reward-model training data.

- Tree-of-Thoughts (Game of 24) · [GitHub](https://github.com/ysymyth/tree-of-thought-llm)
  - description: Repository providing Game of 24 evaluation used in the paper; authors follow Yao et al. (2023b) to evaluate with 100 hard puzzles and build the Game24 environment.

- AgentBench · [GitHub](https://github.com/THUDM/AgentBench) · [Website](https://agentbench.dev)
  - description: Benchmark and Dockerized environments; used to launch WebShop and ALFWorld setups and to follow evaluation protocols.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
  - description: Household instruction-following simulator; used in the appendix experiments to test ARMAP on embodied tasks.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Website](https://vllm.ai)
  - description: LLM serving library used to host open-source models locally (e.g., Llama, Mistral, Phi) for both data synthesis and agent policies.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Planning framework baseline integrated by the authors (denoted ARMAP-R) to compare/test reward-guided planning.

- Meta Llama 3.1 70B Instruct (AWQ INT4) · [Website](https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4)
  - description: Primary LLM used to synthesize reward-model training data and as a strong agent backbone in evaluations.

- Meta Llama 3.1 8B Instruct · [Website](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)
  - description: Smaller LLM agent backbone evaluated with ARMAP planning.

- Mistral 7B Instruct v0.3 · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  - description: Open-source LLM agent backbone used in experiments to test ARMAP’s effectiveness on weaker models.

- Phi-3.5-mini-instruct · [Website](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
  - description: Lightweight LLM agent backbone; used to demonstrate larger relative gains from ARMAP on smaller models.

<!-- paper_id: 0846688b77c8a9f691a9feb4917ba9f6fe4c6360 -->

## 193. ATLaS: Agent Tuning via Learning Critical Steps - ACL - 2025 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_12_Findings_ATLaS_Agent_Tuning_via_Learning_Critical_Steps.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_12_Findings_ATLaS_Agent_Tuning_via_Learning_Critical_Steps.pdf
- Token Usage: input 21948, output 4963, total 26911

### GitHub & Websites

- AgentTraj-L (AgentGym) · [Website](https://huggingface.co/datasets/AgentGym/AgentTraj-L)
  - description: The expert-trajectory dataset the paper uses as its unfiltered source (Do) across all held-in tasks; ATLAS selects critical steps from this dataset for fine-tuning.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting prompting framework adopted for the agent policy format during training and evaluation.

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
  - description: Embodied household text environment used as one of the held-in benchmarks for training/evaluation.

- BabyAI · [GitHub](https://github.com/mila-iqia/babyai)
  - description: Grid-world instruction-following environment used as a held-in benchmark; also used in ablations for value-function and critical-step verification.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld) · [Doc](https://scienceworld.readthedocs.io/)
  - description: Interactive science reasoning environment included among held-in tasks for training/evaluation.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: Web navigation/shopping environment used as a held-in benchmark in the experiments.

- Jericho · [GitHub](https://github.com/microsoft/jericho)
  - description: Text-game framework used as one of the held-out evaluation environments (with tasks shortened per the paper’s setup).

- TextWorld · [GitHub](https://github.com/microsoft/TextWorld)
  - description: Text-based game framework that ALFWorld is built upon; relevant for reproducing ALFWorld experiments described.

- Open-Meteo API (Weather) · [Website](https://open-meteo.com/) · [Doc](https://open-meteo.com/en/docs)
  - description: Weather data API used by the “Weather” tool-using environment for querying meteorological information.

- The Movie Database (TMDB) API (Movie) · [Website](https://www.themoviedb.org/) · [Doc](https://developer.themoviedb.org/reference/intro/getting-started)
  - description: Film metadata API used by the “Movie” tool-using environment.

- Google Sheets API (Sheet) · [Website](https://developers.google.com/sheets/api) · [Doc](https://developers.google.com/sheets/api/guides/concepts)
  - description: Spreadsheet manipulation API used by the held-out “Sheet” environment.

- Llama-3.1-8B-Instruct · [Website](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  - description: Primary open-source backbone model fine-tuned with ATLAS on critical steps.

- Mistral-7B-Instruct-v0.3 · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
  - description: Alternative backbone evaluated to test ATLAS’s robustness across models.

- Qwen2.5-7B-Instruct · [Website](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  - description: Another backbone used to validate ATLAS across different architectures.

- Llama-3.1-70B-Instruct · [Website](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
  - description: Open-source selector model used in an ablation for critical-step selection (compared to GPT-4o).

- AgentTuning / AgentLM · [GitHub](https://github.com/THUDM/AgentTuning)
  - description: Baseline fine-tuned agent models and datasets compared against ATLAS; relevant for reproducing baseline results.

- AgentOhana / xLAM · [GitHub](https://github.com/Agent-Ohana/xLAM)
  - description: Baseline agent family (xLAM-7B-r) used for comparison; useful for practitioners examining alternative agent-tuning pipelines.

- PyTorch FSDP · [Website](https://pytorch.org/docs/stable/fsdp.html)
  - description: Distributed training primitive used to train models efficiently as reported in the implementation details.

<!-- paper_id: 6e15f76e13756478975bb20adfa1e0b9351d6758 -->

## 194. Learn-by-interact: A Data-Centric Framework for Self-Adaptive Agents in Realistic Environments - ICLR - 2025 - citation_count 58 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_58_Poster_Learn-by-interact_A_Data-Centric_Framework_for_Self-Adaptive_Agents_in_Realistic_Environments.pdf
- Tags: agent-framework
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework/2025_ICLR_58_Poster_Learn-by-interact_A_Data-Centric_Framework_for_Self-Adaptive_Agents_in_Realistic_Environments.pdf
- Token Usage: input 34345, output 7143, total 41488

### GitHub & Websites

- SWE-bench · [GitHub](https://github.com/princeton-nlp/SWE-bench) · [Website](https://www.swebench.com)
  - description: Benchmark of real-world GitHub issues; used as one of the four evaluation environments (Lite version by default).

- WebArena · [Website](https://arxiv.org/abs/2307.13854)
  - description: Realistic web environment benchmark; used as an evaluation environment for the agents.

- OSWorld · [Website](https://arxiv.org/abs/2404.07972)
  - description: Benchmark for open-ended tasks in real computer environments; used as an evaluation environment.

- Spider2-V · [Website](https://arxiv.org/abs/2407.10956)
  - description: Multimodal agent benchmark for data science/engineering workflows; used as an evaluation environment.

- Self-Instruct · [GitHub](https://github.com/yizhongw/self-instruct)
  - description: Used to generate diverse task instructions from documentation during data synthesis.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion) · [Website](https://arxiv.org/abs/2303.11366)
  - description: Baseline method providing self-feedback for agents; compared against in experiments.

- Language Agent Tree Search (LATS) · [Website](https://arxiv.org/abs/2310.04406)
  - description: Baseline that integrates tree search with ReAct; compared against in training-free evaluations.

- ReAct · [Website](https://arxiv.org/abs/2210.03629)
  - description: Reasoning-and-acting paradigm that LATS expands upon; referenced in baseline setup.

- STELLA retriever (stella_en_1.5B_v5) · [Website](https://huggingface.co/dunzhang/stella_en_1.5B_v5)
  - description: Dense retriever used for model-based agentic retrieval to fetch demonstrations.

- MDN Accessibility tree · [Doc](https://developer.mozilla.org/en-US/docs/Glossary/Accessibility_tree)
  - description: Documentation referenced for the observation space choice in web/desktop environments.

- PyAutoGUI · [GitHub](https://github.com/asweigart/pyautogui)
  - description: GUI automation library; actions in OSWorld/Spider2-V examples are expressed with PyAutoGUI commands.

- LoRA (Low-Rank Adaptation) · [Website](https://arxiv.org/abs/2106.09685)
  - description: Fine-tuning technique used to train CodeGemma-7B and Codestral-22B on synthesized data.

- Gemini 1.5 Pro · [Website](https://ai.google.dev/gemini-api/docs/models/gemini)
  - description: Commercial LLM used both as an evaluator and as part of the LLM committee for data filtering.

- Claude 3.5 Sonnet · [Website](https://www.anthropic.com/news/claude-3-5-sonnet)
  - description: Commercial LLM used as generator for data synthesis, as LLM committee member, and for evaluation.

- CodeGemma · [Website](https://arxiv.org/abs/2406.11409)
  - description: Open code model used for ICL and fine-tuning evaluations.

- Codestral-22B · [Website](https://mistral.ai/news/codestral/)
  - description: Open code model used for ICL and fine-tuning; showed large gains when trained on synthesized data.

- GitLab Tutorials · [Doc](https://docs.gitlab.com/ee/tutorials/)
  - description: One of the documentation sources used to self-generate tasks for WebArena data synthesis.

- Google Maps Help Center · [Doc](https://support.google.com/maps)
  - description: Documentation source used to derive tasks for WebArena.

- Amazon Help Gateway · [Doc](https://www.amazon.com/hz/contact-us/foresight/hubgateway)
  - description: Documentation source used to derive tasks for WebArena’s shopping scenarios.

- Reddit Help Articles · [Doc](https://support.reddithelp.com/hc/en-us/articles)
  - description: Documentation source used to derive tasks for WebArena forum scenarios.

- Google Chrome Help · [Doc](https://support.google.com/chrome/?hl=en)
  - description: Documentation used to generate tasks for OSWorld’s browser-related workflows.

- GIMP Tutorials · [Doc](https://www.gimp.org/tutorials/)
  - description: Documentation used to generate image-editing tasks for OSWorld.

- LibreOffice Calc Guide · [Doc](https://books.libreoffice.org/en/CG72/CG72.html)
  - description: Documentation used to generate spreadsheet tasks for OSWorld.

- LibreOffice Writer Guide · [Doc](https://books.libreoffice.org/en/WG73/WG73.html)
  - description: Documentation used to generate document-editing tasks for OSWorld.

- Ubuntu Command Line for Beginners · [Doc](https://ubuntu.com/tutorials/command-line-for-beginners)
  - description: Documentation used to synthesize terminal/CLI tasks for OSWorld.

- Mozilla Thunderbird Support · [Doc](https://support.mozilla.org/en-US/products/thunderbird)
  - description: Documentation used to generate email client tasks for OSWorld.

- VLC Wiki Documentation · [Doc](https://wiki.videolan.org/Documentation:Documentation)
  - description: Documentation used to derive media player tasks for OSWorld.

- Visual Studio Code Docs · [Doc](https://code.visualstudio.com/docs)
  - description: Documentation used to synthesize IDE tasks for OSWorld.

- dbt Docs · [Doc](https://docs.getdbt.com/)
  - description: Documentation used to create data-transformation tasks for Spider2-V.

- Dagster Docs (v1.7.2) · [Doc](https://release-1-7-2.dagster.dagster-docs.io/)
  - description: Documentation used to formulate orchestration tasks in Spider2-V.

- Astronomer Docs · [Doc](https://docs.astronomer.io/)
  - description: Documentation used to craft Airflow-related tasks in Spider2-V.

- Airbyte Docs · [Doc](https://docs.airbyte.com/) · [Website](https://airbyte.com/tutorials/)
  - description: Documentation and tutorials used to generate ETL/connector tasks in Spider2-V.

- Airbyte Public API Docs · [Doc](https://airbyte-public-api-docs.s3.us-east-2.amazonaws.com/rapidoc-api-docs.html)
  - description: API documentation referenced for programmatic tasks in Spider2-V.

- Apache Superset Docs · [Doc](https://superset.apache.org/docs/)
  - description: Documentation used to build BI/dashboarding tasks in Spider2-V.

- Metabase Docs v0.49 · [Doc](https://www.metabase.com/docs/v0.49/) · [Website](https://www.metabase.com/learn/)
  - description: Documentation and learning materials used to synthesize analytics tasks in Spider2-V.

- Snowflake Docs · [Doc](https://docs.snowflake.com/en/)
  - description: Documentation used to generate cloud data warehouse tasks in Spider2-V.

- BigQuery Docs · [Doc](https://cloud.google.com/bigquery/docs/)
  - description: Documentation used to create data warehousing tasks (e.g., table creation, uploads) in Spider2-V and examples.

- JupyterLab Docs · [Doc](https://jupyterlab.readthedocs.io/en/4.1.x/)
  - description: Documentation used to derive notebook and data exploration tasks in Spider2-V.

<!-- paper_id: 2e20395786ebaf45b3cee398b3db3531bc4851d6 -->

## 195. AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models - ICLR - 2024 - citation_count 496 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2024_ICLR_496_AutoDAN_Generating_Stealthy_Jailbreak_Prompts_on_Aligned_Large_Language_Models.pdf
- Link: https://arxiv.org/pdf/2310.04451
- Tags: prompt-engineering
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2024_ICLR_496_AutoDAN_Generating_Stealthy_Jailbreak_Prompts_on_Aligned_Large_Language_Models.pdf
- Token Usage: input 22112, output 3969, total 26081

### GitHub & Websites

- AutoDAN · [GitHub](https://github.com/SheltonLiu-N/AutoDAN)
  - description: Official implementation of the paper’s hierarchical genetic algorithm for generating stealthy jailbreak prompts; primary resource for reproducing all methods and experiments.

- GCG (llm-attacks) · [GitHub](https://github.com/llm-attacks/llm-attacks)
  - description: Official codebase for the GCG jailbreak baseline used throughout the paper for comparison, including training scripts and evaluation utilities.

- AdvBench: Harmful Behaviors · [GitHub](https://github.com/llm-attacks/llm-attacks/tree/main/data/advbench)
  - description: Dataset of 520 malicious requests introduced with GCG and used by this paper for evaluation (ASR and Recheck metrics).

- Vicuna · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat) · [Website](https://lmsys.org/blog/2023-03-30-vicuna/) · [Doc](https://chat.lmsys.org)
  - description: Open-source aligned chat model used as a white-box target model (Vicuna-7B) and also shown in examples (Vicuna-33B demo).

- Guanaco-7B (QLoRA) · [GitHub](https://github.com/artidoro/qlora) · [Doc](https://huggingface.co/timdettmers/guanaco-7b)
  - description: Guanaco-7B model and finetuning framework (QLoRA) used as one of the open-source white-box target models in experiments.

- Llama 2 Chat (Llama-2-7b-chat) · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
  - description: Meta’s aligned chat model used as a white-box target (without system prompt) for main evaluations and transfer tests.

- OpenAI GPT-3.5-turbo and GPT-4 APIs · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Proprietary LLM APIs used as black-box targets to evaluate transferability of AutoDAN prompts and to implement the GPT-based Recheck metric.

- GPT-2 (Perplexity scoring) · [GitHub](https://github.com/openai/gpt-2) · [Doc](https://huggingface.co/gpt2)
  - description: Model used to compute sentence-level perplexity (PPL) for assessing the stealthiness of jailbreak prompts.

- Wizard-Vicuna-30B-Uncensored · [Doc](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-fp16)
  - description: Uncensored LLM used in the paper’s ablation to study the Recheck metric agreement with human judgments.

- EasyJailbreak · [GitHub](https://github.com/EasyJailbreak/EasyJailbreak)
  - description: An open-source framework for jailbreaking LLMs cited as a recent benchmark; relevant for extending evaluations beyond the paper’s main setup.

- HarmBench · [GitHub](https://github.com/centerforaisafety/HarmBench)
  - description: A standardized evaluation framework for automated red teaming and robust refusal referenced in the paper’s discussion/limitations; useful for broader benchmarking of jailbreak methods.

- Princeton WordNet · [Website](https://wordnet.princeton.edu)
  - description: Lexical database used in an ablation as a synonym-replacement baseline for mutation/initialization diversity.

- Azure/OpenAI Content Filter · [Doc](https://learn.microsoft.com/azure/ai-services/openai/concepts/content-filter)
  - description: Safety filtering referenced when discussing black-box API safeguards and the interpretation of flagged responses in transfer experiments.

<!-- paper_id: f3f23f7f9f5369aade19f20bc5d028cce7b9c9aa -->

## 196. Beyond Demographics: Aligning Role-playing LLM-based Agents Using Human Belief Networks - EMNLP - 2024 - citation_count 30 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_30_Findings_Beyond_Demographics_Aligning_Role-playing_LLM-based_Agents_Using_Human_Belief_Networks.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_30_Findings_Beyond_Demographics_Aligning_Role-playing_LLM-based_Agents_Using_Human_Belief_Networks.pdf
- Token Usage: input 18804, output 3739, total 22543

### GitHub & Websites

- Controversial Beliefs Survey (Frigo, 2022)
  - description: The 64-topic belief dataset used to build the human belief network and to evaluate alignment. The paper states: “The survey dataset can be obtained by contacting its authors (Frigo, 2022).”

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Website](https://www.langchain.com) · [Doc](https://python.langchain.com)
  - description: Framework the authors used to construct LLM agents and manage prompts/messages for in-context learning experiments.

- OpenAI API (ChatGPT gpt-3.5-turbo-0125 fine-tuning) · [Website](https://platform.openai.com) · [Doc](https://platform.openai.com/docs/guides/fine-tuning)
  - description: API used for supervised fine-tuning of agents on training-topic responses and for running ChatGPT in experiments.

- OpenAI GPT-4o mini · [Website](https://platform.openai.com) · [Doc](https://platform.openai.com/docs/models#gpt-4o-mini)
  - description: One of the LLMs evaluated as an agent; queried via the OpenAI API in the alignment experiments.

- Mistral 7B Instruct v0.2 · [GitHub](https://github.com/mistralai) · [Website](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - description: Open-source LLM baseline the authors ran as an agent in their evaluations.

- Llama 3.1 8B Instruct · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com) · [Doc](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  - description: Open-source LLM baseline evaluated by the paper as an agent; used to test alignment under the proposed prompting conditions.

- Amazon Mechanical Turk · [Website](https://www.mturk.com)
  - description: Platform used to collect the original human survey responses that constitute the Controversial Beliefs Survey dataset.

<!-- paper_id: 04e5c40f897098d1781e2d6ee721f5fafcbf0417 -->

## 197. Prospector: Improving LLM Agents with Self-Asking and Trajectory Ranking - EMNLP - 2024 - citation_count 11 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_11_Findings_Prospector_Improving_LLM_Agents_with_Self-Asking_and_Trajectory_Ranking.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_11_Findings_Prospector_Improving_LLM_Agents_with_Self-Asking_and_Trajectory_Ranking.pdf
- Token Usage: input 19515, output 4698, total 24213

### GitHub & Websites

- ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
  - description: Interactive text-based embodied environment used as a primary benchmark; the paper evaluates Prospector on ALFWorld and uses it to generate trajectories for training the Critic.

- ALFRED · [GitHub](https://github.com/askforalfred/alfred) · [Website](https://askforalfred.com/)
  - description: Instruction-following dataset underlying ALFWorld; authors use its 3,553 training tasks and 134 unseen test tasks, and sample 3K training tasks to fine-tune the LLM Critic.

- TextWorld · [GitHub](https://github.com/microsoft/TextWorld) · [Website](https://www.textworld.ca/)
  - description: Text-based game simulator that ALFWorld builds upon; relevant for reproducing the environment setup.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop) · [Website](https://princeton-nlp.github.io/WebShop/)
  - description: Large-scale simulated online shopping environment; the paper evaluates on its 500 test instructions and uses 12K human instructions to collect trajectories and fine-tune the Critic.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct) · [Website](https://react-lm.github.io/)
  - description: Baseline ICL agent framework combining reasoning and acting; used as the base prompting approach that Prospector extends with AskAct and compares against.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Baseline agent using self-reflection and iterative refinement; used for comparison in experiments.

- LoRA (Low-Rank Adaptation) · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA) · [Doc](https://arxiv.org/abs/2106.09685)
  - description: Parameter-efficient fine-tuning method employed to fine-tune the relatively small LLM Critics on trajectory–reward data.

- Llama 2 · [Website](https://ai.meta.com/llama/) · [Doc](https://huggingface.co/meta-llama/Llama-2-70b-hf)
  - description: Open-source LLM family used as the large Actor (Llama‑2‑70B) and as the small Critic model variant (Llama‑2‑7B‑Chat) before/after fine-tuning.

- T5 (Text-to-Text Transfer Transformer) · [GitHub](https://github.com/google-research/text-to-text-transfer-transformer) · [Doc](https://huggingface.co/t5-large)
  - description: Open-source model used as a Critic backbone; the authors fine-tune T5 (3B) for trajectory reward prediction.

- FLAN-T5 · [Doc](https://huggingface.co/google/flan-t5-xl) · [Website](https://arxiv.org/abs/2210.11416)
  - description: Instruction-tuned T5 variant used as a Critic and fine-tuned to improve reward prediction and ranking.

- BLOOM · [GitHub](https://github.com/bigscience-workshop/bloom) · [Doc](https://huggingface.co/bigscience/bloom)
  - description: Open multilingual LLM evaluated as a fine-tuned Critic for reward prediction.

- BLOOMZ · [Doc](https://huggingface.co/bigscience/bloomz)
  - description: Multitask fine-tuned BLOOM variant considered as a fine-tuned Critic model.

- GPT-J 6B · [GitHub](https://github.com/kingoflolz/mesh-transformer-jax) · [Doc](https://huggingface.co/EleutherAI/gpt-j-6B)
  - description: Open-source model used as a Critic backbone; fine-tuned with LoRA on trajectory–reward data.

- OpenAI GPT-3 (text-davinci-002/003) · [Website](https://openai.com/research/gpt-3) · [Doc](https://platform.openai.com/docs/models/gpt-3)
  - description: Closed-source API models used as the large Actor in some settings and as a few-shot Critic for trajectory reward prediction.

<!-- paper_id: 61349d9d243e344246502c461c21817ec94d4ba0 -->

## 198. WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions - ICLR - 2024 - citation_count 1133 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2024_ICLR_1133_WizardLM_Empowering_Large_Pre-Trained_Language_Models_to_Follow_Complex_Instructions.pdf
- Tags: prompt-engineering
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2024_ICLR_1133_WizardLM_Empowering_Large_Pre-Trained_Language_Models_to_Follow_Complex_Instructions.pdf
- Token Usage: input 22790, output 5922, total 28712

### GitHub & Websites

- WizardLM · [GitHub](https://github.com/nlpxucan/WizardLM)
  - description: Official code/checkpoints for the paper’s instruction-evolved models; used to reproduce WizardLM training and inference.

- Azure OpenAI Service (gpt-3.5-turbo) · [Website](https://oai.azure.com/portal) · [Doc](https://learn.microsoft.com/azure/ai-services/openai/)
  - description: API used to run Evol-Instruct (evolution and response generation) throughout data construction.

- OpenAI ChatGPT / GPT-3.5 · [Website](https://chat.openai.com/) · [Doc](https://platform.openai.com/docs/)
  - description: The underlying LLM used for data evolution and response generation; also a comparison point in evaluations.

- Meta LLaMA (Llama 1/2) · [Website](https://ai.meta.com/llama/)
  - description: Base pre-trained models (e.g., LLaMA-13B, Llama-2-70B-Chat) used for fine-tuning WizardLM and ablations.

- Mistral-7B · [Website](https://mistral.ai/)
  - description: Alternative base model used to show Evol-Instruct generality (WizardLM-7B, Mistral).

- DeepSpeed (ZeRO-3) · [GitHub](https://github.com/microsoft/DeepSpeed) · [Doc](https://www.deepspeed.ai/)
  - description: Distributed training system used for fine-tuning (8×V100 with ZeRO-3).

- Stanford Alpaca · [GitHub](https://github.com/tatsu-lab/stanford_alpaca) · [Codewiki](https://codewiki.google/github.com/tatsu-lab/stanford_alpaca)
  - description: Seed instruction data and training code; authors expanded Alpaca to 70k and retrained Alpaca-13B as a baseline and initial seed for evolution.

- Vicuna / FastChat · [GitHub](https://github.com/lm-sys/FastChat) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat) · [Website](https://vicuna.lmsys.org/)
  - description: Baseline model and evaluation framework; the paper compares WizardLM to Vicuna-13B (v1.1) and uses FastChat for MT-Bench.

- ShareGPT · [Website](https://sharegpt.com/)
  - description: Source of 70k human-shared conversations used to train Vicuna and as an alternative seed in ablations.

- Super-NaturalInstructions (SNI) · [GitHub](https://github.com/allenai/natural-instructions)
  - description: Alternative instruction dataset used in ablations (random 70k sampled to train an LLaMA-13B baseline).

- Baize · [GitHub](https://github.com/project-baize/baize)
  - description: Open-source instruction-tuned baseline model evaluated against WizardLM.

- CAMEL · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel)
  - description: Open-source instruction-tuned baseline model compared in automatic benchmarks.

- Tulu / Open-Instruct · [GitHub](https://github.com/allenai/open-instruct)
  - description: Open-source instruction-tuned baseline model suite used for comparison.

- Open LLM Leaderboard (Hugging Face) · [Website](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
  - description: Benchmarking hub whose tasks (MMLU, ARC, HellaSwag, TruthfulQA) and evaluation setup were used.

- EleutherAI LM Evaluation Harness · [GitHub](https://github.com/EleutherAI/lm-evaluation-harness)
  - description: Evaluation codebase referenced (Gao et al., 2021) for running standard LLM benchmarks.

- MMLU (Massive Multitask Language Understanding) · [GitHub](https://github.com/hendrycks/test)
  - description: Multiple-choice academic benchmark used in automatic evaluation.

- ARC (AI2 Reasoning Challenge) · [GitHub](https://github.com/allenai/ai2-arc) · [Website](https://allenai.org/data/arc)
  - description: Grade-school science QA benchmark used in automatic evaluation.

- HellaSwag · [GitHub](https://github.com/rowanz/hellaswag)
  - description: Commonsense inference benchmark used in automatic evaluation.

- TruthfulQA · [GitHub](https://github.com/sylinrl/TruthfulQA)
  - description: Benchmark measuring propensity to produce falsehoods; used in automatic evaluation.

- HumanEval · [GitHub](https://github.com/openai/human-eval)
  - description: Code generation benchmark (pass@1) used to evaluate programming ability.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Grade school math benchmark used (4-shot, pass@1) for math reasoning evaluation.

- AlpacaEval · [GitHub](https://github.com/tatsu-lab/alpaca_eval)
  - description: GPT-4-based evaluator used to compare instruction-following quality.

- MT-Bench (LMSYS) · [GitHub](https://github.com/lm-sys/FastChat#mt-bench-and-chatbot-arena) · [Codewiki](https://codewiki.google/github.com/lm-sys/FastChat#mt-bench-and-chatbot-arena)
  - description: GPT-4 judging benchmark used to evaluate multi-turn conversational ability.

- BERT · [GitHub](https://github.com/google-research/bert) · [Codewiki](https://codewiki.google/github.com/google-research/bert)
  - description: Used to obtain instruction embeddings for t-SNE/k-means analysis of topic breadth in the evolved datasets.

<!-- paper_id: 08a80cb34d785258c770acecd302ab41ead46eed -->

## 199. Watch Every Step! LLM Agent Learning via Iterative Step-level Process Refinement - EMNLP - 2024 - citation_count 58 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_58_Main_Watch_Every_Step!_LLM_Agent_Learning_via_Iterative_Step-level_Process_Refinement.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_58_Main_Watch_Every_Step!_LLM_Agent_Learning_via_Iterative_Step-level_Process_Refinement.pdf
- Token Usage: input 19211, output 2868, total 22079

### GitHub & Websites

- IPR (Iterative step-level Process Refinement) · [GitHub](https://github.com/WeiminXiong/IPR)
  - description: Official code and data release for the paper; contains the implementation of the IPR framework, training scripts, and resources to reproduce the reported results.

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)
  - description: E-commerce web navigation environment and dataset used as one of the main benchmarks; the paper trains/evaluates agents on WebShop and uses its scoring rules.

- InterCode / InterCodeSQL · [GitHub](https://github.com/princeton-nlp/InterCode) · [Website](https://intercode-bench.github.io/)
  - description: Interactive coding benchmark framework used to build the InterCodeSQL environment evaluated in the paper; authors adapt and evaluate their agents on this SQL task setup.

- ALFWorld · [GitHub](https://github.com/princeton-vl/ALFWorld) · [Website](https://alfworld.github.io/)
  - description: Text-based embodied household environment used as a benchmark (seen/unseen splits) for evaluating IPR-trained agents.

- Spider (Text-to-SQL) · [GitHub](https://github.com/taoyds/spider) · [Website](https://yale-lily.github.io/spider)
  - description: Large-scale text-to-SQL dataset from which the InterCodeSQL database is constructed; provides schemas and queries underlying the interactive SQL evaluations.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reason+Act prompting framework used to collect expert trajectories and to structure agent interactions; the paper fine-tunes agents on ReAct-style trajectories.

- vLLM · [GitHub](https://github.com/vllm-project/vllm) · [Codewiki](https://codewiki.google/github.com/vllm-project/vllm) · [Doc](https://docs.vllm.ai/)
  - description: High-throughput LLM serving/inference engine used for all generations in the experiments (e.g., to sample trajectories and run MC step-reward estimation).

- Direct Preference Optimization (DPO) · [GitHub](https://github.com/eric-mitchell/direct-preference-optimization)
  - description: Reference implementation of DPO; the paper optimizes agents with outcome-level and step-level DPO losses as part of its mixed objective.

<!-- paper_id: 394e14ae60bae4c41162056717d9e30a8168abaa -->

## 200. Decomposed Prompting: A Modular Approach for Solving Complex Tasks - ICLR - 2023 - citation_count 557 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2023_ICLR_557_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf
- Tags: prompt-engineering
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3prompt-engineering/2023_ICLR_557_Decomposed_Prompting_A_Modular_Approach_for_Solving_Complex_Tasks.pdf
- Token Usage: input 86230, output 5133, total 91363

### GitHub & Websites

- Decomposed Prompting (DecomP) · [GitHub](https://github.com/allenai/DecomP)
  - description: Official repository released by the paper; provides datasets, prompts, and code to run the decomposer, sub-task handlers, and all experiments.

- Elasticsearch · [GitHub](https://github.com/elastic/elasticsearch) · [Website](https://www.elastic.co/elasticsearch) · [Doc](https://www.elastic.co/guide/index.html)
  - description: Open-source search engine used as the symbolic retrieval module in the decomposition framework for open-domain QA experiments.

- OpenAI API (GPT‑3 / Codex) · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: The paper prompts GPT‑3 (text-davinci-002/001) and Codex (code-davinci-002) models for decomposition and sub-tasks across all case studies.

- Flan‑T5 (Instruction‑tuned T5) · [GitHub](https://github.com/google-research/FLAN) · [Doc](https://huggingface.co/google/flan-t5-xxl)
  - description: Smaller open models (Large/XL/XXL) used to test model-scale effects for the decomposition framework in open-domain QA.

- HotpotQA · [GitHub](https://github.com/hotpotqa/hotpotqa) · [Website](https://hotpotqa.github.io/)
  - description: Open-domain multi-hop QA dataset; used in the paper’s retrieval-augmented decomposition experiments (fullwiki setting).

- MuSiQue · [GitHub](https://github.com/StonyBrookNLP/musique) · [Website](https://stonybrooknlp.github.io/musique/)
  - description: Multi-hop QA via single-hop question composition; used as an evaluation dataset in the open-domain experiments.

- 2WikiMultihopQA · [Doc](https://huggingface.co/datasets/2wikimultihopqa)
  - description: Two‑Wikipedia multi-hop QA dataset; used to evaluate the retrieval-augmented decomposition approach.

- GSM8K · [GitHub](https://github.com/openai/grade-school-math)
  - description: Grade-school math word problems; used to evaluate a simple decomposition that post-processes CoT to fix answer-extraction errors.

- CommaQA · [GitHub](https://github.com/allenai/CommaQA)
  - description: Long-context synthetic multi-hop QA dataset (CommaQA‑E variant) used to test decomposition vs. CoT in reading‑comprehension settings.

- Wikipedia (corpus provider) · [Website](https://www.wikipedia.org/) · [Doc](https://dumps.wikimedia.org/)
  - description: Source corpus for open-domain QA; the paper retrieves paragraphs from Wikipedia (e.g., HotpotQA fullwiki) via Elasticsearch.

- Forebears Name Lists · [Website](https://forebears.io/earth/forenames)
  - description: Public lists of forenames/surnames used to sample words for synthetic k-th letter concatenation tests.

- Vocabulary.com Word List · [Website](https://www.vocabulary.com/lists/189583)
  - description: Word list used as the vocabulary for the list-reversal task to construct evaluation sequences.

<!-- paper_id: 07955e96cbd778d0ae2a68f09d073b866dd84c2a -->

## 201. Quantifying and Mitigating Unimodal Biases in Multimodal Large Language Models: A Causal Perspective - EMNLP - 2024 - citation_count 24 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_24_Findings_Quantifying_and_Mitigating_Unimodal_Biases_in_Multimodal_Large_Language_Models_A_Causal_Perspective.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_24_Findings_Quantifying_and_Mitigating_Unimodal_Biases_in_Multimodal_Large_Language_Models_A_Causal_Perspective.pdf
- Token Usage: input 24639, output 3781, total 28420

### GitHub & Websites

- MORE (Quantifying and Mitigating Unimodal Biases) · [GitHub](https://github.com/OpenCausaLab/MORE)
  - description: Official project page/repository for this paper; releases the MORE dataset (12k MCQ VQA with causal rationales) and CAVE framework for bias mitigation and evaluation.

- INFOSEEK · [GitHub](https://github.com/google-research-datasets/infoseek) · [Website](https://github.com/google-research-datasets/infoseek)
  - description: Visual information-seeking VQA dataset used as the image source for constructing MORE (images/entities linked to Wikipedia).

- Wikidata5M · [Website](https://deepgraphlearning.github.io/project/wikidata5m/)
  - description: Knowledge graph used to sample subgraphs and generate multi-hop reasoning chains and options for MORE.

- Google Image Search / Lens · [Website](https://images.google.com) · [Doc](https://support.google.com/lens/answer/9255925)
  - description: Used in the CAVE verifier for image retrieval (via Google Image Search/Lens) to gather captions and related entity information for answer verification.

- Dense Passage Retrieval (DPR) · [GitHub](https://github.com/facebookresearch/DPR)
  - description: Off‑the‑shelf dense retriever employed in CAVE’s text-retrieval module to fetch Wikipedia passages for verification (cited as the retriever used).

- InstructBLIP · [GitHub](https://github.com/salesforce/LAVIS) · [Website](https://salesforce.github.io/LAVIS/)
  - description: Open-source MLLM baseline evaluated on MORE; authors use the InstructBLIP-Vicuna-13B variant.

- mPLUG-Owl · [GitHub](https://github.com/X-PLUG/mPLUG-Owl)
  - description: Open-source multimodal model baseline assessed on MORE (Llama-7B variant).

- LLaVA · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io/)
  - description: Open-source MLLM baseline evaluated on MORE (v1.5, 13B).

- Qwen-VL · [GitHub](https://github.com/QwenLM/Qwen-VL) · [Website](https://qwenlm.github.io)
  - description: Open-source vision-language model baseline tested on MORE (7B).

- OpenAI GPT‑4V and GPT‑4o · [Website](https://openai.com/research/gpt-4) · [Doc](https://platform.openai.com/docs/models#gpt-4o-and-gpt-4-turbo)
  - description: Closed-source MLLM baselines; GPT‑4 (text-only) is also used to generate language-bias options when constructing MORE.

- Gemini Pro Vision · [Website](https://deepmind.google/technologies/gemini/) · [Doc](https://ai.google.dev/gemini-api/docs/vision)
  - description: Proprietary multimodal baseline evaluated on MORE; also used (text-only Gemini Pro) to analyze language-bias option selection effects.

- LexicalRichness · [GitHub](https://github.com/LSYS/lexicalrichness)
  - description: Python package used in dataset quality analysis to compute lexical diversity metrics for generated questions.

- GPT‑2 · [GitHub](https://github.com/openai/gpt-2)
  - description: Pretrained language model used to compute question perplexity for fluency assessment in the dataset quality analysis.

<!-- paper_id: 0a39a9d3d884501ae103ec47d48608b6b642203a -->

## 202. ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning - NeurIPS - 2025 - citation_count 25 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_NeurIPS_25_Poster_ThinkAct_Vision-Language-Action_Reasoning_via_Reinforced_Visual_Latent_Planning.pdf
- Tags: planning
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/3planning/2025_NeurIPS_25_Poster_ThinkAct_Vision-Language-Action_Reasoning_via_Reinforced_Visual_Latent_Planning.pdf
- Token Usage: input 23688, output 7477, total 31165

### GitHub & Websites

- ThinkAct (Project Page) · [Website](https://jasper0314-huang.github.io/thinkact-vla/)
  - description: Official project page for ThinkAct with method overview, demos, and updates. The paper notes “We plan to release the source code after acceptance,” so this is the primary official resource at submission time.

- Open X-Embodiment (OXE) · [GitHub](https://github.com/google-deepmind/open_x_embodiment) · [Website](https://robotics-transformer-x.github.io/)
  - description: Large-scale robot learning dataset used to pre-train the DiT-based action policy and for SFT/RL data in ThinkAct.

- SimplerEnv · [GitHub](https://github.com/google-research/simpler-env)
  - description: Simulation benchmark (Google-VM, Google-VA, Bridge-VM) where ThinkAct is evaluated for robot manipulation robustness.

- LIBERO · [GitHub](https://github.com/UT-Austin-RPL/LIBERO) · [Website](https://libero-project.github.io/)
  - description: Long-horizon, compositional manipulation benchmark used to evaluate ThinkAct, including few-shot adaptation experiments.

- OpenEQA · [GitHub](https://github.com/allenai/OpenEQA) · [Website](https://openeqa.github.io)
  - description: Embodied QA benchmark used to evaluate ThinkAct’s embodied reasoning performance.

- RoboVQA · [Website](https://robovqa.github.io)
  - description: Dataset for multimodal long-horizon robotic reasoning; used both for SFT/RL training and evaluation of ThinkAct.

- EgoPlan (EgoPlan-IT / EgoPlan-Bench2) · [Website](https://egoplan-bench.github.io/)
  - description: Planning datasets/benchmarks used for QA-style rewards and evaluation of ThinkAct’s multi-step planning ability.

- Reflect · [Website](https://robot-reflect.github.io/)
  - description: Dataset on failure explanation/correction; used to enhance ThinkAct’s failure detection and reasoning via QA-style rewards.

- Something-Something V2 · [Website](https://20bn.com/datasets/something-something-v2)
  - description: Human video dataset used during RL to provide visual context for reinforcing embodied reasoning.

- LLaVA-Video-178K · [Website](https://llava-vl.github.io/)
  - description: Synthetic video instruction dataset included during RL to improve general video-instruction reasoning.

- Qwen2.5-VL · [GitHub](https://github.com/QwenLM/Qwen2.5-VL) · [Website](https://qwenlm.github.io)
  - description: Multimodal LLM backbone for ThinkAct’s reasoning module; cold-started via SFT and then optimized with GRPO.

- Diffusion Policy (DiT-based policy) · [GitHub](https://github.com/real-stanford/diffusion_policy) · [Website](https://diffusion-policy.cs.columbia.edu)
  - description: Transformer-based action policy architecture used as ThinkAct’s action model and as a baseline (DiT-Policy) in experiments.

- DINOv2 · [GitHub](https://github.com/facebookresearch/dinov2)
  - description: Visual encoder used in ThinkAct’s state encoder for the action policy.

- CLIP · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Text encoder used in ThinkAct’s state encoder to process language instructions.

- BLIP-2 / LAVIS (Q-Former) · [GitHub](https://github.com/salesforce/LAVIS) · [Doc](https://lavis.readthedocs.io/)
  - description: Q-Former module (from BLIP-2/LAVIS) is used as the latent projector to inject ThinkAct’s visual plan latent into the action model.

- OpenVLA · [GitHub](https://github.com/openvla/openvla) · [Website](https://openvla.github.io)
  - description: Vision-language-action baseline compared against ThinkAct on LIBERO and SimplerEnv; also used for inference speed comparison.

- TraceVLA · [GitHub](https://github.com/microsoft/TraceVLA)
  - description: Baseline that uses visual traces prompting for spatial-temporal awareness; compared to ThinkAct in manipulation benchmarks.

- CoT-VLA · [Website](https://cot-vla.github.io)
  - description: Visual chain-of-thought VLA baseline; compared to ThinkAct, especially on long-horizon LIBERO tasks.

- Octo · [GitHub](https://github.com/octo-model/Octo)
  - description: Generalist robot policy baseline included in SimplerEnv comparisons.

- RT-1 / RT-1-X · [GitHub](https://github.com/google-research/robotics_transformer) · [Website](https://robotics-transformer.github.io)
  - description: Robotics Transformer baselines; RT‑1‑X/RT‑X variants are included as comparison methods in SimplerEnv evaluations.

- DeepSpeed · [GitHub](https://github.com/microsoft/DeepSpeed) · [Website](https://www.deepspeed.ai/)
  - description: Training system used in ThinkAct’s SFT cold-start (ZeRO-3) to scale MLLM optimization.

<!-- paper_id: 996f5d92e6c6a6b88a98657e7a664bcc29cb5c14 -->

## 203. Shall We Team Up: Exploring Spontaneous Cooperation of Competing LLM Agents - EMNLP - 2024 - citation_count 26 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_26_Findings_Shall_We_Team_Up_Exploring_Spontaneous_Cooperation_of_Competing_LLM_Agents.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_26_Findings_Shall_We_Team_Up_Exploring_Spontaneous_Cooperation_of_Competing_LLM_Agents.pdf
- Token Usage: input 29011, output 4723, total 33734

### GitHub & Websites

- SABM_ShallWeTeamUp · [GitHub](https://github.com/wuzengqing001225/SABM_ShallWeTeamUp)
  - description: Official code release for this paper; implements the three simulations (Keynesian Beauty Contest, Bertrand Competition, Emergency Evacuation), prompts, and SABM-based multi-agent workflow used in the experiments.

- OpenAI Python API (Chat Completions) · [GitHub](https://github.com/openai/openai-python) · [Codewiki](https://codewiki.google/github.com/openai/openai-python) · [Doc](https://platform.openai.com/docs/api-reference/chat)
  - description: The authors run GPT-4-0314 via openai==0.28.0 and ChatCompletion.create for all core simulations; these docs and client library are required to reproduce the LLM calls.

- Anthropic Claude 3 (Sonnet) · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/claude)
  - description: Used for comparative runs in the KBC case study; practitioners can use the Claude API to replicate the cross-model analysis.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Website](https://www.langchain.com/) · [Doc](https://python.langchain.com/docs/)
  - description: Cited as a well-developed platform for LLM multi-agent simulations; relevant alternative toolkit for extending or re-implementing the authors’ agent workflows.

- AutoGen · [GitHub](https://github.com/microsoft/autogen) · [Website](https://microsoft.github.io/autogen/) · [Doc](https://microsoft.github.io/autogen/docs/Getting-Started)
  - description: Referenced multi-agent conversation framework; useful for practitioners exploring different infrastructure for agent communication and coordination.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Referenced agent-oriented framework for multi-agent collaboration; an alternative platform to build and extend the social simulations described.

- AgentLite · [GitHub](https://github.com/salesforce/AgentLite)
  - description: Referenced lightweight library for task-oriented LLM agent systems; a related open-source option for reproducing or scaling agent-based experiments.

- New York Times “Are you smarter than 61,139 other readers?” (Beauty-Contest Experiment) · [Website](https://www.nytimes.com/interactive/2015/07/03/upshot/are-you-smarter-than-other-new-york-times-readers.html)
  - description: Human data used for comparison in the KBC case study; provides the empirical distribution of guesses that the paper aligns with.

<!-- paper_id: 11d4478587c4d2ecb195fe911809946928767657 -->

## 204. Simulating Human-like Daily Activities with Desire-driven Autonomy - ICLR - 2025 - citation_count 12 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_12_Poster_Simulating_Human-like_Daily_Activities_with_Desire-driven_Autonomy.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_12_Poster_Simulating_Human-like_Daily_Activities_with_Desire-driven_Autonomy.pdf
- Token Usage: input 44641, output 4041, total 48682

### GitHub & Websites

- Desire-driven Autonomy (D2A) · [Website](https://sites.google.com/view/desire-driven-autonomy)
  - description: Official project page for this paper; hosts the main entry point for demos/materials of the Desire-driven Autonomous Agent used in all experiments.

- Concordia · [GitHub](https://github.com/google-deepmind/concordia) · [Website](https://arxiv.org/abs/2312.03664)
  - description: Text-based simulator framework the authors build upon to implement their daily-activity simulator, memory, planning, and environment control; also used for an ablation with its multi-step planning component.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct) · [Website](https://arxiv.org/abs/2210.03629)
  - description: Baseline agent that reasons before acting; the paper compares D2A against ReAct on human-likeness and dissatisfaction metrics.

- BabyAGI · [GitHub](https://github.com/yoheinakajima/babyagi)
  - description: Baseline maintaining a prioritized task list; used for comparison with D2A in activity generation quality.

- LLMob
  - description: Baseline that generates activity plans from profile-derived motivations; included in the paper’s comparisons in both indoor and outdoor settings.

- Tree of Thoughts (ToT) · [GitHub](https://github.com/princeton-nlp/tree-of-thought-llm) · [Website](https://arxiv.org/abs/2305.10601)
  - description: Decision-making paradigm that inspires D2A’s multi-candidate Activity Proposal/Evaluation/Selection process.

- Meta Llama 3.1 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://llama.meta.com/)
  - description: Default backbone LLM (Llama 3.1-70B) for both agents and the environment controller in the experiments.

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5) · [Website](https://qwenlm.github.io)
  - description: Alternative backbone model used to test framework adaptability; integrated via Ollama in the paper’s experiments.

- Ollama · [GitHub](https://github.com/ollama/ollama) · [Codewiki](https://codewiki.google/github.com/ollama/ollama) · [Website](https://ollama.com/)
  - description: Local serving/runtime the authors note for integrating Qwen 2.5 into their simulator setup.

- GPT-4o · [Doc](https://platform.openai.com/docs/models#gpt-4o)
  - description: Evaluation model used as the external judge for pairwise “human-likeness” comparisons of action sequences.

<!-- paper_id: 74ffc0667698691685e33e7a412e5dcdffda04d3 -->

## 205. Modeling Human Subjectivity in LLMs Using Explicit and Implicit Human Factors in Personas - EMNLP - 2024 - citation_count 18 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_18_Findings_Modeling_Human_Subjectivity_in_LLMs_Using_Explicit_and_Implicit_Human_Factors_in_Personas.pdf
- Link: http://arxiv.org/pdf/2406.14462
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_18_Findings_Modeling_Human_Subjectivity_in_LLMs_Using_Explicit_and_Implicit_Human_Factors_in_Personas.pdf
- Token Usage: input 18730, output 6225, total 24955

### GitHub & Websites

- DLATK (Differential Language Analysis ToolKit) · [GitHub](https://github.com/wwbp/dlatk) · [Website](https://dlatk.wwbp.org)
  - description: Python toolkit the authors used for feature extraction, correlation analysis, and word‑cloud visualization of n‑grams and lexicon features in their belief-generation analyses.

- OpenAI GPT‑4o · [Website](https://openai.com/index/gpt-4o/) · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: The LLM used to simulate “Persona‑LLMs” and to run the validation classifier that labeled generated texts for stance.

- LIWC‑22 (Linguistic Inquiry and Word Count) · [Website](https://liwc.net)
  - description: Psycholinguistic lexicon the paper used to compute category correlations with personas (reported in Appendix tables).

- Moral Foundations Dictionary (MFD) · [Website](https://moralfoundations.org)
  - description: Lexicon of moral language the paper applied to the generated texts to examine moral-category correlations across personas.

- U.S. Census Bureau — Frequently Occurring Surnames in the 2010 Census · [Website](https://www.census.gov/topics/population/genealogy/data/2010_surnames.html)
  - description: Source for racially distinctive surnames used to construct implicit race personas.

- Racially Distinctive Names (Crabtree et al., 2022) · [Website](https://www.sociologicalscience.com/articles-v9-21-454/)
  - description: Research source for racially distinctive first names; the paper used these names to build implicit race personas.

- Pew Research Center — Younger Americans’ views on the Israel–Hamas war (Silver, 2024) · [Website](https://www.pewresearch.org/short-reads/2024/05/23/younger-americans-stand-out-in-their-views-of-the-israel-hamas-war/)
  - description: Survey findings used to ground the age-related belief question (Palestine domain) and expected associations.

- Pew Research Center — Gender pay gap hasn’t changed much in two decades (Aragao, 2023) · [Website](https://www.pewresearch.org/short-reads/2023/03/01/gender-pay-gap-facts/)
  - description: Cited as a contemporary Pew resource to contextualize gender differences; the study’s parenting-domain question on pressure at home is evaluated against known gender trends.

- Pew Research Center — How Americans view the situation at the U.S.–Mexico border (2024) · [Website](https://www.pewresearch.org/short-reads/2024/02/29/how-americans-view-the-situation-at-the-us-mexico-border-its-causes-and-consequences/)
  - description: Provides the immigration-domain framing and partisan expectation patterns used for belief generation and validation.

- Pew Research Center — Police views, public views (2017) · [Website](https://www.pewresearch.org/politics/2017/01/11/police-views-public-views/)
  - description: Supplies the policing-domain belief contrasts (e.g., “protectors” vs “enforcers”) across racial groups used to evaluate persona outputs.

- Pew Research Center — Most Americans favor legalizing marijuana for medical, recreational use (2024) · [Website](https://www.pewresearch.org/short-reads/2024/03/26/americans-favor-legalizing-marijuana/)
  - description: Basis for the legalization-domain expectations; used to compare persona generations for substance-use factors.

<!-- paper_id: 661c0467b442479a3891bbfd9a6ef3924af2aa70 -->

## 206. MindSearch: Mimicking Human Minds Elicits Deep AI Searcher - ICLR - 2025 - citation_count 45 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_45_Poster_MindSearch_Mimicking_Human_Minds_Elicits_Deep_AI_Searcher.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_45_Poster_MindSearch_Mimicking_Human_Minds_Elicits_Deep_AI_Searcher.pdf
- Token Usage: input 20077, output 4075, total 24152

### GitHub & Websites

- MindSearch · [GitHub](https://github.com/InternLM/MindSearch)
  - description: Official code release of the paper; contains the multi-agent WebPlanner/WebSearcher implementation used for all experiments.

- InternLM (InternLM2.5) · [GitHub](https://github.com/InternLM/InternLM) · [Website](https://internlm.org/) · [Doc](https://internlm.readthedocs.io/en/latest/)
  - description: Open-source LLM family; the paper uses InternLM2.5-7B-Chat as a backend model for MindSearch.

- GPT-4o · [Doc](https://platform.openai.com/docs/models/gpt-4o)
  - description: Closed-source LLM backend used for experiments and comparisons (ChatGPT-Web).

- Bing Web Search API · [Doc](https://learn.microsoft.com/bing/search-apis/bing-web-search/overview)
  - description: Search engine API used during evaluation; all models are restricted to Internet access via Bing for fair comparison.

- Google Custom Search JSON API · [Website](https://developers.google.com/custom-search/v1/overview)
  - description: One of the search APIs WebSearcher can call in the hierarchical retrieval pipeline.

- DuckDuckGo Instant Answer API · [Website](https://duckduckgo.com/api)
  - description: Another search API option mentioned for WebSearcher’s internet retrieval.

- Bamboogle (dataset) · [GitHub](https://github.com/ofirpress/self-ask/tree/main/bamboogle)
  - description: Closed-set QA dataset used for evaluation; introduced with the self-ask work.

- Musique (dataset) · [GitHub](https://github.com/stonybrooknlp/musique) · [Website](https://musique-data.github.io/)
  - description: Multi-hop QA dataset used to assess MindSearch on closed-set tasks.

- HotpotQA (dataset) · [GitHub](https://github.com/hotpotqa/hotpot) · [Website](https://hotpotqa.github.io/)
  - description: Multi-hop QA dataset; the paper reports main closed-set results and ablations on it.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Reasoning-and-acting baseline framework; used as a comparison and ablation baseline for WebPlanner/WebSearcher.

- self-ask · [GitHub](https://github.com/ofirpress/self-ask)
  - description: Baseline method compared in the appendix; also hosts the Bamboogle dataset.

- CodeAct
  - description: Executable code actions baseline compared against WebPlanner in ablations.

- Searchain
  - description: Chain-of-query method compared in the appendix; used as a competitive web-search baseline.

- Perplexity.ai · [Website](https://www.perplexity.ai/)
  - description: Proprietary AI search application used as a comparison system in open-set human preference evaluation.

- ChatGPT-Web · [Website](https://chat.openai.com/)
  - description: Comparison system (GPT-4o with web/search plugin) used in open-set human preference evaluation.

- DeepSeek-V2 · [GitHub](https://github.com/deepseek-ai/DeepSeek-V2)
  - description: Additional LLM backend tested in the appendix to demonstrate MindSearch generalization.

- Qwen2.5 · [GitHub](https://github.com/QwenLM/Qwen2.5)
  - description: Additional LLM backend (Qwen-2.5-7B) evaluated in the appendix for generalization.

- GLM-4 · [GitHub](https://github.com/THUDM/GLM-4)
  - description: Additional LLM backend (GLM4-9B) evaluated in the appendix for generalization.

<!-- paper_id: 56088d8601afdd524029c5c151f7d13ed3f1a5ee -->

## 207. Is this the real life? Is this just fantasy? The Misleading Success of Simulating Social Interactions With LLMs - EMNLP - 2024 - citation_count 50 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_50_Main_Is_this_the_real_life_Is_this_just_fantasy_The_Misleading_Success_of_Simulating_Social_Interactions_With_LLMs.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_50_Main_Is_this_the_real_life_Is_this_just_fantasy_The_Misleading_Success_of_Simulating_Social_Interactions_With_LLMs.pdf
- Token Usage: input 24051, output 4778, total 28829

### GitHub & Websites

- AGSCR (Agents vs Script) project page · [Website](https://agscr.sotopia.world)
  - description: Official project site for this paper; the authors point here for resources related to the AGENTS vs SCRIPT study, data, and artifacts.

- Sotopia · [GitHub](https://github.com/sotopia-lab/sotopia) · [Website](https://sotopia.world) · [Doc](https://pypi.org/project/sotopia/)
  - description: The simulation/evaluation framework the paper builds on; used to run AGENTS and MINDREADERS modes via its state-space agent library and to compute goal completion metrics.

- Generative Agents (Park et al., 2023) · [GitHub](https://github.com/joonspk-research/generative_agents) · [Codewiki](https://codewiki.google/github.com/joonspk-research/generative_agents) · [Website](https://generativeagents.dev)
  - description: Referenced codebase the authors inspected (Appendix C) to determine simulation modes; earlier versions used a SCRIPT-like mode, informing their analysis of omniscient simulations.

- COCOA Datasets: MutualFriends and CraigslistBargain · [GitHub](https://github.com/stanfordnlp/cocoa) · [Website](https://stanfordnlp.github.io/cocoa)
  - description: Original datasets for the cooperative (MutualFriends) and competitive (Craigslist Bargain) tasks used as representative scenarios in the paper’s evaluations and analyses.

- OpenAI API (GPT-3.5 fine-tuning and GPT-4 evaluation) · [Website](https://platform.openai.com) · [Doc](https://platform.openai.com/docs/guides/fine-tuning)
  - description: Used to fine-tune GPT-3.5 on SCRIPT data and to evaluate goal completion with GPT-4; model versions gpt-3.5-turbo-0613 and gpt-4-0613 are specified for reproducibility.

- Together AI Inference API (Mixtral-8x7B) · [Website](https://www.together.ai) · [Doc](https://docs.together.ai)
  - description: Service the authors used to run Mixtral-8x7B (Mixtral-MoE) for simulations.

- Mixtral-8x7B (Mistral) · [Website](https://mistral.ai) · [Doc](https://docs.mistral.ai) · [GitHub](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Open LLM evaluated alongside GPT-3.5 across SCRIPT/AGENTS/MINDREADERS modes; accessible via Together AI or the released weights. 

- AI2 Impact License · [Website](https://allenai.org/impact-license)
  - description: License under which the authors state they will release data to mitigate misuse; relevant for dataset reuse and redistribution.

<!-- paper_id: bfc2aee63d20fb19c9a851da9e97fec40c454124 -->

## 208. Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View - ICLR - 2025 - citation_count 20 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_20_Poster_Exploring_Prosocial_Irrationality_for_LLM_Agents_A_Social_Cognition_View.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_20_Poster_Exploring_Prosocial_Irrationality_for_LLM_Agents_A_Social_Cognition_View.pdf
- Token Usage: input 27812, output 3837, total 31649

### GitHub & Websites

- CogMir · [GitHub](https://github.com/XuanL17/CogMir)
  - description: Official repository for the paper’s open-ended Multi-LLM Agents framework, including code, prompts, and the constructed datasets (Known/Unknown MCQ, Inform, CogScene, CogAction, CogIdentity) used to mirror cognitive biases and reproduce experiments.

- SimCSE · [GitHub](https://github.com/princeton-nlp/SimCSE) · [Doc](https://huggingface.co/princeton-nlp/sup-simcse-roberta-large)
  - description: Contrastive sentence-embedding method used as a technical discriminator (SimCSE-RoBERTa-large) to compute semantic similarity in the Rumor Chain experiments.

- SelfCheckGPT · [GitHub](https://github.com/potsawee/selfcheckgpt)
  - description: Zero-resource hallucination detection toolkit cited as one of the “state-of-the-art technical discriminators” within CogMir’s evaluator set.

- FACTSCORE · [GitHub](https://github.com/google-research/factscore)
  - description: Fine-grained factuality evaluation metric referenced as a technical discriminator option for objective assessments in the framework.

- SocialIQA · [Website](https://allenai.org/data/socialiqa)
  - description: Social commonsense reasoning dataset referenced when constructing CogMir’s evaluation materials and scenarios.

- Sotopia · [Website](https://sotopia.world)
  - description: Interactive social intelligence evaluation environment cited as a recent benchmark that informed the design of CogMir’s social-evaluation settings.

- AgentBench · [GitHub](https://github.com/THUDM/AgentBench)
  - description: Benchmark for evaluating LLMs as agents; referenced as an existing agent evaluation dataset that informed CogMir’s dataset construction.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Multi-agent collaboration framework referenced in related work; useful for practitioners extending CogMir’s multi-agent interactions.

- MetaGPT · [GitHub](https://github.com/geekan/MetaGPT)
  - description: Multi-agent collaborative framework cited in related work; relevant for implementing alternative multi-agent orchestration comparable to CogMir.

- CAMEL (Communicative Agents for “Mind” Exploration) · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel)
  - description: Multi-agent role-playing environment referenced in related work; provides reusable agent interaction patterns for extensions.

- Character-LLM · [GitHub](https://github.com/thu-coai/Character-LLM)
  - description: Trainable role-playing agent framework referenced as related work; helpful for persona/role setups akin to CogIdentity in CogMir.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Language agents with verbal reinforcement learning referenced in Appendix as an aligned single-agent architecture; useful for adding reflective reasoning to CogMir agents.

- Generative Agents · [GitHub](https://github.com/joonspk-research/generative_agents) · [Codewiki](https://codewiki.google/github.com/joonspk-research/generative_agents)
  - description: Proof-of-concept single-LLM agent system cited when describing individual agent architecture; practitioners can compare memory and behavior modules with CogMir’s design.

- OpenAI GPT-4/GPT-3.5 API · [Website](https://platform.openai.com/) · [Doc](https://platform.openai.com/docs)
  - description: Closed-weight LLMs used as primary agents and evaluators (gpt-4-0125-preview, gpt-3.5-turbo) in the experiments.

- Mistral AI (Mixtral-8x7B, Mistral-Medium) · [Website](https://mistral.ai) · [Doc](https://docs.mistral.ai)
  - description: Models used as agents (open-mixtral-8x7b and mistral-medium-2312); doc pages provide access and deployment details for reproduction.

- Mixtral-8x7B Instruct · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: Open model variant corresponding to the paper’s “open-mixtral-8x7b,” enabling local or HF-based replication.

- Anthropic Claude (Claude 2, Claude 3 Opus) · [Website](https://www.anthropic.com) · [Doc](https://docs.anthropic.com/)
  - description: Proprietary LLMs used as agents and evaluators in several experiments.

- Google Gemini 1.0 Pro · [Website](https://ai.google.dev/) · [Doc](https://ai.google.dev/gemini-api/docs)
  - description: Proprietary multimodal LLM used as one of the evaluated agents in CogMir.

<!-- paper_id: 9f8da87ee4416d57a2cc044bdf8223c7728d74d7 -->

## 209. Advancing Social Intelligence in AI Agents: Technical Challenges and Open Questions - EMNLP - 2024 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_16_Main_Advancing_Social_Intelligence_in_AI_Agents_Technical_Challenges_and_Open_Questions.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2024_EMNLP_16_Main_Advancing_Social_Intelligence_in_AI_Agents_Technical_Challenges_and_Open_Questions.pdf
- Token Usage: input 24390, output 6084, total 30474

### GitHub & Websites

- Social-AI Community Resource · [GitHub](https://github.com/l-mathur/social-ai)
  - description: Official repository released with the paper; a continually updated collection of papers, datasets, benchmarks, simulators, and courses to support research on the technical challenges outlined in the position paper.

- Semantic Scholar Open Data Platform (API) · [Website](https://www.semanticscholar.org/product/api) · [Doc](https://api.semanticscholar.org/api-docs/graph)
  - description: Bibliographic/data API used by the authors to collect and analyze 3,257 papers for their trend analysis (Figure 2) and literature sampling.

- S2ORC: Semantic Scholar Open Research Corpus · [GitHub](https://github.com/allenai/s2orc) · [Website](https://allenai.org/data/s2orc)
  - description: Large corpus of scholarly literature cited as the data backbone for paper/meta-data access; relevant for reproducing the paper’s literature analysis.

- SOTOPIA · [GitHub](https://github.com/sotopia-lab/sotopia) · [Website](https://sotopia.world)
  - description: Interactive environment for evaluating social intelligence in language agents; cited as a dynamic setting to study dyadic and multi-party interactions.

- CAMEL (Communicative Agents for “Mind” Exploration) · [GitHub](https://github.com/camel-ai/camel) · [Codewiki](https://codewiki.google/github.com/camel-ai/camel) · [Website](https://www.camel-ai.org/)
  - description: Multi-agent LLM framework used to study emergent social behaviors; referenced as a platform for dynamic text-agent interactions.

- AgentVerse · [GitHub](https://github.com/OpenBMB/AgentVerse)
  - description: Platform for multi-agent collaboration and emergent behavior analysis; cited as an interactive environment for Social-AI research.

- Habitat 3.0 (AI Habitat) · [GitHub](https://github.com/facebookresearch/habitat-lab) · [Website](https://aihabitat.org/)
  - description: Simulation platform for human–avatar–robot co-habitation; referenced for embodied Social-AI and human-robot interaction studies.

- SOCIALIQA · [GitHub](https://github.com/rowanz/social-iqa)
  - description: Commonsense reasoning benchmark about social interactions; used in the paper to evaluate LLM social knowledge/reasoning limits.

- SOCIAL-IQ (Video QA for Social Intelligence)
  - description: Video question answering benchmark for artificial social intelligence; cited as a static VideoQA benchmark (versions 1.0 and 2.0) to probe social understanding.

- IEMOCAP · [Website](https://sail.usc.edu/iemocap/)
  - description: Multimodal dyadic interaction dataset for affect/emotion research; referenced as a key resource for predicting affect in multimodal conversations.

- SEWA DB · [Website](https://sewa-db.org/)
  - description: In-the-wild audio-visual dataset for emotion and sentiment; cited as a resource for multimodal affective/social signal processing.

- CMU-MOSEI · [GitHub](https://github.com/A2Zadeh/CMU-MultimodalSDK) · [Website](https://multicomp.cs.cmu.edu/resources/cmu-mosei/)
  - description: Large-scale multimodal sentiment/emotion dataset and SDK; referenced for modeling emotion/sentiment in conversations and broader multimodal social understanding.

- Ego-Exo4D · [Website](https://ego-exo4d-data.org/)
  - description: Multimodal egocentric–exocentric dataset of skilled human activity (audio, gaze, vision, pose, language); highlighted as a richer, naturalistic source of social interaction data.

- Caltech Conte Center (Multimodal Social Cognition Dataset) · [Website](https://conte.caltech.edu/)
  - description: Longitudinal multimodal resource on social cognition and decision-making; cited as an example of rich, ethically collected data to advance Social-AI.

- NormBank · [GitHub](https://github.com/behavioral-data/NormBank)
  - description: Knowledge bank of situational social norms; referenced when discussing how norms shape interactions and as a resource for modeling social context.

<!-- paper_id: 6860e27d1731fd88b639589592afe1fb01633a9d -->

## 210. AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection - ACL - 2025 - citation_count 16 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_16_Long_AGrail_A_Lifelong_Agent_Guardrail_with_Effective_and_Adaptive_Safety_Detection.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ACL_16_Long_AGrail_A_Lifelong_Agent_Guardrail_with_Effective_and_Adaptive_Safety_Detection.pdf
- Token Usage: input 28865, output 5304, total 34169

### GitHub & Websites

- AGrail · [Website](https://eddyluo1232.github.io/AGrail/)
  - description: Official project page for the paper’s framework and benchmark; the paper points here as the main project website for AGrail.

- AgentBench · [GitHub](https://github.com/THUDM/AgentBench)
  - description: Benchmark and agent suite used as the OS agent base; the Safe-OS benchmark in the paper is constructed on top of AgentBench’s OS agent and data format.

- SeeAct · [GitHub](https://github.com/OSU-NLP-Group/SeeAct)
  - description: Web agent used for web tasks; the paper evaluates task-specific and systemic risks against web agents by referencing SeeAct.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Realistic web interaction dataset behind Mind2Web-SC (the safety-control split used in the paper via GuardAgent); supports reproducing the web-agent safety evaluations.

- WebArena · [GitHub](https://github.com/WebArena-Lab/WebArena) · [Website](https://webarena.dev/)
  - description: Realistic web environment cited for building and evaluating autonomous web agents; relevant as the underlying environment ecosystem for web-agent experiments like those with SeeAct.

- eICU Collaborative Research Database (eICU-CRD) · [Website](https://eicu-crd.mit.edu/)
  - description: Official clinical database that underlies the EICU-AC access-control tasks evaluated with EHRAgent; relevant for reproducing the healthcare agent experiments.

- Llama Guard 3 · [GitHub](https://github.com/meta-llama/llama-guard) · [Doc](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama-guard-3/)
  - description: Safety classifier model used as a guardrail baseline (LLaMA-Guard 3) in the paper’s comparisons. 

- Docker · [Website](https://www.docker.com/) · [Doc](https://docs.docker.com/)
  - description: Containerization platform used to simulate the OS environments and user privilege settings for Safe-OS; necessary to reproduce the paper’s OS-agent evaluation setup.

<!-- paper_id: d83cd5896c58682aaa186f2fc791cd82bfd0cb9e -->

## 211. Dissecting Adversarial Robustness of Multimodal LM Agents - ICLR - 2025 - citation_count 67 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_67_Poster_Dissecting_Adversarial_Robustness_of_Multimodal_LM_Agents.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICLR_67_Poster_Dissecting_Adversarial_Robustness_of_Multimodal_LM_Agents.pdf
- Token Usage: input 23408, output 2859, total 26267

### GitHub & Websites

- agent-attack (VWA-Adv, attacks, defenses) · [GitHub](https://github.com/ChenWu98/agent-attack)
  - description: Official release from the paper containing the curated VWA-Adv adversarial tasks (200 tasks), evaluation scripts, attack/defense implementations, and code to reproduce the ARE analysis.

- VisualWebArena (VWA) · [GitHub](https://github.com/web-arena-x/visualwebarena) · [Website](https://webarena.dev)
  - description: Realistic multimodal web-agent environment the paper extends to create VWA-Adv; provides the web environments, evaluation primitives, and baseline agents used throughout the experiments.

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: The broader suite of realistic web environments that VWA builds upon; relevant for setting up and extending the web-based evaluation used in the paper.

- LLaVA (Large Language and Vision Assistant) · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io)
  - description: Open-weight VLM used as the white-box captioner component; the paper performs gradient-based attacks against this captioner and uses it to preprocess images for policy models.

- CLIP (Contrastive Language-Image Pre-training) · [GitHub](https://github.com/openai/CLIP) · [Codewiki](https://codewiki.google/github.com/openai/CLIP)
  - description: Vision-language model family used as black-box surrogates for the targeted image attack (CLIP attack); the paper ensembles several CLIP variants to generate transferable adversarial perturbations.

- OpenAI GPT-4o / GPT-4V · [Website](https://openai.com) · [Doc](https://platform.openai.com/docs)
  - description: Frontier multimodal LMs used as the policy model and for self-captioning; primary black-box targets whose robustness is measured under text and image attacks.

- Google Gemini 1.5 Pro · [Website](https://ai.google.dev) · [Doc](https://ai.google.dev/gemini-api)
  - description: Multimodal LM baseline evaluated as a policy model within the VWA-Adv tasks to compare robustness and benign performance.

- Anthropic Claude 3 Opus · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/claude/docs)
  - description: Multimodal LM baseline evaluated as a policy model; included in the robustness-utility trade-off comparisons.

<!-- paper_id: 4f27fc2ea3d3491deded642a5de247d167a03d15 -->

## 212. Efficient Performance Tracking: Leveraging Large Language Models for Automated Construction of Scientific Leaderboards - EMNLP - 2024 - citation_count 10 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_10_Main_Efficient_Performance_Tracking_Leveraging_Large_Language_Models_for_Automated_Construction_of_Scientific_Leaderboards.pdf
- Tags: agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2024_EMNLP_10_Main_Efficient_Performance_Tracking_Leveraging_Large_Language_Models_for_Automated_Construction_of_Scientific_Leaderboards.pdf
- Token Usage: input 20972, output 4630, total 25602

### GitHub & Websites

- SCILEAD (Scientific Leaderboard dataset) / Leaderboard Generation · [GitHub](https://github.com/UKPLab/leaderboard-generation) · [Website](https://www.tudatalib.ulb.tu-darmstadt.de/)
  - description: Official code and data release for the paper; implements PDF parsing, RAG-based TDMR extraction, normalization, and leaderboard construction. The dataset is hosted on TUdatalib as stated in the paper.

- AxCell · [GitHub](https://github.com/paperswithcode/axcell)
  - description: Baseline system for automatic extraction of results from ML papers; used in this work for comparison in TDMR extraction and leaderboard construction.

- LangChain · [GitHub](https://github.com/langchain-ai/langchain) · [Codewiki](https://codewiki.google/github.com/langchain-ai/langchain) · [Doc](https://python.langchain.com/)
  - description: Framework used to orchestrate LLM prompting and Retrieval-Augmented Generation pipelines for TDMR extraction and normalization.

- Unstructured · [GitHub](https://github.com/Unstructured-IO/unstructured) · [Codewiki](https://codewiki.google/github.com/Unstructured-IO/unstructured) · [Doc](https://unstructured-io.github.io/unstructured/)
  - description: PDF processing toolkit used to parse papers and extract text and tables for downstream retrieval and extraction.

- Chroma · [GitHub](https://github.com/chroma-core/chroma) · [Codewiki](https://codewiki.google/github.com/chroma-core/chroma) · [Doc](https://docs.trychroma.com/)
  - description: Vector database used to store embeddings and perform similarity retrieval of paper chunks and tables.

- Sentence-Transformers · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Doc](https://www.sbert.net/)
  - description: Embedding library used for dense retrieval and cosine-similarity baseline; includes the embedding models used in the paper.

- multi-qa-mpnet-base-dot-v1 (Sentence-Transformers) · [Website](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1)
  - description: Pretrained embedding model used to create vector embeddings for paper chunks and tables in the RAG pipeline.

- rbo (Ranked Biased Overlap) · [GitHub](https://github.com/changyaochen/rbo)
  - description: Library used to compute ranking similarity (Average Overlap) between gold and predicted leaderboards.

- NLP-progress · [Website](https://nlpprogress.com/)
  - description: Community leaderboard site used to select popular tasks/datasets for constructing the SCILEAD dataset.

- Papers with Code · [Website](https://paperswithcode.com/)
  - description: Community leaderboard/resource site referenced for prior datasets and related analysis; provides context and baselines in the domain.

- Llama 2 · [GitHub](https://github.com/meta-llama/llama) · [Website](https://ai.meta.com/llama/)
  - description: One of the evaluated LLMs (Llama‑2 Chat 70B) used for zero-shot prompting in TDMR extraction and normalization.

- Llama 3 · [GitHub](https://github.com/meta-llama/llama3) · [Website](https://ai.meta.com/llama/)
  - description: Evaluated LLM (Llama‑3 Instruct 70B) used for extraction and normalization in all settings.

- Mixtral‑8x7B‑Instruct · [GitHub](https://github.com/mistralai) · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) · [Doc](https://mistral.ai/)
  - description: Evaluated LLM from Mistral AI used for TDMR extraction and normalization experiments.

- GPT‑4 Turbo · [Website](https://openai.com/) · [Doc](https://platform.openai.com/docs/models#gpt-4-turbo)
  - description: Closed-source LLM evaluated as the strongest model for TDMR extraction and leaderboard construction; accessed via Azure/OpenAI API.

<!-- paper_id: c7d43357593ec96c4a18845a413ffe5073a47589 -->

## 213. SelfGoal: Your Language Agents Already Know How to Achieve High-level Goals - NAACL - 2025 - citation_count 15 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_15_Long_SelfGoal_Your_Language_Agents_Already_Know_How_to_Achieve_High-level_Goals.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_15_Long_SelfGoal_Your_Language_Agents_Already_Know_How_to_Achieve_High-level_Goals.pdf
- Token Usage: input 30222, output 5050, total 35272

### GitHub & Websites

- SELFGOAL (project page) · [Website](https://selfgoal-agent.github.io)
  - description: Official project page for the paper; central hub with resources for reproducing the SELFGOAL agent framework introduced and evaluated in the work.

- GAMA-Bench (multi-agent game environments)
  - description: Benchmark used for Public Goods Game and Guess 2/3 of the Average; provides the multi-agent environments referenced for evaluation in the paper.

- AucArena · [GitHub](https://github.com/jiangjiechen/auc-arena)
  - description: Auction simulation environment used for the First-price Auction experiments; the paper evaluates agents in this arena to measure strategic planning and profit.

- Deal or No Deal (Negotiation) · [GitHub](https://github.com/facebookresearch/end-to-end-negotiator)
  - description: Official repository for the negotiation task used to implement the Bargaining environment (DealOrNotDeal) in the paper.

- ScienceWorld · [GitHub](https://github.com/allenai/ScienceWorld) · [Website](https://allenai.org/data/scienceworld)
  - description: Embodied, long-horizon textual environment used in additional single-agent evaluations to measure performance on complex, decomposable tasks.

- ReAct · [GitHub](https://github.com/ysymyth/ReAct)
  - description: Baseline agent framework combining reasoning and acting; used for comparison against SELFGOAL across tasks.

- Reflexion · [GitHub](https://github.com/noahshinn024/reflexion)
  - description: Baseline method that adds self-reflection from past attempts; the paper reimplements it for comparisons.

- ADAPT (As-needed Decomposition and Planning) · [GitHub](https://github.com/allenai/adapt) · [Website](https://adapt-agent.github.io)
  - description: Baseline hierarchical decomposition/planning approach; compared with SELFGOAL on the same environments.

- CLIN · [GitHub](https://github.com/allenai/clin)
  - description: Continually learning language agent baseline that stores causal abstractions; used as a comparison to SELFGOAL.

- TrueSkill · [GitHub](https://github.com/sublee/trueskill) · [Website](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) · [Doc](https://trueskill.org/)
  - description: Bayesian skill-rating system used to compute rankings/scores in the Auction and Bargaining evaluations.

- OpenAI API (GPT-3.5, GPT-4) · [Doc](https://platform.openai.com/docs)
  - description: LLM backends used to power agents and, in some setups, opponents; documentation for reproducing API-based runs.

- Google Gemini API (Gemini 1.0 Pro) · [Doc](https://ai.google.dev/)
  - description: Multimodal LLM backend used as one of the agent models in experiments; API docs for replication.

- Mistral 7B Instruct v0.2 · [Website](https://mistral.ai) · [Doc](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
  - description: Open-source instruction-tuned LLM used as a backbone in the experiments.

- Mixtral 8x7B Instruct v0.1 · [Website](https://mistral.ai) · [Doc](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
  - description: MoE instruction-tuned LLM used as a backbone in the experiments.

- Qwen1.5-7B-Chat · [Website](https://qwenlm.ai) · [Doc](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)
  - description: Open-source chat model used as a backbone for agents.

- Qwen1.5-72B-Chat · [Website](https://qwenlm.ai) · [Doc](https://huggingface.co/Qwen/Qwen1.5-72B-Chat)
  - description: Larger Qwen variant used as a backbone for agents.

<!-- paper_id: 09a6e95e87e569dca187f86ce3b155c8be145a9e -->

## 214. GuardAgent: Safeguard LLM Agents by a Guard Agent via Knowledge-Enabled Reasoning - ICML - 2025 - citation_count 65 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICML_65_Poster_GuardAgent_Safeguard_LLM_Agents_by_a_Guard_Agent_via_Knowledge-Enabled_Reasoning.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_ICML_65_Poster_GuardAgent_Safeguard_LLM_Agents_by_a_Guard_Agent_via_Knowledge-Enabled_Reasoning.pdf
- Token Usage: input 25994, output 5548, total 31542

### GitHub & Websites

- GuardAgent · [Website](https://guardagent.github.io/)
  - description: Official project page for the paper; hosts resources for GuardAgent and the two new benchmarks (EICU-AC and Mind2Web-SC) used to evaluate the guard agent.

- EICU-AC (Access Control benchmark) · [Website](https://guardagent.github.io/)
  - description: Benchmark introduced by the paper for assessing role-based access control of healthcare agents; built from the EICU dataset and used to evaluate GuardAgent safeguarding EHRAgent.

- Mind2Web-SC (Safety Control benchmark) · [Website](https://guardagent.github.io/)
  - description: Benchmark introduced by the paper for enforcing safety policies on web agents; constructed from Mind2Web tasks with user profiles and rule checks to evaluate GuardAgent on SeeAct.

- eICU Collaborative Research Database (eICU-CRD) · [Website](https://physionet.org/content/eicu-crd/) · [Doc](https://eicu-crd.mit.edu/)
  - description: Multi-center critical care database cited and used as the source for questions and tables; the paper derives EICU-AC from this dataset for evaluating healthcare agent access control.

- Mind2Web · [GitHub](https://github.com/OSU-NLP-Group/Mind2Web) · [Website](https://osu-nlp-group.github.io/Mind2Web/)
  - description: Large benchmark of real web tasks across many sites; the paper filters Mind2Web tasks and augments them with user profiles to create Mind2Web-SC and evaluate GuardAgent with SeeAct.

- SeeAct · [GitHub](https://github.com/OSU-NLP-Group/SeeAct) · [Website](https://osu-nlp-group.github.io/SeeAct/)
  - description: Generalist web agent used by the paper as a target agent; GuardAgent moderates SeeAct’s predicted actions and reasoning traces on Mind2Web-SC.

- LlamaGuard (Llama Guard 3) · [GitHub](https://github.com/meta-llama/PurpleLlama) · [Doc](https://ai.meta.com/resources/models-and-libraries/llama-guard/)
  - description: Input–output safety classifier from Meta used in the paper as a model-guarding baseline to compare against GuardAgent.

- CommonsenseQA (CSQA) · [Website](https://www.tau-nlp.org/commonsenseqa)
  - description: Multiple-choice commonsense QA dataset used in the paper’s appendix to test GuardAgent’s code-based guardrails on a standalone LLM setting.

<!-- paper_id: 2fd69fa8ebbcbdf23f3ca6dff00706df5719a156 -->

## 215. xLAM: A Family of Large Action Models to Empower AI Agent Systems - NAACL - 2025 - citation_count 81 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_81_Long_xLAM_A_Family_of_Large_Action_Models_to_Empower_AI_Agent_Systems.pdf
- Tags: applications
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/5applications/2025_NAACL_81_Long_xLAM_A_Family_of_Large_Action_Models_to_Empower_AI_Agent_Systems.pdf
- Token Usage: input 20688, output 6727, total 27415

### GitHub & Websites

- xLAM (Large Action Models) · [GitHub](https://github.com/SalesforceAIResearch/xLAM) · [Website](https://huggingface.co/Salesforce)  
  - description: Official release of the xLAM model family introduced in the paper (weights/checkpoints and usage). The models are used throughout the experiments and are the main contribution.

- APIGen · [GitHub](https://github.com/SalesforceAIResearch/APIGen) · [Website](https://arxiv.org/abs/2409.10019)  
  - description: Automated pipeline the authors use to synthesize 50k verifiable function-calling datapoints from 3,673 executable APIs; central to the paper’s data generation in Section 3.4.

- DialogStudio · [GitHub](https://github.com/salesforce/DialogStudio)  
  - description: Unified multi-domain instruction/conversation datasets used as part of the general instruction-tuning mixture for xLAM (Section 3.5).

- Data Provenance Initiative · [Website](https://dataprovenance.org)  
  - description: Instruction-tuning data source incorporated into the training mixture to improve xLAM’s general abilities (Section 3.5).

- Hugging Face Transformers · [GitHub](https://github.com/huggingface/transformers) · [Codewiki](https://codewiki.google/github.com/huggingface/transformers)  
  - description: Core training/inference library the authors base their SFT/DPO code on (Section 4.1).

- Hugging Face Accelerate · [GitHub](https://github.com/huggingface/accelerate)  
  - description: Training launcher/runtime the paper uses to scale experiments (Section 4.1).

- PyTorch FSDP · [Doc](https://pytorch.org/docs/stable/fsdp.html)  
  - description: Fully Sharded Data Parallel used for full-parameter SFT to train xLAM efficiently (Section 4.1).

- LoRA · [GitHub](https://github.com/microsoft/LoRA) · [Codewiki](https://codewiki.google/github.com/microsoft/LoRA)  
  - description: Low-Rank Adaptation used for alignment and to preserve pretrained abilities, especially for xLAM-8x22B and during DPO (Section 4.1).

- Berkeley Function-Calling Leaderboard/Benchmark (BFCL v2) · [Website](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html)  
  - description: Official benchmark/leaderboard where xLAM is evaluated and ranked; used for the function-calling experiments (Sections 5.1, 5.2.2).

- ToolBench / ToolLLM · [GitHub](https://github.com/THUDM/ToolBench)  
  - description: Benchmark and evaluation suite for multi-turn tool-use the authors use to test pass rate and generalization (Sections 5.1, A.2).

- WebShop · [GitHub](https://github.com/princeton-nlp/WebShop)  
  - description: Interactive web shopping environment used to evaluate agent navigation and task completion (Sections 5.1, 5.2.1).

- AgentBoard (incl. ToolQuery config) · [Website](https://arxiv.org/abs/2401.13178)  
  - description: Evaluation framework whose testing configurations the paper follows for WebShop and ToolQuery; ToolQuery is one of the evaluated settings (Sections 5.1, A.1).

- Gorilla OpenFunctions v2 · [Website](https://gorilla.cs.berkeley.edu/blogs/7_open_functions_v2.html)  
  - description: Function-calling resource referenced in the data unification discussion as a strong function-calling model/dataset format the paper compares/relates to (Section 3.1).

- NexusRaven · [GitHub](https://github.com/NexusflowAI/NexusRaven)  
  - description: Function-calling model cited in the data unification section as a prior strong approach, motivating a universal function-calling format (Section 3.1).

- AgentOhana · [GitHub](https://github.com/SalesforceAIResearch/agentohana)  
  - description: Prior open-source agent training pipeline cited for unified data and used as a comparison baseline in experiments (Sections 2.1, 5.2.1).

- RapidAPI · [Website](https://rapidapi.com)  
  - description: API marketplace used by ToolBench for real-time multi-turn tool-use evaluation (Section 5.1).

- Mixtral Instruct (base models) · [Website](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)  
  - description: Pretrained mixture-of-experts instruction models that xLAM builds upon for general agent capabilities (Section 4.2).

- DeepSeek-Coder Instruct (base models) · [Website](https://huggingface.co/deepseek-ai/DeepSeek-Coder-7B-instruct-v1.5)  
  - description: Code-focused instruction models that serve as bases for the specialized function-calling variants xLAM-7b-fc-r and xLAM-1b-fc-r (Section 4.2).

<!-- paper_id: a75da880b921d81426800a9893ce7c743339b278 -->

## 216. Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents - ICML - 2025 - citation_count 25 - /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICML_25_Poster_Proposer-Agent-Evaluator(PAE)_Autonomous_Skill_Discovery_For_Foundation_Model_Internet_Agents.pdf
- Tags: agent-science
- PDF Path: /Users/lingzhi/Code/AI/Ref_Papers/conferences/6agent-science/2025_ICML_25_Poster_Proposer-Agent-Evaluator(PAE)_Autonomous_Skill_Discovery_For_Foundation_Model_Internet_Agents.pdf
- Token Usage: input 37239, output 4896, total 42135

### GitHub & Websites

- WebArena · [GitHub](https://github.com/web-arena-x/webarena) · [Website](https://webarena.dev)
  - description: Realistic, self-hosted multi-website environment with ground-truth functional verifiers; used as a major training/evaluation environment (OpenStreetMap, PostMill, OneStopMarket) and for success/failure detection.

- WebVoyager · [Doc](https://arxiv.org/abs/2401.13919)
  - description: Real‑world web navigation benchmark with human-annotated tasks across popular sites; used as a primary evaluation suite, task source, and to define the set-of-marks observation setup.

- LLaVA (LLaVA-1.6 / LLaVA-Next) · [GitHub](https://github.com/haotian-liu/LLaVA) · [Codewiki](https://codewiki.google/github.com/haotian-liu/LLaVA) · [Website](https://llava-vl.github.io) · [Doc](https://llava-vl.github.io/blog/2024-01-30-llava-next/)
  - description: Open-source VLM used as the base agent policy (LLaVA‑1.6‑7B and LLaVA‑1.6‑34B) that PAE fine-tunes via RL/Filtered-BC.

- Qwen2-VL · [GitHub](https://github.com/QwenLM/Qwen2-VL) · [Website](https://qwenlm.github.io/)
  - description: Open-source multimodal LLM family used both as baselines (7B/72B) and, in ablations, as the autonomous task proposer and outcome evaluator.

- InternLM-XComposer 2.5 (InternVL2.5-XComposer) · [GitHub](https://github.com/InternLM/InternLM-XComposer)
  - description: Open-source VLM baseline (InternVL‑2.5‑XComposer‑7B/8B) evaluated under the same web-navigation setup.

- Claude 3 / 3.5 (Anthropic) · [Website](https://www.anthropic.com/claude) · [Doc](https://docs.anthropic.com/)
  - description: Proprietary VLMs used as strong baselines; also used as autonomous task proposer and outcome evaluator for PAE (0/1 success judgment from final screenshots/answers).

- ChromeDriver · [Website](https://chromedriver.chromium.org/) · [Doc](https://chromedriver.chromium.org/getting-started)
  - description: WebDriver used to automate the browser during trajectories (the paper notes ChromeDriver crashes during runs), relevant for reproducing the browser-based environment.

- Selenium · [GitHub](https://github.com/SeleniumHQ/selenium) · [Codewiki](https://codewiki.google/github.com/SeleniumHQ/selenium) · [Website](https://www.selenium.dev/)
  - description: Standard browser automation toolkit underpinning ChromeDriver-based control; applicable to reproducing the simulated browsing environment used for data collection and rollouts.

- Gradio · [GitHub](https://github.com/gradio-app/gradio) · [Codewiki](https://codewiki.google/github.com/gradio-app/gradio) · [Website](https://www.gradio.app/)
  - description: UI toolkit used by the authors to build the human annotation interface for evaluator alignment and error analysis.

<!-- paper_id: 52bca2ff9159bd9f704adffa5fd9c51a198574df -->

