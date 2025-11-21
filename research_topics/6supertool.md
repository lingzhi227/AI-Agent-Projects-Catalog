# 2025 顶会中与工程 “Supertool” 相关论文梳理

> 更新日期：2025-11-21
> 只关注“我们讨论的那种 supertool”：**LLM 驱动工程类重工具（EDA / CFD / 仿真 / CAD 等）做长链自动化流程**，不列通用 Browser / Calculator 之类的广义 ToolUse。

---

## 1. 检索范围与筛选标准

**顶会范围**

* NLP / AI 顶会：ACL 2025、NAACL 2025、EMNLP 2025、Findings of ACL/EMNLP 2025（通过 ACL Anthology 检索）([ACL Anthology][1])
* NeurIPS 2025（主会 + Datasets & Benchmarks Track + Workshops）([ML4PhysicalSciences][2])

**只保留符合你 “supertool” 定义的：**

* LLM 不是只写一点代码 / 给建议，而是：

  * 直接控制 **工程级专业软件**（EDA 工具、OpenFOAM、MATLAB、CADENCE 等），且
  * 形成 **自动化、长链、多步骤** 的工程流程（综合、仿真、调参、迭代等）
* 不收录：

  * 一般意义上的 function-calling 框架（如 EMNLP 2025 的 Self-Guided Function Calling）([ACL Anthology][3])
  * 仅做代码表示学习 / 代码生成、但不真正 orchestrate 工程工具链的工作（会放到“相关工作”里简要提一下）

---

## 2. 真正命中的“工程 supertool”类工作（2025 顶会）

### 2.1 NAACL 2025 — EDAid：LLM 多智能体自动化 EDA 流程（最接近我们方案）

**论文：**
**Divergent Thoughts toward One Goal: LLM-based Multi-Agent Collaboration System for Electronic Design Automation**
Haoyuan Wu et al., NAACL 2025 long paper（主会长文）([ACL Anthology][4])

**关键点（从 supertool 视角）：**

* 目标：让 LLM 通过**脚本 + API** 自动化完整的 EDA 流程（基于 OpenROAD / iEDA 等平台），而不是只写一小段 Verilog。([ACL Anthology][4])
* 系统：**EDAid 多智能体系统**：

  * 多个 “divergent-thoughts agents” 负责提出不同的 EDA flow 方案
  * 一个 decision-making agent 负责在候选方案中选择最靠谱的一条
  * 每个 agent 背后是专门为 EDA fine-tune 的 **ChipLlama** 模型
* 工具交互方式：

  * LLM 产生 **EDA 脚本**（比如 TCL / Python）→ 调用 OpenROAD / iEDA 等 EDA 工具 API
  * 脚本执行结果（log、错误）再反馈给 LLM，形成闭环自动化。([ACL Anthology][4])
* 任务类型：

  * Simple flow calls：完成完整设计 flow（如 floorplan→placement→CTS→route）
  * Complex flow calls：比如寻找满足时序约束的最小 clock period
  * Parameter tuner calls：设计空间搜索 / 参数调优等复杂任务([ACL Anthology][4])

**和你的 supertool 架构的对应关系：**

* 它做的事情，本质是**“LLM + 多 agent + 工程工具 API = 专用 EDA supertool”**：

  * LLM 负责解析自然语言需求、规划 flow、生成脚本
  * 工具（OpenROAD 等）负责真正的综合、布局布线、时序分析
* 它目前主要集中在 **EDA 流水线**，没有跨 MATLAB / CFD / MCU 等多领域，但“**LLM 调多个专业工程工具、走长链流程**”的设计理念和我们讨论的 MCP+Worker 框架高度一致。

> 如果你在论文里要 cite 一个**最像我们设想的工程 supertool 的 2025 顶会工作**，这个 NAACL 2025 的 EDAid 可以说是第一优先。

---

### 2.2 NeurIPS 2025 Datasets & Benchmarks — EngDesign：LLM + 仿真软件的工程设计基准

**论文 / 项目：**
**Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs (EngDesign)**
NeurIPS 2025 Datasets & Benchmarks Track（官方网站已标明接收）([arXiv][5])

**核心内容：**

* 提出 **EngDesign benchmark**，评估 LLM 在九大工程领域的设计能力：

  * 操作系统设计 / 计算机体系结构 / 控制系统 / 机械设计 / 结构工程
  * 数字硬件设计 / 模拟 IC 设计 / 机器人 / 信号处理等([arXiv][5])
* 最大的特点：**仿真驱动的评估范式**：

  * LLM 给出一个设计（电路、控制器、结构等）
  * 系统用**领域仿真软件**去跑：从 SPICE 电路仿真、有限元分析，到控制系统验证、机器人规划等
  * 根据仿真结果自动打分（pass/fail / 分数 / 日志）([AGI for Engineering][6])
* 网站明确写了：**48 个任务依赖商业科学软件（MATLAB、Cadence 等），53 个任务用开源脚本复现**。([AGI for Engineering][6])

**从 supertool 的角度理解：**

* 他们主要做的是 **“LLM 生成设计 → 后端仿真工具自动验证”**：

  * 用脚本把 MATLAB / Cadence / SPICE / FEM 工具封装成**评测 pipeline**
  * LLM 输出的设计方案，由 pipeline 自动喂给这些仿真工具运行
* 这是一种“**单向调用 supertool**”：

  * 仿真工具并不是 LLM 在对话中任意调用的 Tool，而是**评测 pipeline 的一部分**
  * LLM 不一定能在一次对话内多轮读取仿真 log、自主调参再反复仿真（论文强调的是 benchmark，而不是完整 Agent 系统）([arXiv][7])

**对你有用的地方：**

* 它已经做了你在第 1～3 章里设想的事情的一半：
  **把 MATLAB / Cadence 等工程工具封装成“可编程仿真服务”，并和 LLM 输出自动对接。**
* 你可以在你的系统中：

  * 直接借鉴它的 **“任务定义 + 仿真脚本 + 自动打分”** 结构
  * 把这些 evaluation pipeline 挂到我们的 MCP / Worker 上，让 LLM 不只是“被动被 benchmark”，而是**主动驱动这些仿真**完成真实设计任务。

---

### 2.3 NeurIPS 2025 Workshop（Machine Learning and the Physical Sciences）— FoamGPT：LLM 自动化控制 OpenFOAM CFD 仿真

**论文：**
**FoamGPT: Fine-Tuning Large Language Model for Agentic Automation of CFD Simulations with OpenFOAM**
Ling Yue et al., NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences（ML4Phys）([ResearchGate][8])

**关键点：**

* 场景：CFD（计算流体力学）工程仿真，基于 **OpenFOAM** 框架。([ResearchGate][8])
* 目标：让 LLM **自动完成复杂 CFD 仿真任务**：

  * 从自然语言描述问题 → 生成 OpenFOAM case 文件 → 配置网格 / 边界条件 / 求解器参数 → 运行仿真 → 解读错误并修正
* 技术路径：

  * 对 LLM 微调，形成专用于 CFD 的 FoamGPT
  * 构建 **CFDLLMBench 数据集**，以便系统性评估 LLM 控制 OpenFOAM 的能力([ResearchGate][9])

**从 supertool 的角度：**

* 这是一个非常典型的 **“单领域 supertool”**：

  * LLM 通过命令行 / 脚本全程控制 OpenFOAM
  * 自动化程度很高，几乎实现“自然语言 → CFD 仿真结果”的端到端流程
* 虽然是在 NeurIPS Workshop（不是主会），但在“LLM 控制重型工程仿真软件”这一点上，非常贴近我们想要的风格。

---

### 2.4 2025 年 arXiv / 系统论文 — ChatCFD：OpenFOAM 端到端 Agent 系统

**论文：**
**ChatCFD: an End-to-End CFD Agent with Domain-specific Structured Thinking**
E. Fan et al., arXiv 2025（CFD Agent，OpenFOAM）([arXiv][10])

虽然目前看到的是 arXiv + GitHub / seminar 记录，尚未明确挂在某个顶会主会，但它在“supertool”层面的意义很大，所以这里一并列出，供你参考实现细节。

**内容要点：**

* 目标：让用户用自然语言或论文 PDF 描述一个 CFD 问题，**ChatCFD 自动构建并运行 OpenFOAM 仿真**。([arXiv][10])
* 系统结构（四个阶段）：([arXiv][10])

  1. **知识库构建**：从 OpenFOAM 教程和手册构建结构化 JSON 知识库
  2. **用户输入处理**：接受对话、文档、mesh 文件等多模态输入
  3. **Case 文件生成**：基于知识库自动写出完整 OpenFOAM case
  4. **仿真执行 + 错误反思**：运行仿真，基于 log 自动 debug 和修正配置
* 实现上采用 **多 agent + 结构化思维 + RAG**，并给出了实际仿真 Case 的验证结果（如 NACA0012 翼型等）。([arXiv][10])

**和 FoamGPT 的关系：**

* ChatCFD 更偏 **系统工程**（完整 pipeline + 界面 + 多模态输入）
* FoamGPT 更偏 **模型方法 + Benchmarks**（fine-tune + dataset + workshop 发表）
* 两者都可以看成是“**OpenFOAM 领域 supertool**”，你可以把这类工作类比成“你要做的 MATLAB/Vivado supertool 在 CFD 领域的版本”。

---

## 3. 相关但不完全符合“supertool”定义的 2025 顶会工作（可作为背景引用）

这里列的是“**LLM + EDA/工程**”但不直接控制完整工具链的工作，它们是写你论文 survey 的好素材，但不是你要的“终极 supertool”。

### 3.1 ACL 2025 Findings — DeepRTL2：面向 RTL 的通用模型

**论文：**
**DeepRTL2: A Versatile Model for RTL-Related Tasks**
Yi Liu et al., Findings of ACL 2025([ACL Anthology][11])

* 核心：为 RTL（硬件描述语言）相关任务（生成、理解、质量估计等）构建统一的 LLM / 表征模型。
* 贡献点在于 **RTL 代码层面的 representation & downstream tasks**，并不直接 orchestrate EDA 工具的综合 / P&R 流程。([ACL Anthology][11])
* 对你来说：属于“**supertool 里的语言子模块**”，可以借鉴其 RTL 表示和任务设定，但不解决“多软件协同”的系统问题。

### 3.2 TODAES 2025 — Survey：A Survey of Research in LLMs for EDA

**论文：**
**A Survey of Research in Large Language Models for Electronic Design Automation**
Jingyu Pan et al., ACM TODAES 2025（arXiv:2501.09655, comments: accepted by TODAES）([arXiv][12])

* 系统性总结 LLM 在 EDA 中的应用：

  * 助手 chatbot、HDL/script 生成、验证与分析等
  * 讨论 LLM 与 EDA 工作流融合的挑战与机会([arXiv][12])
* 不是具体 supertool，但可以作为你论文里“**LLM4EDA 相关工作综述**”的一条重要引用。

### 3.3 其他

* **FEA-Bench（ACL 2025）**：侧重代码库级别特征实现生成，不直接控制 CAE 软件，因此不算 supertool，只是对“工程类代码”的 benchmark。([ACL Anthology][13])
* **Self-Guided Function Calling in Large Language Models via Stepwise Experience Recall（EMNLP 2025 Findings）**：是通用 function-calling 强化框架，不特定于工程 supertool，被你划在“广义 tool use”里，可做方法背景引用。([ACL Anthology][3])

---

## 4. 小结：目前顶会上的 supertool 版图 & 和你要做的东西的关系

**2025 顶会中真正“对齐你设想”的 supertool 工作，大致有三条线：**

1. **NAACL 2025 — EDAid（ChipLlama, Multi-agent EDA flow）**

   * 多 agent + 专用 LLM + EDA 工具 API
   * 和你要做的 “LLM 调度 Worker（Vivado/Quartus/ModelSim/MCU 编译链）” 在架构理念上几乎一样，只是它聚焦在单一 EDA 域。([ACL Anthology][4])

2. **NeurIPS 2025 — EngDesign（Simulation-based Benchmark）**

   * 把 MATLAB / Cadence / SPICE / FEM 等**工程仿真软件整合成自动评测 pipeline**
   * LLM 生成设计 → 仿真软件评估 → 自动评分，为你构建“仿真回路 + 评分”的 MCP 工具提供现成范式。([AGI for Engineering][6])

3. **NeurIPS 2025 Workshop & 2025 系统工作 — FoamGPT / ChatCFD（CFD supertool）**

   * 面向 OpenFOAM 的专用智能体 / 微调模型，让 LLM 自动化 CFD 仿真，构成单领域 supertool。([ResearchGate][8])

**空白点（也是你系统有创新空间的地方）：**

* 目前还**没有看到**一个在顶会上发表的系统，能像你设计的那样：

  * 把 **MATLAB / Simulink / Vivado / Quartus / MCU toolchain / CFD** 等一整个工程技术栈
  * 通过统一的 MCP / job 调度 / 多机 Worker 架构拼接成 **一个“跨领域、跨软件”的 supertool 平台**。
* 已有工作大多：

  * 要么聚焦单一领域（EDA 或 CFD），做得很深
  * 要么做 benchmark / survey，而不是可扩展的工程平台

**对你后续写第 4 章 / 相关工作章节的建议：**

* 作为“**顶会工程 supertool**”示例重点讲：

  * NAACL 2025：EDAid（multi-agent EDA flow automation）
  * NeurIPS 2025：EngDesign（simulation-based engineering benchmark）
  * NeurIPS 2025 ML4Phys：FoamGPT（OpenFOAM agentic automation）
* 再用 TODAES survey + DeepRTL2 这类工作**补充 LLM4EDA 生态**，形成完整 related work。
* 然后明确你要做的是：

  > **把这些单领域 supertool 和仿真 pipeline 上升到“统一 MCP + 多 Worker 工程操作系统”的层级**，实现“LLM 一次对话里把 MATLAB / FPGA / MCU / CFD 串成真正的工程流水线”。

如果你愿意，我可以直接把这一段整理成你书稿里“相关工作（工程 supertool 方向）”的小节，一起并入你前面几章的 md 结构里。

[1]: https://aclanthology.org/2025.naacl-long.83/?utm_source=chatgpt.com "Divergent Thoughts toward One Goal: LLM-based Multi ..."
[2]: https://ml4physicalsciences.github.io/?utm_source=chatgpt.com "Machine Learning and the Physical Sciences, NeurIPS 2025"
[3]: https://aclanthology.org/2025.findings-emnlp.574.pdf?utm_source=chatgpt.com "Self-Guided Function Calling in Large Language Models ..."
[4]: https://aclanthology.org/2025.naacl-long.83.pdf?utm_source=chatgpt.com "LLM-based Multi-Agent Collaboration System for Electronic ..."
[5]: https://arxiv.org/abs/2509.16204?utm_source=chatgpt.com "Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs"
[6]: https://agi4engineering.github.io/Eng-Design/?utm_source=chatgpt.com "EngDesign Benchmark"
[7]: https://arxiv.org/html/2509.16204v2?utm_source=chatgpt.com "Benchmarking the Engineering Design Capabilities of LLMs"
[8]: https://www.researchgate.net/publication/397738920_FoamGPT_Fine-Tuning_Large_Language_Model_for_Agentic_Automation_of_CFD_Simulations_with_OpenFOAM?utm_source=chatgpt.com "(PDF) FoamGPT: Fine-Tuning Large Language Model for ..."
[9]: https://www.researchgate.net/figure/Performance-comparison-of-models-on-the-CFDLLMBench_tbl1_397738920?utm_source=chatgpt.com "Performance comparison of models on the CFDLLMBench"
[10]: https://arxiv.org/html/2506.02019v1?utm_source=chatgpt.com "ChatCFD: an End-to-End CFD Agent with Domain-specific ..."
[11]: https://aclanthology.org/2025.findings-acl.336/?utm_source=chatgpt.com "DeepRTL2: A Versatile Model for RTL-Related Tasks"
[12]: https://arxiv.org/abs/2501.09655 "[2501.09655] A Survey of Research in Large Language Models for Electronic Design Automation"
[13]: https://aclanthology.org/2025.acl-long.839.pdf?utm_source=chatgpt.com "FEA-Bench: A Benchmark for Evaluating Repository-Level ..."
