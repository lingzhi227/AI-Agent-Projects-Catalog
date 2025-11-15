# OpenAI Cookbook Notes

## Prompt Engineering
- [Meta Prompting: A Guide to Automated Prompt Optimization](https://github.com/openai/openai-cookbook/blob/main/examples/Enhance_your_prompts_with_meta_prompting.ipynb)
  - 2025-01

- [+] [GPT-4.1 - prompt guide](https://cookbook.openai.com/examples/gpt4-1_prompting_guide)
  - time: 2025-04
  - model: gpt-4.1
  - general idea
    - providing context examples, 
    - making instructions as specific and clear as possible, 
    - inducing planning via prompting to maximize model intelligence
  - prompt structure
    ```
    # Role and Objective

    # Instructions

    ## Sub-categories for more detailed instructions

    # Reasoning Steps

    # Output Format

    # Examples
    ## Example 1

    # Context

    # Final instructions and prompt to think step by step
    ```
  - Delimiters
    - Markdown: We recommend starting here
      - using markdown titles for major sections and subsections (including deeper hierarchy, to H4+). 
      - Use inline backticks or backtick blocks to precisely wrap code
      - and standard numbered or bulleted lists as needed.
    - XML performed well in our long context testing.
      - XML is convenient to precisely wrap a section including start and end, 
      - add metadata to the tags for additional context, 
      - and enable nesting.

- [Optimize Prompts](https://github.com/openai/openai-cookbook/blob/main/examples/Optimize_Prompts.ipynb)
  - 2025-07

- [Prompt Migration Guide](https://github.com/openai/openai-cookbook/blob/main/examples/Prompt_migration_guide.ipynb)
  - 2025-07

- [+] [GPT-5 - prompt guide](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide)
  - time: 2025-08
  - model: gpt-5
  - Combined with [response api](https://platform.openai.com/docs/api-reference/responses)
    - Reusing reasoning context with the Responses API
  - spectrum: 
    ```
    GPT-5 is trained to operate anywhere along this spectrum, from making high-level decisions under ambiguous circumstances (proactivity) to handling focused, well-defined tasks(awaiting explicit guidance). 
    ```
  - Prompting the model how to explore the problem space
    - Prompting for less eagerness
      ```prompt
      <context_gathering>
      Goal: Get enough context fast. Parallelize discovery and stop as soon as you can act.

      Method:
      - Start broad, then fan out to focused subqueries.
      - In parallel, launch varied queries; read top hits per query. Deduplicate paths and cache; don’t repeat queries.
      - Avoid over searching for context. If needed, run targeted searches in one parallel batch.

      Early stop criteria:
      - You can name exact content to change.
      - Top hits converge (~70%) on one area/path.

      Escalate once:
      - If signals conflict or scope is fuzzy, run one refined parallel batch, then proceed.

      Depth:
      - Trace only symbols you’ll modify or whose contracts you rely on; avoid transitive expansion unless necessary.

      Loop:
      - Batch search → minimal plan → complete task.
      - Search again only if validation fails or new unknowns appear. Prefer acting over more searching.
      </context_gathering>
      ```
      ```prompt
      <context_gathering>
      - Search depth: very low
      - Bias strongly towards providing a correct answer as quickly as possible, even if it might not be fully correct.
      - Usually, this means an absolute maximum of 2 tool calls.
      - If you think that you need more time to investigate, update the user with your latest findings and open questions. You can proceed if the user confirms.
      </context_gathering>
      ```
    - Prompting for more eagerness 
      ```prompt
      <persistence>
      - You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user.
      - Only terminate your turn when you are sure that the problem is solved.
      - Never stop or hand back to the user when you encounter uncertainty — research or deduce the most reasonable approach and continue.
      - Do not ask the human to confirm or clarify assumptions, as you can always adjust later — decide what the most reasonable assumption is, proceed with it, and document it for the user's reference after you finish acting
      </persistence>`
      ```
    - Tool preambles
      ```prompt
      <tool_preambles>
      - Always begin by rephrasing the user's goal in a friendly, clear, and concise manner, before calling any tools.
      - Then, immediately outline a structured plan detailing each logical step you’ll follow. - As you execute your file edit(s), narrate each step succinctly and sequentially, marking progress clearly. 
      - Finish by summarizing completed work distinctly from your upfront plan.
      </tool_preambles>
      ```
    - Coding任务避免的错误
      - 修改要minimal
      - 要修根原因、而不是表面拼接
      - 必须跑pre-commit
      - 在完成任务前不能停 - 不算搜索、修改、尝试
      - 先分析、再搜索、再改代码
      - 禁止提问 - 尽量自己推断

- [+] [GPT-5 - prompt optimizer guide](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/prompt-optimization-cookbook.ipynb)
  - time: 2025-08
  - model: gpt-5
  - common failure modes
    - Contradictions in the prompt instructions
    - Missing or unclear format specifications
    - Inconsistencies between the prompt and few-shot examples
  - Using playground tools
  - LLM as a judge to provide qualitative scoring for results

- [+] [GPT-5 New Params and Tools](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/gpt-5_new_params_and_tools.ipynb)
  - time: 2025-09
  - new parameters
  - Verbosity Parameter
  - Minimal Reasoning
  - Free‑Form Function Calling
    - Directly return the written code
    - Use the written code as a tool and execute it directly locally
    - Get the result, return it, and continue to the next step
  - Context‑Free Grammar (CFG)

- [+] [GPT-5 Troubleshooting Guide](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/gpt-5_troubleshooting_guide.ipynb)
  - time: 2025-09
  - Overthinking
  - Laziness / underthinking
  - Overly deferential
  - Too verbose
  - Latency
  - Calling too many tools
  - Malformed tool calling
  - General troubleshooting

- [] [GPT-5 - codex prompt guide](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5-codex_prompting_guide.ipynb)
  - time: 2025-09
  - adaptive reasoning
  - Planning

- [+] [GPT-5.1 - prompting guide](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/gpt-5-1_prompting_guide.ipynb)
  - time: 2025-11
  - model: gpt5.1
  - The model's steerability is enhanced, with better instruction following
    - verbosity: parameter control, or direct prompting
    - personality and style work best when you define a clear agent persona
  - Tool-calling format
    - tool definition defines tools
    - At the same time, prompt directly specifies when to use tools
      - In the form of multiple examples
    - parallel tool calls
      - Declare in the system prompt
      - you can reinforce parallel tool usage by providing some examples of permissible parallelism. 
    - planning
      - planning tool
    - specific tools that are commonly used in coding use cases
      - apply_patch
      - shell
  - metaprompt GPT-5.1 to debug prompt

## response api

- [response api intro](https://github.com/openai/openai-cookbook/blob/main/examples/responses_api/responses_example.ipynb)

- [+] [reasoning](https://github.com/openai/openai-cookbook/blob/main/examples/responses_api/reasoning_items.ipynb)
  - Caching
  - summaries

- [Using file search tool in the Responses API](https://github.com/openai/openai-cookbook/blob/main/examples/File_Search_Responses.ipynb)

- [Multi-Tool Orchestration with RAG approach using OpenAI's Responses API](https://github.com/openai/openai-cookbook/blob/main/examples/responses_api/responses_api_tool_orchestration.ipynb)

## api tricks
- [multicore parallel processing](https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py)

- [batch processing](https://github.com/openai/openai-cookbook/blob/main/examples/batch_processing.ipynb)

- [token usage statistics](https://github.com/openai/openai-cookbook/blob/main/examples/completions_usage_api.ipynb)

- [How to handle rate limits](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb)

## Reasoning

- [+] [o1 - Routine Generation](https://github.com/openai/openai-cookbook/blob/main/examples/o1/Using_reasoning_for_routine_generation.ipynb)
  - time: 2024
  - model: o1
  - generate routine for customer service

- [+] [o1 - data validation](https://github.com/openai/openai-cookbook/blob/main/examples/o1/Using_reasoning_for_data_validation.ipynb)
  - time: 2024
  - model: o1
  - Generate data, then use o1's reasoning capability for data validation

## embeddings

- [User Product Embeddings](https://github.com/openai/openai-cookbook/blob/main/examples/User_and_product_embeddings.ipynb)

## structural output

- [+] [o1 - Structured Output](https://github.com/openai/openai-cookbook/blob/main/examples/o1/Using_chained_calls_for_o1_structured_outputs.ipynb)
  - time: 2024
  - model: o1
  - structural output using prompt

## RAG
- [Retrieval Augmented Generation with a Graph Database](https://github.com/openai/openai-cookbook/blob/main/examples/RAG_with_graph_db.ipynb)

## function call

- [Managing Function Calls With Reasoning Models](https://github.com/openai/openai-cookbook/blob/main/examples/reasoning_function_calls.ipynb)
  - time: 2025-05
  - o3, o4-mini
  - conversation orchestration

## Eval
- [Evaluation Intro](https://github.com/openai/openai-cookbook/tree/main/examples/evaluation)

- [Eval Driven System Design - From Prototype to Production](https://cookbook.openai.com/examples/partners/eval_driven_system_design/receipt_inspection)

## agent sdk
- [object oriented agentic approach](https://github.com/openai/openai-cookbook/blob/main/examples/object_oriented_agentic_approach/Secure_code_interpreter_tool_for_LLM_agents.ipynb)
  - time: 2025-02

- [+] [agentkit](https://github.com/openai/openai-cookbook/blob/main/examples/agentkit/agentkit_walkthrough.ipynb)
  - time: 2025-10
  - Agent Builder: visually build and iterate on agent workflows
  - ChatKit: easily embed chat-based workflows into your app
  - Evals: improve the performance of your LLM-powered apps
  - Improving system performance using prompt optimization and trace grading
    - Single agent optimization
    - Entire workflow optimization

## multi-agent
- [Structured Outputs for Multi-Agent Systems](https://github.com/openai/openai-cookbook/blob/main/examples/Structured_outputs_multi_agent.ipynb)
  - time: 2024-11

- [Multi-Agent Orchestration with OpenAI Agents SDK: Financial Portfolio Analysis Example](https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/multi-agent-portfolio-collaboration/multi_agent_portfolio_collaboration.ipynb)
  - time: 2025-05
  - Collaboration Patterns: Handoff vs. Agent-as-Tool
  - Architecture Overview
  - Supported Tool Types

- [codex_mcp_agents_sdk - Codex Coding Agent](https://github.com/openai/openai-cookbook/blob/main/examples/codex/codex_mcp_agents_sdk/building_consistent_workflows_codex_cli_agents_sdk.ipynb)
  - time: 2025-10
  - Project Manager: Breaks down task list, creates requirements, and coordinates work.
  - Designer: Produces UI/UX specifications.
  - Frontend Developer: Implements UI/UX.
  - Backend Developer: Implements APIs and logic.
  - Tester: Validates outputs against acceptance criteria.


## Context Engineering
- [How to use functions with a knowledge base](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_for_knowledge_retrieval.ipynb)

- [Summarizing Long Documents](https://github.com/openai/openai-cookbook/blob/main/examples/Summarizing_long_documents.ipynb)

- [+] [Context Engineering - Short-Term Memory Management with Sessions from OpenAI Agents SDK](https://github.com/openai/openai-cookbook/blob/main/examples/agents_sdk/session_memory.ipynb)
  - time: 2025-09
  - basic memory support: response api
  - automatic memory management: agent sdk
  - Context Trimming
  - Context Summarization

## Routine
- Using reasoning for routine generation. [Website](https://cookbook.openai.com/examples/o1/using_reasoning_for_routine_generation) - [GitHub](https://github.com/openai/openai-cookbook/blob/main/examples/o1/Using_reasoning_for_routine_generation.ipynb)
  - 2024-09
  - Converting the policy from an external facing document to an internal SOP routine
  - Breaking down the policy in specific actions and sub-actions
  - Outlining specific conditions for moving between steps
  - Determing where external knowledge/actions may be required, and defining functions that we could use to get that information

- Orchestrating Agents - Routines and Handoffs: [website](https://cookbook.openai.com/examples/orchestrating_agents) - [Github](https://github.com/openai/openai-cookbook/blob/main/examples/Orchestrating_agents.ipynb)
  - 2024-10

## Agent - coding
- [gpt5 - agent-coding-app](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/gpt-5_frontend.ipynb)
  - building apps

- [gpt-5.1 coding agent](https://github.com/openai/openai-cookbook/blob/main/examples/Build_a_coding_agent_with_GPT-5.1.ipynb)

## Agent - Research

- [Deep Research api](https://github.com/openai/openai-cookbook/blob/main/examples/deep_research_api/introduction_to_deep_research_api.ipynb)

## Agent - customer support related

- [Build with Realtime Mini](https://github.com/openai/openai-cookbook/blob/main/examples/building_w_rt_mini/building_w_rt_mini.ipynb)

## multi-hour exec

- [Using PLANS.md for multi-hour problem solving](https://github.com/openai/openai-cookbook/blob/main/articles/codex_exec_plans.md)
  - 2025-10