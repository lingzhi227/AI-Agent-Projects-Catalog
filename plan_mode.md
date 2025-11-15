
# Log

## 2025-08 Agent - Voice

## 2025-09 Agent - Web Playwright actions

## 2025-10 OpenAI audio; realtime api

## 2025-11-07 arxiv papers: Download; math agent
- [Background Research](/media/lingzhi/Mass3/Code/AI/Background_Code)
  - [Papers](/media/lingzhi/Mass3/Code/AI/Background_Papers)
  - [Tutorials](/media/lingzhi/Mass3/Code/AI/Background_Tutorials)
    - [MarketPlace Tutorials](/media/lingzhi/Mass3/Code/AI/Background_Tutorials/Marketplace_Tutorials)
    - [Ahead of AI](/media/lingzhi/Mass3/Code/AI/Background_Tutorials/Ahead_of_AI)
  - Code Bases
    - [API call](/media/lingzhi/Mass3/Code/AI/Background_Code/api_call)
    - [ToolUniverse](/media/lingzhi/Mass3/AI_Software/Code/AI/Background_Code/ToolUniverse)
    - Agent SDK etc
    
- read [Papers-Agent](/media/lingzhi/Mass3/Code/AI/Background_Papers/Agent)
    
- write [Domain Specific Agent design](Code/AI/plan/design_domain_specific.md)
    
- start research [Math Agent](/media/lingzhi/Mass3/Code/AI/Agent-Math)
  - design [Math Agent](/media/lingzhi/Mass3/Code/AI/plan/design_Agent_Math.md)
  - MathAgent [dataset preparation](/media/lingzhi/Mass3/Code/AI/Dataset/agent-math)

- continue research [Bio Agent](/media/lingzhi/Mass3/Code/AI/Agent-Bio) 

## 2025-11-10 paper reader v1
- use [Paper Reader](/Users/lingzhi/Code/AI/Background_Papers/paper_reader) to read papers

- task1: paper reader version 1: accept pdf and picture 
  - write a python program do the following work using gpt-5
  - the OpenAI api key is set using /Users/lingzhi/set_llm_keys.sh
  - read in pdf and jpeg picture files.
  - make use of gpt-5 api's multimodal source information understanding ability.
    - for each pdf paper, call api once
      - some papers are agentic system design, some are trying to enhance specific aspects of the system.
      - no matter what the emphasis, we only care about what the end agentic system should look like? what are all the components of the agentic software?
      - so, for eaach file, export a agentic system description model
        - which is a tree structure, to describe the agentic system
        - must describe the agent node, with LLM model names, roles, prompts, available tools, memory features, etc
        - all system level components, and their description
      - where can we read the code: url, website, pdf link, etc
    - for each picture file
      - export a agentic system description model
      - which is a tree structure, to describe the agentic system
      - must describe the agent node, with LLM model names, roles, prompts, available tools, memory features, etc
      - all system level components, and their description
  - export in md format to a agentic_components.md file with all the information api exported
  - with a list of results from each files, pdfs or pictures
  - in a format:
    - Title of the paper or picture: exact copy of the file_name.pdf. the title should only contain the file_name, path not included.
    - its agentic system description
    - source location. remember, the description should be correct. and the source location is not the location of the pdf/jpeg. but the source code/project page that might be mentioned in the file content. if the api gpt-5 cannot find it, export null. The source location should fill with all the url,source code, website mentioned in the pdf that is helpful/constructive in understanding the paper.


## 2025-11-11 conference papers: Download; project: get info
- 把 所有收藏的本地化
  - 把 twitter的所有论文、diagram本地化
  - 把 浏览器的所有论文本地化
- classify [Papers](/Users/lingzhi/Code/AI/Background_Papers)
- 刷conference papers
  - ACL Jan 6
  - ICML Jan 31
  - NIPS May
  - EMNLP May
  - ICLR Sept


- task2 get arxiv citation
  - 使用Semantic Scholar API写一个论文引用量获取工具get_arxiv_citation.py program
  - 利用这个工具，读取/Users/lingzhi/Code/AI/plan+notes/notes/paper_list_agentic_components.md当中的source location当中的arxiv论文，把获取到的引用量直接空4格+引用数字


- task3 get arxiv paper name
  - 找到一个合适的api， 写一个arxiv论文名称获取工具get_arxiv_name.py
  - 利用这个工具，读取/Users/lingzhi/Code/AI/plan+notes/notes/paper_list_agentic_components.md当中source location当中的arxiv论文链接，把获取到的论文名称直接加在链接前面，用冒号‘：’分隔开
  - 例如：line 525我已经改过了，变成了：
    - GPT-4 Technical Report: https://arxiv.org/abs/2303.08774    19068


- task4 get arxiv paper pdf
  - 找到一个合适的api， 写一个arxiv论文pdf获取工具get_arxiv_pdf.py
  - 利用这个工具，读取/Users/lingzhi/Code/AI/plan+notes/notes/paper_list_agentic_components.md当中source location当中的arxiv论文链接，把获取到的论文pdf统一存储在/Users/lingzhi/Code/AI/Background_Papers/arxiv
  - 命名规则：每一个论文首先保留原arxiv号码， 然后是论文的引用量数字，然后是论文的完整标题，出现冒号统一用空格替代
    - 例如： - Adam: A Method for Stochastic Optimization: https://arxiv.org/abs/1412.6980    158577
    - 命名为：1412.6980_158577_Adam A Method for Stochastic Optimization
  - 下载文件中出现的所有的arxiv论文


- task5 paper scratcher from top conferences
  - learn to use this website: ai-paper-finder.info
  - write a paper_finder.py program that automate paper finder process
  - first you should create a config.yaml file that I can type in all the available parameters
  - second, the py program should read the yaml file, execute on the website, the website will get back a list of papers information
  - the py program should download the information appeared on the website to a jsonl file, each paper is a json.
  - then it should use the url provided on the website, to download the pdf of that paper. titled exactly the format provided on the website: conference name+year+title_of_paper
  - remember, you should store the Author, Affinity Score, link, abstract, BibTeX information all in the jsonl for future use purposes.


- task6 get github title
  - 用合适的方式，写出一个get_github_title.py程序，在/Users/lingzhi/Code/AI/plan+notes/tools/github位置
  - 这个program应该给md文件当中的github网址确认到project的title，添加到网址的前方，用冒号：隔开
  - 比如对于这个文件：/Users/lingzhi/Code/AI/plan+notes/notes/paper_list_agentic_components.md
    - 里面每一篇论文的source location有很多github项目的网址
    - 这个程序要扫描所有的github网址
      - 如果有readme.md就把readme当中的title作为这个github项目的名称。
      - 如果没有readme，就直接把github这个代码库的名字作为这个网址的名称。
  - 最后把所有的论文下面的source location当中的github项目收集到一个文件github_projects.md当中，存储地址在程序本地


## 2025-11-12 projects & papers catalog: categorize 
- 学agent基础知识
  - plan-verify求解器
  - 幂等重试
- categorize github & website project
- categorize arxiv & conference papers


- task7 get project info
  - 用合适的方式，写出一个get_project_title.py程序，在/Users/lingzhi/Code/AI/plan+notes/tools 位置
  - 这个program应该给md文件当中的除了arxiv、github的其他网址确认到这个project的title，添加到网址的前方，用冒号：隔开
  - 比如对于这个文件：/Users/lingzhi/Code/AI/plan+notes/notes/paper_list_agentic_components.md
    - 里面每一篇论文的source location有很多arxiv、github、其他网址，对英语项目的网址
    - 这个程序要扫描所有的网址，排除arxiv和github
      - 如果网站明显位置有title，就把这个title作为这个网址项目的名称。
      - 如果没有，就直接把网址最后这个网址的名字作为这个网址的名称。
  - 最后把所有的论文下面的source location当中的这类网址项目，收集到一个文件projects_websites.md当中，存储地址在程序本地

- task8 project categorization
  - 我在尝试给/Users/lingzhi/Code/AI/Reference_projects/project_catalog.md当中的## GitHub Projects from Source Locations当中的github project归类。
  - 我已经归类了好多了，还剩下很多。
  - 你能否把剩余部分，如果可以归类为我已经确定的门类，就把它搬运到门类的后方append。
    - 如果无法确定分类，就先保留不动。
  - 记住只归类## GitHub Projects from Source Locations而不要动剩余的其他的list，那些我之后会继续研究。 


- task9 put website link next to github
  - 帮助我处理/Users/lingzhi/Code/AI/Reference_Projects/project_catalog.md
  - 有很多website的网址，其实是github project相同的项目，所以我们需要把这类github已经提到的项目，网址跟github repo放在一起
  - 例如对于### Deep Learning - node2vec: Scalable Feature Learning for Networks [Github](https://github.com/aditya-grover/node2vec)  [Website](https://snap.stanford.edu/node2vec/)我已经做过了
    - 我把website里面的它的网站剪切到了跟github repo在一起
  - 我们想要类似于这个例子，把所有的github和website的链接都以超链接形式。
  - 我目前的超链接设计是最简单的，但是易读性不强。你可以修改为更加易读的形式。


- task10 claude code website categorization
  - 用claude对website进行整理


- task11 get conference paper citation
  - 对于/Users/lingzhi/Code/AI/Reference_Papers/conferences当中的paper，包括sub-folder当中的
  - 是否可以根据file的文件名
    - 一般只根据会议名称、年份、文章标题（去掉下划线_）就可以搜索到这篇文章的信息
    - 获取到这篇已经发表的文章目前的引用数量
    - 把引用数量数字 放在文章真正的标题的前面，也就是所发表的会议、年份、发表方式（poster、spotlight等等）的后面
  - 写一个get_conference_paper_citation.pdf程序完成这个任务
  - 忽略2026因为这些文章还没有发表不会存在被引用


- task12 paper categorization
  - 我在试图整理/Users/lingzhi/Code/AI/Reference_Papers/conferences/agent_biology_protein_function这个文件夹的100篇论文
  - 创建话题/标题，然后把相关的论文放在里面。
  - 我的关注点主要是multiagent system for bio-science，特别是protein function。我们要关注这些论文是如何构建起来一个multiagent system的，以及用这类system如何解决science领域问题。
  - 我已经做了一些了。
  - 请你帮助我分类所有未分类的文章 

## 2025-11-13 conference papers: categorize, get info

- 用claude code详细分类papers

- task13 claude code arxiv paper categorization

- task14 claude code conference paper categorization
  - conference paper目前全部堆放在一起：/Users/lingzhi/Code/AI/Reference_Papers/conferences
  - 首先把arxiv的文件夹/Users/lingzhi/Code/AI/Reference_Papers/arxiv内部的子文件夹名称都复制过来，重用一样的名称，创建文件夹
  - 对于conference文件夹的内部的每一个文件（包括已经分类了的子文件夹的内部的pdf）
    - 匹配它特别匹配的最合适的文件夹
    - 如果没有特别合适的已知文件夹，即便他们存在相关性，只要不是特别合适，就需要创建新的合适的文件夹名称，作为这个文件夹的分类名称
  - 把每一个文件搬运到它自己归属的文件夹当中
  - 文件夹可以具有嵌套结构
    - 类似于agent-bio当中做的那样。
    - 但是嵌套结构不能超过两层

- task15 semantic scholar api get citation
  - we need to get citation of the papers located mostly at /Users/lingzhi//Code/AI/Ref_Papers/conferences
  - the real name of the paper is behind the conference name, year, type(poster, long, spotlight, Oral, Main, Findings)
    - the numbers behind are citation numbers we get before, but some are wrong, some are correct. We want to replace that number with our new info get from semantic scholar apis
    - when you get the citation count from the api, you should add or replace the number in middle of the conference-name-year-type tag and the real name of the paper. Just like we did before.
  - write a program get_citation_semantic.py: at /Users/lingzhi/Code/AI/plan+notes/tools/semantic/get_citation_semantic.py
  - the program need to make use of the api of semantic scholar

  ```[api reference](https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/post_graph_get_papers)
  In python:

  r = requests.post(
      'https://api.semanticscholar.org/graph/v1/paper/batch',
      params={'fields': 'referenceCount,citationCount,title'},
      json={"ids": ["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"]}
  )
  print(json.dumps(r.json(), indent=2))

  [
    {
      "paperId": "649def34f8be52c8b66281af98ae884c09aef38b",
      "title": "Construction of the Literature Graph in Semantic Scholar",
      "referenceCount": 27,
      "citationCount": 299
    },
    {
      "paperId": "f712fab0d58ae6492e3cdfc1933dae103ec12d5d",
      "title": "Reinfection and low cross-immunity as drivers of epidemic resurgence under high seroprevalence: a model-based approach with application to Amazonas, Brazil",
      "referenceCount": 13,
      "citationCount": 0
    }
  ]
  Other Examples:

  https://api.semanticscholar.org/graph/v1/paper/batch
  {"ids":["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"]}
  Returns details for 2 papers.
  Each paper has its paperId and title.
  https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,isOpenAccess,openAccessPdf,authors
  {"ids":["649def34f8be52c8b66281af98ae884c09aef38b", "ARXIV:2106.15928"]}
  Returns all requested info plus paper IDs for 2 papers.

  Limitations:
  Can only process 500 paper ids at a time.
  Can only return up to 10 MB of data at a time.
  Can only return up to 9999 citations at a time.
  For a list of supported IDs reference the "Details about a paper" endpoint.
  ```
  - the api key is stored at first line of /Users/lingzhi/Code/AI/plan+notes/tools/semantic/semantic_key.md. 
    - the program should not hard code the api key.
    - the program should read api key from the file. The file will always store with the program in the same PATH/location.
  - remember, later we will run the program on hundreds of papers. So the program must be a batched version. 
    - if we use: python get_citation_semantic.py --batch-size 50 the program should process upto 50 papers per endpoint request. 
  - you should test your program with papers from /Users/lingzhi/Code/AI/Ref_Papers/conferences/3agent-framework
  - make sure your program could work before you finish the work.

- task get conference paper citations
  - 用tools/semantic 处理Papers/conferences当中的所有论文

## 2025-11-14 openai cookbook: categorize, take notes; marktechpost: categorize

- task openai cookbook： 筛选、分类、学习、测试
  - 笔记：/Users/lingzhi/Code/AI/plan+notes/notes/tutorial_openai_cookbook.md
  - 重要概念
    - prompt engineering
    - response api
    - context engineering
    - agent sdk

- task marktechpost：筛选、分类、学习、测试
  - 重要概念
    - agentic
    - multiagent
    - tool
    - agent sdk
    - rl

## 2025-11-15 code base

- task 通过论文获取更加全面的github codebase catalog

- task 筛选出有用的模块
  - tooluniverse

- task Test Programs
  - Tools
  - Agent memory
  - Agent synthesizer and optimizer
  - RL settings

## paper database

- task 本地paper数据库
  - paper_id
  - standard_name
  - publication_method: Conference, arxiv
  - org
  - citation
  - Bib引用格式调取

- task 筛选论文
  - 按照类别
  - 按照机构
  - 按照conference影响力
  - 按照引用
  - 组合：每一个类别当中的高被引的文章

- task 用paper reader2阅读筛选出的论文
  - note
  - 代码库 代码库分析
  - 索引分析 获得高被引文章的引用文献当中的高被引

- task 读论文自动化
  - 做paper reader v2
  - 做paper reader v3
  - 用paper reader v3 生成结构化输出
  - refernce： https://www.alphaxiv.org/overview/2510.26692

- task 读论文写notes
  - Read the exists high-impact papers
    - google
    - math agent
    - others

# TO DO

- 写agentic截止11月综述


- task 做 Deep Virtual Researcher
  - 论文-agent：获取和整理论文
  - 软件工具-agent
  - 软件实验-agent
  - 图表-agent
  - 写paper-agent

- task Deep Physical Researcher
  - desktop-agent
  - social-media-agent
  - 硬件工具-agent

