
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

- task use [Paper Reader](/Users/lingzhi/Code/AI/Background_Papers/paper_reader) to read papers


## 2025-11-11 conference papers: Download; project: get info
- task 收集论文
  - 把所有收藏的本地化
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

- task5 paper finder from top conferences
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
- task 学agent基础知识
  - plan-verify求解器
  - 幂等重试

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

- task16 get conference paper citations
  - 用tools/semantic 处理Papers/conferences当中的所有论文

## 2025-11-14 openai cookbook: categorize, take notes; marktechpost: categorize

- task17 openai cookbook： 分类、筛选、学习、测试
  - 笔记：/Users/lingzhi/Code/AI/plan+notes/notes/tutorial_openai_cookbook.md
  - 重要概念
    - prompt engineering
    - response api
    - context engineering
    - agent sdk

- task18 marktechpost：分类、筛选、学习、测试
  - 重要概念
    - agentic
    - multiagent
    - tool
    - agent sdk
    - rl

## 2025-11-15 paperbase; codebase

- task19 construct paperbase
  - 首先我们已经获取到了大量的conference paper pdf
    - 我们用的/Users/lingzhi/Code/AI/tools/paper-manager/paper-finder/paper_finder.py，获取到conference papers的基本信息和pdf实体
    - 目前基本信息放在了/plan+notes/catalog/data/paper_finder_results.jsonl
      - dimensions: title, venue, year, authors, affinity_score, link, abstract, bibtex, pdf_paths, retrieved_time
    - paper的实体pdf目前都存储在了/Users/lingzhi/Code/AI/Ref_Papers/conferences当中
  - 我又进一步用claude对他们进行了分类，产生多个子文件夹，作为他们的类别名称，每个文件夹放的是那个类别的paper pdf
  - 之后我利用semantic scholar获得了这些paper的进一步的信息
    - 我用/tools/paper-manager/semantic/get_citation_semantic.py对pdf进行semantic scholar api的专属信息检索。
    - 目前检索的专属信息比较局限。目前的专属信息只包括paper_id, standard_name, citation_count
    - 结果放在了/Users/lingzhi/Code/AI/plan+notes/catalog/raw_paper_id_conference.jsonl
  - 我们想构建一个paper数据库
    - 我们要构建一个construct_paperbase.py程序，放在/Users/lingzhi/Code/AI/tools/paper-manager，输出一个paper_base.jsonl
    - 对于每一个paper pdf，他们基本都是从paper-finder.py获得的，所以他们基本上都有paper_finder_results.jsonl信息。你需要想办法进行名称匹配，建立pdf与paper_finder_results.jsonl字段的一一对应关系。
    - 后来我们又对这些paper进行了semantic scholar查询，查询结果放在了raw_paper_id_conference.jsonl。
    - 我们要把paper_finder_results.jsonl与raw_paper_id_conference.jsonl合并为paper_base.jsonl。在合并过程中：
      - 我们基本要保留paper_finder_results.jsonl当中这个paper的基本信息。title，venue，authors，affinity_score, link, abstract, bibtex, pdf_path。但是很多信息需要修改。
      - 从semantic scholar results raw_paper_id_conference.jsonl获得paper_id, citation_coount, arxiv_id
      - 加tag
        - 要把paper_finder_results.jsonl当中的每个paper的query转换为tag
          - 比如multiagent tool science， 转换为3个tag：multiagent，tool，science
          - 比如science agent biology protein-function， 转换为四个tag：science， agent， biology， protein-function
          - 比如multiagent tool mathematics转换为三个tag：multiagent， tool， mathematics
        - 目前paper pdf所在的子文件夹名字，去掉名字中的开头数字，也是paper的tag来源。
        - tag里面的关键词，不可以出现重复。
    - 去重。
      - paper_finder_results.jsonl有大量重复，有些paper用一个query查询到，用另一个query也同样查询到了，导致重复出现在paper_finder_results.jsonl。
      - 我们要把相同的paper合并，但保留它的特殊的tag。每一个tag名字，只保留一次，比如tool， science在tag中分别只能出现一次。
    - title合并
      - 如果有semantic scholar检索到的standard name就命名为standard name。
      - 如果没有standard name匹配成功，就保留原有的title
    - venue修正，并与raw_paper_id_conference.jsonl当中的conference合并，统一命名为conference
      - paper_finder_results的venue太复杂，需要修正格式
        - 目前的格式为：venue+year+type， 比如ACL 2024 Findings
        - 我们要把它劈开成三个：
          - venue保留,venue与raw_paper_id_conference.jsonl当中的conference合并，为conference
          - 把year放在year属性当中
          - 把type放在type属性当中
    - affinity_score：
      - 如果paper在paper_finder_results.jsonl当中有重复，就会出现多个afinity_score，只保留最大值
    - 把原pdf文件（存储在ssd当中的那个pdf文件本身的文件名）重命名：
      - 目前这些paper的命名比较复杂，基本格式上是:
        ```
        "conference" "year" ‘type’ ‘citation_cout’ ‘title’.pdf
        ```
        但是属性值都是旧的。
      - 把原pdf，用最新的属性值，文件命名为：
        ```
        "year"_"conference"_"citation_count"_"type"_"title".pdf
        ```
      - 命名中的"title"主体是title属性
      - 但是替换title当中的冒号和空格统一为“_”，因为pdf文件名中最好不能出现冒号或者空格
      - 如果某个属性是null，pdf命名就跳过，而不要写成NA
        - 比如2025_ICLR_10_NA_I-Con_A_Unifying_Framework_for_Representation_Learning应该命名为2025_ICLR_10_I-Con_A_Unifying_Framework_for_Representation_Learning
    - pdf_path：
      - 不管是paper_finder_results.jsonl还是raw_paper_id_conference.jsonl，它们的pdf path都是旧的了
      - 我们要更新为目前最新的pdf path
      - 最新的pdf path最后文件名也是用最新的title文件名
    - 其它都继续保留
    - 合并后的每一行的paper的属性（维度）就是：
      - title：
      - conference
      - year
      - type
      - authors
      - paper_id(from semantic scholar results raw_paper_id_conference.jsonl)
      - citation_count(from raw_paper_id_conference.jsonl)
      - arxiv_id(from raw_paper_id_conference.jsonl)
      - tag
      - affinity_score
      - link
      - abstract
      - bibtex
      - pdf_path
    - 而且这个jsonl的每一行，对可以随着pdf_path对应到具体的pdf文件
    - 任何时候，都不能破坏上述任务描述当中引用到的任何原文件。一定在原文件基础上，重新构建新的程序、文件。

- task20 paperbase frontend
  - 做一个网页工具，可视化我的paper_base.jsonl结果，并且加入一些互动功能
  - 这个网页工具应该根paper_base.jsonl在相同的位置，目前paper_base.jsonl在/Users/lingzhi/Code/AI/tools/paper-manager/paper_base.jsonl。但是未来我可能会搬到别的地方。但是我会把网页工具和它永远放在一起，所以网页工具应该直接尝试打开自己本地的一个paper_base.jsonl文件。
  - 网页的顶端是一个搜索框，我会输入关键词
    - 关键词匹配机制：整个jsonl那一行的带有文字的部分，比如title，tag等等
  - 搜索框下方是各种属性的选项，year（选项），会议（选项），citation_count最小值（数值，必须大于等于这个值），tag（选项， 根据最新的jsonl而更新）
    - year，conference，tag都要可以多选
  - 下方是paper的一个table表格
    - 同一行 清晰分格，独立每一格显示title、conference。。。。
  - 在paper前方加上一个选择框，default为空
    - 当我点击选择，就打上对勾
  - 在“XXX papers shown. X selected.”这个地方的左侧，添加一个select all的选框，点击打对勾代表当前全选，再点击对勾消失，同时取消全选。这样可以加快我的export selected
  - 网页工具右上角有一个export按钮
    - 当我点击export，要产生一个export_[selected conditions].jsonl是所有选中的paper的paper_base.jsonl里面有的信息的集合。我之后会引用这部分paper。
    - 比如当我选择2024， 2025，EMNLP， NIPS，那么就命名为export_2024_2025_EMNLP_NIPS.jsonl
    - 忽略tag condition，因为会造成文件名称过于冗长。
    - export应当到与paper_base.jsonl相同的位置
  - read status
    - 是否阅读过， 默认是没有读过no
    - 通过paper explorer修改，读过打勾，jsonl把read_status改为yes
  - 当我点击某个文献的这一行，要下拉框，展示这个文献的：
    - authors
    - tag
    - abstract
    - bibtex引用（是一个灰色的latex格式显示的小单元）
      - bibtex部分要易于复制，比如点击bibtex引用（整个小单元），就直接复制为latex格式
  - 当我再次点击这个文献行，下拉框要收回。或者点击另一个文献行，另一个文献行要展开，而此时前一个文献行要收回。也就是说时刻页面上只有一个文献行是展开状态。
  - 当我们terminal退出网页工具的时候，要自动释放端口

- task21 用paper reader v2阅读筛选出的论文
  - 已经筛选出：[tag: tool related; citation_count: minimum 20]，存储在了/Users/lingzhi/Code/AI/tools/paper_base/export_cit_20_tools.jsonl
    - 每一行是一个paper entry，每一个json接近末尾都有它的pdf path可以找到pdf本体
  - 需要做一个paper reader v2
    - paper reader v1：/Users/lingzhi/Code/AI/tools/paper_reader/paper_reader_v1.py
    - 这是v1的处理结果：/Users/lingzhi/Code/AI/plan+notes/notes/papers_arxiv_agentic_components.md
    - 类似的要利用gpt-5 api的pdf阅读功能
    - 程序放在/Users/lingzhi/Code/AI/tools/paper_reader/paper_reader_v2.py
  - 类似于paper reader v1，循环处理每一篇pdf
    - 每一次的输出结果存储在程序本地，名称为paper_reader_v2_results.md
    - 新的一篇在处理过的之后
    - 每处理结束一篇，获得输出，就立刻写入md文件，实时更新md结果文件。
  - 但是我们这次的任务是v2之：GitHub Code源代码和网站介绍综合收集
    - 我们收集github的目的，是在阅读理解论文的idea之后，要快速尝试、实践论文的idea，以及利用已经开源的代码，辅助构造我们自己的项目代码
    - 所以与v1不同，不需要再总结出所谓的tree agentic component
    - 但是生成的Github收集结果至少要跟v1一样，以及更好更详细完整的开源代码收集。
  - 所以我们需要为paper reader v2构造一个新的prompt
    - 不是盲目收集链接，而要阅读文章，理解文章的整体内容，基于理解帮助我收集到我需要的信息，能够总结出一个合理的GitHub列表，并描述每个GitHub项目的名称、项目内容、文章是怎么引用这个项目的
    - 专注于收集论文本身的开源代码， official website；以及任何提到的github， official website
      - 不管是论文本体的项目，代码是否有开源的github地址
      - 亦或是论文叙述过程中引用/提及的项目，是否可以很快找到开源的github地址、website介绍
    - 输出结果格式：
      - md文件格式
      - 每一篇论文是一个单元
        - 论文title - conference - year - citation_count - pdf path（这些信息应该是从jsonl直接获取到的，当开始处理这个paper的时候就立刻写入md文件，帮助我们定位文章在哪里）
        - Github & websites（这个部分是一个列表，下面是一个例子）
          - project name · [GitHub](link-to-GitHub) · [Website](link-to-website) · [Doc](link-to-doc)
            - description：（描述每个GitHub项目的名称、项目内容、文章是怎么引用这个项目的）
          - CLIPort · [GitHub](https://github.com/cliport/cliport) · [Website](https://cliport.github.io)
            - description：（描述每个GitHub项目的名称、项目内容、文章是怎么引用这个项目的）
          - ALFWorld · [GitHub](https://github.com/alfworld/alfworld)
            - description：（描述每个GitHub项目的名称、项目内容、文章是怎么引用这个项目的）
          - Sentence Transformers: Embeddings, Retrieval, and Reranking · [GitHub](https://github.com/UKPLab/sentence-transformers) · [Website](https://www.sbert.net)
            - description：（描述每个GitHub项目的名称、项目内容、文章是怎么引用这个项目的）
          - Malmö # · [GitHub](https://github.com/microsoft/malmo)
            - description：（描述每个GitHub项目的名称、项目内容、文章是怎么引用这个项目的）
  - 因为每一篇论文都发表于AI顶级会议，是pdf格式，gpt-5可能处理时间在3分钟-5分钟左右。所以不要过早停止pdf处理过程。
  - 简单测试一两个pdf，确保你的程序可以运行，然后交给我运行更多的pdf。


## 2025-11-16 gemini codewiki; github copilot
- task22 确认paper codebase url哪些有对应的Gemini Codewiki的代码库分析网页
  - 简要阅读/Users/lingzhi/Code/AI/plan+notes/catalog/raw_paper_codebase.md
    - 这是我从大量paper中提取到的被引用的code 的code base的url集合
    - 对于其中的github代码的url， 我们可以使用https://codewiki.google/，搜索是否有对应的code wiki
  - 我建议你按照下面流程节省你的token：
    - 写一个github url模式匹配，确保匹配到的都是github codebase。把所有的这类github codebase url都提取到/Users/lingzhi/Code/AI/plan+notes/catalog/raw_paper_codebase_url.yaml
    - 然后你再一个一个去code wiki查找，并把查找结果直接放在yaml文件当中跟code github url在一起
      - 我发现只要在github网址前面加上codewiki.google/刚好就是它的wiki地址
      - 也就是说我们只需要检查加上codewiki.google/的网页是否存在，是否404就行了，如果存在，就把这个加了codewiki.google/的地址作为它的codewiki地址
      - 你可以使用playwright来获得render后的网页，从而确定是否404
      - 你要一个个处理yaml，每次处理成功一个，就立刻实时修改yaml，而不要等到全部处理完再统一修改yaml，而要每次匹配成功一个codewiki，就立刻丰富对应的yaml字段
    - 你的所有的程序、中间结果，都要保存在/Users/lingzhi/Code/AI/tools/codebase_analyzer, 而不要修改原文件
    - 获取到的所有github url，以及codewiki都放在：/Users/lingzhi/Code/AI/plan+notes/catalog/raw_paper_codebase_url.yaml
  - 然后我们要利用获取到的code wiki:/Users/lingzhi/Code/AI/plan+notes/catalog/raw_paper_codebase_url.yaml, 把codewiki信息，结合到/Users/lingzhi/Code/AI/plan+notes/catalog/raw_paper_codebase_wiki.md当中
    - md文件要新建一个，叫做raw_paper_codebase_wiki.md，而不要直接修改原文件
    - 如果某个github project/url在yaml中的字段，有对应的codewiki，就在md文件的project后面加上 [CodeWiki](link-to-gemini-codewiki)
      - 例如：openai-python 在md文件中为：
        ```md
        - OpenAI API (GPT-4 / GPT-3.5) · [GitHub](https://github.com/openai/openai-python) · [Doc](https://platform.openai.com/docs/api-reference)
        ```
      - 可以发现openai-python在yaml中有codewiki
        ```yaml
        - name: OpenAI Python API (Chat Completions)
          github: https://github.com/openai/openai-python
          codewiki: https://codewiki.google/github.com/openai/openai-python
        ```
      - 从而在md文件中的openai-python改为：
        ```md
        - OpenAI API (GPT-4 / GPT-3.5) · [GitHub](https://github.com/openai/openai-python) · [Codewiki](https://codewiki.google/github.com/openai/openai-python) · [Doc](https://platform.openai.com/docs/api-reference)
        ```
      - 我认为你需要写一个py程序去解决这个问题

- task copilot 整理 paper codebase url
  - 用copilot分类：raw_paper_codebase_url_categorized.md
  - 用github copilot对code进行问答
  - 测试用copilot替换掉claude code
  - 筛选出有用的github模块
    - tooluniverse

## 2025-11-17 test programs

- task23 进一步归类categorized_project_codebase.md
  - 文件地址：/Users/lingzhi/Code/AI/plan+notes/catalog/categorized_project_codebase.md
  - 任务目标：从 ## Libraries & Frameworks (37) 开始一直到文件结尾的内容需要进一步归类整理
  - 归类原则：参考 ## Libraries & Frameworks (37) 之前已有的类别（如Database, Deep Learning, Infrastructure, Fine-tuning, Agent SDK, Planning, RAG, Simulator, Agent - Coding, Agent - Math, Bio & Protein等），将可以归入这些类别的项目移动到对应的前面的类别中

  - 执行步骤：
    1. **识别重复项目**：检查整个文件，找出在多个位置出现的相同项目（通过GitHub URL判断）
    2. **合并重复项目信息**：
       - 如果同一项目在前面的具体类别和后面的Libraries & Frameworks都出现，需要合并信息
       - 保留前面具体类别中的条目，并将后面条目中缺失的信息（如codewiki链接、website链接等）补充到前面
       - 例如：LangChain在Agent SDK部分没有codewiki，但在Libraries & Frameworks有，应该在Agent SDK部分添加codewiki链接
       - 合并完成后，删除后面的重复条目
       - **重要**：绝对不能删除任何信息，只能删除重复的条目本身
    3. **移动可归类项目**：
       - Database相关（向量数据库、键值数据库等）→ Database类别
         - 例如：Chroma (Vector Database), Milvus, FAISS
       - 生物蛋白质相关 → Bio & Protein类别
         - 例如：Chroma (Protein Design), IgFold
       - Benchmark/评估工具 → LLM Evaluation & Benchmarks类别
         - 例如：FLASK
       - 其他可以明确归类的项目也按此原则移动
    4. **处理特殊情况**：
       - 对于同一项目有多个版本/来源（如AndroidWorld有Google DeepMind版和Microsoft版），保留所有版本，在项目名称后加括号注明来源
       - 对于URL略有差异但指向同一项目的（如android-world vs android_world），去重时保留一个，在名称中注明组织
    5. **保留的项目**：
       - 确实是通用库和框架的项目（如Flask, JAX, PyTorch, NumPy, pandas, Playwright等）保留在Libraries & Frameworks
       - 这些项目不属于前面任何具体类别，是基础设施级别的工具
  - 验证要求：
    - 所有codewiki链接、website链接、docs链接都必须保留
    - 不能丢失任何项目条目，只能移动和合并
    - 每个项目最终只在一个最合适的类别中出现一次

- task24 项目列表：详细归类，描述
  - 背景：目前/Users/lingzhi/Code/AI/plan+notes/catalog/categorized_project_codebase.md当中的github项目链接都是从/Users/lingzhi/Code/AI/plan+notes/catalog/raw_project_codebase.md当中找到的
    - raw_project_codebase.md是用paper reader v2（/Users/lingzhi/Code/AI/tools/paper_reader/paper_reader_v2.py）阅读大量pdf得到的列表
    - 这些categorized_project_codebase.md的github链接有的是有codewiki链接的，这些codewiki链接是验证过有效的链接，需要保留下来
    - 目前这些项目没有复制原raw_project_codebase.md数据集中的项目信息描述，导致我们无法进行详细的分类
  - 请你针对每一个categorized_project_codebase.md项目，找到它的项目描述，复制粘贴到categorized_project_codebase.md当中这个entry下方
    - 比如categorized_project_codebase.md当中有：
      ```md
      - [DeepSeekMath](https://github.com/deepseek-ai/DeepSeek-Math)
      ```
      而这个项目在raw中有一段描述：
      ```md
      - DeepSeek-Math · [GitHub](https://github.com/deepseek-ai/DeepSeek-Math)
        - description: Math-pretrained LLM baseline; authors fine-tune a 7B version to create SCIAGENT-DEEPMATH.
      ```
      那么你需要用原文替换掉categorized_project_codebase.md当中的deepseek-math entry的内容
        - 删除：
          ```md
          - [DeepSeekMath](https://github.com/deepseek-ai/DeepSeek-Math)
          ```
        - 改为：
          ```md
          - DeepSeek-Math · [GitHub](https://github.com/deepseek-ai/DeepSeek-Math)
            - description: Math-pretrained LLM baseline; authors fine-tune a 7B version to create SCIAGENT-DEEPMATH.
          ```
  - 有的时候可能一个项目在不同位置有两个不同的description，那么你就统一名称、并添加两个description在这个entry下方，分别显示
  - 你要保证categorized中的每个项目，都填补它在raw当中对应的description，丰富整个categorized md文件


## 2025-11-18 meeting content

- meeting content
  - paper finder
  - paper explorer
  - openai-cookbook
  - github projects categorized
    - how to understand code
      - codewiki
      - github copilot
      - testing
 

# TO DO

## 2025-11-20   

- task 有些yaml链接不在md当中，完善md确保完整

- task propose一个系统的基本框架
  - focus

- task Test Programs

## paper reader v3

- task pipeline化；整理 paperbase
  - 给paper加上tag
    - 预定一些tag keywords
    - 用claude批量加tag
    - 用paper explorer改tag
  - paper explorer
    - 后端功能
      - 获取论文的github codebase catalog
      - notebook
    - 前端功能展示
  - 多种形式筛选论文
    - 按照category tag
    - 按照conference
    - 按照引用
    - 组合：每一个类别当中的高被引的文章
    - 按照机构
    - 索引分析 获得高被引文章的引用文献当中的高被引

- task 读论文进一步自动化
  - 做paper reader v2
  - 做paper reader v3
  - 用paper reader v3 生成结构化输出
  - refernce： https://www.alphaxiv.org/overview/2510.26692

- task 人工读论文写notes

- 写agentic截止11月综述

## Code Analyzer v1

- task 建立paper - codebase - 笔记系统
  - 对于每一篇paper
    - 它有paper base当中的属性信息
    - 可能会有：用paper reader v1获得的agentic system component信息
    - 可能会有：用paper reader v2筛选到的codebase信息
      - 它的codebase，还进一步可能会有：Gemini Codewiki对于github repo分析得到的信息
  - 我们要建立一个笔记系统
    - 单项可能是paper本身
      - 为了辅助我们阅读paper生成笔记，我们需要几个功能
        - 给LLM instruction，LLM follow instruction生成回答
        - 把回答一键导出到note.md当中
    - 单项可能是code repo本身
      - 为了辅助我们理解code，我们需要几个功能
        - code repo table：title、url、website、codewiki、star、tag

- task 把codebase md文件转换成codebase数据库
  - 完善md
    - 每一个github的引用文献名称
    - 用工具获得github的star_count
  - 把md文件转换成jsonl
    - 把分类作为tag
  - jsonl网页可视化

- task code reader v1
  - repo reader
  - code reader

- task note paper project codebase

## Deep Research Agent

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

## Presentation

- task github site
  - 个人网站链接: lingzhiyang.com

- task landingsite ai project site
  - 项目链接： electric-steam.com

- 跟各种人发邮件合作写paper，贡献我的部分

## AI Stock managing Agent

- Role
  - 专注于AI上上下下供应链的分析、着重于AI application layer的公司的新产品、用户增长、营收、潜力
- Routine
  - 公司定期信息披露
  - 新闻跟踪、归类
  - 价格统计、趋势分析
- tools
  - api获取实时stock价格