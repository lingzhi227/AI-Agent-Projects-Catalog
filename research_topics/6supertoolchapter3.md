# 第 3 章：Worker 角色、资源规划与任务调度设计

（草稿 v0.1）

本章将系统性讨论：

* 两台 Linux Worker 的角色定位
* 资源规划（CPU / 内存 / 磁盘 / 并发数）
* Worker 之间的任务分配策略
* Worker 侧运行环境（用户、权限、目录结构）
* Worker Agent 的运行模式（守护进程 / SSH on-demand）
* Worker 故障恢复与任务重试机制

后续章节将逐步按小节填写。

---

# 3.1 Worker 角色与职责分工（核心设计）

两台 Linux Worker 是整个系统的“实际算力执行者”，LLM 的所有工程任务最终都由 Worker 完成。本节定义 Worker 的职能边界，让架构在扩展、调度、可维护性方面具备高度一致性。

本平台采用 **异构分配 + 对称能力保底** 的架构：

* Worker1：FPGA-heavy / 仿真计算节点
* Worker2：MCU-heavy / 固件编译节点
* 两者都具有 MATLAB / ModelSim 基础环境，让 LLM 可以更灵活调度

## 3.1.1 Worker 统一的职责

无论 Worker1 或 Worker2，都必须具备以下核心职能：

### ● 1）执行工程类软件任务（核心职责）

Worker 将从 `/workspace/jobs/<job-id>/request.json` 中读取任务描述，并执行：

* MATLAB 仿真 / 脚本
* FPGA 综合 / TCL 执行
* MCU 编译 / 链接
* 任意 CLI 构建工具（Python、GCC、CMake 等）

执行流程严格标准化：

1. 解析任务 JSON
2. 根据 tool 字段选择对应执行路径
3. 执行命令并记录日志
4. 生成 result.json
5. 将产物写入统一 workspace

### ● 2）保持完全 headless 的执行环境

所有工程软件必须以 CLI 运行：

* MATLAB：`matlab -batch`
* Vivado：`vivado -mode batch`
* Quartus：`quartus_sh -t`
* ModelSim：`vsim -c`
* GCC/CMake：CLI 本身

### ● 3）保证路径与权限标准一致

Worker 必须在：

```
/workspace/ 
  ├── projects/
  └── jobs/
```

下执行所有任务。

用户统一使用：

```
builduser:buildgroup
```

并保证：

```
/workspace 由 builduser 可写
/opt 下软件目录由 root 安装但 builduser 可读可执行
```

### ● 4）Worker 不负责调度，不负责决策

Worker 是**纯执行器**：

* 不选择任务
* 不规划顺序
* 不协调资源
* 不与 LLM 直接通信

所有调度智能都在 **Mac 中控**。

### ● 5）提供可重启、无状态执行模型

除执行任务外，Worker 不保存上下文。

* 每个任务独立
* 必要状态由 session_id 与 workspace 文件持久化

### ● 6）提供健康状态反馈（可选）

可通过以下方式让中控检测 Worker 健康：

* SSH + `uptime`
* SSH + `test -f /opt/.../vivado`
* 心跳脚本

---

# 3.1.2 Worker1（FPGA-heavy 节点）的职责

### ● 主要负责：

* Vivado 综合 + 实现 + bitstream
* COMSOL（若有安装）大规模仿真
* ModelSim/Questa 仿真
* MATLAB 任务（中等负载）

### ● 软件安装配置：

* `/opt/Xilinx/Vivado/2024.1`
* `/opt/comsol/`（可选）
* `/opt/modelsim/`（或 Questa）
* `/opt/mathworks/matlab/R2024a`

### ● 资源特点：

* 需要高 CPU 并发（≥ 16 核最佳）
* 大量内存（≥ 32GB 或 64GB）
* FPFA 综合任务时间长（分钟到小时）

### ● LLM 对 Worker1 的调度策略：

* 优先用于 Vivado/COMSOL
* MATLAB 与仿真任务作为辅助
* 避免同时调度两个大型 FPGA 工程（容易压满 CPU）

---

# 3.1.3 Worker2（MCU-heavy 节点）的职责

### ● 主要负责：

* MCU 工具链编译（arm-none-eabi-gcc）
* STM32CubeMX 代码生成
* Quartus FPGA 工具（如果需要 Intel FPGA）
* ModelSim/Questa
* MATLAB（轻载任务）

### ● 软件安装配置：

* `/opt/intelFPGA/22.1std/quartus`
* `/opt/stm32cubemx/`
* `/opt/arm-gcc-toolchain/`
* `/opt/modelsim/`
* `/opt/mathworks/matlab/R2024a`

### ● 资源特点：

* MCU 工具链本身轻量
* Quartus 编译中等偏高
* 适合并发多个较小任务

### ● LLM 对 Worker2 的调度策略：

* 优先用于 MCU 相关任务（GCC、CubeMX）
* Quartus 编译需与 MATLAB 任务调节负载
* 可作为 MATLAB 任务 overflow 节点

---

# 3.2 Worker 资源规划（CPU / 内存 / 磁盘 / 并发）

本节将明确 Worker 的硬件需求、任务并发上限、workspace 空间规划等。

## 3.2.1 CPU 规划

不同软件的 CPU 消耗极不相同：

| 软件        | CPU 占用 | 说明                |
| --------- | ------ | ----------------- |
| Vivado    | 极高     | 可使用 16 核以上，并长时间满载 |
| COMSOL    | 极高     | 多核并行求解            |
| MATLAB    | 中-高    | 并行仿真时使用多核         |
| Quartus   | 高      | 8~16 核有明显加速       |
| ModelSim  | 中      | 波形仿真可占多核          |
| GCC/CMake | 低-中    | `-jN` 取决于工程大小     |

**推荐最小配置：**

* Worker1（FPGA-heavy）：16 核（理想 24 核）
* Worker2（MCU-heavy）：12–16 核

## 3.2.2 内存规划

| 工具              | 推荐内存              |
| --------------- | ----------------- |
| Vivado          | ≥ 32GB（大型工程 64GB） |
| Quartus         | ≥ 24GB            |
| MATLAB 大型仿真     | ≥ 32GB            |
| ModelSim/Questa | ≥ 16GB            |

**Worker 推荐内存：**

* Worker1：64GB
* Worker2：32GB

## 3.2.3 磁盘规划

* `/opt` 下安装软件不少于 100GB
* `/workspace` 最少 200GB（FPGA 工程产物很大）
* SSD 强烈推荐（改善 Vivado/Quartus IO 性能）

目录示例：

```
/opt/                 ← 工具安装
/workspace/projects   ← Git repo / 工程
/workspace/jobs       ← Job 执行数据
```

## 3.2.4 任务并发上限

### Worker1：

* Vivado 编译：**最多 1 个**（除非 CPU ≥ 32 核）
* MATLAB 仿真：2–3 个
* ModelSim：2 个

### Worker2：

* MCU 编译：4–6 个
* MATLAB：2 个
* Quartus：1 个（≥16 核时可 2 个）

这些并发规则将在 3.3 调度算法中定义。

---

# 3.3 Worker 调度策略与算法（由 Mac 中控执行）

Mac 中控节点是唯一的调度中心。LLM 发出的每个 MCP tool 调用最终会转换成一个 Job，请求中控为其选择最合适的 Worker。

核心要求：

* 避免将大型任务压在同一台 Worker 上
* 让小任务尽量并发
* 保证大型任务优先级
* 避免 license 冲突（MATLAB / Vivado）

本节将设计完整调度流程。

---

## 3.3.1 调度器输入与输出

输入（由 MCP tool 产生的 job JSON）：

```
{
  "tool": "vivado_synth",
  "project": "projA",
  "params": {...}
}
```

输出：

* `selected_worker`: worker1 / worker2
* `dispatch_method`: ssh / local
* `expected_load`: 调度器根据 CPU/内存估算的负载

---

## 3.3.2 基础调度算法（规则优先级排序）

最小可行调度器基于以下顺序决策：

### **规则 1：根据工具类型选择默认 Worker**

* Vivado → Worker1（FPGA-heavy）
* Quartus → Worker2（MCU-heavy）
* MCU 编译（arm-none-eabi-gcc）→ Worker2
* MATLAB 仿真 → Worker1（优先），Worker2（fallback）
* ModelSim/Questa → Worker1（优先），Worker2（fallback）

这确保同类任务命中最适合的节点。

### **规则 2：防止大型任务并发冲突**

大片任务如 FPGA 综合（Vivado/Quartus）会强占 CPU，因此：

```
if worker.running_vivado_jobs >= 1:
    不能再在 worker 上调度 Vivado
```

Quartus 同理：

```
if worker.running_quartus_jobs >= 1:
    worker 不接收新的 quartus_synth
```

### **规则 3：根据 CPU/内存即时负载选择“最轻的可用 Worker”**

在满足规则 1 & 2 的前提下：

```
select worker with min( cpu_load_weighted )
where cpu_load_weighted = cpu_usage + job_cost_factor(tool_type)
```

例如：

* MATLAB job_cost_factor = 2
* Vivado job_cost_factor = 10
* MCU compile job_cost_factor = 1

### **规则 4：避免 license 冲突**

若：

```
MATLAB license 已占满 → 不调度新的 MATLAB job
Vivado license 已占满 → 延迟 FPGA 任务
```

License 状态由中控维护：

```
license_status{"matlab": free_slots, "vivado": free_slots}
```

### **规则 5：遵从会话偏好（session-aware scheduling）**

如果 LLM 在当前 session 中多次调用 MATLAB，则应保持在同一 Worker：

```
if session.preferred_worker != None:
    return preferred_worker
```

避免 MATLAB 多次初始化开销。

---

## 3.3.3 动态调度（基于 Worker 实时状态）

调度器周期性（每 3 秒）收集 Worker 指标：

* `cpu_usage`
* `mem_usage`
* `running_jobs[]`
* `queue_size`
* `healthy`（通过 ssh 心跳）

生成 Worker 状态表：

```
{
  "worker1": {"cpu": 78%, "mem": 61%, "vivado_jobs": 1, "matlab_jobs": 2},
  "worker2": {"cpu": 42%, "mem": 37%, "quartus_jobs": 0, "mcu_jobs": 3}
}
```

动态调度逻辑：

```
if target_worker overloaded:
    route job to other worker
else:
    use preferred worker from base rules
```

Overloaded 定义：

```
(cpu > 85% and running_jobs > threshold) or (mem > 90%)
```

---

## 3.3.4 “大任务 vs 小任务”公平性策略（防止饥饿）

如果连续到达 FPGA 综合任务，可能让 MCU 编译任务“饿死”。

因此加入 fairness queue：

```
small_job_queue
large_job_queue
```

并按比例选择：

```
每调度 1 个大任务，必须调度 2 个小任务
```

权重举例：

* 大任务：Vivado、Quartus
* 小任务：GCC 编译、MATLAB、ModelSim

---

## 3.3.5 任务亲和性（Task Affinity）

某些 pipeline 要求多个步骤在同一 Worker：

* MATLAB 自动生成 HDL → Vivado 综合
* MATLAB 生成 C → MCU 编译

LLM 会在 `job.request.json` 中加入：

```
"affinity": "worker1"
```

调度器必须严格遵循 affinity。

Affinity 也可以基于文件位置自动计算：

```
如果上一个 job 的输出位于 worker1/local_cache → 再次选择 worker1
```

---

## 3.3.6 调度算法伪代码（最终版）

```
function dispatch(job):
    tool = job.tool

    # 1. 规则1：根据工具类型选择默认 worker
    candidate_workers = default_workers(tool)

    # 2. 规则2：检查任务类型限制（FPGA/Quartus 并发）
    candidate_workers = filter_by_task_limits(candidate_workers, tool)

    # 3. 规则3：负载过滤
    candidate_workers = filter(lambda w: w.cpu < 90% and w.mem < 90%, candidate_workers)

    # 4. 规则4：license 检查
    candidate_workers = filter(check_license_available(tool), candidate_workers)

    # 5. 规则5：session 亲和
    if job.session.preferred_worker in candidate_workers:
        return job.session.preferred_worker

    # 6. 若多个候选 worker：选择 CPU 最低的
    return argmin(candidate_workers, key=lambda w: w.cpu_usage + w.job_cost(tool))
```

调度输出：

```
{
  "selected_worker": "worker1",
  "reason": "vivado task; worker1 idle"
}
```

---

## 3.3.7 调度器的软失败与回退策略

当所有 Worker 满载或不健康时：

### ● 回退 1：延迟执行（排队）

```
job.status = "delayed"
job.retry_in = 10 seconds
```

调度器会在 10 秒后再次尝试。

### ● 回退 2：强制 fallback 至兼容节点

例如 MATLAB 可以在两个 Worker 任意执行：

```
if worker1 overloaded:
    route to worker2
```

### ● 回退 3：向 LLM 反馈可等待或可拆分

MCP 返回：

```
{"status": "queued", "message": "All workers busy; retry in 20s"}
```

LLM 通常会选择等待或重规划任务。

---

接下来将继续撰写：

---

# 3.4 Worker 运行环境（用户、权限、目录结构）

本节将详细描述 Worker 节点操作系统层面的标准化要求，使整个系统具备：

* **可复制性**（相同配置的 Worker 可横向扩容）
* **可维护性**（权限/目录统一，减少调试成本）
* **安全性**（任务隔离、权限隔离）
* **一致性**（工具路径、用户、环境变量在所有 Worker 上一致）

Worker 的运行环境包括以下五部分：

* 统一用户体系（builduser）
* 标准化目录结构
* 权限与组策略
* 环境变量与 profile 配置
* SSH 配置与安全策略

下面逐项展开。

---

## 3.4.1 Worker 用户体系设计（builduser）

为了保证执行任务时的权限一致性，两台 Worker 必须形成统一的用户体系：

### ✔️ 定义一个专门执行工程任务的用户：

```
builduser:buildgroup
UID: 2001
GID: 2001
Home: /home/builduser
```

不使用 root 来执行任务，原因：

* 避免工程脚本误改系统文件
* 防止工程工具链写入 `/opt` 等系统目录
* 限制安全风险

### 创建方式：

```
sudo groupadd -g 2001 buildgroup
sudo useradd -m -u 2001 -g 2001 -s /bin/bash builduser
```

### builduser 的作用：

* 所有任务执行均以 builduser 身份进行
* MCP 调度器通过 SSH 使用该用户运行 worker_agent
* `/workspace` 的所有文件均归属 builduser
* 工程工具（Vivado、Quartus、MATLAB）由 root 安装，但 builduser 可以读写工程区

---

## 3.4.2 Worker 标准化目录结构

所有 Worker **必须完全一致**：

```
/opt/                         ← 软件安装目录（root 管理）
  ├── Xilinx/
  ├── intelFPGA/
  ├── mathworks/
  ├── modelsim/
  ├── arm-gcc-toolchain/
  └── stm32cubemx/

/workspace/                   ← 全局工程与任务空间（NFS 共享）
  ├── projects/
  │     ├── projA/
  │     ├── projB/
  │     └── ...
  └── jobs/
        ├── job-20250101-0001/
        │      ├── request.json
        │      ├── result.json
        │      ├── logs/
        │      └── outputs/
        └── job-20250101-0002/

/home/builduser/              ← Worker 本地缓存/临时文件
  ├── .cache/
  ├── .local/
  └── .ssh/
```

### 目录约束：

* `/workspace` 必须能被所有 Worker 所读写（NFS/SSHFS/Samba 容器均可）
* `/opt` 中工具仅 root 可写
* `/home/builduser` 用于 Worker 本地执行时的缓存

---

## 3.4.3 权限与组策略

```
/workspace                → builduser:buildgroup (770)
/opt                      → root:root (755)
/home/builduser           → builduser:buildgroup (700)
```

原因：

* builduser 可继续在 workspace 中生成中间文件
* root 安装工具，但普通用户可执行
* 禁止其他系统用户访问工程文件（安全）

### 授权示例：

```
sudo chown -R builduser:buildgroup /workspace
sudo chmod -R 770 /workspace
sudo chmod -R 755 /opt
```

---

## 3.4.4 环境变量统一（/etc/profile.d/*.sh）

所有 Worker 的工具环境必须统一，确保执行结果一致、调度无差异。

典型文件：

```
/etc/profile.d/matlab.sh
/etc/profile.d/vivado.sh
/etc/profile.d/quartus.sh
/etc/profile.d/arm-gcc.sh
```

示例（matlab.sh）：

```
export MATLAB_ROOT=/opt/mathworks/matlab/R2024a
export PATH="$MATLAB_ROOT/bin:$PATH"
```

四大原则：

1. **所有 Worker 一致性**（必须完全相同）
2. **PATH 统一前缀顺序**（避免不同版本工具被误调用）
3. **调度器无需关心 Worker 的环境差异**
4. **便于新 Worker 加入**（只需拷贝 profile.d）

---

## 3.4.5 SSH 配置与安全策略（中控 → Worker）

Mac 中控需要无密码 SSH 登录 Worker 执行任务。

### 1）生成 SSH key（在 Mac 上）：

```
ssh-keygen -t ed25519 -C "mcp-controller"
```

生成：

```
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

### 2）将公钥复制到 Worker：

```
ssh-copy-id builduser@worker1
ssh-copy-id builduser@worker2
```

### 3）ssh 配置：

在 MacBook：

```
Host worker1
    HostName 192.168.1.101
    User builduser
    IdentityFile ~/.ssh/id_ed25519

Host worker2
    HostName 192.168.1.102
    User builduser
    IdentityFile ~/.ssh/id_ed25519
```

### 4）必要的 SSH 限制：

Worker 上 `/etc/ssh/sshd_config` 可配置：

```
AllowUsers builduser
PasswordAuthentication no
PermitRootLogin no
```

这样：

* 只有 builduser 可被中控调度
* 禁止密码登录
* 禁止 root 登录

提高整体安全性。

---

## 3.4.6 Worker 本地执行缓存与 Temp 目录

某些工具（MATLAB / Vivado / Quartus）默认会在用户目录下写大量文件：

```
/home/builduser/.matlab/
/home/builduser/.Xilinx/
/home/builduser/.config/
```

建议清理策略：

* 每周自动清理 `.matlab` 中旧日志
* 每次 Vivado 任务结束后清理 `.Xilinx` 中缓存

或在 profile 中重定向：

```
export XILINX_LOCAL_USERDATA=/workspace/.cache/builduser/xilinx
```

---

## 3.4.7 Worker 运行模式：守护进程 vs on-demand 调用

Worker Agent 有两种运行模式：

### ● 模式 A：SSH On-demand（最简单）

Mac 中控执行：

```
ssh worker1 "python3 /opt/worker_agent/run_job.py /workspace/jobs/<job-id>/request.json"
```

特点：

* ***无需常驻进程***，实现最简单
* 自动继承 SSH 环境变量
* 适合小规模系统

### ● 模式 B：常驻守护进程（进阶方案）

Worker 本地运行一个常驻的 `worker_agent`：

```bash
sudo systemctl enable worker-agent.service
sudo systemctl start  worker-agent
```

`worker-agent.service` 示例：

```ini
[Unit]
Description=LLM Worker Agent
After=network.target

[Service]
User=builduser
Group=buildgroup
WorkingDirectory=/home/builduser
ExecStart=/usr/bin/python3 /opt/worker_agent/agent_daemon.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

中控调度方式变为：

* 不再直接 ssh 执行脚本
* 改为写入 `jobs/<job-id>/request.json` 后，守护进程自动监听并执行

守护进程模式的优点：

* 能更好地控制并发
* 可维护本地队列
* 更好地向中控上报心跳状态

小结：

* PoC 阶段推荐 **SSH on-demand 模式**（简单、易实现）
* 生产阶段可逐步升级为 **守护进程 + 队列** 模式

---

# 3.5 Worker Agent 设计（run_job.py / agent_daemon.py）

Worker Agent 是“连接中控调度逻辑与实际工程软件”的关键组件。它运行在 Worker 节点上，负责：

* 读取 `/workspace/jobs/<job-id>/request.json`
* 根据 `tool` 字段选择具体执行函数
* 调用 MATLAB / Vivado / Quartus / GCC 等 CLI 工具
* 收集日志与产物，生成 `result.json`
* 在守护进程模式下，维护任务队列与心跳

本节将从架构、数据流、错误处理与并发四个方面详细设计。

---

## 3.5.1 Agent 的总体架构

Agent 可以拆分为三个层次：

1. **入口层（Entry / Main）**

   * 解析命令行参数（on-demand 模式）或轮询 jobs 目录（daemon 模式）
   * 加载 request.json
   * 构造标准化的 `Job` 对象

2. **调度层（Dispatcher）**

   * 根据 `job.tool` 查找对应 handler
   * 负责统一的日志捕获、异常包装、执行时间统计

3. **工具执行层（Tool Handlers）**

   * 一组函数/类，每个负责一类工具：

     * `handle_matlab(job)`
     * `handle_vivado(job)`
     * `handle_quartus(job)`
     * `handle_mcu_build(job)`
     * ...

整体伪代码：

```python
def main(job_path):
    job = load_job(job_path)
    result = init_result(job)
    try:
        handler = TOOL_HANDLERS[job["tool"]]
    except KeyError:
        result["status"] = "failed"
        result["error"] = f"Unknown tool: {job['tool']}"
    else:
        result = handler(job, result)
    finally:
        save_result(job_path, result)
```

---

## 3.5.2 request.json 的标准结构

调度器在 Mac 中控生成的 `request.json` 建议统一结构：

```jsonc
{
  "job_id": "job-20251121-0001",
  "tool": "matlab_run_script",
  "project": "projA",
  "session_id": "sess-xyz",
  "affinity": "worker1",
  "created_at": "2025-11-21T10:00:00Z",
  "params": {
    "script_path": "projects/projA/matlab/controller.m",
    "entry_function": "run_controller",
    "inputs": {
      "Kp": 1.2,
      "Ki": 0.01
    }
  }
}
```

关键字段说明：

* `job_id`：任务唯一编号
* `tool`：决定使用哪个 handler
* `project`：工程名称，用于定位 `projects/<project>/`
* `session_id`：LLM 会话，用于偏好同一 Worker
* `affinity`：强制调度到特定 Worker（可选）
* `params`：具体工具相关参数，由 MCP tool schema 定义

---

## 3.5.3 result.json 的标准结构

Agent 执行结束后写入：

```jsonc
{
  "job_id": "job-20251121-0001",
  "status": "success",         // 或 "failed" / "running"
  "worker": "worker1",
  "tool": "matlab_run_script",
  "started_at": "2025-11-21T10:00:01Z",
  "finished_at": "2025-11-21T10:00:10Z",
  "outputs": {
    "log_file": "jobs/job-20251121-0001/logs/matlab.log",
    "generated_files": [
      "projects/projA/firmware/generated/controller.c"
    ],
    "metrics": {
      "rise_time": 0.12,
      "overshoot": 0.05
    }
  },
  "error": null
}
```

中控在收到 `result.json` 后：

* 向 LLM 返回结构化结果
* 写入 build manifest / pipeline 记录

---

## 3.5.4 Tool Handler 注册机制

为便于扩展，Agent 内部可以使用一个字典式注册表：

```python
TOOL_HANDLERS = {}

def register_tool(name):
    def deco(fn):
        TOOL_HANDLERS[name] = fn
        return fn
    return deco

@register_tool("matlab_run_script")
def handle_matlab(job, result):
    # 解析参数
    # 构造命令
    # 执行并填充 result
    return result

@register_tool("vivado_synth")
def handle_vivado(job, result):
    # 类似流程
    return result
```

好处：

* 新增工具时只需添加一个 handler 函数 + 装饰器
* 不需要修改主调度逻辑

---

## 3.5.5 日志捕获与错误处理

每个 handler 都应遵循统一日志策略：

1. 在 `jobs/<job-id>/logs/` 下创建专用日志文件：

   * `matlab.log`
   * `vivado.log`
   * `quartus.log`

2. 使用 `subprocess.run` 时将 stdout/stderr 重定向到该文件：

```python
with open(log_path, "w") as f:
    subprocess.run(cmd, cwd=project_dir, stdout=f, stderr=f, check=False)
```

3. 根据 `returncode` 决定任务状态：

```python
if proc.returncode == 0:
    result["status"] = "success"
else:
    result["status"] = "failed"
    result["error"] = f"Command failed with code {proc.returncode}"
```

4. 不在 Agent 内部解析详细报错，将解析交给 LLM：

* LLM 可读取 `log_file` 内容
* 理解具体错误信息并给出修复策略

---

## 3.5.6 并发与队列（守护进程模式下）

在 `agent_daemon.py` 中，可以设计一个简单任务队列：

1. 周期性扫描：

```python
while True:
    pending_jobs = scan_jobs_directory()
    for job in pending_jobs:
        if not is_over_capacity():
            start_worker_thread(job)
    sleep(2)
```

2. 使用线程或进程池执行任务：

```python
from concurrent.futures import ThreadPoolExecutor

EXECUTOR = ThreadPoolExecutor(max_workers=MAX_JOBS)

EXECUTOR.submit(run_single_job, job_path)
```

3. `is_over_capacity()` 根据当前运行的 Vivado/Quartus/MATLAB 任务数量决定是否再接新任务。

---

## 3.5.7 心跳与 Worker 状态上报

守护进程模式下，Agent 可以定期写入 Worker 状态：

```jsonc
{
  "worker": "worker1",
  "timestamp": "2025-11-21T10:00:00Z",
  "cpu_usage": 0.72,
  "mem_usage": 0.61,
  "running_jobs": [
    "job-20251121-0001",
    "job-20251121-0002"
  ]
}
```

状态文件存放在：

```
/workspace/worker_status/worker1.json
```

Mac 中控定期读取这些状态文件，以实现 3.3 中描述的动态调度算法。

---

## 3.5.8 小结

本节定义了 Worker Agent 的：

* 三层架构（入口 / 调度 / handler）
* request.json / result.json 标准
* Tool Handler 注册和扩展机制
* 日志捕获与错误处理
* 守护进程下的并发队列
* Worker 状态心跳上报

至此，Mac 中控 → Worker → 工具软件 的完整路径打通：

* LLM → MCP tool → 生成 Job → 调度器选 Worker
* Worker Agent 读取 Job → 调用工具 → 写回结果
* LLM 继续根据结果进行下一步规划

下一节（3.6）将讨论：

# 3.6 Worker 故障恢复与任务重试机制

在一个分布式、多 Worker 的工程自动化系统中，故障是必然会发生的情况，包括：

* Worker 节点掉线 / 无响应
* 工具链崩溃（例如 Vivado 内部异常）
* 网络闪断导致 SSH 中途中断
* 任务执行中机器重启
* result.json 未生成或部分生成
* 工具 License 获取失败

本节将定义完整的故障检测策略、恢复策略、自动重试规则、任务幂等性机制以及跨 Worker 回退策略。

本节目标：**让 LLM 与中控的工程任务具备高度鲁棒性，不因 Worker 故障导致整体停摆。**

---

## 3.6.1 故障类型分类

我们将 Worker 故障分为 3 大类：

### **Ⅰ类：轻度故障（可自动修复）**

例如：

* MATLAB 启动失败（偶发）
* Vivado 运行 TCL 时中途异常退出
* Quartus license 服务器延迟
* 超时等待文件锁

处理方式：自动重试。

---

### **Ⅱ类：中度故障（节点仍在线但不健康）**

例如：

* Worker CPU 100% 卡死
* I/O 过载导致任务长时间无输出
* 某工具（如 Vivado）进入死循环模式
* Agent 无法写入 result.json

处理方式：

* 标记 Worker 为 *degraded*
* 当前任务尝试重调度到另一个 Worker
* 对 Worker 进行轻量运维（清理缓存等）

---

### **Ⅲ类：重度故障（节点掉线或不可用）**

例如：

* Worker SSH 连接不上
* Worker 重启 / 死机
* 磁盘满
* `/workspace` 挂载断开

处理方式：

* 标记 Worker 为 *offline*
* 所有属于该 Worker 的任务移入重试队列
* 调度器避开该 Worker
* 等 Worker 恢复后再重新纳入系统

---

## 3.6.2 故障检测机制

中控与 Worker 的心跳系统可以检测 90% 的故障。

### **1）SSH 主动健康检查（默认）**

中控每隔 5 秒执行：

```
ssh worker1 "echo ok"
```

若 2 次失败 → 标记 offline

---

### **2）Worker 心跳文件检查（守护进程模式）**

Worker Agent 每 3 秒写入：

```
/workspace/worker_status/worker1.json
```

中控检测：

```
更新超过 10 秒 → unhealthy
```

---

### **3）任务超时机制**

request.json 可包含：

```
"timeout": 3600   # 秒
```

Agent 执行时强制 kill：

* 超出时间
* 无 stdout/stderr 超过 N 分钟

---

### **4）工具级健康检测**

例如 Vivado：

```
vivado -mode tcl -source check_version.tcl
```

返回非 0 → Vivado 不健康

中控在调度前可主动检查工具是否可用。

---

## 3.6.3 自动重试机制（幂等）

所有任务默认具备重试机制：

```
"retry": 3,
"retry_interval": 10
```

Agent 接收到失败时：

* 若 returncode != 0 且是轻度故障
* 则等待 retry_interval 秒后重新运行

任务重试必须**幂等**：

* 输入与 output 目录必须可安全重复运行
* 不依赖随机路径、不写 root 目录、不修改系统文件

例如 Vivado：

* 同一工程多次综合不会产生互相影响（只覆盖工程缓存）

MATLAB 仿真：

* 多次执行脚本无副作用

---

## 3.6.4 重调度（fallback 到另一 Worker）

若 Worker 发生中度故障：

* 不立即失败任务
* 中控将任务转移到另一 Worker

判断逻辑：

```
if worker1 unhealthy:
    fallback to worker2
```

特别适用于 MATLAB、MCU 编译等工具，因它们在两个 Worker 上都安装了。

### 对于 FPGA 工具：

Vivado（仅 Worker1）→ 无法 fallback，只能排队重试或等待 worker 恢复。
Quartus（仅 Worker2）→ 同理。

---

## 3.6.5 result.json 缺失的恢复策略

可能出现执行失败但未生成 result.json。

中控会生成 emergency_result.json：

```
{
  "status": "failed",
  "error": "Worker stopped responding before result.json was written",
  "worker": "worker1",
  "job_id": "..."
}
```

LLM 可根据 emergency result 推断下一步修复。

---

## 3.6.6 Worker 重启后的恢复策略

若 Worker 意外重启：

a) 检查 `/workspace/jobs/<job-id>/result.json` 是否存在

```
存在 → 视为成功
不存在 → 移入重试队列
```

b) 清理部分执行痕迹：

* 清除临时 lock 文件
* 强制停止所有 orphan 进程（Vivado 僵尸进程等）

c) 将 worker 标记为 healthy 后重新加入调度池。

---

## 3.6.7 LLM 层面的容错：自动恢复 pipeline

当任务失败时，LLM 会：

* 解析错误日志
* 分析问题来源（脚本错误、编译错误、资源不足、工具崩溃）
* 自动尝试重新生成改进后的代码
* 发起新的 MCP 工具调用

因此整个系统可实现：

* 自动修复脚本
* 自动重新编译
* 自动尝试不同 Worker

形成 **自愈式工程流水线**。

---

## 3.6.8 小结：完整的故障恢复链路

最终形成如下完整恢复链路：

```
Worker 心跳异常 → 标记 unhealthy → 中控禁止调度该 Worker →
当前任务 fallback 到另一 Worker（如支持） →
若不支持（Vivado/Quartus）→ 放入重试队列 →
Worker 恢复后继续任务
```

加上 LLM 自动修复代码的能力，即使工具链行为复杂、易崩溃，也可以形成高度可靠的工程自动化系统。

# 3.7 Worker 与中控之间的通信协议与作业流关系

本节将定义：

* 中控如何向 Worker 派发任务
* Worker 如何回传结果给中控
* 作业的生命周期（Job Lifecycle）
* 状态同步机制（包括心跳、状态文件、workers.json）
* SSH / 文件系统 / 守护进程三者之间的关系

本节的目标是：**让整个系统拥有明确、可视、可监控的 Job → Worker → Result 数据流，使得 LLM 可以成为系统的“头脑”，而 Worker 是“手臂”。**

---

## 3.7.1 整体数据流概述（从 LLM 到 Worker 再返回）

系统核心通信方式基于：

1. **文件协议（request.json / result.json）** —— 中控与 Worker 的“语言”
2. **SSH 调用（on-demand 模式）** —— 中控触发执行
3. **心跳状态文件（worker_status/<worker>.json）** —— Worker 健康与负载报告

整个过程可以用如下序列图表示：

```
LLM →（MCP 调用）→ 中控 → request.json → Worker
Worker → 执行工具 → result.json → 中控 → 返回 LLM
```

这是一个典型的 **文件驱动 workflow**，具备：

* 高透明度（可轻松 debug）
* 高兼容性（适合各种工具链）
* 高可靠性（掉线仍可恢复）

---

## 3.7.2 通信介质：以 `/workspace` 为中心的“共享文件协议”

两台 Worker 和 Mac 中控共享同一个：

```
/workspace/
```

通过 NFS（或 SSHFS / SMB）挂载。此目录是**通信核心设施**。

### 为什么采用“共享文件协议”而非 RPC 或 HTTP？

因为工程工具（Vivado、MATLAB、Quartus）天然基于文件：

* 输入依赖多个目录（工程、脚本、依赖）
* 输出产物是文件（bitstream、C code、waveform）
* 日志是文本文件

HTTP/RPC 适合写应用，不适合工程工具链。**文件是最自然的协议。**

---

## 3.7.3 作业生命周期（Job Lifecycle）

一个 Job 从创建到结束共有 **7 个阶段**：

```
[1] LLM 生成任务请求
[2] 中控分配 Worker → 创建 request.json
[3] Worker 读取 request.json
[4] Worker 执行任务
[5] Worker 写入 result.json
[6] 中控读取 result.json
[7] LLM 获取结构化结果，决定下一步动作
```

下面逐一解释。

---

### **阶段 1：LLM 产生任务请求（MCP Tool 调用）**

LLM 发出：

```
run_vivado_synth(project="projA", top="fir_filter")
```

中控收到后，将其解析为内部 Job。

---

### **阶段 2：中控创建 request.json（Job 标准化输入）**

路径：

```
/workspace/jobs/<job-id>/request.json
```

示例：

```jsonc
{
  "job_id": "job-20251121-0003",
  "tool": "vivado_synth",
  "project": "projA",
  "params": {"top": "fir_filter"},
  "session_id": "sess123",
  "created_at": "2025-11-21T10:10:00Z"
}
```

中控会在此阶段：

* 将 job 分配给 worker1 或 worker2
* 更新 job 状态表（例如 jobs.json）

---

### **阶段 3：Worker 获取任务（两种模式）**

#### ● On-Demand 模式

中控直接：

```
ssh worker1 "python3 /opt/worker_agent/run_job.py /workspace/jobs/<job-id>/request.json"
```

Worker 不需要轮询。

#### ● Daemon 模式（生产环境推荐）

Worker 守护进程扫描 jobs 目录：

```
/workspace/jobs/*/request.json
```

每当遇到一个新 job：

* 加入队列等待执行
* 由线程池启动 run_single_job(job)

---

### **阶段 4：Worker 执行任务**

Worker Agent 根据 `job.tool` 决定要调用的工具：

* Vivado → `vivado -mode batch -source synth.tcl`
* MATLAB → `matlab -batch ...`
* GCC → `arm-none-eabi-gcc ...`
* ModelSim → `vsim -c -do ...`

所有输出（stdout/stderr）被写入：

```
/workspace/jobs/<job-id>/logs/<tool>.log
```

---

### **阶段 5：Worker 写入 result.json**

路径：

```
/workspace/jobs/<job-id>/result.json
```

该文件是中控判断 job 是否完成的“信号”。

完成后，Worker 会将 job 状态写为：

```
status = success/failure
```

---

### **阶段 6：中控回收结果**

中控以轮询或事件监听方式发现：

```
result.json 出现 → 任务完成
```

然后执行：

* 解析 result.json
* 将结果返回给 LLM
* 更新 pipeline 状态表

---

### **阶段 7：LLM 决策下一步**

根据 result.json：

* 再生成新的代码
* 再发起新的 tool 调用
* 调整设计参数（例如 PID 参数）
* 自动推进整个工程流水线

LLM 是整个系统的“智能控制器”。

---

## 3.7.4 中控的任务状态表（jobs.json / workers.json）

为了让系统可观测、可追踪，中控维护两个核心文件：

### **1）jobs.json（所有任务的状态）**

示例：

```jsonc
{
  "job-20251121-0001": {
    "worker": "worker1",
    "status": "running",
    "started": "10:00",
    "tool": "vivado_synth"
  },
  "job-20251121-0002": {
    "worker": "worker2",
    "status": "success",
    "finished": "10:05",
    "tool": "mcu_build"
  }
}
```

### **2）workers.json（Worker 健康状态）**

来自 Worker 心跳：

```jsonc
{
  "worker1": {
    "healthy": true,
    "cpu": 0.61,
    "mem": 0.42,
    "running_jobs": 2
  },
  "worker2": {
    "healthy": false,
    "cpu": null,
    "mem": null
  }
}
```

中控据此执行 3.3 所述调度算法。

---

## 3.7.5 通信协议：基于文件的“事件触发”逻辑

Worker 与中控没有 RPC，而是基于文件的“事件”。

| 事件类型                 | 触发条件            | 行为                 |
| -------------------- | --------------- | ------------------ |
| **job_created**      | request.json 出现 | Worker 开始执行（或加入队列） |
| **job_finished**     | result.json 出现  | 中控读取结果             |
| **worker_heartbeat** | worker1.json 更新 | 中控更新 Worker 状态     |
| **worker_offline**   | 超时未更新           | 中控重新调度             |

整个系统的通信协议可以总结为：

> *“文件是事件，状态即文件”*

---

## 3.7.6 Worker ↔ 中控通信中的一致性问题

必须保证多 Worker 访问 `/workspace` 时保持一致性。

### 可以选择的共享方式：

* NFS（最简单）
* SSHFS（性能一般）
* GlusterFS（可扩展）
* CephFS（大规模）

对于两台 Worker，推荐：

```
NFS（Read-Write, sync enabled）
```

并将 `/workspace` 挂载到：

```
/opt/nfs_master:/workspace
**（如需要）。**

接下来将继续撰写：

---

# 3.8 中控调度器的实现细节（Scheduler Architecture）

中控调度器是整个系统的大脑，它负责：
- 接收 LLM 的 MCP tool 调用
- 将请求转化为标准化 Job（request.json）
- 根据 Worker 状态做调度决策
- 调用 Worker 执行任务
- 收集 result.json 并返回给 LLM
- 管理任务队列、license 状态、心跳检测等

本节将从架构、模块、数据结构、核心算法流程、调度器运行循环等方面展开描述。

---

## 3.8.1 调度器的总体架构

调度器可拆分成 5 个子模块：

```

Scheduler (中控)
├── MCP Interface (LLM 接口)
├── Job Manager (任务生命周期管理)
├── Worker Manager (Worker 状态管理)
├── Dispatch Engine (调度决策核心)
└── FS Interface (共享文件系统读写接口)

```

以下逐个说明。

---

### **1）MCP Interface（LLM → 中控）**
LLM 发送工具调用，例如：
```

run_matlab_sim(project="projA", script="controller.m")

````
MCP Interface：
- 解析参数
- 生成内部 Job 对象
- 分配 job_id
- 调用 Job Manager 创建 job 目录

---

### **2）Job Manager（任务生命周期管理）**
Job Manager 负责：
- `/workspace/jobs/<job-id>/` 的初始化
- 写入 request.json
- 记录任务状态
- 清理旧的 job
- 管理 jobs.json

模块 API 示例：
```python
create_job(tool, project, params, session_id)
update_job_status(job_id, status)
mark_finished(job_id)
````

---

### **3）Worker Manager（管理 Worker 健康状态）**

Worker Manager 定期读取：

```
/workspace/worker_status/worker1.json
/workspace/worker_status/worker2.json
```

并维护 Worker 状态表：

```python
workers = {
  "worker1": {"healthy": True, "cpu": 0.52, "mem": 0.41},
  "worker2": {"healthy": False}
}
```

Worker Manager 还负责：

* SSH 健康检查
* 工具 License 检查（matlab、vivado）
* 检测 Worker 离线
* 触发 3.6 中描述的恢复机制

---

### **4）Dispatch Engine（调度核心）**

Dispatch Engine 是整个系统最关键的部分。

职责：

* 根据 `job.tool` 查找默认 Worker（规则 1）
* 检查并发限制（规则 2）
* 计算 CPU/内存权重（规则 3）
* 检查 License（规则 4）
* 应用 session 亲和（规则 5）
* 选择最优 Worker
* 决定 dispatch 方法（ssh 或 daemon）

核心函数：

```python
select_worker(job, workers)
```

---

### **5）FS Interface（文件系统接口）**

与 `/workspace` 强绑定，职责：

* 创建目录
* 写 request.json
* 检测 result.json 出现
* 清理旧目录

该模块可以封装 NFS/SSHFS 的差异，使调度器代码无需关心挂载方式。

---

## 3.8.2 调度器的核心数据结构

调度器内部使用两类核心对象：

---

### **Job 对象**

```python
Job {
  job_id: str,
  tool: str,
  project: str,
  params: dict,
  session_id: str,
  affinity: str or None,
  status: str,
  created_at: datetime,
  assigned_worker: str or None
}
```

---

### **Worker 对象**

```python
Worker {
  name: "worker1",
  healthy: bool,
  cpu: float,
  mem: float,
  running_jobs: list,
  supported_tools: [...]
}
```

调度时只关注：

* 工具支持情况
* 当前任务类型占用情况（Vivado/Quartus 限制）
* CPU/内存负载
* Worker 是否 healthy
* Worker 是否在线

---

## 3.8.3 调度器的运行循环（Scheduler Event Loop）

调度器是一个循环运行的程序：

```
while True:
    # 1. 更新 worker 状态
    worker_manager.refresh()

    # 2. 处理等待任务队列
    for job in pending_jobs:
        worker = dispatch_engine.select_worker(job, workers)
        if worker != None:
            dispatch(job, worker)

    # 3. 收集结果
    collect_results()

    sleep(1)
```

循环频率可设为：

```
1 Hz（每秒）或 2 Hz
```

足以处理大部分工程任务（因为任务执行时间通常是分钟级）。

---

## 3.8.4 中控如何执行一个 Job（全流程）

以 Vivado 综合为例：

### 步骤 1：LLM 调用 MCP

```
run_vivado_synth(project="projA", top="fir_filter")
```

### 步骤 2：Job Manager 创建 job 目录

```
/workspace/jobs/job-20251121-0007/
```

并写入 request.json

### 步骤 3：调度器选择 Worker1

因为：

* Vivado 默认 Worker1
* Worker1 healthy
* 并发限制允许

### 步骤 4：执行任务（SSH 或 Daemon）

SSH 模式：

```
ssh worker1 "python3 /opt/worker_agent/run_job.py /workspace/jobs/..."
```

### 步骤 5：等待 result.json

调度器轮询：

```
/workspace/jobs/job-20251121-0007/result.json
```

### 步骤 6：解析结果并返回给 LLM

若成功：LLM 生成下一步（如生成 bitstream 分析、继续 FPGA 流程）
若失败：LLM 分析日志并修复代码

---

## 3.8.5 调度器的 License 管理

某些工具（MATLAB/Vivado）有 license 限制。调度器必须监控：

```
license_status{"matlab": free_slots, "vivado": free_slots}
```

若 free_slots == 0：

* 将任务排在等待队列
* 或 fallback 到另一个 Worker（如果工具允许）

license_status 来源：

* Worker 心跳文件
* 工具命令行检查（如 `matlab -license`）
* 调度器主动检查

---

## 3.8.6 调度器安全性设计

调度器作为系统“大脑”，必须保证安全：

### ● 禁止任意命令执行（LLM 必须通过 MCP tool）

中控不允许 LLM 提交 shell 命令。

### ● request.json 参数严格 whitelist

所有参数都必须符合 MCP tool schema。

### ● Worker 使用非 root 用户（builduser）

避免权限破坏。

### ● SSH 加密传输 + 禁止 root 登录

确保调度器与 Worker 之间链路安全。

### ● NFS 权限隔离

对 `/workspace` 的写权限严格控制。

---

## 3.8.7 调度器的可扩展性（支持更多 Worker）

该设计天然支持扩容：

* Worker3：GPU-heavy 节点（用于深度学习推理）
* Worker4：EDA-heavy 节点（Synopsys / Cadence）
* Worker5：大内存节点（仿真集群）

无需修改核心架构，只需：

* 新 Worker 的心跳加入 worker_status/
* Dispatch Engine 增加对新工具类型的默认路由
* supported_tools 列表扩展

整个调度器具备**线性扩展**能力。

---

## 3.8.8 小结

中控调度器具备：

* 明确的模块结构（MCP、Job、Worker、Dispatch Engine）
* 完整的数据结构体系（Job/Worker objects）
* 健壮的运行循环（event loop）
* 强大的调度策略（规则 + 动态调度 + 亲和性 + license 管理）
* 良好的扩展性（多 worker）

至此，中控作为“大脑”的调度能力已经完善。

接下来可以撰写：

# 3.9 本章总结与整体架构图

本章内容非常庞大，几乎构成整个系统的“基础设施核心设计”。本节将以结构化方式，总结并统一整个第 3 章内容，并给出多个架构图，以便读者或系统实施者能够快速理解全局运作方式。

本节包含：

* 3.9.1 核心思想总结
* 3.9.2 完整系统架构图（Worker + 中控 + LLM）
* 3.9.3 作业生命周期序列图（Job Lifecycle Sequence）
* 3.9.4 Worker 端架构图
* 3.9.5 中控调度器内部模块图
* 3.9.6 本章关键要点回顾

---

## 3.9.1 核心思想总结

第 3 章的核心目标：**构建一个可扩展、可恢复、可管理的多节点工程计算系统，使 LLM 成为“编排者”，Worker 成为“执行器”。**

整个体系的理念：

* **LLM 负责智能与规划**（代码生成、错误修复、自动优化）
* **中控调度器负责决策**（资源分配、Worker 健康管理、任务调度）
* **Worker 节点负责执行**（具体调用工程软件，生成产物）
* **共享文件系统是通信纽带**（request.json / result.json / 状态文件）
* **日志与结果文件为真相来源**（LLM 自主分析问题与改进）

通过上述角色分离，实现：

* 故障可恢复
* 任务可并发
* Worker 可扩展
* 流程可自动继续
* LLM 可全自动完成端到端工程流程

---

## 3.9.2 完整系统架构图（Worker + 中控 + LLM）

采用 ASCII 图（后续可在第 5 章加入正式绘图）：

```
                 ┌──────────────────────────────┐
                 │            LLM (大脑)         │
                 │  - 生成任务 / 代码 / 修复        │
                 │  - 调用 MCP Tools           │
                 └──────────────┬──────────────┘
                                │
                                ▼
                   ┌────────────────────────┐
                   │      Mac 中控调度器      │
                   │ ------------------------ │
                   │  MCP Interface           │
                   │  Job Manager             │
                   │  Worker Manager          │
                   │  Dispatch Engine         │
                   │  FS Interface            │
                   └──────────────┬───────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
   ┌──────────────────────────┐       ┌──────────────────────────┐
   │       Worker1 (FPGA)     │       │      Worker2 (MCU)        │
   │--------------------------│       │---------------------------│
   │  Vivado / MATLAB /       │       │ Quartus / GCC / MATLAB    │
   │  ModelSim / COMSOL       │       │ ModelSim / CubeMX         │
   │  Agent(run_job / daemon) │       │ Agent(run_job / daemon)   │
   └───────────────┬──────────┘       └───────────────┬───────────┘
                   │                                  │
                   └──────────────┬───────────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │    /workspace (NFS)     │
                     │  projects/              │
                     │  jobs/ (req + res)      │
                     │  worker_status/         │
                     └────────────────────────┘
```

解释：

* LLM → 中控：以 MCP Tool 的结构化调用方式传递任务
* 中控 → Worker：通过 request.json + ssh/daemon 执行
* Worker → 中控：通过 result.json 回传
* Worker → 中控：通过 worker_status.json 报告健康
* 中控 → LLM：返回结果并触发下一步（继续工程流水线）

整个结构自然可横向扩展更多 Worker。

---

## 3.9.3 作业生命周期序列图（Job Lifecycle Sequence）

```
LLM                中控                 Worker
│                   │                    │
│ (1) MCP 调用       │                    │
├──────────────────►│                    │
│                   │ (2) 创建 request.json
│                   ├───────────────────►│
│                   │                    │
│                   │      (3) 执行 run_job/daemon
│                   │◄───────────────────┤
│                   │                    │
│                   │      (4) 工具执行 (Vivado/MATLAB/GCC)
│                   │                    │
│                   │      (5) 写 result.json
│                   ├───────────────────►│
│                   │                    │
│                   │ (6) 读取 result.json
│                   ◄────────────────────┤
│                   │                    │
│ (7) 返回结果给 LLM │                    │
◄────────────────────┤                    │
```

序列图说明：

* 所有关键交互都是文件事件
* Worker 仅执行 request.json 的内容
* LLM 最终获得 result.json 的结构化结果

---

## 3.9.4 Worker 端架构图（Agent 内部结构）

```
                   Worker Agent
                ┌─────────────────┐
                │ run_job.py       │
                │ agent_daemon.py  │
                ├─────────────────┤
                │ Dispatcher       │
                ├─────────────────┤
                │ Tool Handlers    │
                │  - MATLAB        │
                │  - Vivado        │
                │  - Quartus       │
                │  - GCC           │
                │  - ModelSim      │
                ├─────────────────┤
                │ Log Manager      │
                │ Result Writer    │
                ├─────────────────┤
                │ Heartbeat Writer │
                └─────────────────┘
```

Worker 端设计重点：

* 统一入口
* 多工具 handler 注册机制
* 一致的日志与结果写入流程
* 支持并发队列（daemon 模式）
* 提供心跳文件给中控

---

## 3.9.5 中控调度器内部模块图

```
               Scheduler Architecture
┌────────────────────────────────────────────┐
│                MCP Interface               │
│  - 接收 LLM 调用                           │
│  - 参数解析                                │
├────────────────────────────────────────────┤
│                 Job Manager                │
│  - 创建 job 目录                           │
│  - 写 request.json                         │
│  - 维护 jobs.json                           │
├────────────────────────────────────────────┤
│                Worker Manager              │
│  - 心跳检测                                │
│  - Worker 状态维护                          │
│  - License 检查                            │
├────────────────────────────────────────────┤
│                Dispatch Engine             │
│  - 默认规则                                │
│  - 动态调度                                │
│  - 会话亲和                                │
│  - fallback                                │
├────────────────────────────────────────────┤
│                  FS Interface              │
│  - 统一文件读写                            │
│  - 探测 result.json                         │
└────────────────────────────────────────────┘
```

该模块化架构使中控具备：

* 高扩展性
* 明确边界
* 便于维护
* 可替换性（未来可加入 HTTP/RPC 作为优化）

---

## 3.9.6 本章关键要点回顾

本章实现了一个完整、工业级的多 Worker 架构：

### **1）Worker 的角色（重点分工设计）**

* Worker1：FPGA-heavy（Vivado / COMSOL）
* Worker2：MCU-heavy（Quartus / GCC / CubeMX）
* 两者都具备 MATLAB/ModelSim

### **2）统一运行环境**

* builduser 权限体系
* `/opt` 工具路径标准化
* `/workspace` 多 Worker 共享
* profile.d 工具环境统一
* SSH 安全配置

### **3）Worker Agent 的标准化执行模型**

* request.json → 工具 CLI → result.json
* handler 注册机制
* 一致日志结构
* 并发队列（daemon）
* 心跳

### **4）中控的调度系统**

* 多规则决策 + 动态调度
* 任务亲和性
* License-aware 调度
* 高级回退策略（fallback、delay、retry）

### **5）故障恢复能力**

* Worker offline / degraded 检测
* 自动重试、回退、迁移
* LLM 的自动修复闭环

### **6）整体架构透明、可扩展**

* 通过文件协议自然连接 Worker
* 可横向扩展多个 Worker 节点
* 可容纳更多工具链（Synopsys / Cadence / GPU 工具等）

---

第 3 章至此完整收尾，为第 4 章（MCP 工具设计）奠定坚实基础。

如需继续撰写第 4 章，请告诉我：“开始第 4 章”。

包括 Worker 掉线、任务中断、部分结果保留、重试策略等。
