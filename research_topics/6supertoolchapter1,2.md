# LLM 多工具协同计算平台（Mac 中控 + 两台 Linux Worker）设计与部署指南

（草稿 v0.2）

---

# 第 1 章：总体架构设计

本章将完整描述本平台的总体结构，包括：系统角色、节点之间的职责分工、通信方式、任务调度模式、工具协同方式、文件系统结构、以及整个系统为何适合用于“大量工程软件 + LLM 协同生产”这一目标场景。

本章为全局性设计，后续所有章节（工具安装规划、调度实现方式、工作区设计、MCP 工具 schema、安装部署指南等等）都将以本章结构为核心框架。

---

## 1.1 系统角色与核心思想

本平台由 **1 台 Mac 中控节点** + **2 台 Linux Worker 节点** 组成，责任划分如下：

### ● Mac 中控节点（LLM + MCP Server + 调度器）

* 作为“平台大脑”，执行所有逻辑：

  * 运行 LLM（本地或远程 API）
  * 运行 MCP server 暴露工具接口
  * 进行任务调度（选择 worker）
  * 通过 SSH 调用 worker 执行工具任务
  * 管理任务生命周期、日志、结果、错误
  * 管理统一 workspace 的元数据（manifest、system.yaml 等）

### ● 两台 Linux Worker 节点（工程软件的实际执行者）

* 安装与运行各种工程类、仿真类、硬件相关软件：

  * MATLAB（含 Simulink）
  * FPGA 工具（Vivado / Quartus）
  * MCU 工具链（Keil、IAR、GCC、STM32CubeIDE 等）
  * 其他需要大量 CPU / 内存的工程软件
* 每台 worker 接收中控的任务指令，通过本地脚本执行对应软件的 batch/headless 模式，将结果写回 workspace。

### ● 统一 Workspace（共享文件系统）

* 所有机器（Mac + Worker1 + Worker2）都挂载同一个 `/workspace` 目录
* 是平台的“真相来源”（single source of truth）
* LLM 通过阅读/写入 workspace 中的文件，完成真正的工程协作
* 多个工具通过 workspace 实现“紧耦合协同”：

  * MATLAB 输出 C 文件 → MCU 工具链编译
  * MATLAB 导出参数 → Vivado 生成约束
  * FPGA bitstream + MCU firmware → 测试节点自动运行测试脚本

---

## 1.2 整体系统拓扑图（概念性）

```
                     ┌───────────────────────────┐
                     │      Mac 中控（主控）       │
                     │  - LLM / MCP Server       │
                     │  - 调度器 (Dispatcher)     │
                     │  - SSH Client              │
                     └─────────────┬─────────────┘
                                   │ SSH
          ┌─────────────────────────┼──────────────────────────┐
          │                         │                          │
┌─────────▼──────────┐   ┌──────────▼─────────┐
│   Linux Worker 1    │   │   Linux Worker 2    │
│  - MATLAB           │   │  - MATLAB（可选）    │
│  - Vivado/Quartus   │   │  - MCU toolchain    │
│  - MCU 工具链       │   │  - 其他工具          │
│  - worker_agent.py  │   │  - worker_agent.py  │
└─────────┬──────────┘   └───────────┬────────┘
          │                          │
          └──────────────┬───────────┘
                         │ NFS / NAS / 共享 FS
                 ┌────────▼────────┐
                 │   /workspace     │
                 │ 项目/任务/日志等 │
                 └──────────────────┘
```

---

## 1.3 为什么需要这种三节点结构？

### 理由 1：Mac 适合作为轻量但高效的“中控 + LLM 主脑”

* Mac M 系列芯片在推理、开发环境上表现很好
* 图形界面方便开发与监控工具运行
* 适合运行 MCP Server + WebUI + 日志查看器 + 调度控制台

### 理由 2：工程类工具普遍对 Linux 支持更成熟

* Vivado / Quartus / MATLAB headless / GCC / Keil Linux CLI 等软件在 Linux 上性能最佳
* Linux Worker 拿来跑这些工具，可以减少兼容性麻烦
* 工具授权（license server）也更容易配置在 Linux

### 理由 3：分布式架构可以线性扩展计算能力

* 随时新增 Worker3 / Worker4
* 任务按照 CPU / 内存 / 工具类型调度到最合适的 worker
* LLM 完全不需要知道有几台机器：只和 MCP Server 交互

### 理由 4：工具之间“协同加工”的核心是共享 workspace

* MATLAB 生成代码 → MCU 编译 → FPGA 工具生成 bitstream
* 所有输入输出统一放在 `/workspace/projects/...` 下
* LLM 能理解整个工程状态，就能承担“自动流水线”的角色

---

## 1.4 核心设计哲学

### ● 工具不直接暴露给 LLM

LLM 只调用 MCP tool，不直接操作 SSH 或具体软件。
调度逻辑完全隐藏在 Mac 上的 MCP Server 中，保证：

* 安全性
* 可控性
* 可替换性
* 可复现性

### ● Worker 是“黑盒执行器”

* 它仅按照 request.json 指令执行特定 task
* 写 result.json 给中控
* 完全 stateless 或轻状态
* 易于横向扩展

### ● Workspace 是全局状态机

* 所有项目、任务、日志、产物都写入 workspace
* 所有工具通过文件协同
* LLM 仅需理解文件与描述文档（system.yaml / manifest.json）即可协同跨工具生产

### ● 任务以“Job JSON 协议”标准化

* MATLAB/Vivado/Keil/... 的执行指令统一格式
* LLM 调一个工具，本质就是让中控生成一个 Job 请求

---

## 1.5 系统具备的能力（目标能力）

当后续章节全部部署完成后，本系统应实现：

### 1）自动构建与仿真

* LLM 自动调用 MATLAB 计算并导出 C
* LLM 自动调用 MCU 工具链编译 firmware
* LLM 自动调用 FPGA 工具链综合 bitstream

### 2）自动跨工具联动

例如：

* MATLAB → C 模型 → Keil/IAR 编译 → MCU 固件
* MATLAB → 参数 → Vivado 约束文件自动生成

### 3）自动错误处理

* 读取编译/综合/仿真日志
* 推断错误原因
* 修改源文件，重新提交任务

### 4）自动流水线 DAG（后续章节会写）

* LLM 自己规划“先 MATLAB → 再 MCU → 再 FPGA”顺序
* 并在 workspace 写入 build-manifest.json

### 5）多 worker 并发计算

* Worker1 跑 FPGA 综合
* Worker2 跑 MCU 编译
* Mac 中控任务调度 + 结果收集

### 6）可横向扩容

新增 Worker3/4（GPU 节点、Windows 节点、更多 Linux）仅需注册到调度器即可。

---

## 1.6 本章总结

第 1 章完成了平台总体架构的设计：

* 三节点系统（Mac 中控 + Linux Worker 双节点）
* SSH 调用 + 统一 workspace
* Worker 作为黑盒工具执行器
* LLM 通过 MCP tool 指挥任务
* 整套系统天然适合“工程类软件链条式生产”

# 第 2 章：工程软件生态与 Worker 安装规划

* 列举平台要支持的所有软件（MATLAB, Vivado, Keil, GCC …）
* 哪些适合装在 worker1 / worker2
* 使用方式（CLI/headless）
* 依赖、license 与资源消耗

---

## 2.1 工具生态概览：为什么需要这么多软件？

在“LLM 参与工程设计与自动化”背景下，一个完整的工程工具链往往跨越多个领域：

* **算法层**（控制仿真、滤波器设计、信号处理）
* **代码生成层**（自动生成 C/C++/HDL）
* **MCU 固件层**（单片机代码、驱动、实时系统）
* **FPGA 层**（RTL 综合、实现、bitstream）
* **系统集成层**（链接脚本、硬件配置、接口定义）
* **测试验证层**（自动化测试、仪器控制、仿真回归）

LLM 需要通过工具 API 调用整个链条，因此我们必须在两个 Linux Worker 上安装尽可能齐全的工具，涵盖上述所有领域。

---

## 2.2 常用软件大列表（按领域分类）

下面我们从五大类工具进行列举。

---

### 2.2.1 数值计算 / 建模仿真类（MATLAB / Simulink / COMSOL / Mathematica）

#### ① MATLAB（含 Simulink）

* 用途：控制系统设计、信号处理、优化、系统建模、代码生成等。
* Linux 支持：非常成熟。
* Headless：完全支持 `matlab -batch` 与 MATLAB Engine。
* License：需要能访问 license server（FlexLM）。
* 资源占用：中等到高（取决于仿真规模）。

**安装建议：Worker1 与 Worker2 都安装。**

* 原因：任务调度更灵活（MATLAB 常被调用）。
* 可配置不同 toolbox 组合（可选）。

#### ② COMSOL Multiphysics

* 用途：多物理场仿真（电磁/流体/结构/热）。
* 资源占用：非常高。
* Headless：支持（`comsol batch`）。
* License：需要 License Server。

**安装建议：仅安装在 Worker1。**

* 原因：此类任务极度占用 CPU，不适合多机重复安装。

#### ③ Mathematica / Maple

* 用途：符号数学、数学建模。
* 资源占用：中。
* Headless：支持。

**安装建议：Worker2（轻量任务为主）。**

---

### 2.2.2 FPGA / 数字电路设计类（Vivado / Quartus / Libero / ModelSim 等）

#### ① Xilinx Vivado (FPGA)

* 用途：RTL 综合、实现、生成 bitstream。
* Headless：完全支持 TCL 模式 `vivado -mode batch -source script.tcl`。
* 资源占用：非常高（多核高并发、长时间运行）。
* 依赖：Java 环境、主机内核参数等。

**安装建议：Worker1。**

* Worker1 作为“FPGA-heavy 节点”。

#### ② Intel Quartus Prime

* 用途：另一个 FPGA 工具链。
* Headless：支持（.tcl）。
* 资源占用：高。

**安装建议：Worker2（避免与 Vivado 同机争资源）。**

#### ③ ModelSim / Questa

* 用途：RTL 仿真器。
* Headless：完全支持 CLI。
* 资源占用：中等偏高。

**安装建议：两台 Worker 都可安装。**

* 因为仿真是高频步骤。

---

### 2.2.3 MCU / 嵌入式开发环境（Keil / IAR / GCC / STM32CubeIDE）

#### ① ARM Keil MDK（需 Wine 或只运行命令行工具）

* 用途：ARM Cortex-M 开发。
* Linux 支持：官方无原生支持，只能靠 wine/容器或换用 ARMClang + CMSIS。

**建议：用 `arm-none-eabi-gcc` 替代 Keil，除非必须 Keil MDK。**

#### ② IAR Embedded Workbench

* Linux 支持：较弱，CLI 辅助工具少。

**建议：不作为主力工具，仅使用 GCC / Clang MCU toolchain。**

#### ③ GCC ARM (arm-none-eabi-gcc)

* 用途：ARM MCU 主力编译器。
* 完全支持 Linux。
* CLI 完全可控。

**安装建议：Worker2（MCU-heavy 节点）。**

#### ④ STM32CubeIDE / CubeMX

* 用途：STM32 配置工具和 HAL 代码生成。
* Headless：CubeMX 支持 CLI 生成工程。

**安装建议：Worker2。**

---

### 2.2.4 测试与测量工具（LabVIEW / TestStand / PyVISA）

LabVIEW/TestStand 大多需要 Windows GUI，不适合放在 Linux Worker。

**解决方案：未来可扩展 Windows Worker 节点，不在本次最小设计部署。**

**当前 Linux Worker 可安装：**

* PyVISA（仪器控制）
* SCPI 控制脚本

---

### 2.2.5 系统工具 / 通用构建环境

所有 Worker 推荐安装：

* Python3 + pip
* OpenJDK（Vivado 需要）
* CMake / Ninja
* GCC/G++ / Clang
* Git
* Make
* Docker（可选，用于隔离某些工具）
* SSH server

这些提供基础构建、文件操作、日志处理、解析脚本的能力。

---

## 2.3 Worker1 与 Worker2 的软件安装规划（最终版）

下面是推荐的最优分配方案。

| 工具                   | Worker1 | Worker2 | 说明                       |
| -------------------- | ------- | ------- | ------------------------ |
| MATLAB + Simulink    | ✔️      | ✔️      | 常用高频，双机安装提高调度能力          |
| COMSOL               | ✔️      | ❌       | 重型任务集中在 Worker1          |
| Vivado               | ✔️      | ❌       | FPGA-heavy 节点            |
| Quartus              | ❌       | ✔️      | 避免 Vivado/Quartus 共存减少冲突 |
| ModelSim / Questa    | ✔️      | ✔️      | 仿真常用，双机均可                |
| ARM GCC Toolchain    | ✔️      | ✔️      | 轻量通用，两边都装                |
| STM32CubeMX          | ❌       | ✔️      | MCU-heavy 节点             |
| CMake / GCC / Python | ✔️      | ✔️      | 标配                       |

总结：

* **Worker1 = FPGA-heavy 节点（Vivado + COMSOL）**
* **Worker2 = MCU-heavy 节点（CubeMX + Quartus）**
* 共同承担 MATLAB 与仿真任务

---

# 2.4 各软件 Linux 安装细节（MATLAB / Vivado / CubeMX / GCC 等）

* 获取方式
* 安装命令
* 文件路径规划
* 环境变量设置
* 常见问题（headless、license、依赖）

## 2.4 各软件 Linux 安装细节（MATLAB / Vivado / Quartus / CubeMX / GCC 等）

本节将逐个软件详细说明：获取方式、安装步骤、目录规划、环境变量、license 配置、headless 使用方式，以及推荐的安装路径规范，让两个 Linux Worker 具备完全可自动化执行的能力。

本节内容较长，将按“从最关键的软件开始”逐段写入。

---

# 2.4.1 MATLAB（含 Simulink）在 Linux Worker 上的安装细节

MATLAB 是整个系统中被 LLM 调用频率非常高的软件之一，且 MATLAB 官方对 Linux 的 headless 执行支持极好（`matlab -batch` + MATLAB Engine API）。因此我们需确保 Worker1 & Worker2 都安装完整稳定的 MATLAB 环境。

---

## 一、MATLAB 获取方式

两种正式方式：

### ● 方式 A：使用 MathWorks 官方安装包（推荐）

从 MathWorks 账户下载：

```
matlab_R2024a_glnxa64.zip
```

或使用 ISO：

```
matlab_R2024a_glnxa64.iso
```

### ● 方式 B：离线安装包（本地部署时常用）

企业/科研机构常维护离线下载器，可直接放入 `/opt/matlab_installers/`。

---

## 二、安装前准备（两台 Worker 都需要）

```
sudo apt update
sudo apt install -y build-essential libxt6 libxmu6 libxpm4 libglu1-mesa libgtk2.0-0
sudo apt install -y unzip xauth
```

MATLAB 依赖少，主要是：

* X11 库（虽然是 headless，但 MATLAB 本体仍依赖）
* GCC（用于 mex 编译）
* Java（MATLAB 自带，不需额外安装）

---

## 三、解压与安装目录规范

为了保证自动化脚本容易识别，建议：

```
/opt/mathworks/matlab/R2024a
```

操作示例：

```
sudo mkdir -p /opt/mathworks/matlab
sudo unzip matlab_R2024a_glnxa64.zip -d /opt/mathworks/matlab_installer
sudo /opt/mathworks/matlab_installer/install
```

安装过程中：

* 选择 **Linux x86_64**
* 勾选所需 toolbox（至少包括 Simulink、Control System、Signal Processing）
* License 采用 **网络授权（license server）**

---

## 四、License Server 设置

在所有 Worker 的 `/usr/local/MATLAB/R2024a/licenses/network.lic` 或 `/etc/matlab/licenses/` 中写入类似：

```
SERVER license.mycompany.com 001122334455 27000
USE_SERVER
```

如果 MacBook 作为 server，也可以写：

```
SERVER mac-controller.local <mac-address> 27000
USE_SERVER
```

验证：

```
matlab -batch "disp('test')"
```

如果显示 license 错误，则检查：

* 端口是否开放 27000
* hostid 是否匹配
* 防火墙是否阻挡

---

## 五、环境变量配置（强烈推荐写入 /etc/profile.d/matlab.sh）

创建：

```
sudo nano /etc/profile.d/matlab.sh
```

内容：

```
export MATLAB_ROOT=/opt/mathworks/matlab/R2024a
export PATH="$MATLAB_ROOT/bin:$PATH"
```

使其生效：

```
source /etc/profile.d/matlab.sh
```

---

## 六、Headless / Batch 模式（LLM 调用必须使用）

MATLAB headless 执行方式：

### ● batch 模式（最推荐）

```
matlab -batch "myscript"
```

特点：

* 完全无需图形界面
* 不产生 GUI 依赖
* 错误有清晰 stdout/stderr
* 可直接重定向日志

### ● MATLAB Engine（Python 调用）

Worker agent 可选用：

```
python3 -c "import matlab.engine; eng=matlab.engine.start_matlab(); eng.myfunc(nargout=0)"
```

但 batch 模式更稳定，建议以 batch 为主。

---

## 七、Worker Agent 中对 MATLAB 的调用方式

将来在 `worker_agent.py` 中典型调用为：

```
cmd = [
    "matlab",
    "-batch",
    f"addpath(genpath('{project_path}')); {entry_fn}({arg_str});"
]
subprocess.run(cmd, ...)
```

并将 stdout 重定向到：

```
/workspace/jobs/<job-id>/logs/matlab.log
```

这为 LLM 提供完全可解析的、结构化的执行结果。

---

## 八、MATLAB 常见问题（FAQ）

### ● 1）matlab: command not found

检查 `/etc/profile.d/matlab.sh` 是否正确配置。

### ● 2）Error: No display found

必须使用 `-batch` 而不是 `-r` 或 `-nodesktop`。

### ● 3）License error -15

表示 license server 不可达，检查端口与 hostid。

### ● 4）Simulink 模型无法加载

需要在 batch 指令中调用：

```
load_system('model')
```

---

# 2.4.2 Vivado 在 Linux Worker 上的安装细节（安装于 Worker1）

Vivado 是 FPGA-heavy 节点（Worker1）的核心工具之一，用于综合、实现与 bitstream 生成，资源消耗极高。本节提供完整的安装与配置流程。

---

## 一、Vivado 获取方式

从 Xilinx/AMD 官方下载：

```
Xilinx_Unified_2024.1_0420_1238_Lin64.bin
```

或离线安装包（企业内部镜像）：

```
Xilinx_Unified_2024.1_Offline.tar.gz
```

---

## 二、安装前准备

```
sudo apt install -y libtinfo5 libncurses5 libxft2 libxrender1 libxi6 libxtst6
sudo apt install -y build-essential python3
```

Vivado 自带 JRE，但对 Linux 某些图形包有隐式依赖，因此即便 headless，也要安装相关库。

---

## 三、推荐安装路径

```
/opt/Xilinx/Vivado/2024.1
```

安装：

```
chmod +x Xilinx_Unified_2024.1_0420_1238_Lin64.bin
sudo ./Xilinx_Unified_2024.1_0420_1238_Lin64.bin --target /opt/Xilinx --batch Install
```

---

## 四、环境变量配置

创建：

```
sudo nano /etc/profile.d/vivado.sh
```

内容：

```
export XILINX_VIVADO=/opt/Xilinx/Vivado/2024.1
export PATH="$XILINX_VIVADO/bin:$PATH"
```

---

## 五、Vivado headless 使用方式（LLM 必须使用）

### ● Batch 模式（最关键）

```
vivado -mode batch -source run_synth.tcl
```

典型 TCL 脚本内容：

```
read_verilog top.v
read_xdc constraints.xdc
synth_design -top top
opt_design
place_design
route_design
write_bitstream top.bit
```

### ● 输出重定向

Worker agent 中通常写：

```
subprocess.run([
  "vivado", "-mode", "batch", "-source", script,
], stdout=logfile, stderr=logfile)
```

---

## 六、Vivado 常见问题（FAQ）

### ● 1）Missing 32-bit libraries

安装：

```
sudo apt install libstdc++6:i386 libc6:i386
```

某些版本需要。

### ● 2）Tcl 解析失败

通常是 Windows 风格换行，应在 worker 端执行：

```
dos2unix *.tcl
```

### ● 3）Bitstream 路径找不到

LLM 需要写入 manifest.json，worker 依据 manifest 路径运行。

---

# 2.4.3 Quartus Prime（安装于 Worker2）

Quartus 是 Intel FPGA 工具链，对应于 Xilinx Vivado。由于资源占用高，**

本章将系统性列举整个平台所能支持的工程类软件生态，包括：数值计算、建模仿真、FPGA/数字逻辑设计、MCU 工具链、测试测量系统、数据处理环境等。每种工具不仅会介绍用途，还会详细说明：

* 适合安装在哪一台 Linux Worker 上（worker1 / worker2）
* 资源占用与推荐分配
* Headless/CLI 模式是否成熟
* Linux 环境的依赖项（Python、Java、环境变量、library 等）
* License Server 注意事项

本章规模较大，将分多节逐步写入。

---
# 2.4.3 Quartus Prime（Worker2）安装细节

本文档将专门记录 Quartus Prime 在 Worker2 上的安装流程、依赖、headless 使用方法、环境变量设置以及常见问题排查。

---

# 一、Quartus Prime 获取方式

Quartus Prime 分为三个版本：Lite / Standard / Pro。工程自动化与 LLM 集成建议使用 **Standard 或 Pro**，因为：

* 支持更完整的 Tcl 执行接口
* 适配主流 Cyclone / Arria / Stratix 系列 FPGA
* 编译速度更快、稳定性更好

**下载方式：**

* Intel 官方软件下载中心：[https://www.intel.com/content/www/us/en/collections/products/fpga/software/downloads.html](https://www.intel.com/content/www/us/en/collections/products/fpga/software/downloads.html)
* 企业内部镜像服务器（推荐放置在 `/opt/intel_installers/`）

典型文件名：

```
Quartus-lite-21.1.0.842-linux.tar
Quartus-standard-22.1std.0.169-linux.tar
Quartus-pro-23.4.0.88-linux.tar
```

---

# 二、系统依赖安装（Worker2 必须）

Quartus 依赖大量旧版 Linux 库，即使以 headless 模式运行，也需要安装完整依赖：

```
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install -y libxext6 libxft2 libx11-6 libxrender1 libxi6 libxtst6 libxss1 \
    libc6:i386 libstdc++6:i386 libx11-6:i386 libxext6:i386 libxrender1:i386
```

说明：

* Quartus 使用大量 X11 组件，即使不启动 GUI 也必须存在这些库
* 部分 Quartus 工具（如 nios2-*）依赖 32-bit 版本

---

# 三、推荐安装路径

本平台约定所有大型工程软件放在 `/opt` 下：

```
/opt/intelFPGA/22.1std/quartus
```

安装示例：

```
mkdir -p /opt/intelFPGA
tar -xf Quartus-standard-22.1std.0.169-linux.tar -C /opt/intelFPGA
mv /opt/intelFPGA/quartus /opt/intelFPGA/22.1std
```

Lite 版也可以：

```
/opt/intelFPGA_lite/21.1/quartus
```

---

# 四、环境变量配置

创建：

```
sudo nano /etc/profile.d/quartus.sh
```

内容：

```
export QUARTUS_ROOTDIR=/opt/intelFPGA/22.1std/quartus
export PATH="$QUARTUS_ROOTDIR/bin:$PATH"
```

立即生效：

```
source /etc/profile.d/quartus.sh
```

验证是否安装成功：

```
quartus_sh --version
```

输出类似：

```
Quartus Prime Shell
Version 22.1std.0 Build 169
```

---

# 五、Headless 模式（LLM 必须使用）

LLM 调用 Quartus 不会使用 GUI，全部通过 Tcl 脚本与 `quartus_sh` 运行。

## ● 1）编译项目

```
quartus_sh --flow compile <project_name>
```

示例：

```
quartus_sh --flow compile fpga_top
```

运行结果会在：

```
<project_name>/output_files/
```

生成：

* `.sof`（下载文件）
* `.pof`（可编程对象文件）
* `.rpt`（报告）

## ● 2）运行 Tcl 脚本

```
quartus_sh -t run_flow.tcl
```

典型的 `run_flow.tcl`：

```
project_open fpga_top
execute_flow -compile
project_close
```

## ● 3）生成约束文件、读取 pinout、修改配置

LLM 可以自动写 Tcl，然后让 worker 执行：

```
quartus_sh -t gen_pin.tcl
```

---

# 六、Worker Agent 调用 Quartus 的标准流程

`worker_agent.py` 中执行 Quartus 任务一般如下：

```
cmd = [
    "quartus_sh",
    "--flow", "compile",
    project_name
]

with open(log_path, "w") as f:
    subprocess.run(cmd, cwd=project_dir, stdout=f, stderr=f)
```

执行结束后 agent 写入：

* `result.json`（任务结果）
* 日志文件路径（供 LLM 分析）

---

# 七、License 说明（如使用 Standard / Pro）

Quartus Prime Standard/Pro **需要许可证**：

* 文件位置：`/opt/intelFPGA/licenses/license.dat`
* 或网络授权：

```
LM_LICENSE_FILE=<server>:1800
```

Lite 版 **完全免费**，无需 license。

---

# 八、Quartus 常见问题（FAQ）

## ① 启动时报 libpng12.so.0 缺失

```
sudo apt install libpng12-0
```

（Ubuntu 22+ 可能需从兼容仓库安装）

## ② 编译太慢

* 增加 Worker2 CPU（≥16 核）
* 使用 `--read_settings_files=off` 加速重复编译

## ③ Tcl 脚本报错：未找到工程

检查是否正确：

```
project_open fpga_top
```

路径是否正确、文件扩展名 `.qpf` 是否存在。

## ④ 运行 Tcl 提示 32-bit lib 缺失

按照本文前述依赖安装 32-bit 库。

---

# 后续将继续补充：

* STM32CubeMX 安装
* ARM GCC Toolchain 安装
* ModelSim 安装
* MCU 调试工具（OpenOCD / pyOCD）
* 通用构建环境（CMake / Ninja / Python）

（下一次更新继续撰写）


本报告将逐章展开撰写，逐步扩充内容，包括：

* 系统总体架构设计
* 常用工程/仿真/硬件工具软件生态及其在两台 Linux worker 的安装规划
* Mac 中控节点的 LLM + MCP 调度系统设计
* worker agent 设计
* 统一 workspace 规范
* 工具调用协议（Job JSON）
* 多工具紧耦合的流水线示例
* 部署步骤（从零开始）
* 扩展、并发与安全性设计

后续章节将依次补充。本页现为空白章，后续将逐章填充。


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

3.3 Worker 调度策略与算法（由 Mac 中控执行）

Mac 中控节点是唯一的调度中心。LLM 发出的每个 MCP tool 调用最终会转换成一个 Job，请求中控为其选择最合适的 Worker。

核心要求：

避免将大型任务压在同一台 Worker 上

让小任务尽量并发

保证大型任务优先级

避免 license 冲突（MATLAB / Vivado）

本节将设计完整调度流程。

3.3.1 调度器输入与输出

输入（由 MCP tool 产生的 job JSON）：

{
  "tool": "vivado_synth",
  "project": "projA",
  "params": {...}
}

输出：

selected_worker: worker1 / worker2

dispatch_method: ssh / local

expected_load: 调度器根据 CPU/内存估算的负载

3.3.2 基础调度算法（规则优先级排序）

最小可行调度器基于以下顺序决策：

规则 1：根据工具类型选择默认 Worker

Vivado → Worker1（FPGA-heavy）

Quartus → Worker2（MCU-heavy）

MCU 编译（arm-none-eabi-gcc）→ Worker2

MATLAB 仿真 → Worker1（优先），Worker2（fallback）

ModelSim/Questa → Worker1（优先），Worker2（fallback）

这确保同类任务命中最适合的节点。

规则 2：防止大型任务并发冲突

大片任务如 FPGA 综合（Vivado/Quartus）会强占 CPU，因此：

if worker.running_vivado_jobs >= 1:
    不能再在 worker 上调度 Vivado

Quartus 同理：

if worker.running_quartus_jobs >= 1:
    worker 不接收新的 quartus_synth

规则 3：根据 CPU/内存即时负载选择“最轻的可用 Worker”

在满足规则 1 & 2 的前提下：

select worker with min( cpu_load_weighted )
where cpu_load_weighted = cpu_usage + job_cost_factor(tool_type)

例如：

MATLAB job_cost_factor = 2

Vivado job_cost_factor = 10

MCU compile job_cost_factor = 1

规则 4：避免 license 冲突

若：

MATLAB license 已占满 → 不调度新的 MATLAB job
Vivado license 已占满 → 延迟 FPGA 任务

License 状态由中控维护：

license_status{"matlab": free_slots, "vivado": free_slots}

规则 5：遵从会话偏好（session-aware scheduling）

如果 LLM 在当前 session 中多次调用 MATLAB，则应保持在同一 Worker：

if session.preferred_worker != None:
    return preferred_worker

避免 MATLAB 多次初始化开销。

3.3.3 动态调度（基于 Worker 实时状态）

调度器周期性（每 3 秒）收集 Worker 指标：

cpu_usage

mem_usage

running_jobs[]

queue_size

healthy（通过 ssh 心跳）

生成 Worker 状态表：

{
  "worker1": {"cpu": 78%, "mem": 61%, "vivado_jobs": 1, "matlab_jobs": 2},
  "worker2": {"cpu": 42%, "mem": 37%, "quartus_jobs": 0, "mcu_jobs": 3}
}

动态调度逻辑：

if target_worker overloaded:
    route job to other worker
else:
    use preferred worker from base rules

Overloaded 定义：

(cpu > 85% and running_jobs > threshold) or (mem > 90%)

3.3.4 “大任务 vs 小任务”公平性策略（防止饥饿）

如果连续到达 FPGA 综合任务，可能让 MCU 编译任务“饿死”。

因此加入 fairness queue：

small_job_queue
large_job_queue

并按比例选择：

每调度 1 个大任务，必须调度 2 个小任务

权重举例：

大任务：Vivado、Quartus

小任务：GCC 编译、MATLAB、ModelSim

3.3.5 任务亲和性（Task Affinity）

某些 pipeline 要求多个步骤在同一 Worker：

MATLAB 自动生成 HDL → Vivado 综合

MATLAB 生成 C → MCU 编译

LLM 会在 job.request.json 中加入：

"affinity": "worker1"

调度器必须严格遵循 affinity。

Affinity 也可以基于文件位置自动计算：

如果上一个 job 的输出位于 worker1/local_cache → 再次选择 worker1

3.3.6 调度算法伪代码（最终版）

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

调度输出：

{
  "selected_worker": "worker1",
  "reason": "vivado task; worker1 idle"
}

3.3.7 调度器的软失败与回退策略

当所有 Worker 满载或不健康时：

● 回退 1：延迟执行（排队）

job.status = "delayed"
job.retry_in = 10 seconds

调度器会在 10 秒后再次尝试。

● 回退 2：强制 fallback 至兼容节点

例如 MATLAB 可以在两个 Worker 任意执行：

if worker1 overloaded:
    route to worker2

● 回退 3：向 LLM 反馈可等待或可拆分

MCP 返回：

{"status": "queued", "message": "All workers busy; retry in 20s"}

LLM 通常会选择等待或重规划任务。

接下来将继续撰写：

3.4 Worker 运行环境（用户、权限、目录结构）

本节将详细描述 Worker 节点操作系统层面的标准化要求，使整个系统具备：

可复制性（相同配置的 Worker 可横向扩容）

可维护性（权限/目录统一，减少调试成本）

安全性（任务隔离、权限隔离）

一致性（工具路径、用户、环境变量在所有 Worker 上一致）

Worker 的运行环境包括以下五部分：

统一用户体系（builduser）

标准化目录结构

权限与组策略

环境变量与 profile 配置

SSH 配置与安全策略

下面逐项展开。

3.4.1 Worker 用户体系设计（builduser）

为了保证执行任务时的权限一致性，两台 Worker 必须形成统一的用户体系：

✔️ 定义一个专门执行工程任务的用户：

builduser:buildgroup
UID: 2001
GID: 2001
Home: /home/builduser

不使用 root 来执行任务，原因：

避免工程脚本误改系统文件

防止工程工具链写入 /opt 等系统目录

限制安全风险

创建方式：

sudo groupadd -g 2001 buildgroup
sudo useradd -m -u 2001 -g 2001 -s /bin/bash builduser

builduser 的作用：

所有任务执行均以 builduser 身份进行

MCP 调度器通过 SSH 使用该用户运行 worker_agent

/workspace 的所有文件均归属 builduser

工程工具（Vivado、Quartus、MATLAB）由 root 安装，但 builduser 可以读写工程区

3.4.2 Worker 标准化目录结构

所有 Worker 必须完全一致：

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

目录约束：

/workspace 必须能被所有 Worker 所读写（NFS/SSHFS/Samba 容器均可）

/opt 中工具仅 root 可写

/home/builduser 用于 Worker 本地执行时的缓存

3.4.3 权限与组策略

/workspace                → builduser:buildgroup (770)
/opt                      → root:root (755)
/home/builduser           → builduser:buildgroup (700)

原因：

builduser 可继续在 workspace 中生成中间文件

root 安装工具，但普通用户可执行

禁止其他系统用户访问工程文件（安全）

授权示例：

sudo chown -R builduser:buildgroup /workspace
sudo chmod -R 770 /workspace
sudo chmod -R 755 /opt

3.4.4 环境变量统一（/etc/profile.d/*.sh）

所有 Worker 的工具环境必须统一，确保执行结果一致、调度无差异。

典型文件：

/etc/profile.d/matlab.sh
/etc/profile.d/vivado.sh
/etc/profile.d/quartus.sh
/etc/profile.d/arm-gcc.sh

示例（matlab.sh）：

export MATLAB_ROOT=/opt/mathworks/matlab/R2024a
export PATH="$MATLAB_ROOT/bin:$PATH"

四大原则：

所有 Worker 一致性（必须完全相同）

PATH 统一前缀顺序（避免不同版本工具被误调用）

调度器无需关心 Worker 的环境差异

便于新 Worker 加入（只需拷贝 profile.d）

3.4.5 SSH 配置与安全策略（中控 → Worker）

Mac 中控需要无密码 SSH 登录 Worker 执行任务。

1）生成 SSH key（在 Mac 上）：

ssh-keygen -t ed25519 -C "mcp-controller"

生成：

~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub

2）将公钥复制到 Worker：

ssh-copy-id builduser@worker1
ssh-copy-id builduser@worker2

3）ssh 配置：

在 MacBook：

Host worker1
    HostName 192.168.1.101
    User builduser
    IdentityFile ~/.ssh/id_ed25519

Host worker2
    HostName 192.168.1.102
    User builduser
    IdentityFile ~/.ssh/id_ed25519

4）必要的 SSH 限制：

Worker 上 /etc/ssh/sshd_config 可配置：

AllowUsers builduser
PasswordAuthentication no
PermitRootLogin no

这样：

只有 builduser 可被中控调度

禁止密码登录

禁止 root 登录

提高整体安全性。

3.4.6 Worker 本地执行缓存与 Temp 目录

某些工具（MATLAB / Vivado / Quartus）默认会在用户目录下写大量文件：

/home/builduser/.matlab/
/home/builduser/.Xilinx/
/home/builduser/.config/

建议清理策略：

每周自动清理 .matlab 中旧日志

每次 Vivado 任务结束后清理 .Xilinx 中缓存

或在 profile 中重定向：

export XILINX_LOCAL_USERDATA=/workspace/.cache/builduser/xilinx

3.4.7 Worker 运行模式：守护进程 vs on-demand 调用

Worker Agent 有两种运行模式：

● 模式 A：SSH On-demand（最简单）

Mac 中控执行：

ssh worker1 "python3 /opt/worker_agent/run_job.py /workspace/jobs/<job-id>/request.json"

特点：

无需常驻进程，实现最简单

自动继承 SSH 环境变量

适合小规模系统

● 模式 B：常驻守护进程（进阶方案）

Worker 本地运行一个常驻的 worker_agent：

sudo systemctl enable worker-agent.service
sudo systemctl start  worker-agent

worker-agent.service 示例：

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

中控调度方式变为：

不再直接 ssh 执行脚本

改为写入 jobs/<job-id>/request.json 后，守护进程自动监听并执行

守护进程模式的优点：

能更好地控制并发

可维护本地队列

更好地向中控上报心跳状态

小结：

PoC 阶段推荐 SSH on-demand 模式（简单、易实现）

生产阶段可逐步升级为 守护进程 + 队列 模式

3.5 Worker Agent 设计（run_job.py / agent_daemon.py）

Worker Agent 是“连接中控调度逻辑与实际工程软件”的关键组件。它运行在 Worker 节点上，负责：

读取 /workspace/jobs/<job-id>/request.json

根据 tool 字段选择具体执行函数

调用 MATLAB / Vivado / Quartus / GCC 等 CLI 工具

收集日志与产物，生成 result.json

在守护进程模式下，维护任务队列与心跳

本节将从架构、数据流、错误处理与并发四个方面详细设计。

3.5.1 Agent 的总体架构

Agent 可以拆分为三个层次：

入口层（Entry / Main）

解析命令行参数（on-demand 模式）或轮询 jobs 目录（daemon 模式）

加载 request.json

构造标准化的 Job 对象

调度层（Dispatcher）

根据 job.tool 查找对应 handler

负责统一的日志捕获、异常包装、执行时间统计

工具执行层（Tool Handlers）

一组函数/类，每个负责一类工具：

handle_matlab(job)

handle_vivado(job)

handle_quartus(job)

handle_mcu_build(job)

...

整体伪代码：

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

3.5.2 request.json 的标准结构

调度器在 Mac 中控生成的 request.json 建议统一结构：

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

关键字段说明：

job_id：任务唯一编号

tool：决定使用哪个 handler

project：工程名称，用于定位 projects/<project>/

session_id：LLM 会话，用于偏好同一 Worker

affinity：强制调度到特定 Worker（可选）

params：具体工具相关参数，由 MCP tool schema 定义

3.5.3 result.json 的标准结构

Agent 执行结束后写入：

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

中控在收到 result.json 后：

向 LLM 返回结构化结果

写入 build manifest / pipeline 记录

3.5.4 Tool Handler 注册机制

为便于扩展，Agent 内部可以使用一个字典式注册表：

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

好处：

新增工具时只需添加一个 handler 函数 + 装饰器

不需要修改主调度逻辑

3.5.5 日志捕获与错误处理

每个 handler 都应遵循统一日志策略：

在 jobs/<job-id>/logs/ 下创建专用日志文件：

matlab.log

vivado.log

quartus.log

使用 subprocess.run 时将 stdout/stderr 重定向到该文件：

with open(log_path, "w") as f:
    subprocess.run(cmd, cwd=project_dir, stdout=f, stderr=f, check=False)

根据 returncode 决定任务状态：

if proc.returncode == 0:
    result["status"] = "success"
else:
    result["status"] = "failed"
    result["error"] = f"Command failed with code {proc.returncode}"

不在 Agent 内部解析详细报错，将解析交给 LLM：

LLM 可读取 log_file 内容

理解具体错误信息并给出修复策略

3.5.6 并发与队列（守护进程模式下）

在 agent_daemon.py 中，可以设计一个简单任务队列：

周期性扫描：

while True:
    pending_jobs = scan_jobs_directory()
    for job in pending_jobs:
        if not is_over_capacity():
            start_worker_thread(job)
    sleep(2)

使用线程或进程池执行任务：

from concurrent.futures import ThreadPoolExecutor

EXECUTOR = ThreadPoolExecutor(max_workers=MAX_JOBS)

EXECUTOR.submit(run_single_job, job_path)

is_over_capacity() 根据当前运行的 Vivado/Quartus/MATLAB 任务数量决定是否再接新任务。

3.5.7 心跳与 Worker 状态上报

守护进程模式下，Agent 可以定期写入 Worker 状态：

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

状态文件存放在：

/workspace/worker_status/worker1.json

Mac 中控定期读取这些状态文件，以实现 3.3 中描述的动态调度算法。

3.5.8 小结

本节定义了 Worker Agent 的：

三层架构（入口 / 调度 / handler）

request.json / result.json 标准

Tool Handler 注册和扩展机制

日志捕获与错误处理

守护进程下的并发队列

Worker 状态心跳上报

至此，Mac 中控 → Worker → 工具软件 的完整路径打通：

LLM → MCP tool → 生成 Job → 调度器选 Worker

Worker Agent 读取 Job → 调用工具 → 写回结果

LLM 继续根据结果进行下一步规划