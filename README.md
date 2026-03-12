# 🔬 LLMEmbeddingBO for Mg-Mediated Hydroalkylation
# Mg 介导加氢烷基化反应的 LLM 嵌入贝叶斯优化框架

<p align="center">
  <strong>利用大语言模型（LLM）的语义理解 + 贝叶斯优化（BO），快速找到最优反应条件</strong>
</p>

---

## 📖 目录

1. [项目背景与目标](#-项目背景与目标)
2. [我们在做什么？](#-我们在做什么)
3. [核心算法原理（小白版）](#-核心算法原理小白版)
4. [目录结构说明](#-目录结构说明)
5. [快速开始（新手推荐）](#-快速开始新手推荐)
6. [完整安装与配置](#-完整安装与配置)
7. [使用说明](#-使用说明)
8. [逐步上手路线图](#-逐步上手路线图)
9. [技术路线与优化思路](#-技术路线与优化思路)
10. [常见问题 FAQ](#-常见问题-faq)

---

## 🎯 项目背景与目标

### 我们在研究什么反应？

我们正在研究一种**由金属镁（Mg）介导的还原偶联反应（加氢烷基化）**：

```
取代苯乙烯 (0.2 mmol)
    +
碘代烷烃 (0.3 mmol)
    ──[Mg (5 eq), AcOH (8 eq), DMAc, 室温, 24 h]──→
    烷基苯类产物（双键加氢烷基化）
```

**基准条件**：Mg 粉（5当量）+ AcOH（8当量）在 DMAc 溶剂中，室温24小时

**存在的问题**：
- 反应高度依赖特定溶剂（DMAc，有生殖毒性）
- 需要大量还原剂和酸（不经济、不绿色）
- 某些含敏感官能团底物的收率仍有提升空间

### 我们想解决什么问题？

通过**智能优化算法**，自动搜索最优的反应条件组合，解决以下三个目标：

| 目标 | 说明 |
|------|------|
| 🏆 **收率最大化** | 调整还原剂/质子源/溶剂组合，寻找高产率条件 |
| 🌿 **绿色经济性** | 在保证产率的前提下，用绿色溶剂替代 DMAc，减少用量 |
| 🔬 **底物韧性提升** | 找到对各类官能团（-CF₃、-OMe、-Br、-Cl）都兼容的通用条件 |

---

## 🤔 我们在做什么？

### 问题的挑战

如果手动测试所有条件组合：
- 10种还原剂 × 10种质子源 × 12种溶剂 = **1200 个实验**
- 每个实验耗时约1天 → **需要 3年多**！

### 我们的解决方案：LLMEmbeddingBO

```
                     ┌─────────────────────────────────────────┐
  化学物质名称       │         LLM（大语言模型）                │
  "AcOH", "DMAc"  → │  理解化学含义，生成"化学语义向量"        │ → 1536维嵌入向量
  "Mg powder"      │  （类似 Word2Vec，但懂化学）              │
                     └─────────────────────────────────────────┘
                                       ↓
                     ┌─────────────────────────────────────────┐
                     │         贝叶斯优化（BO）                  │
  历史实验结果      → │  高斯过程预测哪个条件可能产率最高        │ → 推荐下一个实验
  (产率数据)        │  采集函数平衡探索与利用                   │
                     └─────────────────────────────────────────┘
```

**关键优势**：
- LLM 能"理解"分子结构和化学性质，而不仅仅是把名字当作字符串
- 贝叶斯优化能从每次实验中学习，越做越聪明
- 通常只需 **20-50 个实验** 就能找到接近最优的条件

---

## 🧮 核心算法原理（小白版）

### 什么是嵌入向量（Embedding）？

把化学物质名称变成数字向量，让计算机能理解化学含义：

```
"AcOH" → LLM理解 → "弱有机酸，pKa 4.76，质子供体..." → [0.12, -0.34, 0.56, ...]（1536个数字）
"TFA"  → LLM理解 → "强有机酸，pKa 0.23，腐蚀性强..." → [0.09, -0.41, 0.48, ...]
"DMAc" → LLM理解 → "极性非质子溶剂，高供体数..." → [-0.23, 0.67, 0.12, ...]
```

语义相近的物质，向量也相近。这让优化器能**插值和泛化**！

### 什么是贝叶斯优化？

类比：你在一个黑暗的房间里寻找最高点，每次只能踩一脚测量高度。

```
轮次0（初始）：随机踩5个点，测量高度（=做初始实验，测产率）
轮次1：根据已知高度，GP模型预测哪里可能更高 → 选最有希望的点踩
轮次2：更新模型 → 再选 → 越来越精准
...
通常20-50轮后，找到最高点（≈最优条件）
```

**高斯过程（GP）** = 预测每个位置的"高度"和"不确定性"
**采集函数（UCB）** = 分数 = 预测高度 + κ × 不确定性（鼓励探索未知区域）

### 先运行演示脚本直观感受！

```bash
python examples/quick_demo.py
```

---

## 📁 目录结构说明

```
AI-assisted-PET-glycolysis/
│
├── 📄 README.md                    ← 你现在看的这个文件
├── 📄 requirements.txt             ← Python 依赖包列表
├── 📄 .gitignore                   ← Git 忽略文件配置
│
├── 📂 examples/                    ← 示例脚本（新手入口）
│   └── quick_demo.py               ← ★★★ 快速演示，无需API，直接运行！
│
├── 📂 tests/                       ← 环境测试脚本
│   ├── test_api_connection.py      ← 测试 OpenAI API 是否可用
│   ├── test_embedding.py           ← 测试嵌入功能是否正常
│   └── README.md                   ← 测试说明
│
├── 📂 src/                         ← 核心源代码
│   ├── llm_embedding_bo.py         ← ★ 主程序：LLM嵌入+贝叶斯优化
│   ├── llm_prompt_generation.py    ← LLM提示词生成（假设生成）
│   ├── data.py                     ← 嵌入向量获取与缓存
│   ├── utils.py                    ← 工具函数（API Key读取、采集函数）
│   ├── openai_api_key              ← [可选] 存放 API Key 的文件
│   │
│   └── 📂 data/                    ← 数据目录（★重要★）
│       ├── 数据文件夹说明.md          ← 详细说明每个文件的用途
│       ├── reductant.xlsx           ← 还原剂候选列表（可编辑）
│       ├── proton_source.xlsx       ← 质子源候选列表（可编辑）
│       ├── solvent.xlsx             ← 溶剂候选列表（可编辑）
│       ├── init_experiments.xlsx    ← ★ 初始实验数据（需要你填写！）
│       └── embeddings/              ← LLM嵌入缓存（自动生成）
│           └── 嵌入缓存说明.md
│
├── 📂 results/                     ← 实验结果
│   ├── 结果文件夹说明.md             ← 说明结果文件格式
│   └── sample_bo_results.xlsx       ← 样例结果文件（格式参考）
│
├── 📂 figures/                     ← 可视化图表
└── 📂 notebooks/                   ← Jupyter 分析笔记本
```

> **重要文件说明**：
> - `src/data/init_experiments.xlsx`：你的**初始实验数据**，格式见 [数据文件说明](#)
> - `src/data/reductant.xlsx`：定义**搜索空间**（哪些还原剂可以被推荐）
> - `src/llm_embedding_bo.py`：**核心程序**，直接运行即可获得推荐

---

## 🚀 快速开始（新手推荐）

### 第0步：先运行演示，理解原理（不需要任何配置！）

```bash
# 克隆仓库
git clone https://github.com/byLennert/AI-assisted-PET-glycolysis.git
cd AI-assisted-PET-glycolysis

# 安装依赖
pip install -r requirements.txt

# 运行演示脚本（无需 API Key，立刻看到结果）
python examples/quick_demo.py
```

**演示输出示例**：
```
============================================================
  Mg 介导加氢烷基化反应 — 贝叶斯优化快速演示
============================================================

搜索空间大小：
  还原剂 ×  5  种
  质子源 ×  5  种
  溶剂   ×  5  种
  总组合 = 125 种

【第一步】初始实验（人工选择的起始点）
----------------------------------------
  还原剂=Mg powder (325 mesh)  质子源=AcOH  溶剂=DMAc  产率=83.6%
  ...

  🏆 最优条件（经过5轮优化找到）：
     还原剂：Mg powder (325 mesh)
     质子源：AcOH
     溶剂：  DMAc
     产率：  83.6%

  仅测试了 8% 的搜索空间，就找到了产率≥84% 的条件！
```

---

## ⚙️ 完整安装与配置

### 1. 克隆仓库

```bash
git clone https://github.com/byLennert/AI-assisted-PET-glycolysis.git
cd AI-assisted-PET-glycolysis
```

### 2. 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 3. 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 配置 OpenAI API Key

**方法一（推荐，更安全）**：设置环境变量

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-your-api-key-here"

# Windows 命令行
set OPENAI_API_KEY=sk-your-api-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**方法二**：写入文件

```bash
echo "sk-your-api-key-here" > src/openai_api_key
```

> 💡 如何获取 API Key？
> 访问 [platform.openai.com](https://platform.openai.com) → 注册账号 → API Keys → Create new secret key

### 5. 验证配置

```bash
cd src
python ../tests/test_api_connection.py
```

---

## 📋 使用说明

### 步骤1：准备初始实验数据

编辑 `src/data/init_experiments.xlsx`，填入你已经做过的实验数据：

| Reductant | ProtonSource | Solvent | Yield | 备注 |
|-----------|--------------|---------|-------|------|
| Mg powder (325 mesh) | AcOH | DMAc | 85 | 基准条件 |
| Mg powder (325 mesh) | AcOH | DMF | 72 | 替换溶剂 |
| Zn powder | AcOH | DMAc | 30 | 替换还原剂 |
| ... | ... | ... | ... | ... |

> **重要**：Yield 列填写 0-100 之间的数字（产率百分比）

### 步骤2：确认候选化学品列表

检查 `src/data/` 中的 Excel 文件，确保你想测试的化学品都在列表中：

- `reductant.xlsx`：还原剂候选（如 Mg, Zn, Mn, Al...）
- `proton_source.xlsx`：质子源候选（如 AcOH, TFA, 丙酸...）
- `solvent.xlsx`：溶剂候选（如 DMAc, 2-MeTHF, EtOH...）

### 步骤3：运行贝叶斯优化，获取推荐实验

```bash
cd src
python llm_embedding_bo.py
```

**输出示例**：
```
============================================================
  Top 6 suggested experiments
============================================================
  1. Reductant=Mg powder (325 mesh), ProtonSource=Benzoic acid, Solvent=2-MeTHF  (acq=3.2145)
  2. Reductant=Mg powder (325 mesh), ProtonSource=AcOH, Solvent=DMSO  (acq=2.9872)
  3. Reductant=Zn-Cu couple, ProtonSource=AcOH, Solvent=DMAc  (acq=2.7631)
  ...
```

### 步骤4：做实验，记录产率

按照推荐的条件做实验，测量产率，记录到 `results/` 文件夹中。

### 步骤5：循环迭代

将新实验结果追加到 `src/data/init_experiments.xlsx`，再次运行步骤3。
通常经过 **3-8 轮**迭代，产率会明显提升！

---

## 🗺️ 逐步上手路线图

以下是一个**完整的项目推进计划**，帮助从零开始到拿到优化结果：

### 阶段一：理解工具（无需实验数据）⏱️ 1-2天

- [ ] **1.1** 阅读本 README，了解项目背景和原理
- [ ] **1.2** 运行 `python examples/quick_demo.py` 直观感受贝叶斯优化
- [ ] **1.3** 观察 quick_demo.py 的代码，理解 GP + 采集函数的逻辑
- [ ] **1.4** 配置好 Python 环境（`pip install -r requirements.txt`）

### 阶段二：准备环境（需要 API Key）⏱️ 0.5天

- [ ] **2.1** 注册 OpenAI 账号，获取 API Key
- [ ] **2.2** 配置 API Key（环境变量或文件）
- [ ] **2.3** 运行 `python tests/test_api_connection.py`，确认 API 正常
- [ ] **2.4** 运行 `python tests/test_embedding.py`，确认嵌入功能正常

### 阶段三：准备实验数据⏱️ 1-3天（视实验进展）

- [ ] **3.1** 确定研究目标（优先追求高产率？绿色溶剂？底物适用性？）
- [ ] **3.2** 设计初始实验矩阵（建议 8-15 个点）：
  - 基准条件（Mg + AcOH + DMAc）
  - 替换还原剂（Zn, Mn, Al 各1-2个）
  - 替换质子源（TFA, 丙酸各1-2个）
  - 替换溶剂（DMF, 2-MeTHF, EtOH 各1-2个）
- [ ] **3.3** 完成初始实验，测定产率
- [ ] **3.4** 将数据填入 `src/data/init_experiments.xlsx`

### 阶段四：第一次运行 BO⏱️ 0.5天

- [ ] **4.1** 确认候选列表（reductant.xlsx, proton_source.xlsx, solvent.xlsx）
- [ ] **4.2** 运行 `python src/llm_embedding_bo.py`
  - 第一次运行会调用 API 生成所有候选物质的嵌入向量（会花费几分钟和少量 API 费用）
  - 嵌入向量缓存到本地后，后续运行很快
- [ ] **4.3** 记录推荐的实验条件（Top 5-6 个）
- [ ] **4.4** 检查推荐结果是否合理（用化学直觉判断）

### 阶段五：迭代优化⏱️ 2-8轮（每轮约1-3天）

每轮迭代步骤：
1. 按照 BO 推荐条件做实验（可以做 Top 3-5 个）
2. 将结果追加到 `init_experiments.xlsx`
3. 重新运行 `python src/llm_embedding_bo.py`
4. 记录新的推荐条件

> **收敛判断**：当 BO 连续 2-3 轮推荐的条件都差不多，或者产率已达到你的目标时，可以停止

### 阶段六：结果分析与报告⏱️ 1-2天

- [ ] **6.1** 整理所有实验数据，保存到 `results/` 文件夹
- [ ] **6.2** 使用 `notebooks/analysis.ipynb` 绘制优化曲线
- [ ] **6.3** 确认最优条件的重现性（做3次平行实验）
- [ ] **6.4** 对比最优条件与基准条件的绿色指标（E-factor、溶剂用量）

---

## 💡 技术路线与优化思路

### A. 当前框架说明

```
输入：化学物质名称
  → LLM 生成化学描述
  → text-embedding-3-small 将描述变为 1536 维向量
  → 随机投影降到 20 维（加速 GP 计算）
  → GP 拟合历史数据
  → UCB 采集函数打分
  → 输出：最优推荐条件
```

### B. LLM Embedding 策略改进（进阶）

在当前基础上，可以通过以下方式让嵌入向量更具化学意义：

**1. 语义增强**

除了名称，在提示词中加入物化参数，帮助 LLM 更准确理解：
```python
# 在 data.py 的 ask_llm_prompt 中添加：
"pKa: {pka_value}, Reduction potential: {e0_value}"
```

**2. 先验知识引入**

利用文献数据（镁介导还原反应、格氏试剂原位生成）作为贝叶斯先验：
- 在 `init_experiments.xlsx` 中加入文献中已知的优化条件
- 这相当于告诉优化器"这些区域值得多探索"

### C. 多目标优化（进阶）

修改目标函数，采用**权重评分制**：

$$\text{Score} = w_1 \cdot \text{Yield} + w_2 \cdot (1 - E\_\text{factor}) + w_3 \cdot \text{Scope\_Penalty}$$

其中：
- $w_1, w_2, w_3$：权重（如 0.5, 0.3, 0.2）
- $E\_\text{factor}$：环境因子（越低越绿色）
- $\text{Scope\_Penalty}$：不同底物间产率的标准差（越低代表适用性越广）

在 `src/llm_embedding_bo.py` 的 `register_by_name` 函数中，将 `target` 参数改为 `Score` 而非单纯 `Yield`。

### D. 绿色指标计算

| 指标 | 计算方式 | 绿色目标 |
|------|----------|----------|
| 溶剂评分 | 基于 CHEM21 溶剂选择指南 | 推荐 2-MeTHF, EtOH, H₂O |
| E-factor | (废物质量) / (产物质量) | 越低越好 |
| 还原剂用量 | 等当量数 | 从 5 eq 降到 2-3 eq |

---

## ❓ 常见问题 FAQ

**Q1：第一次运行会花多少钱？**

A：生成全部候选物质的嵌入向量（约 30 种化学品），预计花费 **0.5-2 美元**。之后使用缓存，基本免费。

**Q2：没有 OpenAI API Key 能用吗？**

A：核心的贝叶斯优化部分需要 API。但你可以先运行 `examples/quick_demo.py` 理解流程，再申请 API。

**Q3：我的产率数据是 NMR 产率还是分离产率？**

A：都可以，但整个项目中保持**统一**。建议使用分离产率（更贴近实际）。

**Q4：推荐的实验条件我觉得化学上不合理，要怎么办？**

A：可以：
1. 检查 `reductant.xlsx` 等文件，移除明显不合理的候选
2. 在 `init_experiments.xlsx` 中手动添加你认为有价值的条件（赋予合理产率）
3. 增加初始实验数量（至少 10 个），给 GP 更多信息

**Q5：如何加速计算？**

A：在 `src/llm_embedding_bo.py` 的 `LLMEmbeddingBO` 构造函数中增大 `random_embedding` 参数会降速但精度更高；减小则加速。默认值 20 是个平衡点。

**Q6：能不能用国内的 LLM（如 DeepSeek、通义千问）？**

A：可以！修改 `src/data.py` 中的 API 调用部分，更换客户端即可。这是一个很好的改进方向，特别是 API 费用和访问速度方面。

---

## 📦 依赖包说明

| 包名 | 用途 |
|------|------|
| `openai` | 调用 OpenAI API 生成嵌入和文本 |
| `numpy` | 数值计算、矩阵操作 |
| `pandas` | Excel 数据读写 |
| `scikit-learn` | 高斯过程回归 |
| `scipy` | 采集函数计算（正态分布） |
| `openpyxl` | Excel 文件读写 |
| `bayesian-optimization` | 辅助工具（`ensure_rng`） |
| `matplotlib`, `seaborn` | 可视化 |

```bash
pip install -r requirements.txt
```

---

## 📄 License

本项目基于 MIT License 开源。详见 [LICENSE](LICENSE) 文件。

