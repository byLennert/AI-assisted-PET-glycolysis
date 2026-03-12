"""
quick_demo.py — 贝叶斯优化快速入门演示（无需 API Key）
==========================================================

## 这个文件是做什么的？

本脚本用**模拟数据**演示 LLMEmbeddingBO 的**完整核心流程**，
无需调用 OpenAI API，也无需真实实验数据，新手可直接运行。

## 适合谁？

- 对贝叶斯优化（Bayesian Optimization）不熟悉的化学研究人员
- 想快速理解程序运行逻辑的同学
- 想验证代码能否在本机运行的开发者

## 如何运行？

  # 克隆仓库后，在项目根目录下运行：
  python examples/quick_demo.py

## 核心概念速查

  贝叶斯优化 = 高斯过程（GP）+ 采集函数（Acquisition Function）
  
  - 高斯过程（GP）：用已做过的实验数据，预测未做实验的"可能产率"和"不确定性"
  - 采集函数（UCB）：综合"预测高"和"不确定高"两个因素，给每个候选条件打分
  - 每轮选分数最高的条件去做实验 → 更新模型 → 再推荐 → 逐步收敛最优
"""

import numpy as np
import sys
import os

# ──────────────────────────────────────────────────────────────────────────────
# 依赖检查
# ──────────────────────────────────────────────────────────────────────────────
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from scipy.stats import norm
except ImportError:
    print("缺少依赖包！请运行：pip install scikit-learn scipy")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# 第一步：定义搜索空间（候选化学品）
# ──────────────────────────────────────────────────────────────────────────────

# 真实项目中，这些名称来自 src/data/reductant.xlsx 等文件
# 这里用简单列表模拟
REDUCTANTS = [
    "Mg powder (325 mesh)",  # 基准
    "Mg powder (100 mesh)",
    "Mg turnings",
    "Zn powder",
    "Mn powder",
]

PROTON_SOURCES = [
    "AcOH",            # 基准（pKa 4.76）
    "Propionic acid",  # pKa 4.87
    "Formic acid",     # pKa 3.74
    "Benzoic acid",    # pKa 4.20
    "NH4Cl",           # pKa 9.25（弱酸）
]

SOLVENTS = [
    "DMAc",       # 基准（极性非质子，毒性高）
    "DMF",        # 常见极性非质子
    "2-MeTHF",    # 绿色替代
    "EtOH",       # 绿色醇类
    "MeCN",       # 中等极性
]

print("=" * 60)
print("  Mg 介导加氢烷基化反应 — 贝叶斯优化快速演示")
print("=" * 60)
print(f"\n搜索空间大小：")
print(f"  还原剂 ×  {len(REDUCTANTS)}  种")
print(f"  质子源 ×  {len(PROTON_SOURCES)}  种")
print(f"  溶剂   ×  {len(SOLVENTS)}  种")
print(f"  总组合 = {len(REDUCTANTS) * len(PROTON_SOURCES) * len(SOLVENTS)} 种\n")


# ──────────────────────────────────────────────────────────────────────────────
# 第二步：用随机向量模拟"嵌入向量"
# ──────────────────────────────────────────────────────────────────────────────
# 真实项目中，嵌入向量由 LLM 根据化学物质描述生成（1536维）
# 这里用随机向量（4维）演示，原理完全相同

np.random.seed(42)  # 固定随机种子，保证每次运行结果相同

EMB_DIM = 4  # 演示用低维向量（真实项目用 1536 维）

# 为每种化学品分配一个随机"特征向量"
# 类比：化学品的"化学特征"被压缩成一个数字向量
reductant_emb = {name: np.random.randn(EMB_DIM) for name in REDUCTANTS}
proton_emb    = {name: np.random.randn(EMB_DIM) for name in PROTON_SOURCES}
solvent_emb   = {name: np.random.randn(EMB_DIM) for name in SOLVENTS}


def get_feature_vector(reductant, proton_source, solvent):
    """
    将三种化学品的特征向量拼接为一个联合特征向量。
    
    真实项目中：每个向量是 1536 维，拼接后是 4608 维
    演示项目中：每个向量是 4 维，拼接后是 12 维
    """
    return np.concatenate([
        reductant_emb[reductant],
        proton_emb[proton_source],
        solvent_emb[solvent]
    ])


# ──────────────────────────────────────────────────────────────────────────────
# 第三步：定义"真实产率"函数（黑盒）
# ──────────────────────────────────────────────────────────────────────────────
# 真实实验中，这个函数就是"去实验室做实验"
# 这里用一个简单的数学函数模拟（新手不需要理解细节）

# 真实最优条件（我们假装不知道，让 BO 去发现）
BEST_REDUCTANT    = "Mg powder (325 mesh)"
BEST_PROTON_SOURCE = "AcOH"
BEST_SOLVENT      = "DMAc"

def simulate_yield(reductant, proton_source, solvent):
    """
    模拟实验产率（黑盒函数，只有做实验才能知道结果）。
    
    真实应用中，这个函数不存在——你需要去实验室做实验，然后手动填入产率。
    """
    # 基础产率
    base_yield = 85.0
    
    # 还原剂惩罚（偏离最优条件的处罚）
    reductant_penalty = {
        "Mg powder (325 mesh)": 0,   # 最优
        "Mg powder (100 mesh)": -15,
        "Mg turnings":          -30,
        "Zn powder":            -55,
        "Mn powder":            -60,
    }.get(reductant, -50)
    
    # 质子源惩罚
    proton_penalty = {
        "AcOH":           0,    # 最优
        "Propionic acid": -5,
        "Benzoic acid":   -12,
        "Formic acid":    -20,
        "NH4Cl":          -45,
    }.get(proton_source, -30)
    
    # 溶剂惩罚
    solvent_penalty = {
        "DMAc":    0,    # 最优（但毒性高）
        "DMF":     -13,
        "MeCN":    -25,
        "2-MeTHF": -37,
        "EtOH":    -47,
    }.get(solvent, -40)
    
    # 加入少量随机噪声（模拟实验误差）
    noise = np.random.normal(0, 3)
    
    yield_val = base_yield + reductant_penalty + proton_penalty + solvent_penalty + noise
    return max(0.0, min(100.0, yield_val))  # 产率限制在 0-100%


# ──────────────────────────────────────────────────────────────────────────────
# 第四步：初始化贝叶斯优化——先做几个实验
# ──────────────────────────────────────────────────────────────────────────────

print("【第一步】初始实验（人工选择的起始点）")
print("-" * 40)

# 初始实验：手动选几个条件做实验
init_conditions = [
    ("Mg powder (325 mesh)", "AcOH",          "DMAc"),    # 基准
    ("Zn powder",            "AcOH",          "DMAc"),    # 换还原剂
    ("Mg powder (325 mesh)", "NH4Cl",         "DMAc"),    # 换质子源
    ("Mg powder (325 mesh)", "AcOH",          "EtOH"),    # 换溶剂（绿色）
    ("Mg turnings",          "Propionic acid","2-MeTHF"), # 全换
]

# 存储观测数据
X_observed = []  # 特征向量
y_observed = []  # 产率（目标值）
conditions_observed = []  # 条件记录

for red, ps, sol in init_conditions:
    x = get_feature_vector(red, ps, sol)
    y = simulate_yield(red, ps, sol)
    X_observed.append(x)
    y_observed.append(y)
    conditions_observed.append((red, ps, sol))
    print(f"  还原剂={red:<25} 质子源={ps:<16} 溶剂={sol:<8}  产率={y:.1f}%")

print(f"\n  初始最高产率: {max(y_observed):.1f}%\n")


# ──────────────────────────────────────────────────────────────────────────────
# 第五步：贝叶斯优化主循环
# ──────────────────────────────────────────────────────────────────────────────

# 高斯过程模型（GP）
gp = GaussianProcessRegressor(
    kernel=Matern(nu=2.5),  # Matern核，适合非平滑函数
    alpha=1e-6,             # 数值稳定性参数
    normalize_y=True,       # 标准化目标值
    n_restarts_optimizer=5, # 优化器重启次数（越多越精确但越慢）
)

N_ROUNDS = 5  # 优化轮数（真实实验中每轮需要去实验室做实验）

print("【第二步】贝叶斯优化迭代")
print("=" * 60)

for round_num in range(1, N_ROUNDS + 1):
    print(f"\n  第 {round_num} 轮优化")
    print("  " + "-" * 50)
    
    # ── 拟合高斯过程模型 ────────────────────────────────────────────────────
    # GP 从历史数据中"学习"哪些条件可能产率更高
    X = np.array(X_observed)
    y = np.array(y_observed)
    gp.fit(X, y)
    
    # ── 采集函数：UCB（上置信界） ───────────────────────────────────────────
    # UCB = 预测均值 + kappa × 预测标准差
    # 这个公式平衡了"利用"（均值高）和"探索"（不确定性高）
    kappa = 2.576  # 探索-利用权衡参数（越大越倾向探索未知区域）
    y_max = max(y_observed)
    
    # 遍历所有未测试的组合，计算采集函数值
    best_acq = -np.inf
    best_condition = None
    
    candidate_scores = []
    
    for red in REDUCTANTS:
        for ps in PROTON_SOURCES:
            for sol in SOLVENTS:
                if (red, ps, sol) in conditions_observed:
                    continue  # 跳过已测试的组合
                
                x = get_feature_vector(red, ps, sol)
                mean, std = gp.predict(x.reshape(1, -1), return_std=True)
                
                # UCB 采集函数（越高 = 越值得做实验）
                ucb = float((mean + kappa * std).ravel()[0])
                candidate_scores.append((ucb, red, ps, sol))
    
    # 按 UCB 值排序，选最高的推荐
    candidate_scores.sort(reverse=True)
    
    # 显示 Top-3 推荐
    print(f"  Top-3 推荐实验（UCB采集函数值越高越推荐）：")
    for rank, (acq, red, ps, sol) in enumerate(candidate_scores[:3], 1):
        print(f"    {rank}. {red} + {ps} + {sol}  (UCB={acq:.3f})")
    
    # ── "做实验"：测量最推荐条件的产率 ────────────────────────────────────
    best_acq_val, best_red, best_ps, best_sol = candidate_scores[0]
    new_yield = simulate_yield(best_red, best_ps, best_sol)
    
    print(f"\n  ✅ 选择第1推荐做实验：")
    print(f"     {best_red} + {best_ps} + {best_sol}")
    print(f"     实测产率 = {new_yield:.1f}%  （历史最高 = {max(y_observed):.1f}%）")
    
    # 将新结果加入观测数据
    X_observed.append(get_feature_vector(best_red, best_ps, best_sol))
    y_observed.append(new_yield)
    conditions_observed.append((best_red, best_ps, best_sol))
    
    if new_yield > y_max:
        print(f"     🎉 新纪录！产率提升了 {new_yield - y_max:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 第六步：显示最终结果
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  优化结果汇总")
print("=" * 60)

best_idx = np.argmax(y_observed)
best_red, best_ps, best_sol = conditions_observed[best_idx]
best_yield = y_observed[best_idx]

print(f"\n  🏆 最优条件：")
print(f"     还原剂：{best_red}")
print(f"     质子源：{best_ps}")
print(f"     溶剂：  {best_sol}")
print(f"     产率：  {best_yield:.1f}%")

print(f"\n  优化过程中测试了 {len(y_observed)} 个条件（共 {len(REDUCTANTS)*len(PROTON_SOURCES)*len(SOLVENTS)} 个）")
print(f"  仅测试了 {len(y_observed)/len(REDUCTANTS)/len(PROTON_SOURCES)/len(SOLVENTS)*100:.0f}% 的搜索空间，")
print(f"  就找到了产率 ≥{best_yield:.0f}% 的条件！")

print("\n" + "=" * 60)
print("  演示完成！真实项目使用方式与此相同，区别在于：")
print("  1. 嵌入向量由 LLM 生成（需要 API Key）")
print("  2. 产率由真实实验测定（需要去实验室做实验）")
print("  3. 搜索空间更大（数百种候选化学品）")
print("=" * 60)
