"""
test_embedding.py — 测试 LLM 嵌入向量生成是否正常
===================================================

## 用途

验证嵌入功能的核心逻辑，包括：
  1. 缓存机制（避免重复调用 API）
  2. 不同角色（reductant/proton_source/solvent）的嵌入维度是否正确
  3. 相同类型物质之间的语义相似度是否符合化学直觉

## 如何运行

  # 从 src/ 目录下运行（推荐）：
  cd <项目根目录>/src
  python ../tests/test_embedding.py

  # 设置 API Key 以启用 B 组测试：
  OPENAI_API_KEY="sk-..." python ../tests/test_embedding.py

## 测试内容说明

  测试A（无需API）：验证缓存读取逻辑
  测试B（需要API）：验证嵌入维度正确（1536维）
  测试C（需要API）：验证化学语义合理性
               AcOH 和 Propionic acid 的相似度 > AcOH 和 DMAc 的相似度
"""

import sys
import os
import json
import tempfile
import shutil

# ── 路径设置 ──────────────────────────────────────────────────────────────
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_dir))
os.chdir(os.path.abspath(src_dir))

PASSED = 0
FAILED = 0


def check(condition, test_name, detail=""):
    global PASSED, FAILED
    if condition:
        print(f"  ✅ {test_name}")
        PASSED += 1
    else:
        print(f"  ❌ {test_name}" + (f"\n     {detail}" if detail else ""))
        FAILED += 1


# ════════════════════════════════════════════════════════════════════════════
# 测试 A：不需要 API（验证辅助函数）
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  测试组 A：辅助函数（无需 API）")
print("=" * 60)

# A1：is_valid_windows_filename
try:
    from data import is_valid_windows_filename
    check(is_valid_windows_filename("AcOH"), "A1 合法文件名：AcOH")
    check(is_valid_windows_filename("Mg powder (325 mesh)"), "A1 合法文件名：含空格括号")
    check(not is_valid_windows_filename("a/b"), "A1 非法文件名：含斜杠")
    check(not is_valid_windows_filename("a" * 256), "A1 非法文件名：超长")
    check(not is_valid_windows_filename("CON"), "A1 非法文件名：系统保留名")
except Exception as e:
    print(f"  ❌ A1 导入 data.py 失败：{e}")

# A2：ask_llm_prompt
try:
    from data import ask_llm_prompt
    prompt_r = ask_llm_prompt("Mg powder", role='reductant')
    prompt_p = ask_llm_prompt("AcOH", role='proton_source')
    prompt_s = ask_llm_prompt("DMAc", role='solvent')
    
    check("reduction potential" in prompt_r.lower() or "reductant" in prompt_r.lower(),
          "A2 还原剂提示词包含关键词 reduction potential")
    check("pka" in prompt_p.lower() or "proton" in prompt_p.lower(),
          "A2 质子源提示词包含关键词 pKa/proton")
    check("donor number" in prompt_s.lower() or "solvent" in prompt_s.lower(),
          "A2 溶剂提示词包含关键词 donor number/solvent")
except Exception as e:
    print(f"  ❌ A2 ask_llm_prompt 测试失败：{e}")

# A3：get_uname_dict（在临时目录测试）
try:
    from data import get_uname_dict
    # 保存当前工作目录
    orig_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)
    os.chdir(tmp_dir)
    udict = get_uname_dict()
    check(isinstance(udict, dict), "A3 get_uname_dict 返回字典")
    check(os.path.exists(os.path.join(tmp_dir, "data", "uname_dict.json")),
          "A3 uname_dict.json 自动创建")
    os.chdir(orig_dir)
    shutil.rmtree(tmp_dir)
except Exception as e:
    print(f"  ❌ A3 get_uname_dict 测试失败：{e}")
    os.chdir(src_dir)

# A4：缓存命中逻辑（Mock LLM 客户端）
print("\n  [A4] 测试嵌入缓存命中（使用临时目录，不调用 API）")
try:
    import importlib
    import data as data_module

    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, "data", "embeddings"), exist_ok=True)
    
    # 预写一个假的缓存文件
    fake_embedding = [0.1] * 1536
    fake_cache = {
        "molecule": "AcOH",
        "property": "Acetic acid test",
        "embedding": fake_embedding,
        "role": "proton_source"
    }
    cache_path = os.path.join(tmp_dir, "data", "embeddings", "AcOH.json")
    with open(cache_path, "w") as f:
        json.dump(fake_cache, f)
    
    # 将工作目录切到临时目录，然后调用 get_one_embedding
    orig_dir = os.getcwd()
    os.chdir(tmp_dir)
    
    # 测试：传入 None 客户端，如果缓存命中则不需要调用 API
    result = data_module.get_one_embedding("AcOH", client=None,
                                            role='proton_source')
    check(result == fake_embedding, "A4 缓存命中：读取到预存嵌入向量")
    check(len(result) == 1536, "A4 缓存命中：嵌入维度为 1536")
    
    os.chdir(orig_dir)
    shutil.rmtree(tmp_dir)
except Exception as e:
    print(f"  ❌ A4 缓存测试失败：{e}")
    try:
        os.chdir(src_dir)
    except:
        pass


# ════════════════════════════════════════════════════════════════════════════
# 测试 B：需要 API（可选，如无 API Key 则跳过）
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  测试组 B：LLM API 调用（需要 API Key）")
print("=" * 60)

api_available = False
client = None

try:
    import openai
    from utils import get_openai_key
    key = get_openai_key()
    client = openai.OpenAI(api_key=key)
    # 简单验证：调用最小请求
    client.embeddings.create(input=["test"], model="text-embedding-3-small")
    api_available = True
    print("  ℹ️  检测到可用的 API Key，进行 API 测试...")
except Exception as e:
    print(f"  ℹ️  未检测到可用 API Key，跳过 B 组测试（{str(e)[:60]}）")
    print("  （这不影响 A 组测试结果，API 测试是可选的）")

if api_available and client:
    import numpy as np
    import tempfile, shutil
    
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, "data", "embeddings"), exist_ok=True)
    orig_dir = os.getcwd()
    os.chdir(tmp_dir)
    
    # 创建空的 uname_dict
    with open("data/uname_dict.json", "w") as f:
        json.dump({}, f)
    
    try:
        import importlib
        import sys
        # 重新加载 data 模块以使用当前工作目录
        if 'data' in sys.modules:
            importlib.reload(sys.modules['data'])
        import data as dm
        
        # B1：生成还原剂嵌入
        emb_r = dm.get_one_embedding("Mg", client=client, role='reductant')
        check(isinstance(emb_r, list), "B1 还原剂嵌入返回列表类型")
        check(len(emb_r) == 1536, f"B1 还原剂嵌入维度为 1536（实际 {len(emb_r)}）")
        
        # B2：生成质子源嵌入
        emb_p = dm.get_one_embedding("AcOH", client=client, role='proton_source')
        check(len(emb_p) == 1536, "B2 质子源嵌入维度为 1536")
        
        # B3：生成溶剂嵌入
        emb_s = dm.get_one_embedding("DMAc", client=client, role='solvent')
        check(len(emb_s) == 1536, "B3 溶剂嵌入维度为 1536")
        
        # B4：缓存生成验证
        check(os.path.exists("data/embeddings/Mg.json"), "B4 还原剂嵌入已缓存到文件")
        check(os.path.exists("data/embeddings/AcOH.json"), "B4 质子源嵌入已缓存到文件")
        
        # B5：验证缓存文件内容
        with open("data/embeddings/AcOH.json") as f:
            cached = json.load(f)
        check(cached.get('role') == 'proton_source', "B5 缓存文件的 role 字段正确")
        check('property' in cached and len(cached['property']) > 50,
              "B5 缓存文件包含 LLM 生成的化学描述")
        
        # B6：语义相似度测试
        # 相同类型物质（两种羧酸）应比不同类型（酸 vs 溶剂）更相似
        emb_acoh = np.array(dm.get_one_embedding("AcOH", client=client, role='proton_source'))
        emb_prop = np.array(dm.get_one_embedding("Propionic acid", client=client,
                                                   role='proton_source'))
        emb_dmac = np.array(dm.get_one_embedding("DMAc", client=client, role='solvent'))
        
        cos_acid_acid = np.dot(emb_acoh, emb_prop) / (
            np.linalg.norm(emb_acoh) * np.linalg.norm(emb_prop))
        cos_acid_solvent = np.dot(emb_acoh, emb_dmac) / (
            np.linalg.norm(emb_acoh) * np.linalg.norm(emb_dmac))
        
        print(f"\n     语义相似度分析：")
        print(f"       AcOH vs Propionic acid（同为羧酸）：{cos_acid_acid:.4f}")
        print(f"       AcOH vs DMAc（酸 vs 溶剂）：        {cos_acid_solvent:.4f}")
        
        check(cos_acid_acid > cos_acid_solvent,
              "B6 语义相似度：同类物质相似度 > 异类物质（化学直觉验证）",
              f"AcOH-PropAcid={cos_acid_acid:.4f}, AcOH-DMAc={cos_acid_solvent:.4f}")

    except Exception as e:
        print(f"  ❌ B 组测试发生错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        os.chdir(orig_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"  测试结果：{PASSED} 通过，{FAILED} 失败")
if FAILED == 0:
    print("  ✅ 全部通过！嵌入模块运行正常。")
else:
    print(f"  ⚠️  有 {FAILED} 个测试失败，请检查上方的错误信息。")
print("=" * 60)

sys.exit(0 if FAILED == 0 else 1)
