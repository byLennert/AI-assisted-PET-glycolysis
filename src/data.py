import pandas as pd
import numpy as np
import os
from utils import get_openai_key
from openai import OpenAI
import json
import re


# ────────────────────────────────────────────────────────────────────────────
# 提示词生成
# ────────────────────────────────────────────────────────────────────────────

def ask_llm_prompt(name, role='reductant'):
    """
    根据化学物质的角色（role）生成面向 LLM 的化学描述提示词。
    
    role 可选值：
      - 'reductant'     : 还原剂（金属粉末，如 Mg, Zn）
      - 'proton_source' : 质子源（有机酸或弱酸，如 AcOH, TFA）
      - 'solvent'       : 溶剂（如 DMAc, 2-MeTHF, EtOH）
    
    为什么要根据角色提问？
    不同角色的化学物质在反应中扮演不同功能。给 LLM 更具体的上下文，
    能让其生成更贴合反应机理的化学描述，从而使嵌入向量更有化学意义。
    """
    context = {
        'reductant': (
            "In the context of a Mg-mediated reductive hydroalkylation reaction "
            "(combining styrenes with alkyl iodides at room temperature), "
            "describe the chemical and physical properties of {name} as a metallic reductant. "
            "Include: reduction potential (E0), particle size effects, surface reactivity, "
            "ability to generate carbon radicals via SET with alkyl iodides, and any known "
            "compatibility issues with functional groups."
        ),
        'proton_source': (
            "In the context of a Mg-mediated reductive hydroalkylation reaction, "
            "describe the chemical properties of {name} as a proton source. "
            "Include: pKa value, role in protonating organometallic intermediates, "
            "effect on metal surface activation, stoichiometry considerations, "
            "and potential side reactions (e.g., premature protonation, ester formation)."
        ),
        'solvent': (
            "In the context of a Mg-mediated reductive hydroalkylation reaction, "
            "describe the solvent properties of {name}. "
            "Include: polarity, donor number (DN), dielectric constant, "
            "ability to solvate metal surfaces and radical intermediates, "
            "green chemistry metrics (toxicity, renewability), and "
            "Hansen solubility parameters if available."
        ),
    }
    template = context.get(role, "Please give me some chemical knowledge and properties of {name}.")
    return template.format(name=name)


# ────────────────────────────────────────────────────────────────────────────
# 核心嵌入函数
# ────────────────────────────────────────────────────────────────────────────

def get_one_embedding(text, client, embedding_model='text-embedding-3-small',
                      chat_model='gpt-4o', role='reductant', type=None):
    """
    为单个化学物质生成 LLM 语义嵌入向量。

    工作流程：
      1. 检查本地缓存（data/embeddings/文件夹）→ 有则直接返回，节省 API 费用
      2. 若无缓存 → 调用 LLM 生成化学描述文本 → 对该文本生成嵌入向量
      3. 将结果保存到本地缓存，供下次使用

    参数说明：
      text             : 化学物质名称（如 "AcOH", "Mg powder (325 mesh)"）
      client           : OpenAI 客户端对象
      embedding_model  : 用于生成嵌入的模型（默认 text-embedding-3-small，1536维）
      chat_model       : 用于生成化学描述的对话模型（默认 gpt-4o）
      role             : 该物质的角色（'reductant'/'proton_source'/'solvent'）
      type             : 旧版兼容参数，优先使用 role

    返回：
      list[float]：1536 维嵌入向量
    """
    # 兼容旧版 type 参数（'acid' → 'reductant'，'base' → 'proton_source'）
    if type is not None:
        type_to_role = {'acid': 'reductant', 'base': 'proton_source', 'solvent': 'solvent'}
        role = type_to_role.get(type, role)

    # ── 步骤1：检查本地缓存 ──────────────────────────────────────────────────
    cache_file = f"data/embeddings/{text}.json"
    if os.path.exists(cache_file):
        cached = json.load(open(cache_file, "r"))
        return cached['embedding']

    # 检查文件名映射字典（处理含特殊字符的名称）
    uname_dict = get_uname_dict()
    if text in uname_dict:
        mapped_file = f"data/embeddings/{uname_dict[text]}.json"
        if os.path.exists(mapped_file):
            return json.load(open(mapped_file, "r"))['embedding']

    # ── 步骤2：调用 LLM 生成化学描述 ────────────────────────────────────────
    prompt = ask_llm_prompt(text, role=role)
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system",
             "content": "You are an expert organic chemist specializing in radical reactions."},
            {"role": "user", "content": prompt}
        ]
    )
    description = completion.choices[0].message.content

    # ── 步骤3：对描述文本生成嵌入向量 ───────────────────────────────────────
    embedding = client.embeddings.create(
        input=[description], model=embedding_model
    ).data[0].embedding

    # ── 步骤4：保存到本地缓存 ────────────────────────────────────────────────
    json_data = {
        "molecule": text,
        "property": description,  # LLM 生成的化学性质描述（供调试/理解使用）
        "embedding": embedding,
        "role": role               # 统一使用 role 字段（新版格式）
    }

    os.makedirs("data/embeddings", exist_ok=True)
    if is_valid_windows_filename(text):
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)
    else:
        # 文件名含特殊字符时，用 'name0', 'name1' 等替代
        uname_dict = get_uname_dict()
        value = 'name' + str(len(uname_dict))
        uname_dict[text] = value
        with open(f"data/embeddings/{value}.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)
        with open('data/uname_dict.json', 'w', encoding="utf-8") as f:
            json.dump(uname_dict, f, ensure_ascii=False)

    return embedding


def get_uname_dict():
    if not os.path.exists('data/uname_dict.json'):
        data = {}
        json.dump(data, open('data/uname_dict.json', 'w'))
        return data
    return json.load(open('data/uname_dict.json', 'r'))


def is_valid_windows_filename(filename):
    # 检查长度
    if len(filename) > 255:
        return False
    # 检查特殊字符
    invalid_chars = r'[<>:"/\\|?*]'
    if re.search(invalid_chars, filename):
        return False
    # 检查是否与系统保留名称相同
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
        'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3',
        'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    if filename.upper() in reserved_names:
        return False
    # 检查是否以空格或点结束
    if filename.endswith(' ') or filename.endswith('.'):
        return False
    return True
