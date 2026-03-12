"""
test_api_connection.py — 测试 OpenAI API 连接是否正常
=======================================================

## 用途

在开始正式实验之前，用此脚本快速验证：
  1. API Key 是否正确配置
  2. 网络是否能访问 OpenAI 服务
  3. 指定的模型是否可用

## 如何运行

  cd <项目根目录>
  python tests/test_api_connection.py

  # 或者在 src/ 目录下运行：
  cd src
  python ../tests/test_api_connection.py

  或者设置 API Key 后运行：
  OPENAI_API_KEY="sk-..." python tests/test_api_connection.py

## 预期输出（成功）

  ✅ 测试1通过：API Key 读取成功
  ✅ 测试2通过：OpenAI 客户端创建成功
  ✅ 测试3通过：对话模型 gpt-4o 可用，响应正常
  ✅ 测试4通过：嵌入模型 text-embedding-3-small 可用
  ✅ 所有测试通过！环境配置正确，可以开始使用。
"""

import sys
import os

# ── 设置路径（从任意目录运行） ─────────────────────────────────────────────
# 将 src/ 加入路径，以便导入 utils.py
src_dir = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_dir))
os.chdir(os.path.abspath(src_dir))  # 切换到 src/ 目录（utils 需要相对路径）


def test_api_key():
    """测试1：检查 API Key 是否可以读取"""
    print("\n[测试1] 检查 API Key 配置...")
    try:
        from utils import get_openai_key
        key = get_openai_key()
        if not key.startswith("sk-"):
            print(f"  ⚠️  警告：API Key 格式不标准（应以 'sk-' 开头）")
            print(f"      当前值：{key[:10]}...")
            return False
        print(f"  ✅ 测试1通过：API Key 读取成功（{key[:10]}...）")
        return True
    except FileNotFoundError as e:
        print(f"  ❌ 测试1失败：{e}")
        print("\n  解决方法：")
        print("    方法1（推荐）：设置环境变量")
        print("      Linux/Mac: export OPENAI_API_KEY='sk-...'")
        print("      Windows:   set OPENAI_API_KEY=sk-...")
        print("    方法2：将 API Key 写入文件")
        print("      echo 'sk-...' > src/openai_api_key")
        return False
    except Exception as e:
        print(f"  ❌ 测试1失败（未知错误）：{e}")
        return False


def test_client_creation():
    """测试2：检查能否创建 OpenAI 客户端"""
    print("\n[测试2] 创建 OpenAI 客户端...")
    try:
        import openai
        from utils import get_openai_key
        key = get_openai_key()
        client = openai.OpenAI(api_key=key)
        print("  ✅ 测试2通过：OpenAI 客户端创建成功")
        return client
    except ImportError:
        print("  ❌ 测试2失败：openai 包未安装")
        print("      解决方法：pip install openai")
        return None
    except Exception as e:
        print(f"  ❌ 测试2失败：{e}")
        return None


def test_chat_model(client, model='gpt-4o'):
    """测试3：测试对话模型是否可用"""
    print(f"\n[测试3] 测试对话模型 {model}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user",
                 "content": "Reply with exactly: 'API connection successful'"}
            ],
            max_tokens=20
        )
        reply = response.choices[0].message.content.strip()
        if "successful" in reply.lower() or "connection" in reply.lower():
            print(f"  ✅ 测试3通过：对话模型 {model} 可用，响应：'{reply}'")
            return True
        else:
            print(f"  ✅ 测试3通过：对话模型 {model} 可用，响应：'{reply}'")
            return True
    except Exception as e:
        error_str = str(e)
        if "model_not_found" in error_str or "does not exist" in error_str:
            print(f"  ❌ 测试3失败：模型 {model} 不可用（可能需要升级账号）")
            print(f"      尝试改用 gpt-3.5-turbo？")
        elif "insufficient_quota" in error_str:
            print(f"  ❌ 测试3失败：API 额度不足，请充值")
        elif "invalid_api_key" in error_str:
            print(f"  ❌ 测试3失败：API Key 无效，请检查是否正确")
        else:
            print(f"  ❌ 测试3失败：{error_str[:200]}")
        return False


def test_embedding_model(client, model='text-embedding-3-small'):
    """测试4：测试嵌入模型是否可用"""
    print(f"\n[测试4] 测试嵌入模型 {model}...")
    try:
        response = client.embeddings.create(
            input=["Acetic acid is a weak organic acid used as proton source."],
            model=model
        )
        embedding = response.data[0].embedding
        dim = len(embedding)
        print(f"  ✅ 测试4通过：嵌入模型 {model} 可用，向量维度 = {dim}")
        return True
    except Exception as e:
        print(f"  ❌ 测试4失败：{e}")
        return False


def main():
    print("=" * 60)
    print("  OpenAI API 连接测试")
    print("=" * 60)

    results = []

    # 测试1：API Key
    results.append(test_api_key())
    if not results[-1]:
        print("\n❌ API Key 未配置，后续测试跳过。")
        sys.exit(1)

    # 测试2：创建客户端
    client = test_client_creation()
    results.append(client is not None)
    if not results[-1]:
        print("\n❌ 客户端创建失败，后续测试跳过。")
        sys.exit(1)

    # 测试3：对话模型
    results.append(test_chat_model(client))

    # 测试4：嵌入模型
    results.append(test_embedding_model(client))

    # 汇总
    print("\n" + "=" * 60)
    if all(results):
        print("  ✅ 所有测试通过！环境配置正确，可以开始使用。")
        print("\n  下一步：")
        print("    1. 检查 src/data/init_experiments.xlsx 中是否有初始实验数据")
        print("    2. 运行：cd src && python llm_embedding_bo.py")
    else:
        failed = sum(1 for r in results if not r)
        print(f"  ❌ {failed} 个测试失败，请按照提示修复后重试。")
    print("=" * 60)


if __name__ == '__main__':
    main()
