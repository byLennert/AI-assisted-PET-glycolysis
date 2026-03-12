# 🧪 tests/ — 测试文件说明

## 这个文件夹是做什么的？

`tests/` 文件夹包含帮助你**验证环境配置是否正确**的测试脚本。
在开始使用项目之前，建议先运行这些测试，确认一切就绪。

---

## 📄 测试文件说明

| 文件名 | 用途 | 是否需要 API Key |
|--------|------|----------------|
| `test_api_connection.py` | 验证 OpenAI API 是否能正常连接 | ✅ 需要 |
| `test_embedding.py` | 验证嵌入向量生成功能是否正常 | 部分需要（A组不需要） |

---

## 🚀 如何运行

### 推荐方式（从项目根目录）

```bash
# 测试1：检查 API 连接
cd src
python ../tests/test_api_connection.py

# 测试2：检查嵌入功能
python ../tests/test_embedding.py
```

### 设置 API Key

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-your-key-here"

# Windows 命令行
set OPENAI_API_KEY=sk-your-key-here

# Windows PowerShell
$env:OPENAI_API_KEY="sk-your-key-here"
```

---

## ✅ 预期输出（全部通过）

```
============================================================
  OpenAI API 连接测试
============================================================

[测试1] 检查 API Key 配置...
  ✅ 测试1通过：API Key 读取成功（sk-proj-...）

[测试2] 创建 OpenAI 客户端...
  ✅ 测试2通过：OpenAI 客户端创建成功

[测试3] 测试对话模型 gpt-4o...
  ✅ 测试3通过：对话模型 gpt-4o 可用，响应：'API connection successful'

[测试4] 测试嵌入模型 text-embedding-3-small...
  ✅ 测试4通过：嵌入模型 text-embedding-3-small 可用，向量维度 = 1536

============================================================
  ✅ 所有测试通过！环境配置正确，可以开始使用。
============================================================
```

---

## ❌ 常见错误处理

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `未找到 OpenAI API Key` | 没有设置 API Key | 参考上方设置方法 |
| `insufficient_quota` | API 额度不足 | 去 platform.openai.com 充值 |
| `invalid_api_key` | API Key 格式错误 | 检查是否完整复制（以 `sk-` 开头） |
| `Connection error` | 网络问题 | 检查网络/VPN/防火墙设置 |
| `model_not_found` | 账号无权访问该模型 | 换用 `gpt-3.5-turbo` 或升级账号 |
| `No module named 'openai'` | 未安装依赖 | 运行 `pip install -r requirements.txt` |
