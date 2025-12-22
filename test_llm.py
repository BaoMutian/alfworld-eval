#!/usr/bin/env python3
"""Simple test script for LLM API."""

import httpx
from openai import OpenAI

# ============ 配置 ============
API_BASE_URL = "http://localhost:8001/v1"  # 修改为你的 vLLM 地址
MODEL_NAME = "Qwen/Qwen3-8B"  # 修改为你的模型名称
ENABLE_THINKING = False
# ==============================

def test_with_httpx():
    """使用 httpx 直接测试（用户提供的可工作方式）"""
    print("=" * 50)
    print("Test 1: Using httpx.Client (user's working method)")
    print("=" * 50)
    
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=API_BASE_URL,
            http_client=httpx.Client(
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                timeout=60.0,
            ),
        )
        
        params = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello, say 'test ok' in one word"}],
            "temperature": 0.3,
            "stream": False,
        }
        
        if ENABLE_THINKING:
            params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True}
            }
        
        response = client.chat.completions.create(**params)
        print(f"✓ Success: {response.choices[0].message.content[:100]}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_simple():
    """最简单的测试"""
    print("\n" + "=" * 50)
    print("Test 2: Simple OpenAI client (no httpx)")
    print("=" * 50)
    
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=API_BASE_URL,
            timeout=60.0,
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.3,
        )
        print(f"✓ Success: {response.choices[0].message.content[:100]}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_with_extra_body():
    """测试带 extra_body 的情况"""
    print("\n" + "=" * 50)
    print("Test 3: With extra_body (enable_thinking)")
    print("=" * 50)
    
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=API_BASE_URL,
            http_client=httpx.Client(
                transport=httpx.HTTPTransport(local_address="0.0.0.0"),
                timeout=60.0,
            ),
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.3,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        print(f"✓ Success: {response.choices[0].message.content[:100]}")
        return True
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_requests_directly():
    """直接用 requests 测试"""
    print("\n" + "=" * 50)
    print("Test 4: Direct requests (bypass OpenAI client)")
    print("=" * 50)
    
    try:
        import requests
        
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.3,
            },
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success: {result['choices'][0]['message']['content'][:100]}")
            return True
        else:
            print(f"✗ HTTP {response.status_code}: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


if __name__ == "__main__":
    print(f"Testing API: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print()
    
    test_with_httpx()
    test_simple()
    test_with_extra_body()
    test_requests_directly()
    
    print("\n" + "=" * 50)
    print("如果 Test 4 成功但其他失败，问题在 OpenAI 客户端配置")
    print("如果都失败，检查 API_BASE_URL 和 MODEL_NAME")
    print("=" * 50)

