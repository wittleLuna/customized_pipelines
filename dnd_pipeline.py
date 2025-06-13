"""
title: Custom RAG Pipeline
author: your_name
date: 2025-06-13
version: 1.0
license: MIT
description: A custom RAG pipeline using ChromaDB and Qwen for generating structured markdown answers.
requirements: chromadb, openai, pypdf, python-docx, langchain
"""

from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import chromadb
from openai import OpenAI

# 嵌入函数封装
class OpenAIEmbeddingFunction:
    def __init__(self, client):
        self.client = client

    def __call__(self, input):
        resp = self.client.embeddings.create(
            input=input,
            model="text-embedding-v3"
        )
        return [item.embedding for item in resp.data]

    def name(self):
        return "text-embedding-v3"

# RAG 核心
class ReportGenerator:
    def __init__(self):
        self.openai_client = OpenAI(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-442562cd6b6b4b2896ebdac8ce8d047e"  # 替换为你的 key
        )
        self.embed_fn = OpenAIEmbeddingFunction(self.openai_client)
        self.client = chromadb.PersistentClient(path="db")
        self.collection = self.client.get_collection(
            "linux_course",
            embedding_function=self.embed_fn
        )

    def generate_report(self, topic: str):
        print("🔍 正在进行嵌入")
        query_vector = self.embed_fn([topic])
        print("✅ 嵌入完成")

        results = self.collection.query(query_embeddings=query_vector, n_results=5)
        context = "\n\n".join(results['documents'][0])

        prompt = f"""
你是一个经验丰富的DnD 5e 主持人，你需要根据以下知识库中的内容对玩家的提问提供丰富，恰当的知识和介绍

## 知识库片段 ##
{context}

## 报告要求 ##
- 保持专业但友好的语调。如果玩家询问规则，请提供准确的D&D 5E规则信息。
- 可使用markdown格式进行回复
"""

        print("🤖 正在调用 qwen")
        response = self.openai_client.chat.completions.create(
            model='qwen-plus',
            messages=[{'role': 'user', 'content': prompt}]
        )
        ans = response.choices[0].message.content
        print("✅ qwen 响应完成")
        return ans

# Pipeline 主体
class Pipeline:
    def __init__(self):
        self.generator = ReportGenerator()

    async def on_startup(self):
        print("✅ RAG Pipeline 启动")

    async def on_shutdown(self):
        print("🛑 RAG Pipeline 关闭")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body  # 可加入预处理逻辑

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        return body  # 可加入后处理逻辑

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        print(f"📩 收到请求: {user_message}")
        return self.generator.generate_report(user_message)
