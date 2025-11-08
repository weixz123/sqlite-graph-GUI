#!/usr/bin/env python3
"""
图数据库查询与嵌入生成工具集
"""

import sqlite3
import logging
import json
import re
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional

from simple_graph_sqlite import database as db

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graph_query.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== LMStudio & Embedding Functions ====================

# 配置OpenAI客户端连接到LMStudio的本地服务器
client = OpenAI(
    base_url="http://localhost:1234/v1/",
    api_key="not-needed"
)


def call_lmstudio_chat(prompt="你好,请介绍一下自己"):
    """
    使用OpenAI聊天补全API格式调用LMStudio的语言模型
    """
    try:
        response = client.chat.completions.create(
            model="qwen3_30ba3b",
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=32768
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"调用LMStudio聊天API时出错: {e}")
        return None


def get_embedding(text, model="bge-m3:latest"):
    """
    使用LMStudio的嵌入API生成文本的向量表示
    """
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"生成嵌入向量时出错: {e}")
        return None


# ==================== Graph Query System ====================

class GraphQuerySystem:
    """图查询系统"""

    def __init__(self, graph_db_path: str):
        """
        初始化查询系统
        
        参数:
            graph_db_path: 图数据库路径
        """
        self.graph_db_path = graph_db_path
        self._all_nodes_cache = None

    def get_all_nodes_with_embeddings(self, force_reload: bool = False) -> List[Dict]:
        """
        获取所有节点及其嵌入向量 (带缓存)
        """
        if self._all_nodes_cache is not None and not force_reload:
            return self._all_nodes_cache

        logger.info("正在加载并生成所有节点的嵌入向量...")
        all_nodes = db.atomic(self.graph_db_path, db.find_nodes([], ()))
        
        nodes_with_embeddings = []
        for node in all_nodes:
            text_for_embedding = f"{node.get('name', '')} {node.get('description', '')}"
            embedding = get_embedding(text_for_embedding)
            
            if embedding:
                nodes_with_embeddings.append({
                    'id': node['id'],
                    'name': node.get('name', 'Unknown'),
                    'type': node.get('type', 'unknown'),
                    'description': node.get('description', ''),
                    'embedding': np.array(embedding)
                })
        
        self._all_nodes_cache = nodes_with_embeddings
        logger.info(f"已加载 {len(self._all_nodes_cache)} 个节点的嵌入向量。")
        return self._all_nodes_cache

    def find_similar_nodes(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """
        使用BGE嵌入查找与查询最相似的节点
        """
        logger.info(f"使用BGE查找与 '{query}' 相似的节点 (阈值: {threshold})...")
        
        query_embedding = get_embedding(query)
        if not query_embedding:
            logger.error("无法生成查询嵌入向量")
            return []
        
        query_vec = np.array(query_embedding)
        
        nodes = self.get_all_nodes_with_embeddings()
        if not nodes:
            logger.warning("数据库中没有可用的节点")
            return []
        
        similarities = []
        for node in nodes:
            node_vec = node['embedding']
            similarity = np.dot(query_vec, node_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(node_vec))
            
            if similarity >= threshold:
                similarities.append({
                    'id': node['id'],
                    'name': node['name'],
                    'type': node['type'],
                    'description': node['description'],
                    'score': float(similarity)
                })
        
        similarities.sort(key=lambda x: x['score'], reverse=True)
        top_results = similarities[:top_k]
        
        logger.info(f"找到 {len(top_results)} 个相似节点 (阈值以上)")
        return top_results

    def get_neighbors(self, node_id: int) -> List[Dict]:
        """获取节点的邻居及连接关系"""
        conn = sqlite3.connect(self.graph_db_path)
        cursor = conn.cursor()
        
        neighbors = []
        try:
            # 出边: node_id -> target
            cursor.execute("SELECT target, properties FROM edges WHERE source = ?", (node_id,))
            for row in cursor.fetchall():
                target_id, properties = row
                neighbors.append({
                    'source': node_id,
                    'target': target_id,
                    'edge': json.loads(properties) if properties else {}
                })

            # 入边: source -> node_id
            cursor.execute("SELECT source, properties FROM edges WHERE target = ?", (node_id,))
            for row in cursor.fetchall():
                source_id, properties = row
                neighbors.append({
                    'source': source_id,
                    'target': node_id,
                    'edge': json.loads(properties) if properties else {}
                })
        finally:
            conn.close()
        return neighbors

    def n_hop_query(self, start_node_id: int, n_hops: int = 1) -> Dict:
        """
        从起始节点进行N跳查询
        """
        logger.info(f"从节点 {start_node_id} 开始进行 {n_hops}-跳查询...")
        
        if n_hops <= 0:
            return {'nodes': [], 'edges': []}

        all_nodes = {start_node_id}
        all_edges = []
        
        current_frontier = {start_node_id}
        
        for _ in range(n_hops):
            next_frontier = set()
            for node_id in current_frontier:
                neighbors = self.get_neighbors(node_id)
                for neighbor_info in neighbors:
                    edge = (neighbor_info['source'], neighbor_info['target'])
                    # 确保边的方向唯一性，并处理潜在的类型问题
                    source_id, target_id = int(edge[0]), int(edge[1])
                    if source_id > target_id:
                        edge = (target_id, source_id)
                    else:
                        edge = (source_id, target_id)

                    if edge not in all_edges:
                        all_edges.append(edge)
                    
                    # 添加邻居到下一个前沿
                    neighbor_id = neighbor_info['target'] if neighbor_info['source'] == node_id else neighbor_info['source']
                    if neighbor_id not in all_nodes:
                        next_frontier.add(neighbor_id)
                        all_nodes.add(neighbor_id)
            
            current_frontier = next_frontier
            if not current_frontier:
                break # 没有新的节点可以扩展

        # 获取所有涉及的节点信息
        nodes_data = []
        for node_id in all_nodes:
            node = db.atomic(self.graph_db_path, db.find_node(node_id))
            if node:
                nodes_data.append(node)
        
        # 获取所有涉及的边的信息
        edges_data = []
        conn = sqlite3.connect(self.graph_db_path)
        cursor = conn.cursor()
        try:
            for source_id, target_id in all_edges:
                cursor.execute(
                    "SELECT properties FROM edges WHERE (source = ? AND target = ?) OR (source = ? AND target = ?)",
                    (source_id, target_id, target_id, source_id)
                )
                row = cursor.fetchone()
                properties = json.loads(row[0]) if row and row[0] else {}
                
                edges_data.append({
                    'source': source_id,
                    'target': target_id,
                    'properties': properties
                })
        finally:
            conn.close()

        logger.info(f"N跳查询找到 {len(nodes_data)} 个节点, {len(edges_data)} 条边")
        return {'nodes': nodes_data, 'edges': edges_data}
