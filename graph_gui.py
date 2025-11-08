#!/usr/bin/env python3
"""
航空知识图谱交互式GUI查询系统 - Flask后端
"""

import json
import logging
from flask import Flask, render_template, request, jsonify
from graph_utils import GraphQuerySystem
from simple_graph_sqlite import database as db

# ==================== 配置 ====================
GRAPH_DB_PATH = 'aviation_graph.sqlite'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("graph_gui.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== Flask 应用初始化 ====================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False # 支持中文

# 初始化图查询系统
try:
    query_system = GraphQuerySystem(GRAPH_DB_PATH)
    # 预加载所有节点的嵌入，加快首次查询速度
    query_system.get_all_nodes_with_embeddings() 
    logger.info("Flask后端初始化成功，图查询系统已就绪。")
except Exception as e:
    logger.error(f"初始化GraphQuerySystem失败: {e}")
    query_system = None

# ==================== 路由和API端点 ====================

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """
    处理搜索请求，返回子图数据
    """
    if not query_system:
        return jsonify({"error": "查询系统未初始化"}), 500

    data = request.json
    query = data.get('query', '')
    threshold = float(data.get('threshold', 0.5))
    n_hops = int(data.get('n_hops', 1))
    top_k = int(data.get('top_k', 5))

    if not query:
        return jsonify({"error": "查询内容不能为空"}), 400

    try:
        # 1. 语义相似度查询
        similar_nodes = query_system.find_similar_nodes(query, top_k=top_k, threshold=threshold)
        if not similar_nodes:
            return jsonify({'nodes': [], 'edges': []})

        # 2. N跳查询并合并结果
        combined_nodes = {}
        combined_edges = {}

        # 提取初始相似节点ID及其相似度分数
        initial_nodes_with_scores = {
            node['id']: node.get('score', 0.0) for node in similar_nodes
        }

        for node_id in initial_nodes_with_scores.keys():
            subgraph = query_system.n_hop_query(node_id, n_hops)
            
            for n in subgraph['nodes']:
                if n['id'] not in combined_nodes:
                    combined_nodes[n['id']] = n
            
            for e in subgraph['edges']:
                # 使用frozenset确保边的唯一性，忽略方向
                edge_key = frozenset([e['source'], e['target']])
                if edge_key not in combined_edges:
                    combined_edges[edge_key] = e
        
        return jsonify({
            'nodes': list(combined_nodes.values()),
            'edges': list(combined_edges.values()),
            'initial_nodes': initial_nodes_with_scores
        })

    except Exception as e:
        logger.error(f"搜索过程中发生错误: {e}")
        return jsonify({"error": "服务器内部错误"}), 500

@app.route('/api/neighbors/<int:node_id>', methods=['GET'])
def get_neighbors(node_id):
    """获取单个节点的1跳邻居"""
    if not query_system:
        return jsonify({"error": "查询系统未初始化"}), 500
    
    try:
        subgraph = query_system.n_hop_query(node_id, 1)
        return jsonify(subgraph)
    except Exception as e:
        logger.error(f"获取邻居节点失败: {e}")
        return jsonify({"error": "获取邻居节点失败"}), 500

# ==================== CRUD API ====================

@app.route('/api/node', methods=['POST'])
def add_node():
    """添加新节点"""
    data = request.json
    name = data.get('name')
    node_type = data.get('type')
    description = data.get('description', '')

    if not name or not node_type:
        return jsonify({"error": "节点名称和类型不能为空"}), 400

    try:
        # 获取当前最大的ID
        all_nodes = db.atomic(GRAPH_DB_PATH, db.find_nodes([], ()))
        max_id = max([n['id'] for n in all_nodes]) if all_nodes else 0
        new_id = max_id + 1

        node_data = {'name': name, 'type': node_type, 'description': description}
        db.atomic(GRAPH_DB_PATH, db.add_node(node_data, new_id))
        
        # 更新缓存
        if query_system:
            query_system.get_all_nodes_with_embeddings(force_reload=True)
            
        return jsonify({"message": "节点添加成功", "id": new_id}), 201
    except Exception as e:
        logger.error(f"添加节点失败: {e}")
        return jsonify({"error": "添加节点失败"}), 500

@app.route('/api/node/<int:node_id>', methods=['PUT'])
def update_node(node_id):
    """更新节点"""
    data = request.json
    try:
        db.atomic(GRAPH_DB_PATH, db.upsert_node(node_id, data))
        if query_system:
            query_system.get_all_nodes_with_embeddings(force_reload=True)
        return jsonify({"message": f"节点 {node_id} 更新成功"})
    except Exception as e:
        logger.error(f"更新节点 {node_id} 失败: {e}")
        return jsonify({"error": "更新节点失败"}), 500

@app.route('/api/node/<int:node_id>', methods=['DELETE'])
def delete_node(node_id):
    """删除节点"""
    try:
        db.atomic(GRAPH_DB_PATH, db.delete_node(node_id))
        if query_system:
            query_system.get_all_nodes_with_embeddings(force_reload=True)
        return jsonify({"message": f"节点 {node_id} 删除成功"})
    except Exception as e:
        logger.error(f"删除节点 {node_id} 失败: {e}")
        return jsonify({"error": "删除节点失败"}), 500

@app.route('/api/edge', methods=['POST'])
def add_edge():
    """添加关系"""
    data = request.json
    source = data.get('source')
    target = data.get('target')
    properties = data.get('properties', {})

    if source is None or target is None:
        return jsonify({"error": "源节点和目标节点不能为空"}), 400

    try:
        db.atomic(GRAPH_DB_PATH, db.connect_nodes(int(source), int(target), properties))
        return jsonify({"message": "关系添加成功"}), 201
    except Exception as e:
        logger.error(f"添加关系失败: {e}")
        return jsonify({"error": "添加关系失败"}), 500

@app.route('/api/edge', methods=['DELETE'])
def delete_edge():
    """删除关系"""
    data = request.json
    source = data.get('source')
    target = data.get('target')

    if source is None or target is None:
        return jsonify({"error": "源节点和目标节点不能为空"}), 400

    try:
        db.atomic(GRAPH_DB_PATH, db.disconnect_nodes(int(source), int(target)))
        return jsonify({"message": "关系删除成功"})
    except Exception as e:
        logger.error(f"删除关系失败: {e}")
        return jsonify({"error": "删除关系失败"}), 500

# ==================== 主程序入口 ====================

if __name__ == '__main__':
    logger.info("启动Flask服务器，请在浏览器中访问 http://127.0.0.1:5000")
    app.run(debug=True)
