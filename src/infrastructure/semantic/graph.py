"""
Graph Pathfinding Module
BFSベースのグラフ探索とパス発見
"""
from collections import deque
from typing import Dict, List, Set, Tuple, Optional, Any
from ...domain.semantic import Edge


def build_graph(edges: List[Edge]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, Any]]]:
    """
    エッジリストから隣接リストとエッジ情報マップを構築

    Args:
        edges: Edgeオブジェクトのリスト

    Returns:
        (graph, edge_info):
            - graph: 隣接リスト {node_id: [connected_node_ids]}
            - edge_info: エッジ情報マップ {"from_id|to_id": {join_type, join_condition, ...}}
    """
    graph: Dict[str, List[str]] = {}
    edge_info: Dict[str, Dict[str, Any]] = {}

    for edge in edges:
        from_id = edge.from_node_id
        to_id = edge.to_node_id

        # 隣接リストの初期化
        if from_id not in graph:
            graph[from_id] = []
        if to_id not in graph:
            graph[to_id] = []

        # パス探索用に双方向に設定
        graph[from_id].append(to_id)
        graph[to_id].append(from_id)

        # エッジ情報を両方向に保存
        join_type = edge.join_type.value if hasattr(edge.join_type, 'value') else edge.join_type

        edge_key = f"{from_id}|{to_id}"
        edge_info[edge_key] = {
            'join_type': join_type,
            'join_condition': edge.join_condition,
            'from_id': from_id,
            'to_id': to_id,
            'reversed': False
        }

        # 逆方向のエッジ情報
        edge_key_rev = f"{to_id}|{from_id}"
        edge_info[edge_key_rev] = {
            'join_type': join_type,
            'join_condition': edge.join_condition,
            'from_id': to_id,
            'to_id': from_id,
            'reversed': True
        }

    return graph, edge_info


def find_path_bfs(graph: Dict[str, List[str]], start: str, end: str) -> Optional[List[str]]:
    """
    BFSで2ノード間の最短パスを探索

    Args:
        graph: 隣接リスト
        start: 開始ノードID
        end: 終了ノードID

    Returns:
        パス（ノードIDリスト）または None（パスが存在しない場合）
    """
    if start == end:
        return [start]

    if start not in graph or end not in graph:
        return None

    visited: Set[str] = {start}
    queue: deque = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()

        for neighbor in graph.get(node, []):
            if neighbor == end:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None


def find_all_required_nodes(
    graph: Dict[str, List[str]],
    target_nodes: Set[str]
) -> Tuple[Set[str], List[List[str]]]:
    """
    全ターゲットノードを接続する最小木を探索

    Args:
        graph: 隣接リスト
        target_nodes: 接続したいノードIDのセット

    Returns:
        (all_nodes, all_paths):
            - all_nodes: 必要な全ノードID
            - all_paths: 接続パスのリスト
    """
    if len(target_nodes) < 2:
        return target_nodes, []

    all_nodes: Set[str] = set()
    all_paths: List[List[str]] = []

    nodes_list = list(target_nodes)
    connected: Set[str] = {nodes_list[0]}

    for node in nodes_list[1:]:
        best_path: Optional[List[str]] = None

        # 既に接続済みのノードからの最短パスを探索
        for connected_node in connected:
            path = find_path_bfs(graph, connected_node, node)
            if path and (best_path is None or len(path) < len(best_path)):
                best_path = path

        if best_path:
            all_nodes.update(best_path)
            all_paths.append(best_path)
            connected.add(node)

    return all_nodes, all_paths


def get_join_order(
    nodes: Set[str],
    edges_info: Dict[str, Dict[str, Any]],
    start_node: str
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    JOIN順序を決定（BFSベースのトポロジカルソート）

    Args:
        nodes: JOINに含めるノードIDセット
        edges_info: エッジ情報マップ
        start_node: 開始ノード（FROMの最初のテーブル）

    Returns:
        [(from_node, to_node, edge_info), ...] の順序付きリスト
    """
    if len(nodes) <= 1:
        return []

    join_order: List[Tuple[str, str, Dict[str, Any]]] = []
    joined: Set[str] = {start_node}
    remaining = nodes - {start_node}

    while remaining:
        found = False
        for node in list(remaining):
            # 既にJOIN済みのノードとの接続を探す
            for joined_node in joined:
                edge_key = f"{joined_node}|{node}"
                if edge_key in edges_info:
                    join_order.append((joined_node, node, edges_info[edge_key]))
                    joined.add(node)
                    remaining.remove(node)
                    found = True
                    break
            if found:
                break

        if not found and remaining:
            # 接続できないノードがある場合は中断
            break

    return join_order
