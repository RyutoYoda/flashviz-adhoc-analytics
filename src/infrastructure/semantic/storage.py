"""
DuckDB-based Semantic Layer Storage
セマンティックレイヤーの永続化ストレージ
"""
import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
import duckdb

from ...domain.semantic import (
    SemanticLayer, Node, Edge, Dimension, Measure,
    NodeType, JoinType, Aggregation
)


class SemanticLayerStorage:
    """DuckDBベースのセマンティックレイヤーストレージ"""

    def __init__(self, db_path: str = "data/semantic_layer.duckdb"):
        """
        ストレージを初期化

        Args:
            db_path: DuckDBファイルパス
        """
        self.db_path = db_path

        # ディレクトリがなければ作成
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 接続とテーブル初期化
        self._init_tables()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """DuckDB接続を取得"""
        return duckdb.connect(self.db_path)

    def _init_tables(self) -> None:
        """テーブルを初期化"""
        conn = self._get_connection()

        # プロジェクトテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                connector_type VARCHAR NOT NULL,
                description VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # ノードテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id VARCHAR PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                node_name VARCHAR NOT NULL,
                node_type VARCHAR NOT NULL,
                source_object VARCHAR,
                transform_sql VARCHAR,
                description VARCHAR,
                group_name VARCHAR,
                columns_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            )
        """)

        # エッジテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                edge_id VARCHAR PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                from_node_id VARCHAR NOT NULL,
                to_node_id VARCHAR NOT NULL,
                join_type VARCHAR NOT NULL,
                join_condition VARCHAR NOT NULL,
                description VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id),
                FOREIGN KEY (from_node_id) REFERENCES nodes(node_id),
                FOREIGN KEY (to_node_id) REFERENCES nodes(node_id)
            )
        """)

        # ディメンションテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dimensions (
                attr_id VARCHAR PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                node_id VARCHAR NOT NULL,
                attr_name VARCHAR NOT NULL,
                column_expression VARCHAR NOT NULL,
                display_name VARCHAR,
                description VARCHAR,
                sort_order INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id),
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)

        # メジャーテーブル
        conn.execute("""
            CREATE TABLE IF NOT EXISTS measures (
                measure_id VARCHAR PRIMARY KEY,
                project_id VARCHAR NOT NULL,
                node_id VARCHAR NOT NULL,
                measure_name VARCHAR NOT NULL,
                aggregation VARCHAR NOT NULL,
                column_expression VARCHAR NOT NULL,
                display_name VARCHAR,
                description VARCHAR,
                format_str VARCHAR,
                sort_order INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id),
                FOREIGN KEY (node_id) REFERENCES nodes(node_id)
            )
        """)

        conn.close()

    def save_semantic_layer(self, layer: SemanticLayer) -> None:
        """
        セマンティックレイヤーを保存

        Args:
            layer: 保存するセマンティックレイヤー
        """
        conn = self._get_connection()

        try:
            # プロジェクト保存（UPSERT）
            conn.execute("""
                INSERT OR REPLACE INTO projects (project_id, name, connector_type, description, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [layer.project_id, layer.name, layer.connector_type, layer.description])

            # 既存データ削除
            conn.execute("DELETE FROM measures WHERE project_id = ?", [layer.project_id])
            conn.execute("DELETE FROM dimensions WHERE project_id = ?", [layer.project_id])
            conn.execute("DELETE FROM edges WHERE project_id = ?", [layer.project_id])
            conn.execute("DELETE FROM nodes WHERE project_id = ?", [layer.project_id])

            # ノード保存
            for node in layer.nodes:
                columns_json = json.dumps([c.to_dict() for c in node.columns]) if node.columns else None
                node_type = node.node_type.value if isinstance(node.node_type, NodeType) else node.node_type
                conn.execute("""
                    INSERT INTO nodes (node_id, project_id, node_name, node_type, source_object,
                                      transform_sql, description, group_name, columns_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [node.node_id, layer.project_id, node.node_name, node_type,
                      node.source_object, node.transform_sql, node.description,
                      node.group_name, columns_json])

            # エッジ保存
            for edge in layer.edges:
                join_type = edge.join_type.value if isinstance(edge.join_type, JoinType) else edge.join_type
                conn.execute("""
                    INSERT INTO edges (edge_id, project_id, from_node_id, to_node_id,
                                      join_type, join_condition, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [edge.edge_id, layer.project_id, edge.from_node_id, edge.to_node_id,
                      join_type, edge.join_condition, edge.description])

            # ディメンション保存
            for dim in layer.dimensions:
                conn.execute("""
                    INSERT INTO dimensions (attr_id, project_id, node_id, attr_name,
                                           column_expression, display_name, description, sort_order)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [dim.attr_id, layer.project_id, dim.node_id, dim.attr_name,
                      dim.column_expression, dim.display_name, dim.description, dim.sort_order])

            # メジャー保存
            for measure in layer.measures:
                aggregation = measure.aggregation.value if isinstance(measure.aggregation, Aggregation) else measure.aggregation
                conn.execute("""
                    INSERT INTO measures (measure_id, project_id, node_id, measure_name,
                                         aggregation, column_expression, display_name,
                                         description, format_str, sort_order)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [measure.measure_id, layer.project_id, measure.node_id, measure.measure_name,
                      aggregation, measure.column_expression, measure.display_name,
                      measure.description, measure.format_str, measure.sort_order])

            conn.commit()
        finally:
            conn.close()

    def load_semantic_layer(self, project_id: str) -> Optional[SemanticLayer]:
        """
        セマンティックレイヤーを読み込み

        Args:
            project_id: プロジェクトID

        Returns:
            SemanticLayerまたはNone
        """
        conn = self._get_connection()

        try:
            # プロジェクト取得
            result = conn.execute(
                "SELECT * FROM projects WHERE project_id = ?",
                [project_id]
            ).fetchone()

            if not result:
                return None

            layer = SemanticLayer(
                project_id=result[0],
                name=result[1],
                connector_type=result[2],
                description=result[3]
            )

            # ノード取得
            nodes = conn.execute(
                "SELECT * FROM nodes WHERE project_id = ? ORDER BY created_at",
                [project_id]
            ).fetchall()

            from ...domain.semantic import ColumnInfo

            for row in nodes:
                columns = []
                if row[8]:  # columns_json
                    columns = [ColumnInfo.from_dict(c) for c in json.loads(row[8])]

                node = Node(
                    node_id=row[0],
                    node_name=row[2],
                    node_type=NodeType(row[3]),
                    source_object=row[4],
                    transform_sql=row[5],
                    description=row[6],
                    group_name=row[7],
                    columns=columns
                )
                layer.nodes.append(node)

            # エッジ取得
            edges = conn.execute(
                "SELECT * FROM edges WHERE project_id = ? ORDER BY created_at",
                [project_id]
            ).fetchall()

            for row in edges:
                edge = Edge(
                    edge_id=row[0],
                    from_node_id=row[2],
                    to_node_id=row[3],
                    join_type=JoinType(row[4]),
                    join_condition=row[5],
                    description=row[6]
                )
                layer.edges.append(edge)

            # ディメンション取得
            dimensions = conn.execute(
                "SELECT * FROM dimensions WHERE project_id = ? ORDER BY sort_order",
                [project_id]
            ).fetchall()

            for row in dimensions:
                dim = Dimension(
                    attr_id=row[0],
                    node_id=row[2],
                    attr_name=row[3],
                    column_expression=row[4],
                    display_name=row[5],
                    description=row[6],
                    sort_order=row[7] or 0
                )
                layer.dimensions.append(dim)

            # メジャー取得
            measures = conn.execute(
                "SELECT * FROM measures WHERE project_id = ? ORDER BY sort_order",
                [project_id]
            ).fetchall()

            for row in measures:
                measure = Measure(
                    measure_id=row[0],
                    node_id=row[2],
                    measure_name=row[3],
                    aggregation=Aggregation(row[4]),
                    column_expression=row[5],
                    display_name=row[6],
                    description=row[7],
                    format_str=row[8],
                    sort_order=row[9] or 0
                )
                layer.measures.append(measure)

            return layer

        finally:
            conn.close()

    def list_projects(self) -> List[Dict[str, Any]]:
        """
        全プロジェクトのリストを取得

        Returns:
            プロジェクト情報のリスト
        """
        conn = self._get_connection()

        try:
            results = conn.execute("""
                SELECT project_id, name, connector_type, description, created_at, updated_at
                FROM projects
                ORDER BY updated_at DESC
            """).fetchall()

            return [
                {
                    "project_id": row[0],
                    "name": row[1],
                    "connector_type": row[2],
                    "description": row[3],
                    "created_at": row[4],
                    "updated_at": row[5]
                }
                for row in results
            ]
        finally:
            conn.close()

    def delete_project(self, project_id: str) -> bool:
        """
        プロジェクトを削除

        Args:
            project_id: 削除するプロジェクトID

        Returns:
            削除成功したかどうか
        """
        conn = self._get_connection()

        try:
            # カスケード削除
            conn.execute("DELETE FROM measures WHERE project_id = ?", [project_id])
            conn.execute("DELETE FROM dimensions WHERE project_id = ?", [project_id])
            conn.execute("DELETE FROM edges WHERE project_id = ?", [project_id])
            conn.execute("DELETE FROM nodes WHERE project_id = ?", [project_id])
            conn.execute("DELETE FROM projects WHERE project_id = ?", [project_id])
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def export_to_json(self, project_id: str) -> Optional[str]:
        """
        セマンティックレイヤーをJSONにエクスポート

        Args:
            project_id: プロジェクトID

        Returns:
            JSON文字列またはNone
        """
        layer = self.load_semantic_layer(project_id)
        if layer:
            return json.dumps(layer.to_dict(), indent=2, ensure_ascii=False)
        return None

    def import_from_json(self, json_str: str) -> Optional[SemanticLayer]:
        """
        JSONからセマンティックレイヤーをインポート

        Args:
            json_str: JSON文字列

        Returns:
            SemanticLayerまたはNone
        """
        try:
            data = json.loads(json_str)
            layer = SemanticLayer.from_dict(data)
            self.save_semantic_layer(layer)
            return layer
        except Exception:
            return None
