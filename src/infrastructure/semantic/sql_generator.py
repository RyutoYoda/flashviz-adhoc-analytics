"""
Dialect-Aware SQL Generator
複数のSQLダイアレクトに対応したSQL生成
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
from ...domain.semantic import (
    SemanticLayer, Node, Edge, Dimension, Measure,
    NodeType, Aggregation
)
from .graph import build_graph, find_all_required_nodes, get_join_order


@dataclass
class DialectConfig:
    """SQLダイアレクト設定"""
    name: str
    identifier_quote: str  # カラム名のクォート文字
    string_concat: str  # 文字列結合関数 (CONCAT or ||)
    date_trunc_format: str  # DATE_TRUNC形式
    supports_full_outer: bool = True

    @classmethod
    def snowflake(cls) -> "DialectConfig":
        return cls(
            name="snowflake",
            identifier_quote='"',
            string_concat="CONCAT",
            date_trunc_format="DATE_TRUNC('{unit}', {column})",
            supports_full_outer=True
        )

    @classmethod
    def bigquery(cls) -> "DialectConfig":
        return cls(
            name="bigquery",
            identifier_quote="`",
            string_concat="CONCAT",
            date_trunc_format="DATE_TRUNC({column}, {unit})",
            supports_full_outer=True
        )

    @classmethod
    def databricks(cls) -> "DialectConfig":
        return cls(
            name="databricks",
            identifier_quote="`",
            string_concat="CONCAT",
            date_trunc_format="date_trunc('{unit}', {column})",
            supports_full_outer=True
        )

    @classmethod
    def duckdb(cls) -> "DialectConfig":
        return cls(
            name="duckdb",
            identifier_quote='"',
            string_concat="||",
            date_trunc_format="DATE_TRUNC('{unit}', {column})",
            supports_full_outer=True
        )

    @classmethod
    def from_connector_type(cls, connector_type: str) -> "DialectConfig":
        """コネクタタイプからダイアレクト設定を取得"""
        configs = {
            "snowflake": cls.snowflake,
            "bigquery": cls.bigquery,
            "databricks": cls.databricks,
            "duckdb": cls.duckdb,
            "local": cls.duckdb,  # ローカルファイルはDuckDB
        }
        factory = configs.get(connector_type.lower(), cls.duckdb)
        return factory()


class SQLGenerator:
    """SQL生成クラス"""

    def __init__(self, semantic_layer: SemanticLayer, dialect: Optional[DialectConfig] = None):
        self.semantic_layer = semantic_layer
        self.dialect = dialect or DialectConfig.from_connector_type(semantic_layer.connector_type)

        # グラフ構築
        self.graph, self.edge_info = build_graph(semantic_layer.edges)

        # ノードマップ構築
        self.node_map: Dict[str, Node] = {
            node.node_id: node for node in semantic_layer.nodes
        }

        # ディメンション/メジャーマップ構築
        self.dimension_map: Dict[str, Dimension] = {
            dim.attr_id: dim for dim in semantic_layer.dimensions
        }
        self.measure_map: Dict[str, Measure] = {
            m.measure_id: m for m in semantic_layer.measures
        }

    def quote_identifier(self, identifier: str) -> str:
        """識別子をクォート"""
        q = self.dialect.identifier_quote
        return f"{q}{identifier}{q}"

    def generate_sql(
        self,
        dimension_ids: List[str],
        measure_ids: List[str],
        filters: Optional[List[str]] = None,
        order_by: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        選択されたディメンションとメジャーからSQLを生成

        Args:
            dimension_ids: 選択されたディメンション属性IDリスト
            measure_ids: 選択されたメジャーIDリスト
            filters: WHERE条件リスト
            order_by: ORDER BY句リスト
            limit: LIMIT値

        Returns:
            (sql, error): 生成されたSQLまたはエラーメッセージ
        """
        if not dimension_ids and not measure_ids:
            return None, "ディメンションまたはメジャーを1つ以上選択してください"

        # ターゲットノードIDを収集
        target_node_ids: Set[str] = set()

        for dim_id in dimension_ids:
            dim = self.dimension_map.get(dim_id)
            if dim:
                target_node_ids.add(dim.node_id)

        for measure_id in measure_ids:
            measure = self.measure_map.get(measure_id)
            if measure:
                target_node_ids.add(measure.node_id)

        if not target_node_ids:
            return None, "有効なノードが選択されていません"

        # 必要なノードを探索
        all_nodes, paths = find_all_required_nodes(self.graph, target_node_ids)

        if not all_nodes:
            return None, "選択されたノードを接続するパスが見つかりません"

        # エイリアス割り当て
        alias_map: Dict[str, str] = {}
        for idx, node_id in enumerate(all_nodes):
            alias_map[node_id] = f"t{idx}"

        # SELECT句構築
        select_parts: List[str] = []
        group_by_parts: List[str] = []

        for dim_id in dimension_ids:
            dim = self.dimension_map.get(dim_id)
            if not dim:
                continue

            alias = alias_map.get(dim.node_id, 't0')
            col_expr = dim.column_expression
            display = dim.display_name or dim.attr_name

            # エイリアスを付ける（関数やドットを含まない場合）
            if '(' not in col_expr and '.' not in col_expr:
                col_expr = f"{alias}.{col_expr}"

            select_parts.append(f"{col_expr} AS {self.quote_identifier(display)}")
            group_by_parts.append(col_expr)

        for measure_id in measure_ids:
            measure = self.measure_map.get(measure_id)
            if not measure:
                continue

            alias = alias_map.get(measure.node_id, 't0')
            col_expr = measure.column_expression
            agg = measure.aggregation
            display = measure.display_name or measure.measure_name

            # エイリアスを付ける
            if '(' not in col_expr and '.' not in col_expr:
                col_expr = f"{alias}.{col_expr}"

            # 集計関数を適用
            agg_value = agg.value if isinstance(agg, Aggregation) else agg
            if agg_value == 'COUNT_DISTINCT':
                select_parts.append(f"COUNT(DISTINCT {col_expr}) AS {self.quote_identifier(display)}")
            else:
                select_parts.append(f"{agg_value}({col_expr}) AS {self.quote_identifier(display)}")

        # FROM/JOIN句構築
        nodes_list = list(all_nodes)
        first_node_id = nodes_list[0]
        first_node = self.node_map.get(first_node_id)

        if not first_node:
            return None, f"ノードが見つかりません: {first_node_id}"

        # FROMテーブル
        if first_node.node_type == NodeType.TRANSFORM:
            from_clause = f"({first_node.transform_sql}) AS {alias_map[first_node_id]}"
        else:
            from_clause = f"{first_node.source_object} AS {alias_map[first_node_id]}"

        # JOIN句を構築
        join_clauses: List[str] = []
        joined_nodes: Set[str] = {first_node_id}

        while len(joined_nodes) < len(all_nodes):
            found = False
            for node_id in nodes_list:
                if node_id in joined_nodes:
                    continue

                for joined_node in joined_nodes:
                    edge_key = f"{joined_node}|{node_id}"
                    if edge_key in self.edge_info:
                        edge = self.edge_info[edge_key]
                        node = self.node_map.get(node_id)

                        if not node:
                            continue

                        # テーブル参照
                        if node.node_type == NodeType.TRANSFORM:
                            table_ref = f"({node.transform_sql})"
                        else:
                            table_ref = node.source_object

                        join_cond = edge['join_condition']

                        join_clauses.append(
                            f"{edge['join_type']} JOIN {table_ref} AS {alias_map[node_id]} ON {join_cond}"
                        )
                        joined_nodes.add(node_id)
                        found = True
                        break

                if found:
                    break

            if not found:
                break

        # SQL組み立て
        sql_parts = [
            "SELECT",
            f"    {', '.join(select_parts)}",
            f"FROM {from_clause}"
        ]

        if join_clauses:
            sql_parts.extend(join_clauses)

        if filters:
            sql_parts.append(f"WHERE {' AND '.join(filters)}")

        if group_by_parts:
            sql_parts.append("GROUP BY")
            sql_parts.append(f"    {', '.join(group_by_parts)}")

        if order_by:
            sql_parts.append(f"ORDER BY {', '.join(order_by)}")

        if limit:
            sql_parts.append(f"LIMIT {limit}")

        return '\n'.join(sql_parts), None

    def get_node_alias(self, node_id: str, alias_map: Dict[str, str]) -> str:
        """ノードのエイリアスを取得"""
        return alias_map.get(node_id, 't0')
