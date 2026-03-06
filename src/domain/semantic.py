"""
Semantic Layer Domain Models
セマンティックレイヤーのデータモデル定義
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class NodeType(str, Enum):
    """ノードタイプ"""
    SOURCE = "SOURCE"
    TRANSFORM = "TRANSFORM"
    DIMENSION = "DIMENSION"
    FACT = "FACT"


class JoinType(str, Enum):
    """JOIN種別"""
    INNER = "INNER"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FULL = "FULL"


class Aggregation(str, Enum):
    """集計関数"""
    SUM = "SUM"
    COUNT = "COUNT"
    AVG = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    COUNT_DISTINCT = "COUNT_DISTINCT"


@dataclass
class ColumnInfo:
    """カラム情報"""
    name: str
    data_type: str
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnInfo":
        return cls(
            name=data["name"],
            data_type=data["data_type"],
            description=data.get("description")
        )


@dataclass
class Node:
    """ノード（テーブル/ビュー/変換）"""
    node_id: str
    node_name: str
    node_type: NodeType
    source_object: Optional[str] = None  # Full table name (DB.SCHEMA.TABLE)
    transform_sql: Optional[str] = None  # For TRANSFORM nodes
    description: Optional[str] = None
    group_name: Optional[str] = None
    columns: List[ColumnInfo] = field(default_factory=list)

    @classmethod
    def create(cls, name: str, node_type: NodeType, source_object: Optional[str] = None,
               transform_sql: Optional[str] = None, description: Optional[str] = None,
               group_name: Optional[str] = None) -> "Node":
        return cls(
            node_id=str(uuid.uuid4())[:8],
            node_name=name,
            node_type=node_type,
            source_object=source_object,
            transform_sql=transform_sql,
            description=description,
            group_name=group_name
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "node_type": self.node_type.value if isinstance(self.node_type, NodeType) else self.node_type,
            "source_object": self.source_object,
            "transform_sql": self.transform_sql,
            "description": self.description,
            "group_name": self.group_name,
            "columns": [c.to_dict() for c in self.columns]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        node_type = data["node_type"]
        if isinstance(node_type, str):
            node_type = NodeType(node_type)
        return cls(
            node_id=data["node_id"],
            node_name=data["node_name"],
            node_type=node_type,
            source_object=data.get("source_object"),
            transform_sql=data.get("transform_sql"),
            description=data.get("description"),
            group_name=data.get("group_name"),
            columns=[ColumnInfo.from_dict(c) for c in data.get("columns", [])]
        )


@dataclass
class Edge:
    """エッジ（JOIN定義）"""
    edge_id: str
    from_node_id: str
    to_node_id: str
    join_type: JoinType
    join_condition: str  # e.g., "t0.customer_id = t1.id"
    description: Optional[str] = None

    @classmethod
    def create(cls, from_node_id: str, to_node_id: str, join_type: JoinType,
               join_condition: str, description: Optional[str] = None) -> "Edge":
        return cls(
            edge_id=str(uuid.uuid4())[:8],
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            join_type=join_type,
            join_condition=join_condition,
            description=description
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "join_type": self.join_type.value if isinstance(self.join_type, JoinType) else self.join_type,
            "join_condition": self.join_condition,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Edge":
        join_type = data["join_type"]
        if isinstance(join_type, str):
            join_type = JoinType(join_type)
        return cls(
            edge_id=data["edge_id"],
            from_node_id=data["from_node_id"],
            to_node_id=data["to_node_id"],
            join_type=join_type,
            join_condition=data["join_condition"],
            description=data.get("description")
        )


@dataclass
class Dimension:
    """ディメンション属性（GROUP BY対象）"""
    attr_id: str
    node_id: str
    attr_name: str
    column_expression: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    sort_order: int = 0

    @classmethod
    def create(cls, node_id: str, attr_name: str, column_expression: str,
               display_name: Optional[str] = None, description: Optional[str] = None,
               sort_order: int = 0) -> "Dimension":
        return cls(
            attr_id=str(uuid.uuid4())[:8],
            node_id=node_id,
            attr_name=attr_name,
            column_expression=column_expression,
            display_name=display_name or attr_name,
            description=description,
            sort_order=sort_order
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attr_id": self.attr_id,
            "node_id": self.node_id,
            "attr_name": self.attr_name,
            "column_expression": self.column_expression,
            "display_name": self.display_name,
            "description": self.description,
            "sort_order": self.sort_order
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dimension":
        return cls(
            attr_id=data["attr_id"],
            node_id=data["node_id"],
            attr_name=data["attr_name"],
            column_expression=data["column_expression"],
            display_name=data.get("display_name"),
            description=data.get("description"),
            sort_order=data.get("sort_order", 0)
        )


@dataclass
class Measure:
    """メジャー（集計対象）"""
    measure_id: str
    node_id: str
    measure_name: str
    aggregation: Aggregation
    column_expression: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    format_str: Optional[str] = None  # e.g., "#,##0"
    sort_order: int = 0

    @classmethod
    def create(cls, node_id: str, measure_name: str, aggregation: Aggregation,
               column_expression: str, display_name: Optional[str] = None,
               description: Optional[str] = None, format_str: Optional[str] = None,
               sort_order: int = 0) -> "Measure":
        return cls(
            measure_id=str(uuid.uuid4())[:8],
            node_id=node_id,
            measure_name=measure_name,
            aggregation=aggregation,
            column_expression=column_expression,
            display_name=display_name or f"{aggregation.value}_{measure_name}",
            description=description,
            format_str=format_str,
            sort_order=sort_order
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "measure_id": self.measure_id,
            "node_id": self.node_id,
            "measure_name": self.measure_name,
            "aggregation": self.aggregation.value if isinstance(self.aggregation, Aggregation) else self.aggregation,
            "column_expression": self.column_expression,
            "display_name": self.display_name,
            "description": self.description,
            "format_str": self.format_str,
            "sort_order": self.sort_order
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Measure":
        aggregation = data["aggregation"]
        if isinstance(aggregation, str):
            aggregation = Aggregation(aggregation)
        return cls(
            measure_id=data["measure_id"],
            node_id=data["node_id"],
            measure_name=data["measure_name"],
            aggregation=aggregation,
            column_expression=data["column_expression"],
            display_name=data.get("display_name"),
            description=data.get("description"),
            format_str=data.get("format_str"),
            sort_order=data.get("sort_order", 0)
        )


@dataclass
class SemanticLayer:
    """セマンティックレイヤー全体"""
    project_id: str
    name: str
    connector_type: str  # snowflake, bigquery, databricks, duckdb
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    dimensions: List[Dimension] = field(default_factory=list)
    measures: List[Measure] = field(default_factory=list)
    description: Optional[str] = None

    @classmethod
    def create(cls, name: str, connector_type: str, description: Optional[str] = None) -> "SemanticLayer":
        return cls(
            project_id=str(uuid.uuid4())[:8],
            name=name,
            connector_type=connector_type,
            description=description
        )

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def add_dimension(self, dimension: Dimension) -> None:
        self.dimensions.append(dimension)

    def add_measure(self, measure: Measure) -> None:
        self.measures.append(measure)

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_node_by_name(self, node_name: str) -> Optional[Node]:
        for node in self.nodes:
            if node.node_name == node_name:
                return node
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "connector_type": self.connector_type,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "dimensions": [d.to_dict() for d in self.dimensions],
            "measures": [m.to_dict() for m in self.measures]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticLayer":
        layer = cls(
            project_id=data["project_id"],
            name=data["name"],
            connector_type=data["connector_type"],
            description=data.get("description")
        )
        layer.nodes = [Node.from_dict(n) for n in data.get("nodes", [])]
        layer.edges = [Edge.from_dict(e) for e in data.get("edges", [])]
        layer.dimensions = [Dimension.from_dict(d) for d in data.get("dimensions", [])]
        layer.measures = [Measure.from_dict(m) for m in data.get("measures", [])]
        return layer


# データ型分類（ディメンション/ファクト判定用）
DIMENSION_TYPES = [
    'VARCHAR', 'TEXT', 'CHAR', 'STRING', 'DATE', 'TIMESTAMP', 'TIMESTAMP_NTZ',
    'TIMESTAMP_LTZ', 'TIMESTAMP_TZ', 'BOOLEAN', 'TIME', 'DATETIME'
]

FACT_TYPES = [
    'NUMBER', 'FLOAT', 'DECIMAL', 'INTEGER', 'INT', 'BIGINT', 'SMALLINT',
    'TINYINT', 'DOUBLE', 'REAL', 'NUMERIC', 'INT64', 'FLOAT64'
]


def classify_column(column_name: str, data_type: str) -> str:
    """カラムをディメンション/ファクト候補に分類"""
    data_type_upper = data_type.upper().split('(')[0]  # NUMBER(10,2) -> NUMBER

    # IDカラムは除外（通常は集計対象外）
    is_id_col = column_name.upper().endswith('_ID') or column_name.upper() == 'ID'

    if data_type_upper in FACT_TYPES and not is_id_col:
        return "FACT"
    elif data_type_upper in DIMENSION_TYPES or is_id_col:
        return "DIMENSION"
    else:
        return "DIMENSION"  # 不明な型はディメンション扱い
