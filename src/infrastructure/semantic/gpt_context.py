"""
GPT Context Formatter
セマンティックレイヤーをGPTのシステムプロンプトにフォーマット
"""
from typing import Dict, List, Optional, Any
from ...domain.semantic import SemanticLayer, Node, Edge, Dimension, Measure, NodeType


class GPTContextFormatter:
    """GPTコンテキストフォーマッター"""

    def __init__(self, semantic_layer: SemanticLayer):
        self.semantic_layer = semantic_layer
        self._node_map: Dict[str, Node] = {
            node.node_id: node for node in semantic_layer.nodes
        }

    def format_system_prompt(self, dialect: str = "snowflake") -> str:
        """
        セマンティックレイヤーをGPT用システムプロンプトにフォーマット

        Args:
            dialect: SQLダイアレクト (snowflake, bigquery, databricks, duckdb)

        Returns:
            システムプロンプト文字列
        """
        sections = [
            self._format_header(dialect),
            self._format_tables(),
            self._format_relationships(),
            self._format_dimensions(),
            self._format_measures(),
            self._format_instructions(dialect)
        ]

        return "\n\n".join(filter(None, sections))

    def _format_header(self, dialect: str) -> str:
        """ヘッダーセクション"""
        return f"""You are a SQL expert specializing in {dialect} SQL.
You have access to a semantic layer model that defines the data structure.
Generate {dialect}-compatible SQL queries based on the user's natural language questions.

## Semantic Layer: {self.semantic_layer.name}
{self.semantic_layer.description or 'No description provided.'}"""

    def _format_tables(self) -> str:
        """テーブル情報セクション"""
        if not self.semantic_layer.nodes:
            return ""

        lines = ["## Available Tables"]

        for node in self.semantic_layer.nodes:
            if node.node_type in [NodeType.SOURCE, NodeType.DIMENSION, NodeType.FACT]:
                alias = node.node_name.lower().replace(' ', '_')
                source = node.source_object or "(derived)"
                desc = f" - {node.description}" if node.description else ""

                lines.append(f"- **{alias}**: `{source}`{desc}")

                # カラム情報があれば追加
                if node.columns:
                    col_info = ", ".join([
                        f"{c.name}({c.data_type})"
                        for c in node.columns[:10]  # 最大10カラム
                    ])
                    if len(node.columns) > 10:
                        col_info += f", ... (+{len(node.columns) - 10} more)"
                    lines.append(f"  Columns: {col_info}")

        return "\n".join(lines)

    def _format_relationships(self) -> str:
        """リレーション情報セクション"""
        if not self.semantic_layer.edges:
            return ""

        lines = ["## Relationships (JOINs)"]

        for edge in self.semantic_layer.edges:
            from_node = self._node_map.get(edge.from_node_id)
            to_node = self._node_map.get(edge.to_node_id)

            if from_node and to_node:
                from_alias = from_node.node_name.lower().replace(' ', '_')
                to_alias = to_node.node_name.lower().replace(' ', '_')
                join_type = edge.join_type.value if hasattr(edge.join_type, 'value') else edge.join_type

                lines.append(f"- {from_alias} **{join_type} JOIN** {to_alias}")
                lines.append(f"  ON: `{edge.join_condition}`")

        return "\n".join(lines)

    def _format_dimensions(self) -> str:
        """ディメンション情報セクション"""
        if not self.semantic_layer.dimensions:
            return ""

        lines = ["## Dimensions (GROUP BY candidates)"]

        # ノードごとにグループ化
        dims_by_node: Dict[str, List[Dimension]] = {}
        for dim in self.semantic_layer.dimensions:
            if dim.node_id not in dims_by_node:
                dims_by_node[dim.node_id] = []
            dims_by_node[dim.node_id].append(dim)

        for node_id, dims in dims_by_node.items():
            node = self._node_map.get(node_id)
            node_name = node.node_name if node else "Unknown"

            lines.append(f"\n### {node_name}")
            for dim in sorted(dims, key=lambda d: d.sort_order):
                display = dim.display_name or dim.attr_name
                desc = f" - {dim.description}" if dim.description else ""
                lines.append(f"- **{display}**: `{dim.column_expression}`{desc}")

        return "\n".join(lines)

    def _format_measures(self) -> str:
        """メジャー情報セクション"""
        if not self.semantic_layer.measures:
            return ""

        lines = ["## Measures (Aggregations)"]

        # ノードごとにグループ化
        measures_by_node: Dict[str, List[Measure]] = {}
        for measure in self.semantic_layer.measures:
            if measure.node_id not in measures_by_node:
                measures_by_node[measure.node_id] = []
            measures_by_node[measure.node_id].append(measure)

        for node_id, measures in measures_by_node.items():
            node = self._node_map.get(node_id)
            node_name = node.node_name if node else "Unknown"

            lines.append(f"\n### {node_name}")
            for m in sorted(measures, key=lambda x: x.sort_order):
                display = m.display_name or m.measure_name
                agg = m.aggregation.value if hasattr(m.aggregation, 'value') else m.aggregation
                desc = f" - {m.description}" if m.description else ""
                lines.append(f"- **{display}**: `{agg}({m.column_expression})`{desc}")

        return "\n".join(lines)

    def _format_instructions(self, dialect: str) -> str:
        """SQL生成指示セクション"""
        dialect_specific = self._get_dialect_instructions(dialect)

        return f"""## SQL Generation Instructions

1. Use the tables, relationships, dimensions, and measures defined above
2. Generate only SELECT queries - no modifications allowed
3. Use the exact column expressions defined in dimensions and measures
4. Apply appropriate JOINs based on the relationships defined
5. Include GROUP BY clause when using aggregation functions
6. Add ORDER BY when results should be sorted

{dialect_specific}

## Response Format
- Return ONLY the SQL query
- No explanations or markdown code blocks
- The query should be executable as-is"""

    def _get_dialect_instructions(self, dialect: str) -> str:
        """ダイアレクト固有の指示"""
        instructions = {
            "snowflake": """### Snowflake-specific:
- Use double quotes for identifiers with special characters: "column_name"
- Date functions: DATE_TRUNC('month', date_col), DATEADD(), DATEDIFF()
- String functions: CONCAT(), SPLIT_PART(), REGEXP_SUBSTR()
- Use QUALIFY for window function filtering""",

            "bigquery": """### BigQuery-specific:
- Use backticks for identifiers: `column_name`
- Full table names: `project.dataset.table`
- Date functions: DATE_TRUNC(date_col, MONTH), DATE_ADD(), DATE_DIFF()
- Use UNNEST for array operations""",

            "databricks": """### Databricks-specific:
- Use backticks for identifiers: `column_name`
- Unity Catalog format: catalog.schema.table
- Date functions: date_trunc('month', date_col), date_add(), datediff()
- Spark SQL syntax""",

            "duckdb": """### DuckDB-specific:
- Use double quotes for identifiers: "column_name"
- Date functions: DATE_TRUNC('month', date_col), DATE_ADD(), DATE_DIFF()
- Supports modern SQL features like QUALIFY, ASOF JOINs"""
        }

        return instructions.get(dialect.lower(), instructions["duckdb"])

    def format_query_context(
        self,
        user_question: str,
        selected_dimensions: Optional[List[str]] = None,
        selected_measures: Optional[List[str]] = None
    ) -> str:
        """
        ユーザー質問に対するクエリコンテキストをフォーマット

        Args:
            user_question: ユーザーの自然言語質問
            selected_dimensions: 選択されたディメンション名リスト
            selected_measures: 選択されたメジャー名リスト

        Returns:
            クエリコンテキスト文字列
        """
        context_parts = [f"User Question: {user_question}"]

        if selected_dimensions:
            context_parts.append(f"Selected Dimensions: {', '.join(selected_dimensions)}")

        if selected_measures:
            context_parts.append(f"Selected Measures: {', '.join(selected_measures)}")

        context_parts.append(
            "\nGenerate a SQL query that answers this question using the semantic layer model."
        )

        return "\n".join(context_parts)


def create_enhanced_prompt(
    semantic_layer: SemanticLayer,
    user_question: str,
    dialect: str = "snowflake",
    sample_data: Optional[str] = None
) -> Dict[str, str]:
    """
    強化されたGPTプロンプトを作成

    Args:
        semantic_layer: セマンティックレイヤー
        user_question: ユーザーの質問
        dialect: SQLダイアレクト
        sample_data: サンプルデータ（オプション）

    Returns:
        {"system": system_prompt, "user": user_prompt}
    """
    formatter = GPTContextFormatter(semantic_layer)

    system_prompt = formatter.format_system_prompt(dialect)

    user_parts = [f"Question: {user_question}"]

    if sample_data:
        user_parts.append(f"\nSample Data:\n{sample_data}")

    user_parts.append("\nGenerate the SQL query:")

    return {
        "system": system_prompt,
        "user": "\n".join(user_parts)
    }
