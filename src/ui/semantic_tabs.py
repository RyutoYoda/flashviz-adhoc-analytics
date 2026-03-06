"""
Semantic Layer UI Components
セマンティックレイヤービルダーのUI
"""
import streamlit as st
import pandas as pd
from typing import Optional, Dict, List, Any

from ..domain.semantic import (
    SemanticLayer, Node, Edge, Dimension, Measure,
    NodeType, JoinType, Aggregation, classify_column
)
from ..infrastructure.semantic import (
    SQLGenerator, GPTContextFormatter, SemanticLayerStorage,
    build_graph, find_all_required_nodes
)


def init_semantic_state():
    """セマンティックレイヤーのセッション状態を初期化"""
    if 'semantic_layer' not in st.session_state:
        st.session_state.semantic_layer = None
    if 'semantic_storage' not in st.session_state:
        st.session_state.semantic_storage = SemanticLayerStorage("data/semantic_layer.duckdb")
    if 'generated_query_sql' not in st.session_state:
        st.session_state.generated_query_sql = None
    if 'query_result_df' not in st.session_state:
        st.session_state.query_result_df = None


def render_semantic_sidebar(connector_type: str = "duckdb"):
    """セマンティックレイヤーサイドバー"""
    st.markdown("### セマンティックレイヤー")

    storage = st.session_state.semantic_storage

    # プロジェクト選択
    projects = storage.list_projects()
    project_names = ["新規作成"] + [p["name"] for p in projects]

    selected_project = st.selectbox(
        "プロジェクト",
        project_names,
        key="sl_project_select"
    )

    if selected_project == "新規作成":
        with st.expander("新規プロジェクト作成", expanded=True):
            new_name = st.text_input("プロジェクト名", key="sl_new_name")
            new_desc = st.text_area("説明", height=60, key="sl_new_desc")

            if st.button("作成", key="sl_create"):
                if new_name:
                    layer = SemanticLayer.create(
                        name=new_name,
                        connector_type=connector_type,
                        description=new_desc
                    )
                    storage.save_semantic_layer(layer)
                    st.session_state.semantic_layer = layer
                    st.success(f"'{new_name}' を作成しました")
                    st.rerun()
                else:
                    st.error("プロジェクト名を入力してください")
    else:
        # 既存プロジェクトを読み込み
        project = next((p for p in projects if p["name"] == selected_project), None)
        if project:
            layer = storage.load_semantic_layer(project["project_id"])
            if layer:
                st.session_state.semantic_layer = layer

                col1, col2 = st.columns([3, 1])
                with col2:
                    if st.button("🗑️", key="sl_delete", help="プロジェクトを削除"):
                        storage.delete_project(layer.project_id)
                        st.session_state.semantic_layer = None
                        st.rerun()

    # 現在のセマンティックレイヤー情報
    if st.session_state.semantic_layer:
        layer = st.session_state.semantic_layer
        st.divider()
        st.metric("ノード", len(layer.nodes))
        st.metric("リレーション", len(layer.edges))
        st.metric("ディメンション", len(layer.dimensions))
        st.metric("メジャー", len(layer.measures))


def render_tables_tab(get_table_schema_func=None, get_available_tables_func=None):
    """テーブル（ノード）管理タブ"""
    st.header("テーブル登録")

    layer = st.session_state.semantic_layer
    if not layer:
        st.warning("先にプロジェクトを選択または作成してください")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("ノード追加")

        node_type = st.selectbox(
            "ノードタイプ",
            [t.value for t in NodeType],
            key="sl_node_type"
        )

        node_name = st.text_input("ノード名", key="sl_node_name")
        group_name = st.text_input("グループ名", key="sl_group_name")
        description = st.text_area("説明", height=60, key="sl_node_desc")

        if node_type == "SOURCE":
            # テーブル一覧取得（コネクタから）
            tables = []
            if get_available_tables_func:
                tables = get_available_tables_func()

            if tables:
                # テーブル候補がある場合はセレクトボックス + 手動入力オプション
                table_options = tables + ["（手動入力）"]
                selected_table = st.selectbox("ソーステーブル", table_options, key="sl_source")

                if selected_table == "（手動入力）":
                    source_object = st.text_input(
                        "テーブル名を入力",
                        placeholder="DATABASE.SCHEMA.TABLE",
                        key="sl_source_manual"
                    )
                else:
                    source_object = selected_table
            else:
                # テーブル候補がない場合は手動入力
                st.info("データソースを接続するとテーブル候補が表示されます")
                source_object = st.text_input(
                    "ソースオブジェクト",
                    placeholder="DATABASE.SCHEMA.TABLE",
                    key="sl_source_manual"
                )
            transform_sql = None
        else:
            source_object = None
            transform_sql = st.text_area(
                "変換SQL",
                height=150,
                key="sl_transform_sql"
            )

        if st.button("追加", type="primary", key="sl_add_node"):
            if not node_name:
                st.error("ノード名は必須です")
            else:
                node = Node.create(
                    name=node_name,
                    node_type=NodeType(node_type),
                    source_object=source_object,
                    transform_sql=transform_sql,
                    description=description,
                    group_name=group_name
                )
                layer.add_node(node)
                st.session_state.semantic_storage.save_semantic_layer(layer)
                st.success(f"ノード '{node_name}' を追加しました")
                st.rerun()

    with col2:
        st.subheader("登録済みノード")

        if not layer.nodes:
            st.info("ノードが登録されていません")
            return

        for node_type in NodeType:
            type_nodes = [n for n in layer.nodes if n.node_type == node_type]
            if type_nodes:
                with st.expander(f"{node_type.value} ({len(type_nodes)}件)", expanded=True):
                    for node in type_nodes:
                        col_a, col_b = st.columns([4, 1])
                        with col_a:
                            group_badge = f"[{node.group_name}] " if node.group_name else ""
                            st.markdown(f"**{node.node_name}** {group_badge}`{node.node_id}`")
                            if node.source_object:
                                st.caption(f"参照: {node.source_object}")
                            if node.description:
                                st.caption(node.description)
                        with col_b:
                            if st.button("削除", key=f"del_node_{node.node_id}"):
                                layer.nodes = [n for n in layer.nodes if n.node_id != node.node_id]
                                st.session_state.semantic_storage.save_semantic_layer(layer)
                                st.rerun()


def render_relations_tab(get_table_columns_func=None):
    """リレーション（JOIN定義）管理タブ"""
    st.header("リレーション管理")

    layer = st.session_state.semantic_layer
    if not layer:
        st.warning("先にプロジェクトを選択または作成してください")
        return

    if not layer.nodes:
        st.warning("先にテーブルタブでノードを追加してください")
        return

    node_options = {node.node_name: node.node_id for node in layer.nodes}

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("JOIN定義を追加")

        st.markdown("**左テーブル (t0)**")
        from_node_name = st.selectbox(
            "左テーブル",
            list(node_options.keys()),
            key="sl_from_node",
            label_visibility="collapsed"
        )

        st.markdown("**右テーブル (t1)**")
        to_node_name = st.selectbox(
            "右テーブル",
            list(node_options.keys()),
            key="sl_to_node",
            label_visibility="collapsed"
        )

        join_type = st.selectbox(
            "JOIN種別",
            [t.value for t in JoinType],
            key="sl_join_type"
        )

        join_condition = st.text_input(
            "JOIN条件",
            placeholder="t0.CUSTOMER_ID = t1.CUSTOMER_ID",
            key="sl_join_cond"
        )

        description = st.text_input("説明（任意）", key="sl_edge_desc")

        if st.button("追加", type="primary", key="sl_add_edge"):
            if from_node_name == to_node_name:
                st.error("左右のテーブルは異なる必要があります")
            elif not join_condition:
                st.error("JOIN条件は必須です")
            else:
                edge = Edge.create(
                    from_node_id=node_options[from_node_name],
                    to_node_id=node_options[to_node_name],
                    join_type=JoinType(join_type),
                    join_condition=join_condition,
                    description=description
                )
                layer.add_edge(edge)
                st.session_state.semantic_storage.save_semantic_layer(layer)
                st.success("JOIN定義を追加しました")
                st.rerun()

    with col2:
        st.subheader("登録済みJOIN定義")

        if not layer.edges:
            st.info("JOIN定義がありません")
            return

        node_id_to_name = {n.node_id: n.node_name for n in layer.nodes}

        for edge in layer.edges:
            from_name = node_id_to_name.get(edge.from_node_id, "Unknown")
            to_name = node_id_to_name.get(edge.to_node_id, "Unknown")
            join_type = edge.join_type.value if hasattr(edge.join_type, 'value') else edge.join_type

            col_a, col_b = st.columns([5, 1])
            with col_a:
                st.markdown(f"**{from_name}** (t0) {join_type} JOIN **{to_name}** (t1)")
                st.code(f"ON {edge.join_condition}", language="sql")
            with col_b:
                if st.button("削除", key=f"del_edge_{edge.edge_id}"):
                    layer.edges = [e for e in layer.edges if e.edge_id != edge.edge_id]
                    st.session_state.semantic_storage.save_semantic_layer(layer)
                    st.rerun()
            st.divider()


def render_dimensions_tab(get_table_columns_func=None):
    """ディメンション属性管理タブ"""
    st.header("ディメンション属性")

    layer = st.session_state.semantic_layer
    if not layer:
        st.warning("先にプロジェクトを選択または作成してください")
        return

    # SOURCE/DIMENSION/TRANSFORMノードのみ対象
    dim_nodes = [n for n in layer.nodes if n.node_type in [NodeType.SOURCE, NodeType.DIMENSION, NodeType.TRANSFORM]]
    if not dim_nodes:
        st.warning("対象ノードがありません")
        return

    node_options = {node.node_name: node.node_id for node in dim_nodes}

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("属性追加")

        selected_node_name = st.selectbox(
            "ノード",
            list(node_options.keys()),
            key="sl_dim_node"
        )

        # 選択されたノードのカラム候補を取得
        columns = []
        if selected_node_name:
            selected_node = next((n for n in dim_nodes if n.node_name == selected_node_name), None)
            if selected_node and selected_node.source_object and get_table_columns_func:
                columns = get_table_columns_func(selected_node.source_object)

        if columns:
            st.markdown("**カラムから選択:**")
            # 既存のディメンションを除外
            existing_cols = set()
            node_id = node_options[selected_node_name]
            for dim in layer.dimensions:
                if dim.node_id == node_id:
                    existing_cols.add(dim.column_expression)

            available_cols = [c for c in columns if c not in existing_cols]

            if available_cols:
                selected_cols = st.multiselect(
                    "カラムを選択",
                    available_cols,
                    key="sl_dim_cols_select"
                )

                if st.button("選択したカラムを一括追加", type="primary", key="sl_add_dims_bulk"):
                    if selected_cols:
                        for idx, col_name in enumerate(selected_cols):
                            dim = Dimension.create(
                                node_id=node_options[selected_node_name],
                                attr_name=col_name,
                                column_expression=col_name,
                                display_name=col_name,
                                sort_order=idx
                            )
                            layer.add_dimension(dim)
                        st.session_state.semantic_storage.save_semantic_layer(layer)
                        st.success(f"{len(selected_cols)}件の属性を追加しました")
                        st.rerun()
            else:
                st.caption("全カラム登録済み")

            st.markdown("---")

        st.markdown("**手動入力:**")
        attr_name = st.text_input("属性名", key="sl_attr_name")
        column_expr = st.text_input(
            "カラム式",
            placeholder="COLUMN_NAME",
            key="sl_col_expr"
        )
        display_name = st.text_input("表示名（任意）", key="sl_disp_name")
        description = st.text_input("説明（任意）", key="sl_dim_desc")
        sort_order = st.number_input("表示順", min_value=0, value=0, key="sl_dim_sort")

        if st.button("追加", key="sl_add_dim"):
            if not attr_name or not column_expr:
                st.error("属性名とカラム式は必須です")
            else:
                dim = Dimension.create(
                    node_id=node_options[selected_node_name],
                    attr_name=attr_name,
                    column_expression=column_expr,
                    display_name=display_name,
                    description=description,
                    sort_order=sort_order
                )
                layer.add_dimension(dim)
                st.session_state.semantic_storage.save_semantic_layer(layer)
                st.success("属性を追加しました")
                st.rerun()

    with col2:
        st.subheader("登録済み属性")

        if not layer.dimensions:
            st.info("ディメンション属性が登録されていません")
            return

        node_id_to_name = {n.node_id: n.node_name for n in layer.nodes}

        # ノードごとにグループ化
        dims_by_node: Dict[str, List[Dimension]] = {}
        for dim in layer.dimensions:
            if dim.node_id not in dims_by_node:
                dims_by_node[dim.node_id] = []
            dims_by_node[dim.node_id].append(dim)

        for node_id, dims in dims_by_node.items():
            node_name = node_id_to_name.get(node_id, "Unknown")
            with st.expander(f"{node_name} ({len(dims)}件)", expanded=True):
                for dim in sorted(dims, key=lambda d: d.sort_order):
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        display = dim.display_name or dim.attr_name
                        st.markdown(f"**{display}** `{dim.column_expression}`")
                    with col_b:
                        if st.button("削除", key=f"del_dim_{dim.attr_id}"):
                            layer.dimensions = [d for d in layer.dimensions if d.attr_id != dim.attr_id]
                            st.session_state.semantic_storage.save_semantic_layer(layer)
                            st.rerun()


def render_measures_tab(get_table_columns_func=None):
    """ファクトメジャー管理タブ"""
    st.header("ファクトメジャー")

    layer = st.session_state.semantic_layer
    if not layer:
        st.warning("先にプロジェクトを選択または作成してください")
        return

    # SOURCE/FACT/TRANSFORMノードのみ対象
    fact_nodes = [n for n in layer.nodes if n.node_type in [NodeType.SOURCE, NodeType.FACT, NodeType.TRANSFORM]]
    if not fact_nodes:
        st.warning("対象ノードがありません")
        return

    node_options = {node.node_name: node.node_id for node in fact_nodes}

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("メジャー追加")

        selected_node_name = st.selectbox(
            "ノード",
            list(node_options.keys()),
            key="sl_fact_node"
        )

        # 選択されたノードのカラム候補を取得
        columns = []
        if selected_node_name:
            selected_node = next((n for n in fact_nodes if n.node_name == selected_node_name), None)
            if selected_node and selected_node.source_object and get_table_columns_func:
                columns = get_table_columns_func(selected_node.source_object)

        if columns:
            st.markdown("**カラムから選択:**")
            # 既存のメジャーを除外
            existing_cols = set()
            node_id = node_options[selected_node_name]
            for m in layer.measures:
                if m.node_id == node_id:
                    existing_cols.add(m.column_expression)

            available_cols = [c for c in columns if c not in existing_cols]

            if available_cols:
                selected_cols = st.multiselect(
                    "カラムを選択",
                    available_cols,
                    key="sl_measure_cols_select"
                )

                bulk_agg = st.selectbox(
                    "集計関数（一括）",
                    [a.value for a in Aggregation],
                    key="sl_bulk_agg"
                )

                if st.button("選択したカラムを一括追加", type="primary", key="sl_add_measures_bulk"):
                    if selected_cols:
                        for idx, col_name in enumerate(selected_cols):
                            measure = Measure.create(
                                node_id=node_options[selected_node_name],
                                measure_name=col_name,
                                aggregation=Aggregation(bulk_agg),
                                column_expression=col_name,
                                display_name=f"{bulk_agg}_{col_name}",
                                sort_order=idx
                            )
                            layer.add_measure(measure)
                        st.session_state.semantic_storage.save_semantic_layer(layer)
                        st.success(f"{len(selected_cols)}件のメジャーを追加しました")
                        st.rerun()
            else:
                st.caption("全カラム登録済み")

            st.markdown("---")

        st.markdown("**手動入力:**")
        measure_name = st.text_input("メジャー名", key="sl_measure_name")
        aggregation = st.selectbox(
            "集計関数",
            [a.value for a in Aggregation],
            key="sl_agg"
        )
        column_expr = st.text_input(
            "カラム式",
            placeholder="COLUMN_NAME",
            key="sl_fact_col_expr"
        )
        display_name = st.text_input("表示名（任意）", key="sl_fact_disp_name")
        format_str = st.text_input("書式（任意）", placeholder="#,##0", key="sl_format")
        description = st.text_input("説明（任意）", key="sl_fact_desc")
        sort_order = st.number_input("表示順", min_value=0, value=0, key="sl_fact_sort")

        if st.button("追加", key="sl_add_measure"):
            if not measure_name or not column_expr:
                st.error("メジャー名とカラム式は必須です")
            else:
                measure = Measure.create(
                    node_id=node_options[selected_node_name],
                    measure_name=measure_name,
                    aggregation=Aggregation(aggregation),
                    column_expression=column_expr,
                    display_name=display_name,
                    description=description,
                    format_str=format_str,
                    sort_order=sort_order
                )
                layer.add_measure(measure)
                st.session_state.semantic_storage.save_semantic_layer(layer)
                st.success("メジャーを追加しました")
                st.rerun()

    with col2:
        st.subheader("登録済みメジャー")

        if not layer.measures:
            st.info("メジャーが登録されていません")
            return

        node_id_to_name = {n.node_id: n.node_name for n in layer.nodes}

        # ノードごとにグループ化
        measures_by_node: Dict[str, List[Measure]] = {}
        for m in layer.measures:
            if m.node_id not in measures_by_node:
                measures_by_node[m.node_id] = []
            measures_by_node[m.node_id].append(m)

        for node_id, measures in measures_by_node.items():
            node_name = node_id_to_name.get(node_id, "Unknown")
            with st.expander(f"{node_name} ({len(measures)}件)", expanded=True):
                for m in sorted(measures, key=lambda x: x.sort_order):
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        display = m.display_name or m.measure_name
                        agg = m.aggregation.value if hasattr(m.aggregation, 'value') else m.aggregation
                        st.markdown(f"**{display}** = `{agg}({m.column_expression})`")
                    with col_b:
                        if st.button("削除", key=f"del_measure_{m.measure_id}"):
                            layer.measures = [x for x in layer.measures if x.measure_id != m.measure_id]
                            st.session_state.semantic_storage.save_semantic_layer(layer)
                            st.rerun()


def render_query_builder_tab(execute_query_func=None):
    """クエリビルダータブ"""
    st.header("クエリビルダー")

    layer = st.session_state.semantic_layer
    if not layer:
        st.warning("先にプロジェクトを選択または作成してください")
        return

    if not layer.dimensions and not layer.measures:
        st.warning("先にディメンション属性またはファクトメジャーを定義してください")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("フィールド選択")

        node_id_to_name = {n.node_id: n.node_name for n in layer.nodes}

        st.markdown("**ディメンション (GROUP BY)**")
        selected_dims = []
        if layer.dimensions:
            dims_by_node: Dict[str, List[Dimension]] = {}
            for dim in layer.dimensions:
                if dim.node_id not in dims_by_node:
                    dims_by_node[dim.node_id] = []
                dims_by_node[dim.node_id].append(dim)

            for node_id, dims in dims_by_node.items():
                node_name = node_id_to_name.get(node_id, "Unknown")
                st.caption(f"[{node_name}]")
                for dim in sorted(dims, key=lambda d: d.sort_order):
                    display = dim.display_name or dim.attr_name
                    if st.checkbox(display, key=f"cb_dim_{dim.attr_id}"):
                        selected_dims.append(dim.attr_id)

        st.markdown("---")

        st.markdown("**メジャー (集計)**")
        selected_measures = []
        if layer.measures:
            measures_by_node: Dict[str, List[Measure]] = {}
            for m in layer.measures:
                if m.node_id not in measures_by_node:
                    measures_by_node[m.node_id] = []
                measures_by_node[m.node_id].append(m)

            for node_id, measures in measures_by_node.items():
                node_name = node_id_to_name.get(node_id, "Unknown")
                st.caption(f"[{node_name}]")
                for m in sorted(measures, key=lambda x: x.sort_order):
                    display = m.display_name or m.measure_name
                    agg = m.aggregation.value if hasattr(m.aggregation, 'value') else m.aggregation
                    label = f"{display} ({agg})"
                    if st.checkbox(label, key=f"cb_measure_{m.measure_id}"):
                        selected_measures.append(m.measure_id)

        generate_btn = st.button("SQL生成", type="primary", use_container_width=True)

    with col2:
        st.subheader("生成SQL と 実行結果")

        if generate_btn:
            if not selected_dims and not selected_measures:
                st.error("ディメンションまたはメジャーを1つ以上選択してください")
            else:
                generator = SQLGenerator(layer)
                sql, error = generator.generate_sql(selected_dims, selected_measures)

                if error:
                    st.error(error)
                else:
                    st.session_state.generated_query_sql = sql

        if st.session_state.generated_query_sql:
            st.code(st.session_state.generated_query_sql, language="sql")

            col_a, col_b = st.columns(2)
            with col_a:
                if execute_query_func and st.button("クエリ実行", type="primary"):
                    with st.spinner("実行中..."):
                        try:
                            result_df = execute_query_func(st.session_state.generated_query_sql)
                            if result_df is not None and not result_df.empty:
                                st.session_state.query_result_df = result_df
                            else:
                                st.warning("結果が0件でした")
                        except Exception as e:
                            st.error(f"実行エラー: {e}")

            with col_b:
                if st.button("SQLクリア"):
                    st.session_state.generated_query_sql = None
                    st.session_state.query_result_df = None
                    st.rerun()

            if st.session_state.query_result_df is not None:
                result_df = st.session_state.query_result_df

                st.subheader("実行結果")
                st.dataframe(result_df, use_container_width=True)

                csv = result_df.to_csv(index=False)
                st.download_button(
                    "CSVダウンロード",
                    csv,
                    "query_result.csv",
                    "text/csv"
                )
        else:
            st.info("ディメンションやメジャーを選択し、「SQL生成」をクリックしてください")


def render_lineage_tab():
    """リネージタブ"""
    st.header("データリネージ")

    layer = st.session_state.semantic_layer
    if not layer:
        st.warning("先にプロジェクトを選択または作成してください")
        return

    if not layer.nodes:
        st.warning("ノードが登録されていません")
        return

    # ノードタイプ別の色
    type_colors = {
        NodeType.SOURCE: '#22c55e',
        NodeType.TRANSFORM: '#a855f7',
        NodeType.DIMENSION: '#3b82f6',
        NodeType.FACT: '#f97316'
    }

    # ノード一覧テーブル
    st.subheader("All Nodes")

    table_data = []
    for node in layer.nodes:
        attr_count = len([d for d in layer.dimensions if d.node_id == node.node_id])
        measure_count = len([m for m in layer.measures if m.node_id == node.node_id])

        info = []
        if attr_count > 0:
            info.append(f"{attr_count} dims")
        if measure_count > 0:
            info.append(f"{measure_count} measures")

        source = node.source_object or '-'
        if len(source) > 40:
            source = '...' + source[-37:]

        node_type = node.node_type.value if hasattr(node.node_type, 'value') else node.node_type

        table_data.append({
            'NAME': node.node_name,
            'ID': node.node_id,
            'SOURCE': source,
            'TYPE': node_type,
            'ATTRS/MEASURES': ', '.join(info) if info else '-'
        })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True)
    st.caption("TYPE: SOURCE(緑)=データソース, TRANSFORM(紫)=変換, DIMENSION(青)=切り口, FACT(オレンジ)=集計")

    st.markdown("---")

    # DAGグラフ
    st.subheader("Lineage Graph")

    # Graphviz DOT形式
    dot_lines = [
        "digraph G {",
        "  rankdir=LR;",
        "  bgcolor=transparent;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [color="#64748b", arrowsize=0.7, penwidth=1.5];'
    ]

    for node in layer.nodes:
        node_type = node.node_type if isinstance(node.node_type, NodeType) else NodeType(node.node_type)
        color = type_colors.get(node_type, '#6b7280')

        attr_count = len([d for d in layer.dimensions if d.node_id == node.node_id])
        measure_count = len([m for m in layer.measures if m.node_id == node.node_id])

        extra = []
        if attr_count > 0:
            extra.append(f"{attr_count}D")
        if measure_count > 0:
            extra.append(f"{measure_count}M")
        extra_str = f" ({','.join(extra)})" if extra else ""

        label = f"{node.node_name}\\n{node_type.value}{extra_str}"
        dot_lines.append(f'  "{node.node_id}" [label="{label}", fillcolor="{color}", fontcolor="white"];')

    for edge in layer.edges:
        dot_lines.append(f'  "{edge.from_node_id}" -> "{edge.to_node_id}";')

    dot_lines.append("}")
    dot_source = "\n".join(dot_lines)

    try:
        st.graphviz_chart(dot_source, use_container_width=True)
    except Exception as e:
        st.code(dot_source, language="dot")

    if not layer.edges:
        st.info("リレーションを定義すると、ノード間の接続が矢印で表示されます。")


def render_export_tab():
    """エクスポート/インポートタブ"""
    st.header("エクスポート / インポート")

    layer = st.session_state.semantic_layer
    storage = st.session_state.semantic_storage

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("エクスポート")

        if layer:
            json_str = storage.export_to_json(layer.project_id)
            if json_str:
                st.download_button(
                    "JSONダウンロード",
                    json_str,
                    f"{layer.name}_semantic_layer.json",
                    "application/json",
                    type="primary"
                )

                with st.expander("プレビュー"):
                    st.code(json_str, language="json")
        else:
            st.info("プロジェクトを選択してください")

    with col2:
        st.subheader("インポート")

        uploaded_file = st.file_uploader(
            "JSONファイルをアップロード",
            type=["json"],
            key="sl_import_file"
        )

        if uploaded_file:
            if st.button("インポート", type="primary"):
                try:
                    json_str = uploaded_file.read().decode('utf-8')
                    imported_layer = storage.import_from_json(json_str)
                    if imported_layer:
                        st.session_state.semantic_layer = imported_layer
                        st.success(f"'{imported_layer.name}' をインポートしました")
                        st.rerun()
                    else:
                        st.error("インポートに失敗しました")
                except Exception as e:
                    st.error(f"エラー: {e}")


def get_gpt_system_prompt(dialect: str = "snowflake") -> Optional[str]:
    """GPT用システムプロンプトを取得"""
    layer = st.session_state.semantic_layer
    if not layer:
        return None

    formatter = GPTContextFormatter(layer)
    return formatter.format_system_prompt(dialect)
