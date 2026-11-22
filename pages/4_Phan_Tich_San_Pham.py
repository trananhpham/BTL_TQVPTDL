"""Page 4: Phân tích sản phẩm — Top N, Sunburst/Treemap phân cấp & Market Basket (Apriori)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.sidebar import init_page, render_global_sidebar
from data.analytics import (
    build_apriori_network_elements,
    compute_apriori_lift_matrix,
    compute_pareto_abc,
    generate_smart_cross_sell_bundles,
    run_market_basket_apriori,
    summarize_by_category,
    summarize_top_products,
)
from utils.ui_styles import CATEGORY_COLORS, render_dynamic_insight_banner

init_page("Page 4 - Sản phẩm, Sunburst & Giỏ hàng", icon="🏆")
_, _, df_sidebar = render_global_sidebar()


@st.cache_data(show_spinner="Đang khai phá luật kết hợp Apriori...")
def cached_apriori(
    data: pd.DataFrame,
    item_col: str,
    min_support: float,
    min_confidence: float,
) -> pd.DataFrame:
    """Cache kết quả Market Basket Analysis để phản hồi mượt mà."""
    return run_market_basket_apriori(
        data,
        item_col=item_col,
        min_support=min_support,
        min_confidence=min_confidence,
    )


@st.cache_data(show_spinner=False)
def cached_network(rules_data: pd.DataFrame, df_data: pd.DataFrame, min_lift: float) -> dict:
    """Cache cấu trúc mạng lưới liên kết sản phẩm."""
    return build_apriori_network_elements(rules_data, df_data, min_lift=min_lift)


@st.cache_data(show_spinner=False)
def cached_top_products(data: pd.DataFrame, top_n: int, rank_by: str) -> pd.DataFrame:
    """Cache danh sách top sản phẩm."""
    return summarize_top_products(data, top_n=top_n, rank_by=rank_by)


@st.cache_data(show_spinner=False)
def cached_lift_matrix(data: pd.DataFrame, item_col: str) -> pd.DataFrame:
    """Cache ma trận Lift 2 chiều."""
    return compute_apriori_lift_matrix(data, item_col=item_col)


@st.cache_data(show_spinner=False)
def cached_pareto(data: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Cache phân tích Pareto ABC 80/20."""
    return compute_pareto_abc(data, entity_col=entity_col)


@st.cache_data(show_spinner=False)
def cached_hierarchy(data: pd.DataFrame) -> pd.DataFrame:
    """Cache bảng phân cấp Category -> Sub-Category -> Product."""
    h = (
        data.groupby(["Category", "Sub-Category", "Product Name"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
    )
    return h[h["Sales"] > 0]


@st.cache_data(show_spinner=False)
def cached_smart_bundles(data: pd.DataFrame, min_lift: float) -> pd.DataFrame:
    """Cache bảng đề xuất các gói Combo bán kèm tối ưu."""
    return generate_smart_cross_sell_bundles(data, min_lift=min_lift, top_n=5)


st.title("🏆 Page 4: Phân tích sản phẩm, Sunburst & Giỏ hàng (Apriori)")
st.caption(
    "Xếp hạng Top N sản phẩm, khoan sâu phân cấp Category → Sub-Category → Product bằng Sunburst "
    "và tìm sản phẩm mua kèm bằng thuật toán Apriori."
)

if df_sidebar.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc Sidebar hiện tại.")
    st.stop()

# Bộ lọc bổ sung theo Top N, Thời gian và Khu vực
min_dt = df_sidebar["Order Date"].min().date()
max_dt = df_sidebar["Order Date"].max().date()
all_regions = sorted(df_sidebar["Region"].unique().tolist())

c1, c2, c3 = st.columns([1.2, 1.8, 2])
with c1:
    top_n = st.slider("Số lượng Top N sản phẩm", min_value=5, max_value=25, value=10, step=1)
with c2:
    date_range = st.date_input(
        "Lọc thêm theo thời gian",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
    )
with c3:
    selected_regions = st.multiselect(
        "Lọc thêm theo khu vực",
        options=all_regions,
        default=all_regions,
    )

if not selected_regions:
    st.warning("Vui lòng chọn ít nhất một khu vực.")
    st.stop()

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_dt, end_dt = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start_dt, end_dt = pd.Timestamp(min_dt), pd.Timestamp(max_dt)

df_filtered = df_sidebar[
    (df_sidebar["Order Date"] >= start_dt)
    & (df_sidebar["Order Date"] <= end_dt)
    & (df_sidebar["Region"].isin(selected_regions))
]

if df_filtered.empty:
    st.warning("Không có giao dịch nào thỏa mãn điều kiện lọc.")
    st.stop()

render_dynamic_insight_banner(df_filtered)

tab_top, tab_sunburst, tab_apriori, tab_pareto = st.tabs(
    [
        f"🏅 Top {top_n} Sản phẩm & Sub-Category",
        "🌞 Biểu đồ phân cấp (Sunburst / Treemap)",
        "🛒 Phân tích Giỏ hàng (Market Basket - Apriori)",
        "⚖️ Phân tích Pareto 80/20 (ABC)",
    ]
)

with tab_top:
    rank_mode_label = st.radio(
        "Tiêu chí xếp hạng Sản phẩm",
        options=[
            "💰 Top N Doanh thu cao nhất (Best Sellers)",
            "📈 Top N Lợi nhuận cao nhất (Cash Cows)",
            "🚨 Top N Sản phẩm lỗ nặng nhất (Loss Leaders)",
        ],
        horizontal=True,
    )
    if "Lợi nhuận cao nhất" in rank_mode_label:
        rank_code = "Profit_Max"
        x_col = "Total_Profit"
        chart_title = f"Top {top_n} Sản phẩm mang lại Lợi nhuận cao nhất (Cash Cows)"
        x_label = "Tổng lợi nhuận (USD)"
    elif "lỗ nặng nhất" in rank_mode_label:
        rank_code = "Profit_Min"
        x_col = "Total_Profit"
        chart_title = f"Top {top_n} Sản phẩm gây Thua lỗ nặng nhất (Loss Leaders)"
        x_label = "Khoản lỗ ròng (USD)"
    else:
        rank_code = "Sales"
        x_col = "Total_Sales"
        chart_title = f"Top {top_n} Sản phẩm có Doanh thu cao nhất (Best Sellers)"
        x_label = "Tổng doanh thu (USD)"

    top_products = cached_top_products(df_filtered, top_n=top_n, rank_by=rank_code)
    top_sorted = top_products.sort_values(x_col, ascending=(rank_code != "Profit_Min")).copy()
    top_sorted["Short_Name"] = [
        s if len(s) <= 44 else s[:41] + "..." for s in top_sorted["Product Name"]
    ]

    fig_top = px.bar(
        top_sorted,
        x=x_col,
        y="Short_Name",
        color="Category",
        color_discrete_map=CATEGORY_COLORS,
        orientation="h",
        text_auto=".2s",
        hover_data={
            "Product Name": True,
            "Short_Name": False,
            "Sub-Category": True,
            "Total_Sales": ":,.2f",
            "Total_Profit": ":,.2f",
            "Avg_Discount": ":.1%",
            "Profit_Margin_Pct": ":.2f",
        },
        title=chart_title,
        labels={x_col: x_label, "Short_Name": "Sản phẩm", "Category": "Ngành hàng"},
        height=max(420, top_n * 30),
    )
    if x_col == "Total_Profit":
        fig_top.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_top, use_container_width=True)

    subcat_df = summarize_by_category(df_filtered).sort_values("Total_Profit", ascending=True)
    fig_subcat = px.bar(
        subcat_df,
        x="Total_Profit",
        y="Sub-Category",
        color="Total_Profit",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        orientation="h",
        text_auto=".2s",
        title="Hiệu suất Lợi nhuận theo Nhóm sản phẩm (17 Sub-Categories)",
        labels={"Total_Profit": "Tổng lợi nhuận (USD)", "Sub-Category": "Nhóm sản phẩm"},
        height=520,
    )
    fig_subcat.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_subcat, use_container_width=True)

with tab_sunburst:
    st.subheader("🌞 Khoan sâu phân cấp: Category → Sub-Category → Product Name")
    st.caption("Nhấp chuột vào từng lát cắt để phóng to (drill-down) vào nhóm sản phẩm chi tiết.")
    chart_style = st.radio(
        "Kiểu biểu đồ phân cấp",
        options=["Sunburst", "Treemap"],
        horizontal=True,
    )
    hierarchy_df = cached_hierarchy(df_filtered)

    if chart_style == "Sunburst":
        fig_hier = px.sunburst(
            hierarchy_df,
            path=["Category", "Sub-Category", "Product Name"],
            values="Sales",
            color="Profit",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Biểu đồ Sunburst Doanh thu (Kích thước) & Lợi nhuận (Màu sắc)",
            height=650,
        )
    else:
        fig_hier = px.treemap(
            hierarchy_df,
            path=["Category", "Sub-Category", "Product Name"],
            values="Sales",
            color="Profit",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Biểu đồ Treemap Doanh thu (Kích thước) & Lợi nhuận (Màu sắc)",
            height=650,
        )
    st.plotly_chart(fig_hier, use_container_width=True)

with tab_apriori:
    st.subheader("🛒 Phân tích Giỏ hàng (Market Basket Analysis — Thuật toán Apriori)")
    st.caption("Tìm các nhóm/mặt hàng thường xuất hiện trong cùng một đơn hàng (`Order ID`) để đề xuất bán chéo (cross-sell).")

    a1, a2, a3 = st.columns(3)
    with a1:
        item_granularity = st.selectbox("Cấp độ phân tích giỏ hàng", options=["Sub-Category", "Category"])
    with a2:
        min_support = st.slider("Min Support (Tần suất tối thiểu)", 0.005, 0.10, 0.015, 0.005)
    with a3:
        min_confidence = st.slider("Min Confidence (Độ tin cậy tối thiểu)", 0.10, 0.80, 0.20, 0.05)

    rules_df = cached_apriori(df_filtered, item_granularity, min_support, min_confidence)
    if rules_df.empty:
        st.info("Không tìm thấy luật kết hợp nào thỏa mãn ngưỡng Support/Confidence. Hãy thử giảm ngưỡng lọc.")
    else:
        ap_left, ap_right = st.columns([1.2, 1.1])
        with ap_left:
            rules_plot = rules_df.head(12).copy()
            rules_plot["Rule"] = rules_plot["Antecedents"] + " → " + rules_plot["Consequents"]
            fig_rules = px.bar(
                rules_plot.sort_values("Lift", ascending=True),
                x="Lift",
                y="Rule",
                color="Confidence",
                color_continuous_scale="Tealgrn",
                orientation="h",
                hover_data=["Support", "Pair_Orders"],
                title="Top các cặp sản phẩm thường mua cùng nhau (Xếp theo Lift)",
                labels={"Lift": "Chỉ số Lift (> 1 là liên kết dương)", "Rule": "Luật bán chéo"},
                height=460,
            )
            st.plotly_chart(fig_rules, use_container_width=True)

        with ap_right:
            lift_mat = cached_lift_matrix(df_filtered, item_col=item_granularity)
            if not lift_mat.empty:
                fig_lift_hm = px.imshow(
                    lift_mat,
                    text_auto=".2f",
                    color_continuous_scale="YlGnBu",
                    aspect="auto",
                    title=f"Ma trận Nhiệt Hệ số Lift ({item_granularity} × {item_granularity})",
                    labels={"x": "Mua kèm (Consequents)", "y": "Mặt hàng gốc (Antecedents)", "color": "Lift"},
                )
                fig_lift_hm.update_layout(height=460)
                st.plotly_chart(fig_lift_hm, use_container_width=True)

        st.divider()
        st.subheader("🕸️ Đồ thị Mạng lưới Liên kết Sản phẩm (Interactive Product Affinity Network Graph)")
        st.caption(
            "Trực quan hóa cấu trúc mạng lưới giữa các nhóm hàng: Kích thước node đại diện cho Doanh thu, "
            "Màu sắc theo Ngành hàng, và Độ dày đường nối thể hiện sức mạnh liên kết mua kèm (Confidence & Lift)."
        )
        min_net_lift = st.slider("Ngưỡng Lift tối thiểu để hiển thị liên kết trên mạng lưới", 0.8, 2.0, 0.95, 0.05)
        net_data = cached_network(rules_df, df_filtered, min_lift=min_net_lift)
        nodes_df = net_data.get("nodes", pd.DataFrame())
        edges_df = net_data.get("edges", pd.DataFrame())

        if not nodes_df.empty:
            fig_net = go.Figure()

            # Vẽ các cạnh liên kết (Edges)
            for edge in edges_df.to_dict("records"):
                fig_net.add_trace(
                    go.Scatter(
                        x=[edge["x0"], edge["x1"], None],
                        y=[edge["y0"], edge["y1"], None],
                        mode="lines",
                        line=dict(
                            width=max(1.0, min(5.0, float(edge["confidence"]) * 7.0)),
                            color="rgba(148, 163, 184, 0.45)",
                        ),
                        hoverinfo="text",
                        text=(
                            f"{edge['source']} ↔ {edge['target']}<br>"
                            f"Lift: {edge['lift']:.2f} | Confidence: {edge['confidence']*100:.1f}%<br>"
                            f"Đơn mua kèm: {edge['orders']:,}"
                        ),
                        showlegend=False,
                    )
                )

            # Vẽ các đỉnh (Nodes)
            node_colors = [CATEGORY_COLORS.get(c, "#2563eb") for c in nodes_df["Category"]]
            hover_texts = [
                f"<b>{r['Sub-Category']}</b> ({r['Category']})<br>Doanh thu: ${r['Total_Sales']:,.2f}<br>Lợi nhuận: ${r['Total_Profit']:,.2f}<br>Số đơn: {r['Order_Count']:,}"
                for r in nodes_df.to_dict("records")
            ]
            fig_net.add_trace(
                go.Scatter(
                    x=nodes_df["x"],
                    y=nodes_df["y"],
                    mode="markers+text",
                    marker=dict(
                        size=nodes_df["node_size"],
                        color=node_colors,
                        line=dict(width=2, color="white"),
                    ),
                    text=nodes_df["Sub-Category"],
                    textposition="top center",
                    textfont=dict(size=11, color="#1e293b"),
                    hoverinfo="text",
                    hovertext=hover_texts,
                    name="Nhóm hàng",
                    showlegend=False,
                )
            )

            fig_net.update_layout(
                title=f"Mạng lưới Bán kèm ({len(nodes_df)} nhóm hàng, {len(edges_df)} liên kết có Lift ≥ {min_net_lift})",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=560,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="rgba(248, 250, 252, 0.5)",
            )
            st.plotly_chart(fig_net, use_container_width=True)

        st.divider()
        st.subheader("💡 Công cụ Đề xuất Gói Bán Kèm (Cross-Sell Bundle Advisor)")
        st.caption(
            "Chọn một nhóm hàng đang lỗ hoặc biên lợi nhuận thấp (ví dụ: `Tables`, `Bookcases`, `Binders`) "
            "để tìm các nhóm hàng có biên lợi nhuận cao thường được khách mua chung trong cùng giỏ hàng."
        )
        sub_margins = (
            df_filtered.groupby(item_granularity, as_index=False)
            .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"))
        )
        sub_margins["Margin_Pct"] = (sub_margins["Profit"] / sub_margins["Sales"].replace(0, 1)) * 100.0
        margin_map = dict(zip(sub_margins[item_granularity], sub_margins["Margin_Pct"]))

        avail_items = sorted(df_filtered[item_granularity].unique().tolist())
        default_item = "Tables" if "Tables" in avail_items else (avail_items[0] if avail_items else "")
        if avail_items:
            target_item = st.selectbox(
                f"Chọn {item_granularity} cần thiết kế gói Combo bán kèm:",
                options=avail_items,
                index=avail_items.index(default_item),
            )
            all_pair_rules = run_market_basket_apriori(df_filtered, item_col=item_granularity, min_support=0.002, min_confidence=0.05)
            related = all_pair_rules[all_pair_rules["Antecedents"] == target_item].copy()
            if not related.empty:
                related["Biên_LN_Mặt_Hàng_Kèm (%)"] = related["Consequents"].map(margin_map).round(2)
                best_partner = related.sort_values(["Biên_LN_Mặt_Hàng_Kèm (%)", "Lift"], ascending=False).iloc[0]
                st.success(
                    f"🎯 **Gợi ý Combo cho `{target_item}` (Biên LN hiện tại: {margin_map.get(target_item, 0.0):.2f}%):** "
                    f"Nên ghép gói bán kèm với **`{best_partner['Consequents']}`** "
                    f"(Biên LN: **+{best_partner['Biên_LN_Mặt_Hàng_Kèm (%)']:.2f}%**, "
                    f"Lift = **{best_partner['Lift']:.2f}**, Confidence = **{best_partner['Confidence']*100:.1f}%**, "
                    f"xuất hiện chung trong **{int(best_partner['Pair_Orders'])}** đơn hàng) và giới hạn chiết khấu gói ≤ 15%."
                )
                st.dataframe(
                    related.head(8).style.format(
                        {
                            "Support": "{:.2%}",
                            "Confidence": "{:.2%}",
                            "Lift": "{:.2f}",
                            "Biên_LN_Mặt_Hàng_Kèm (%)": "{:+.2f}%",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.divider()
        st.subheader("🎁 Đề xuất Top 5 Gói Combo Bán Kèm Tối Ưu (Smart Cross-Sell Bundles)")
        st.caption(
            "Mô hình thuật toán tự động tính toán các cặp sản phẩm có hệ số Lift cao nhất, "
            "định giá gói combo (chiết khấu 8%) và dự phóng doanh thu gia tăng khi áp dụng tiếp thị chéo."
        )
        bundles_df = cached_smart_bundles(df_filtered, min_lift=1.1)
        if not bundles_df.empty:
            st.dataframe(
                bundles_df.style.format(
                    {
                        "Lift": "{:.2f}",
                        "Giá gốc 2 món ($)": "${:,.2f}",
                        "Giá đề xuất Combo ($)": "${:,.2f}",
                        "Doanh thu gia tăng dự phóng ($)": "+${:,.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("Chưa đủ dữ liệu để tạo gói combo tối ưu.")

        with st.expander("📋 Xem toàn bộ bảng luật kết hợp Apriori"):
            st.dataframe(
                rules_df.style.format({"Support": "{:.2%}", "Confidence": "{:.2%}", "Lift": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

with tab_pareto:
    st.subheader("⚖️ Phân tích Pareto 80/20 & Phân loại ABC")
    pareto_entity = st.radio(
        "Đối tượng phân tích Pareto",
        options=["Sub-Category", "Product Name", "Customer Name"],
        horizontal=True,
    )
    abc_df = cached_pareto(df_filtered, entity_col=pareto_entity)
    if abc_df.empty:
        st.info("Không đủ dữ liệu để tính phân tích Pareto.")
    else:
        a_count = int((abc_df["ABC_Class"] == "Nhóm A (Top 80% doanh thu)").sum())
        total_items = len(abc_df)
        p1, p2, p3 = st.columns(3)
        p1.metric("Tổng số đối tượng", f"{total_items:,}")
        p2.metric("Số lượng Nhóm A (80% doanh thu)", f"{a_count:,}")
        p3.metric("Tỷ lệ Nhóm A / Tổng", f"{(a_count / total_items * 100):.1f}%")

        plot_df = abc_df.head(30)
        fig_pareto = go.Figure()
        fig_pareto.add_trace(
            go.Bar(
                x=plot_df[pareto_entity].astype(str).str.slice(0, 28),
                y=plot_df["Total_Sales"],
                name="Doanh thu (USD)",
                marker_color="#1f77b4",
            )
        )
        fig_pareto.add_trace(
            go.Scatter(
                x=plot_df[pareto_entity].astype(str).str.slice(0, 28),
                y=plot_df["Cumulative_Sales_Pct"],
                name="Tỷ lệ tích lũy (%)",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=2.5),
            )
        )
        fig_pareto.update_layout(
            title=f"Biểu đồ Pareto (Top 30 {pareto_entity}) & Đường tích lũy %",
            yaxis=dict(title="Doanh thu (USD)"),
            yaxis2=dict(title="Tích lũy (%)", overlaying="y", side="right", range=[0, 105]),
            legend=dict(orientation="h", y=1.12),
            height=480,
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.dataframe(
            abc_df.head(50).style.format(
                {
                    "Total_Sales": "${:,.2f}",
                    "Total_Profit": "${:,.2f}",
                    "Sales_Share_Pct": "{:.2f}%",
                    "Cumulative_Sales_Pct": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
