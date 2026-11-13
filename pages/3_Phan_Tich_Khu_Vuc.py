"""Page 3: Phân tích khu vực — Bar chart đa vùng, multiselect và Bản đồ Choropleth Hoa Kỳ."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from components.sidebar import init_page, render_global_sidebar
from data.analytics import (
    compare_groups_ab,
    pivot_region_category,
    summarize_by_region,
    summarize_by_state,
    summarize_discount_bands,
    summarize_state_performance,
    summarize_state_subcategory,
)
from utils.ui_styles import render_dynamic_insight_banner

init_page("Page 3 - Khu vực & Bản đồ Mỹ", icon="🗺️")
_, _, df_sidebar = render_global_sidebar()

st.title("🗺️ Page 3: Phân tích khu vực & Bản đồ địa lý Hoa Kỳ")
st.caption("So sánh doanh thu theo vùng bằng biểu đồ cột (`multiselect`) và bản đồ nhiệt Choropleth theo từng tiểu bang.")

if df_sidebar.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc Sidebar hiện tại.")
    st.stop()

# Bộ lọc bắt buộc: st.multiselect chọn nhiều khu vực cùng lúc
available_regions = sorted(df_sidebar["Region"].unique().tolist())
selected_regions = st.multiselect(
    "Chọn các khu vực muốn so sánh (Region - Multiselect)",
    options=available_regions,
    default=available_regions,
)

if not selected_regions:
    st.warning("Vui lòng chọn ít nhất một khu vực để hiển thị.")
    st.stop()

df_filtered = df_sidebar[df_sidebar["Region"].isin(selected_regions)]
render_dynamic_insight_banner(df_filtered)
region_summary = summarize_by_region(df_filtered)
state_summary = summarize_by_state(df_filtered)

# 1. Biểu đồ cột so sánh giữa các vùng
col_left, col_right = st.columns(2)
with col_left:
    fig_region_bar = px.bar(
        region_summary,
        x="Region",
        y="Total_Sales",
        color="Region",
        text_auto=".2s",
        title="So sánh Tổng doanh thu theo Khu vực (Bar Chart)",
        labels={"Region": "Khu vực", "Total_Sales": "Tổng doanh thu (USD)"},
    )
    fig_region_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_region_bar, use_container_width=True)

with col_right:
    fig_profit_bar = px.bar(
        region_summary,
        x="Region",
        y=["Total_Sales", "Total_Profit"],
        barmode="group",
        title="Đối chiếu Doanh thu và Lợi nhuận theo Khu vực",
        labels={"Region": "Khu vực", "value": "Giá trị (USD)", "variable": "Chỉ số"},
    )
    st.plotly_chart(fig_profit_bar, use_container_width=True)

st.divider()

# 2. Bản đồ địa lý Hoa Kỳ (Choropleth Map)
st.subheader("🇺🇸 Bản đồ nhiệt Doanh thu & Lợi nhuận theo Tiểu bang (Choropleth Map)")
map_metric_label = st.radio(
    "Chọn chỉ số hiển thị trên bản đồ",
    options=["Tổng doanh thu (Total_Sales)", "Tổng lợi nhuận (Total_Profit)", "Biên lợi nhuận (%)"],
    horizontal=True,
)
metric_col = {
    "Tổng doanh thu (Total_Sales)": "Total_Sales",
    "Tổng lợi nhuận (Total_Profit)": "Total_Profit",
    "Biên lợi nhuận (%)": "Profit_Margin_Pct",
}[map_metric_label]

color_scale = "Blues" if metric_col == "Total_Sales" else "RdYlGn"

fig_map = px.choropleth(
    state_summary.dropna(subset=["State_Code"]),
    locations="State_Code",
    locationmode="USA-states",
    color=metric_col,
    scope="usa",
    hover_name="State",
    hover_data={
        "State_Code": True,
        "Region": True,
        "Total_Sales": ":,.2f",
        "Total_Profit": ":,.2f",
        "Profit_Margin_Pct": ":.2f",
        "Order_Count": ":,",
    },
    color_continuous_scale=color_scale,
    title=f"Bản đồ phân bố {map_metric_label} tại Hoa Kỳ (Zoom & Di chuột để xem chi tiết)",
)
fig_map.update_layout(
    geo=dict(lakecolor="rgb(255, 255, 255)"),
    height=540,
    margin=dict(l=0, r=0, t=50, b=0),
)
st.plotly_chart(fig_map, use_container_width=True)

st.subheader("📊 Bảng Pivot Table: Khu vực × Danh mục sản phẩm (Doanh thu)")
pivot_df = pivot_region_category(df_filtered)
st.dataframe(pivot_df.style.format("${:,.2f}"), use_container_width=True)

st.divider()
st.subheader("🚚 Hiệu suất Vận chuyển & Thời gian Giao hàng (Shipping SLA)")
sla_left, sla_right = st.columns(2)

with sla_left:
    fig_ship_box = px.box(
        df_filtered,
        x="Ship Mode",
        y="Shipping_Days",
        color="Ship Mode",
        title="Phân phối Số ngày giao hàng (Shipping Days) theo Hình thức vận chuyển",
        labels={"Ship Mode": "Hình thức giao hàng", "Shipping_Days": "Số ngày giao hàng"},
    )
    fig_ship_box.update_layout(showlegend=False)
    st.plotly_chart(fig_ship_box, use_container_width=True)

with sla_right:
    sla_pivot = (
        df_filtered.pivot_table(
            values="Shipping_Days",
            index="Region",
            columns="Ship Mode",
            aggfunc="mean",
        )
        .round(2)
    )
    fig_sla_heat = px.imshow(
        sla_pivot,
        text_auto=".2f",
        color_continuous_scale="Teal",
        aspect="auto",
        title="Thời gian giao hàng trung bình (Ngày): Khu vực × Hình thức vận chuyển",
        labels={"x": "Hình thức giao hàng", "y": "Khu vực", "color": "Ngày TB"},
    )
    st.plotly_chart(fig_sla_heat, use_container_width=True)

st.divider()
st.subheader("🧭 Ma trận Tứ phân vị Tiểu bang (Doanh thu vs. Biên lợi nhuận %)")
if not state_summary.empty:
    median_sales = float(state_summary["Total_Sales"].median())
    fig_quad = px.scatter(
        state_summary,
        x="Total_Sales",
        y="Profit_Margin_Pct",
        size="Order_Count",
        color="Region",
        text="State_Code",
        hover_name="State",
        hover_data={"Total_Sales": ":,.2f", "Total_Profit": ":,.2f", "Profit_Margin_Pct": ":.2f"},
        title="Định vị hiệu quả các Tiểu bang (Đường dọc: Trung vị Doanh thu | Đường ngang: Hòa vốn 0%)",
        labels={"Total_Sales": "Tổng doanh thu (USD)", "Profit_Margin_Pct": "Biên lợi nhuận (%)"},
        height=500,
    )
    fig_quad.update_traces(textposition="top center")
    fig_quad.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Hòa vốn (0%)")
    fig_quad.add_vline(x=median_sales, line_dash="dot", line_color="gray", annotation_text="Trung vị Doanh thu")
    st.plotly_chart(fig_quad, use_container_width=True)

st.divider()
st.subheader("🔍 Khoan sâu khu vực Central — bang nào và nhóm hàng nào đang lỗ")
central_states = summarize_state_performance(df_filtered, region="Central")
if central_states.empty:
    st.info("Bộ lọc hiện tại không còn dữ liệu khu vực Central. Chọn Central ở multiselect để xem khoan sâu.")
else:
    worst_state = str(central_states.iloc[0]["State"])
    worst_profit = float(central_states.iloc[0]["Total_Profit"])
    worst_discount = float(central_states.iloc[0]["Avg_Discount"])
    st.caption(
        f"Trong Central, **{worst_state}** có lợi nhuận thấp nhất "
        f"({worst_profit:,.0f} USD) với chiết khấu trung bình {worst_discount * 100:.1f}%."
    )
    c_left, c_right = st.columns(2)
    with c_left:
        fig_central = px.bar(
            central_states.sort_values("Total_Profit"),
            x="Total_Profit",
            y="State",
            color="Avg_Discount",
            orientation="h",
            title="Lợi nhuận theo bang thuộc Central (màu = chiết khấu trung bình)",
            labels={"Total_Profit": "Lợi nhuận (USD)", "State": "Tiểu bang", "Avg_Discount": "Chiết khấu TB"},
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig_central, use_container_width=True)
    with c_right:
        sub_state = summarize_state_subcategory(df_filtered, worst_state)
        fig_sub = px.bar(
            sub_state,
            x="Total_Profit",
            y="Sub-Category",
            color="Category",
            orientation="h",
            title=f"Lợi nhuận theo nhóm hàng tại {worst_state}",
            labels={"Total_Profit": "Lợi nhuận (USD)", "Sub-Category": "Nhóm hàng"},
        )
        st.plotly_chart(fig_sub, use_container_width=True)
    st.dataframe(
        central_states.style.format(
            {
                "Total_Sales": "${:,.2f}",
                "Total_Profit": "${:,.2f}",
                "Avg_Discount": "{:.1%}",
                "Profit_Margin_Pct": "{:.2f}%",
                "Loss_Rate_Pct": "{:.1f}%",
                "Order_Count": "{:,}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()
st.subheader("🔬 Khoan sâu Texas — Bang lỗ nhiều nhất trong Central")
texas_sub = summarize_state_subcategory(df_filtered, "Texas")
if texas_sub.empty:
    st.info("Không có dữ liệu Texas trong bộ lọc hiện tại.")
else:
    tx1, tx2 = st.columns([1.5, 2])
    with tx1:
        tx_disc_df = df_filtered[df_filtered["State"] == "Texas"]
        tx_sales = float(tx_disc_df["Sales"].sum())
        tx_profit = float(tx_disc_df["Profit"].sum())
        tx_margin = (tx_profit / tx_sales * 100) if tx_sales else 0
        tx_avg_disc = float(tx_disc_df["Discount"].mean()) * 100 if len(tx_disc_df) else 0
        st.metric("Texas: Doanh thu", f"${tx_sales:,.0f}")
        st.metric("Texas: Lợi nhuận", f"${tx_profit:,.0f}", delta=f"Biên {tx_margin:.1f}%")
        st.metric("Texas: Chiết khấu TB", f"{tx_avg_disc:.1f}%")
        loss_rows = tx_disc_df[tx_disc_df["Profit"] < 0]
        disc_loss = loss_rows[loss_rows["Discount"] > 0.2]
        st.caption(f"Texas có **{len(loss_rows)}** giao dịch lỗ, trong đó **{len(disc_loss)}** ({len(disc_loss)/max(len(loss_rows),1)*100:.0f}%) do chiết khấu > 20%.")
    with tx2:
        fig_tx = px.bar(
            texas_sub.sort_values("Total_Profit"),
            x="Total_Profit",
            y="Sub-Category",
            color="Category",
            orientation="h",
            title="Lợi nhuận theo nhóm hàng tại Texas",
            labels={"Total_Profit": "Lợi nhuận (USD)", "Sub-Category": "Nhóm hàng"},
            height=420,
        )
        fig_tx.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_tx, use_container_width=True)
    tx_band = summarize_discount_bands(tx_disc_df)
    if not tx_band.empty:
        st.caption("**Phân tích ngưỡng chiết khấu tại Texas:**")
        st.dataframe(
            tx_band[["Nguong_chiet_khau", "So_dong", "Loi_nhuan_TB", "Ty_le_lo", "Bien_LN_pct"]].style.format({
                "Loi_nhuan_TB": "${:,.2f}", "Ty_le_lo": "{:.1f}%", "Bien_LN_pct": "{:.2f}%"
            }),
            use_container_width=True, hide_index=True,
        )

st.divider()
st.subheader("⚖️ So sánh Đối chứng 2 Khu vực (A/B Regional Benchmark)")
if len(selected_regions) >= 2:
    r_col1, r_col2 = st.columns(2)
    default_a = "West" if "West" in selected_regions else selected_regions[0]
    default_b = "Central" if "Central" in selected_regions else selected_regions[-1]
    with r_col1:
        reg_a = st.selectbox("Khu vực A (Mục tiêu)", options=selected_regions, index=selected_regions.index(default_a))
    with r_col2:
        reg_b = st.selectbox("Khu vực B (Đối chiếu)", options=selected_regions, index=selected_regions.index(default_b))
    ab_reg_df = compare_groups_ab(df_filtered, "Region", reg_a, reg_b)
    st.dataframe(
        ab_reg_df.style.format(
            {
                f"{reg_a}": "{:,.2f}",
                f"{reg_b}": "{:,.2f}",
                "Chênh lệch (A - B)": "{:+,.2f}",
                "Tương đối (%)": "{:+.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Chọn ít nhất 2 khu vực ở multiselect phía trên để đối chiếu A/B.")
