"""Page 3: Phân tích khu vực — Bar chart đa vùng, multiselect và Bản đồ Choropleth Hoa Kỳ."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.sidebar import init_page, render_global_sidebar
from data.analytics import (
    compare_groups_ab,
    compute_regional_radar_metrics,
    generate_regional_action_playbook,
    pivot_region_category,
    summarize_by_region,
    summarize_by_state,
    summarize_discount_bands,
    summarize_state_performance,
    summarize_state_subcategory,
)
from utils.ui_styles import CATEGORY_COLORS, REGION_COLORS, render_dynamic_insight_banner

init_page("Page 3 - Khu vực & Bản đồ Mỹ", icon="🗺️")
_, _, df_sidebar = render_global_sidebar()


@st.cache_data(show_spinner=False)
def cached_radar(data: pd.DataFrame) -> dict:
    """Cache kết quả tính toán năng lực vận hành 4 vùng (Radar Chart)."""
    return compute_regional_radar_metrics(data)


@st.cache_data(show_spinner=False)
def cached_playbook(radar_data: pd.DataFrame) -> list[dict]:
    """Cache sổ tay hành động chiến lược theo từng vùng."""
    return generate_regional_action_playbook(radar_data)


@st.cache_data(show_spinner=False)
def cached_region_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Cache tổng hợp doanh thu theo vùng."""
    return summarize_by_region(data)


@st.cache_data(show_spinner=False)
def cached_state_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Cache tổng hợp doanh thu theo tiểu bang."""
    return summarize_by_state(data)


@st.cache_data(show_spinner=False)
def cached_pivot_reg_cat(data: pd.DataFrame) -> pd.DataFrame:
    """Cache Pivot Table Vùng x Ngành hàng."""
    return pivot_region_category(data)


@st.cache_data(show_spinner=False)
def cached_central_states(data: pd.DataFrame) -> pd.DataFrame:
    """Cache tổng hợp các bang vùng Central."""
    return summarize_state_performance(data, region="Central")


@st.cache_data(show_spinner=False)
def cached_state_subcat(data: pd.DataFrame, state_name: str) -> pd.DataFrame:
    """Cache chi tiết nhóm hàng theo tiểu bang."""
    return summarize_state_subcategory(data, state_name)


@st.cache_data(show_spinner=False)
def cached_compare_regions(data: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Cache so sánh đối chứng 2 vùng."""
    return compare_groups_ab(data, "Region", a, b)

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
region_summary = cached_region_summary(df_filtered)
state_summary = cached_state_summary(df_filtered)

# 1. Biểu đồ cột so sánh giữa các vùng
col_left, col_right = st.columns(2)
with col_left:
    fig_region_bar = px.bar(
        region_summary,
        x="Region",
        y="Total_Sales",
        color="Region",
        color_discrete_map=REGION_COLORS,
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
        color_discrete_map={"Total_Sales": "#2563eb", "Total_Profit": "#16a34a"},
        text_auto=".2s",
        title="Đối chiếu Doanh thu và Lợi nhuận theo Khu vực",
        labels={"Region": "Khu vực", "value": "Giá trị (USD)", "variable": "Chỉ số"},
    )
    st.plotly_chart(fig_profit_bar, use_container_width=True)

st.divider()

# 1.5. Biểu đồ Radar Đánh giá Năng lực Vận hành 4 Vùng
st.subheader("🕸️ Đánh giá Sức khỏe Vận hành 4 Vùng (Regional Operational Health Radar)")
st.caption(
    "Đánh giá toàn diện các khu vực trên 5 trục năng lực chuẩn hóa (thang điểm 0–100): "
    "Quy mô Doanh thu, Biên Lợi nhuận, Tỷ lệ Đơn Lãi, Tốc độ Giao vận và Kỷ luật Chiết khấu."
)
radar_res = cached_radar(df_filtered)
radar_df = radar_res.get("scores", pd.DataFrame())
dimensions = radar_res.get("dimensions", [])

if not radar_df.empty and dimensions:
    r_left, r_right = st.columns([1.25, 1.0])
    with r_left:
        fig_radar = go.Figure()
        theta_closed = dimensions + [dimensions[0]]
        for r_row in radar_df.to_dict("records"):
            reg_name = str(r_row["Region"])
            r_vals = [float(r_row[d]) for d in dimensions]
            r_vals_closed = r_vals + [r_vals[0]]
            color = REGION_COLORS.get(reg_name, "#2563eb")
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r_vals_closed,
                    theta=theta_closed,
                    fill="toself",
                    name=f"{reg_name} (Điểm: {r_row['Diem_Tong_Hop']})",
                    line=dict(color=color, width=2),
                    opacity=0.6,
                )
            )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], ticksuffix=" đ"),
            ),
            showlegend=True,
            title="Đồ thị Radar Đa giác Năng lực Vận hành (0 - 100 điểm)",
            height=440,
            margin=dict(l=40, r=40, t=40, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with r_right:
        st.markdown("#### 🏆 Bảng Xếp hạng Sức khỏe Vận hành 4 Vùng:")
        st.dataframe(
            radar_df[
                [
                    "Region",
                    "Diem_Tong_Hop",
                    "Quy mô Doanh thu",
                    "Biên Lợi nhuận",
                    "Tỷ lệ Đơn Lãi",
                    "Tốc độ Giao vận",
                    "Kỷ luật Chiết khấu",
                ]
            ].style.format(
                {
                    "Diem_Tong_Hop": "{:.1f} đ",
                    "Quy mô Doanh thu": "{:.1f}",
                    "Biên Lợi nhuận": "{:.1f}",
                    "Tỷ lệ Đơn Lãi": "{:.1f}",
                    "Tốc độ Giao vận": "{:.1f}",
                    "Kỷ luật Chiết khấu": "{:.1f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "📌 **Giải thích trục đo:** *Kỷ luật Chiết khấu* cao thể hiện vùng kiểm soát giảm giá tốt (ít đơn giảm sâu). "
            "*Tốc độ Giao vận* cao thể hiện thời gian hoàn tất đơn hàng nhanh."
        )

    # 1.6 Sổ tay Hành động Chiến lược Điều hành Tự động (Action Playbook)
    playbook = cached_playbook(radar_df)
    if playbook:
        st.markdown("#### 📋 Sổ tay Hành động Chiến lược Điều hành (Automated Strategic Regional Playbook)")
        pb_cols = st.columns(len(playbook))
        for col_item, pb in zip(pb_cols, playbook):
            with col_item:
                reg_name = pb["region"]
                b_color = pb["badge_color"]
                st.markdown(
                    f"<div style='border: 1px solid #cbd5e1; border-top: 4px solid {b_color}; border-radius: 8px; padding: 12px; background: #f8fafc; height: 100%;'>"
                    f"<h4 style='margin: 0; color: #1e293b;'>Khu vực: {reg_name}</h4>"
                    f"<p style='margin: 4px 0 8px 0; font-size: 13px; font-weight: 600; color: {b_color};'>● Trạng thái: {pb['status']}</p>"
                    f"<p style='margin: 0; font-size: 12px; color: #475569;'>Điểm tổng: <b>{pb['overall_score']:.1f}/100</b> | Biên LN: <b>{pb['actual_margin']:.1f}%</b></p>"
                    f"<hr style='margin: 8px 0; border: none; border-top: 1px solid #e2e8f0;'/>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                with st.expander(f"Hành động đề xuất ({reg_name})", expanded=(pb["badge_color"] == "#dc2626")):
                    for act in pb["actions"]:
                        st.markdown(f"- {act}")

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
pivot_df = cached_pivot_reg_cat(df_filtered)
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
        color_discrete_map=REGION_COLORS,
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
st.subheader("🔍 Khoan sâu Khu vực Central & Chẩn đoán Tiểu bang Tương tác (Mặc định: Texas)")
st.caption(
    "Phân tích nguyên nhân tại sao khu vực Central có biên lợi nhuận thấp nhất (7.92%) và khoan sâu vào từng tiểu bang cụ thể "
    "để nhận diện nhóm sản phẩm cùng chính sách chiết khấu gây lỗ."
)
central_states = cached_central_states(df_filtered)
all_avail_states = state_summary.sort_values("Total_Profit", ascending=True)["State"].tolist()
default_state = "Texas" if "Texas" in all_avail_states else (all_avail_states[0] if all_avail_states else "")

c_left, c_right = st.columns([1.15, 1.35])
with c_left:
    if central_states.empty:
        st.info("Bộ lọc hiện tại không bao gồm khu vực Central. Hãy chọn Central ở multiselect phía trên.")
    else:
        worst_central = str(central_states.iloc[0]["State"])
        worst_c_profit = float(central_states.iloc[0]["Total_Profit"])
        worst_c_disc = float(central_states.iloc[0]["Avg_Discount"]) * 100.0
        st.markdown(
            f"**Bức tranh vùng Central:** Tiểu bang lỗ nặng nhất là **{worst_central}** "
            f"(**${worst_c_profit:,.0f}**, chiết khấu TB **{worst_c_disc:.1f}%**), đối lập với **Michigan** và **Indiana** sinh lời cao."
        )
        fig_central = px.bar(
            central_states.sort_values("Total_Profit"),
            x="Total_Profit",
            y="State",
            color="Avg_Discount",
            orientation="h",
            title="Xếp hạng Lợi nhuận 13 Tiểu bang vùng Central (Màu = Chiết khấu TB)",
            labels={"Total_Profit": "Lợi nhuận (USD)", "State": "Tiểu bang", "Avg_Discount": "Chiết khấu TB"},
            color_continuous_scale="YlOrRd",
            height=450,
        )
        fig_central.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_central, use_container_width=True)

with c_right:
    if all_avail_states:
        chosen_state = st.selectbox(
            "🎯 Chọn Tiểu bang để chẩn đoán chi tiết (Xếp từ lỗ nặng nhất → lãi cao nhất)",
            options=all_avail_states,
            index=all_avail_states.index(default_state),
        )
        st_df = df_filtered[df_filtered["State"] == chosen_state]
        st_sub = cached_state_subcat(df_filtered, chosen_state)
        st_sales = float(st_df["Sales"].sum())
        st_profit = float(st_df["Profit"].sum())
        st_margin = (st_profit / st_sales * 100.0) if st_sales else 0.0
        st_avg_disc = float(st_df["Discount"].mean() * 100.0) if len(st_df) else 0.0
        st_loss_rows = st_df[st_df["Profit"] < 0]
        st_disc_loss = st_loss_rows[st_loss_rows["Discount"] > 0.20]

        sk1, sk2, sk3, sk4 = st.columns(4)
        sk1.metric(f"{chosen_state}: Doanh thu", f"${st_sales:,.0f}")
        sk2.metric(f"{chosen_state}: Lợi nhuận", f"${st_profit:,.0f}", delta=f"Biên {st_margin:.1f}%")
        sk3.metric("Chiết khấu TB", f"{st_avg_disc:.1f}%")
        sk4.metric(
            "Giao dịch lỗ",
            f"{len(st_loss_rows):,}",
            delta=f"{len(st_disc_loss)} đơn CK >20%",
            delta_color="inverse",
        )

        fig_tx = px.bar(
            st_sub.sort_values("Total_Profit"),
            x="Total_Profit",
            y="Sub-Category",
            color="Category",
            color_discrete_map=CATEGORY_COLORS,
            orientation="h",
            text_auto=".2s",
            title=f"Lợi nhuận theo Nhóm hàng (Sub-Category) tại {chosen_state}",
            labels={"Total_Profit": "Lợi nhuận (USD)", "Sub-Category": "Nhóm hàng", "Category": "Ngành hàng"},
            height=360,
        )
        fig_tx.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_tx, use_container_width=True)

with st.expander("📋 Xem bảng chi tiết các bang vùng Central & Phân tích ngưỡng chiết khấu tại bang đang chọn"):
    if not central_states.empty:
        st.caption("**1. Bảng hiệu suất 13 tiểu bang thuộc khu vực Central:**")
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
    if all_avail_states:
        tx_band = summarize_discount_bands(df_filtered[df_filtered["State"] == chosen_state])
        if not tx_band.empty:
            st.caption(f"**2. Phân tích 5 ngưỡng chiết khấu tại tiểu bang {chosen_state}:**")
            st.dataframe(
                tx_band[["Nguong_chiet_khau", "So_dong", "Doanh_thu", "Loi_nhuan", "Loi_nhuan_TB", "Ty_le_lo", "Bien_LN_pct"]].style.format(
                    {
                        "Doanh_thu": "${:,.2f}",
                        "Loi_nhuan": "${:,.2f}",
                        "Loi_nhuan_TB": "${:,.2f}",
                        "Ty_le_lo": "{:.1f}%",
                        "Bien_LN_pct": "{:.2f}%",
                    }
                ),
                use_container_width=True,
                hide_index=True,
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
    ab_reg_df = cached_compare_regions(df_filtered, reg_a, reg_b)
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
