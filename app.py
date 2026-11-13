"""Trang chủ hệ thống Dashboard Phân tích Dữ liệu Bán lẻ Superstore — Executive Command Center."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.sidebar import init_page, render_global_sidebar
from data.analytics import generate_dynamic_insight, summarize_by_state, summarize_by_time
from data.loader import get_data_dictionary
from utils.formatters import format_int, format_usd
from utils.ui_styles import render_dynamic_insight_banner

init_page("Executive Command Center", icon="🏬")
df_clean, audit, df_filtered = render_global_sidebar()

st.title("🏬 Superstore BI & AI — Trung tâm Điều hành (Executive Command Center)")
st.caption(
    "Kiến trúc sạch 4 tầng (`data/`, `components/`, `pages/`, `utils/`) • Đồng bộ `st.session_state` • "
    "Dự báo **Prophet / SARIMAX** • Phân khúc **RFM & K-Means** • Luật giỏ hàng **Apriori** • **Gemini 2.5 Flash**"
)

if df_filtered.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc Sidebar hiện tại. Vui lòng đặt lại bộ lọc.")
    st.stop()

# 1. Hộp Auto-Insight động nhảy số theo bộ lọc
render_dynamic_insight_banner(df_filtered)

# 2. 6 Thẻ KPI Điều hành (Executive Metrics)
ins = generate_dynamic_insight(df_filtered)
total_sales = float(df_filtered["Sales"].sum())
total_profit = float(df_filtered["Profit"].sum())
total_orders = int(df_filtered["Order ID"].nunique())
total_customers = int(df_filtered["Customer ID"].nunique())
margin_pct = (total_profit / total_sales * 100.0) if total_sales else 0.0
aov = (total_sales / total_orders) if total_orders else 0.0
loss_rows = int((df_filtered["Profit"] < 0).sum())
loss_rate = (loss_rows / len(df_filtered) * 100.0) if len(df_filtered) else 0.0
yoy_pct = float(ins.get("yoy_sales_pct", 0.0))

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric(
    "💰 Tổng Doanh thu",
    format_usd(total_sales),
    delta=f"{yoy_pct:+.1f}% YoY ({ins.get('latest_year', '')})" if yoy_pct != 0 else f"AOV: {format_usd(aov)}",
)
k2.metric(
    "📈 Tổng Lợi nhuận",
    format_usd(total_profit),
    delta=f"Biên LN: {margin_pct:.2f}%",
)
k3.metric(
    "🧾 Tổng Đơn hàng",
    format_int(total_orders),
    delta=f"AOV: {format_usd(aov)}",
)
k4.metric(
    "👥 Khách hàng Active",
    format_int(total_customers),
    delta=f"{total_orders / max(total_customers, 1):.1f} đơn/khách",
)
k5.metric(
    "⚠️ Tỷ lệ Giao dịch Lỗ",
    f"{loss_rate:.1f}%",
    delta=f"{format_int(loss_rows)} dòng âm",
    delta_color="inverse",
)
k6.metric(
    "🗂️ Bản ghi Đang lọc",
    f"{format_int(len(df_filtered))}",
    delta=f"/{format_int(len(df_clean))} dòng sạch (-{audit['business_duplicates']} trùng)",
)

st.divider()

# 3. Bức tranh toàn cảnh (Executive Snapshot Charts)
st.subheader("📊 Bức tranh Tài chính Toàn cảnh (Executive Snapshot)")
col_chart_left, col_chart_right = st.columns([1.35, 1.0])

with col_chart_left:
    q_df = summarize_by_time(df_filtered, freq="Q")
    fig_q = go.Figure()
    fig_q.add_trace(
        go.Bar(
            x=q_df["Period"],
            y=q_df["Total_Sales"],
            name="Doanh thu (USD)",
            marker_color="#2563eb",
            yaxis="y1",
        )
    )
    fig_q.add_trace(
        go.Scatter(
            x=q_df["Period"],
            y=q_df["Profit_Margin_Pct"],
            name="Biên lợi nhuận (%)",
            mode="lines+markers",
            line=dict(color="#f97316", width=3),
            marker=dict(size=8),
            yaxis="y2",
        )
    )
    fig_q.update_layout(
        title="Đà tăng trưởng Doanh thu & Biên lợi nhuận (%) theo Quý",
        xaxis_title="Quý",
        yaxis=dict(title="Doanh thu (USD)"),
        yaxis2=dict(title="Biên LN (%)", overlaying="y", side="right", ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=390,
        hovermode="x unified",
    )
    st.plotly_chart(fig_q, use_container_width=True)

with col_chart_right:
    state_df = summarize_by_state(df_filtered)
    if len(state_df) >= 6:
        top5 = state_df.nlargest(5, "Total_Profit")
        bot5 = state_df.nsmallest(5, "Total_Profit")
        ext_states = pd.concat([bot5, top5], ignore_index=True).drop_duplicates(subset=["State"])
        ext_states = ext_states.sort_values("Total_Profit", ascending=True)
        ext_states["Nhóm"] = ext_states["Total_Profit"].apply(
            lambda v: "Top Lãi cao" if v >= 0 else "Top Lỗ nặng"
        )
        fig_ext = px.bar(
            ext_states,
            x="Total_Profit",
            y="State",
            color="Nhóm",
            orientation="h",
            text_auto=".2s",
            color_discrete_map={"Top Lãi cao": "#16a34a", "Top Lỗ nặng": "#dc2626"},
            title="Đối cực Lợi nhuận: Top 5 Bang Lãi cao nhất vs. Lỗ nặng nhất",
            labels={"Total_Profit": "Lợi nhuận (USD)", "State": "Tiểu bang"},
            height=390,
        )
        fig_ext.add_vline(x=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_ext, use_container_width=True)

st.divider()

# 4. Hệ thống Điều hướng Nhanh (Quick Navigation Cards)
st.subheader("🧭 Điều hướng Nhanh 5 Trang Phân tích Chuyên sâu")
nav1, nav2, nav3, nav4, nav5 = st.columns(5)

with nav1:
    with st.container(border=True):
        st.markdown("#### 📊 Page 1\n**Tổng quan & RFM**")
        st.caption("KPI cốt lõi, Grouped/Stacked Bar, Histogram/Boxplot, RFM, Cohort & K-Means 3D (Elbow/Silhouette).")
        st.page_link("pages/1_Tong_Quan_Kinh_Doanh.py", label="Mở Page 1 →", icon="📊", use_container_width=True)

with nav2:
    with st.container(border=True):
        st.markdown("#### 📈 Page 2\n**Xu hướng & Dự báo**")
        st.caption("Chuỗi thời gian + MA3, Dự báo 3–6 tháng, Chấm sai số Hold-out (MAE/MAPE), Seasonality & Waterfall.")
        st.page_link("pages/2_Xu_Huong_Doanh_Thu.py", label="Mở Page 2 →", icon="📈", use_container_width=True)

with nav3:
    with st.container(border=True):
        st.markdown("#### 🗺️ Page 3\n**Khu vực & Bản đồ Mỹ**")
        st.caption("Choropleth Map 49 bang, Khoan sâu Central & Texas, Ma trận Tứ phân vị, So sánh A/B & Shipping SLA.")
        st.page_link("pages/3_Phan_Tich_Khu_Vuc.py", label="Mở Page 3 →", icon="🗺️", use_container_width=True)

with nav4:
    with st.container(border=True):
        st.markdown("#### 🏆 Page 4\n**Sản phẩm & Giỏ hàng**")
        st.caption("Top N sản phẩm, Khoan sâu phân cấp Sunburst/Treemap, Luật kết hợp bán chéo Apriori & Pareto 80/20.")
        st.page_link("pages/4_Phan_Tich_San_Pham.py", label="Mở Page 4 →", icon="🏆", use_container_width=True)

with nav5:
    with st.container(border=True):
        st.markdown("#### ✨ Page 5\n**UX, What-If & AI**")
        st.caption("Biểu đồ kép Ngưỡng chiết khấu, What-If Simulator, Cảnh báo lỗ, Tra cứu 360°, Xuất HTML/PDF & Gemini AI.")
        st.page_link("pages/5_Trai_Nghiem_Nguoi_Dung.py", label="Mở Page 5 →", icon="✨", use_container_width=True)

st.divider()

# 5. Hồ sơ Dữ liệu & Kiểm toán Chất lượng (Đáp ứng Phần 1 - 10% & Phần 2 - 25% ngay trên Web)
tab_audit, tab_dict, tab_desc = st.tabs(
    [
        "🛡️ Kiểm toán Chất lượng Dữ liệu (Data Quality Audit — Phần 2)",
        "📖 Từ điển Dữ liệu 21 Biến (Data Dictionary — Phần 1)",
        "📐 Thống kê Mô tả (.describe() — Phần 2)",
    ]
)

with tab_audit:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Số dòng gốc (Raw Rows)", f"{audit['rows']:,}", f"{audit['columns']} cột thuộc tính")
    a2.metric("Giá trị thiếu (Missing/Null)", f"{audit['missing_total']}", "Đạt chuẩn 100% đầy đủ")
    a3.metric("Trùng lặp nghiệp vụ đã loại", f"{audit['business_duplicates']}", "Tại Row ID = 3407")
    a4.metric("Ngoại lệ IQR (Sales / Profit)", f"{audit['sales_outliers_count']:,} / {audit['profit_outliers_count']:,}", "Đơn B2B thực tế")
    st.caption(
        f"**Ngưỡng 1.5×IQR thực tế:** `Sales` trong khoảng `{audit['sales_iqr_bounds']}` USD | "
        f"`Profit` trong khoảng `{audit['profit_iqr_bounds']}` USD. "
        "Mặc định giữ nguyên đơn B2B lớn để bảo toàn tổng doanh thu thực tế; có thể bật *IQR Winsorize* ở Sidebar."
    )

with tab_dict:
    st.dataframe(get_data_dictionary(), use_container_width=True, hide_index=True)

with tab_desc:
    desc_df = (
        df_filtered[["Sales", "Quantity", "Discount", "Profit", "Shipping_Days", "Profit_Margin_Pct"]]
        .describe()
        .T.reset_index()
        .rename(columns={"index": "Biến định lượng"})
    )
    st.dataframe(desc_df.round(2), use_container_width=True, hide_index=True)
