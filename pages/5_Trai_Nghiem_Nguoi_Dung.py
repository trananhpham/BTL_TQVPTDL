"""Page 5: Trải nghiệm người dùng — Sidebar, Tabs/Expander, @st.cache_data & Gemini Flash Chatbot."""

from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib
import data.analytics as _analytics_mod

importlib.reload(_analytics_mod)

from components.ai_chatbot import render_ai_chatbot
from components.sidebar import init_page, render_global_sidebar
from data.analytics import (
    build_executive_html_report,
    compute_elasticity_curve,
    detect_loss_anomalies,
    pivot_region_category,
    pivot_year_segment,
    simulate_discount_what_if,
    summarize_by_time,
    summarize_discount_bands,
    train_loss_risk_model,
)
from utils.formatters import format_int, format_usd
from utils.ui_styles import render_dynamic_insight_banner

init_page("Page 5 - UX, Cache & AI Chatbot", icon="✨")

# Đo thời gian nạp từ RAM nhờ @st.cache_data
start_time = time.perf_counter()
df_clean, audit, df_sidebar = render_global_sidebar()
load_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

# Quy hoạch thêm các bộ lọc nâng cao vào Sidebar
with st.sidebar:
    st.subheader("🔍 Bộ lọc chuyên sâu (Page 5)")
    ship_modes = sorted(df_clean["Ship Mode"].unique().tolist())
    selected_ship_modes = st.multiselect(
        "Hình thức giao hàng (Ship Mode)",
        options=ship_modes,
        default=ship_modes,
    )
    max_discount = st.slider(
        "Chiết khấu tối đa (Discount ≤)",
        min_value=0.0,
        max_value=0.8,
        value=0.8,
        step=0.05,
    )

df_filtered = df_sidebar[
    (df_sidebar["Ship Mode"].isin(selected_ship_modes))
    & (df_sidebar["Discount"] <= max_discount)
]


@st.cache_data(show_spinner=False)
def build_cached_pivots(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache kết quả Pivot Table để phản hồi real-time khi tương tác Dashboard."""
    return pivot_region_category(data), pivot_year_segment(data)


@st.cache_data(show_spinner="Đang huấn luyện mô hình học máy dự báo rủi ro đơn lỗ...")
def cached_loss_ml(data: pd.DataFrame) -> dict[str, object]:
    """Cache kết quả huấn luyện Random Forest & Logistic Regression trên tập đang lọc."""
    return train_loss_risk_model(data)


st.title("✨ Page 5: Trải nghiệm người dùng, Tối ưu Cache & Trợ lý AI")
st.caption(
    "Toàn bộ bộ lọc được quy hoạch tại Sidebar (đồng bộ `st.session_state`), "
    "phân tầng nội dung bằng `Tabs`/`Expander`, tăng tốc với `@st.cache_data` và tích hợp **Gemini 2.5 Flash**."
)

if df_filtered.empty:
    st.warning("Không có bản ghi nào khớp với các tiêu chí lọc trong Sidebar.")
    st.stop()

render_dynamic_insight_banner(df_filtered)

c1, c2, c3, c4 = st.columns(4)
c1.metric("⚡ Thời gian nạp Cache", f"{load_elapsed_ms:.2f} ms")
c2.metric("📄 Bản ghi đang lọc", f"{format_int(len(df_filtered))} / {format_int(len(df_clean))}")
c3.metric("💰 Doanh thu lọc được", format_usd(df_filtered["Sales"].sum()))
c4.metric("📈 Lợi nhuận lọc được", format_usd(df_filtered["Profit"].sum()))

# Tổ chức nội dung bằng Tabs
tab_ai, tab_sim, tab_lookup, tab_disc, tab_pivot, tab_data = st.tabs(
    [
        "🤖 Trợ lý AI Phân tích B2B (Gemini Flash)",
        "🎯 What-If Simulator & Cảnh báo Lỗ",
        "🔎 Tra cứu 360° (Đơn / Khách / SP)",
        "📉 Tác động Chiết khấu & Lợi nhuận",
        "🧮 Pivot Table & GroupBy",
        "🗂️ Dữ liệu sạch & Xuất Báo cáo",
    ]
)

with tab_ai:
    import importlib
    import components.ai_chatbot as _ai_mod

    importlib.reload(_ai_mod)
    _ai_mod.render_ai_chatbot(df_filtered, location="main")

with tab_sim:
    st.subheader("🎯 Mô phỏng Kịch bản Kinh doanh & Độ co giãn của Cầu theo Giá (Price Elasticity)")
    st.caption(
        "Mô hình kinh tế lượng giả lập tác động tài chính khi đặt trần chiết khấu tối đa, điều chỉnh giá nhóm hàng lỗ "
        "và tính đến **hệ số co giãn của cầu theo giá (ε)**: khi giảm chiết khấu (tức giá bán thực tế tăng), sản lượng tiêu thụ "
        "sẽ điều chỉnh theo công thức Q_mới = Q_cũ × (1 + ε × %ΔGiá)."
    )
    s1, s2, s3 = st.columns(3)
    with s1:
        cap_pct = st.slider("Đặt trần chiết khấu tối đa (%)", min_value=5, max_value=50, value=20, step=5)
    with s2:
        uplift_pct = st.slider(
            "Điều chỉnh giá nhóm lỗ Tables/Bookcases/Supplies (%)",
            min_value=-10,
            max_value=25,
            value=5,
            step=1,
        )
    with s3:
        elasticity_val = st.slider(
            "Hệ số co giãn cầu theo giá (ε)",
            min_value=-1.5,
            max_value=0.0,
            value=-0.4,
            step=0.1,
            help="ε = 0: Cầu hoàn toàn không co giãn (sản lượng giữ nguyên). ε = -0.4: Giá thực tế tăng 10% thì sản lượng giảm 4%.",
        )

    sim_res = simulate_discount_what_if(
        df_filtered,
        max_discount_cap=cap_pct / 100.0,
        loss_subcat_uplift_pct=float(uplift_pct),
        price_elasticity=float(elasticity_val),
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Lợi nhuận mô phỏng (Có co giãn)",
        format_usd(sim_res["sim_profit"]),
        delta=f"+{format_usd(sim_res['profit_gain'])}" if sim_res["profit_gain"] >= 0 else format_usd(sim_res["profit_gain"]),
    )
    m2.metric(
        "Biên lợi nhuận mới",
        f"{sim_res['sim_margin_pct']:.2f}%",
        delta=f"{sim_res['sim_margin_pct'] - sim_res['actual_margin_pct']:+.2f}%",
    )
    m3.metric(
        "Doanh thu mô phỏng",
        format_usd(sim_res["sim_sales"]),
        delta=f"Sản lượng giữ lại: {sim_res['volume_retention_pct']:.1f}%",
    )
    m4.metric(
        "Giao dịch bị lỗ còn lại",
        format_int(sim_res["sim_loss_rows"]),
        delta=f"{sim_res['sim_loss_rows'] - sim_res['actual_loss_rows']:,} dòng",
        delta_color="inverse",
    )

    c_sim_left, c_sim_right = st.columns(2)
    with c_sim_left:
        compare_df = pd.DataFrame(
            {
                "Kịch bản": ["Thực tế hiện tại", "Thực tế hiện tại", "Mô phỏng What-If", "Mô phỏng What-If"],
                "Chỉ số": ["Doanh thu", "Lợi nhuận", "Doanh thu", "Lợi nhuận"],
                "Giá trị (USD)": [
                    sim_res["actual_sales"],
                    sim_res["actual_profit"],
                    sim_res["sim_sales"],
                    sim_res["sim_profit"],
                ],
            }
        )
        fig_sim = px.bar(
            compare_df,
            x="Chỉ số",
            y="Giá trị (USD)",
            color="Kịch bản",
            barmode="group",
            text_auto=".2s",
            title=f"Thực tế vs. What-If (Trần CK {cap_pct}%, ε = {elasticity_val})",
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    with c_sim_right:
        curve_df = compute_elasticity_curve(
            df_filtered, uplift_pct=float(uplift_pct), elasticity=float(elasticity_val if elasticity_val != 0 else -0.5)
        )
        if not curve_df.empty:
            fig_curve = go.Figure()
            fig_curve.add_trace(
                go.Scatter(
                    x=curve_df["Tran_chiet_khau_pct"],
                    y=curve_df["Loi_nhuan_khong_co_gian"],
                    mode="lines+markers",
                    name="Không co giãn (ε = 0.0)",
                    line=dict(color="#64748b", width=2, dash="dot"),
                )
            )
            fig_curve.add_trace(
                go.Scatter(
                    x=curve_df["Tran_chiet_khau_pct"],
                    y=curve_df["Loi_nhuan_co_co_gian"],
                    mode="lines+markers",
                    name=f"Có co giãn cầu (ε = {elasticity_val if elasticity_val != 0 else -0.5})",
                    line=dict(color="#16a34a", width=3),
                )
            )
            fig_curve.update_layout(
                title="Đường cong Tối ưu hóa Lợi nhuận theo Trần Chiết khấu (%)",
                xaxis_title="Trần chiết khấu tối đa (%)",
                yaxis_title="Tổng lợi nhuận kỳ vọng (USD)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_curve, use_container_width=True)

    st.divider()
    st.subheader("🧠 Mô hình Học máy Giải thích Rủi ro Đơn hàng Thua lỗ (Explainable ML — Random Forest & Logit Odds Ratio)")
    st.caption(
        "Huấn luyện mô hình phân loại nhị phân dự báo xác suất đơn hàng bị lỗ P(Loss = 1 | X) dựa trên Discount, Quantity, "
        "Shipping_Days, Category, Sub-Category, Region, Segment, Ship Mode (chia 80% Train / 20% Test phân tầng)."
    )
    ml_res = cached_loss_ml(df_filtered)
    ml_metrics = ml_res.get("metrics", {})
    feat_df = ml_res.get("feature_importance", pd.DataFrame())
    if isinstance(ml_metrics, dict) and ml_metrics:
        ml1, ml2, ml3, ml4 = st.columns(4)
        ml1.metric("🎯 Random Forest ROC-AUC", f"{ml_metrics['rf_roc_auc']:.4f}")
        ml2.metric("⚖️ Random Forest F1-Score", f"{ml_metrics['rf_f1']:.4f}")
        ml3.metric("✅ Độ chính xác (Test Accuracy)", f"{ml_metrics['rf_accuracy']:.2f}%")
        ml4.metric("📈 Logistic ROC-AUC", f"{ml_metrics['lr_roc_auc']:.4f}")

        if isinstance(feat_df, pd.DataFrame) and not feat_df.empty:
            ml_left, ml_right = st.columns([1.3, 1.2])
            with ml_left:
                fig_fi = px.bar(
                    feat_df.sort_values("Tầm quan trọng RF (%)", ascending=True),
                    x="Tầm quan trọng RF (%)",
                    y="Biến giải thích (Feature)",
                    orientation="h",
                    color="Tầm quan trọng RF (%)",
                    color_continuous_scale="Tealgrn",
                    text_auto=".1f",
                    title="Top 12 Yếu tố Quyết định Rủi ro Thua lỗ (Random Forest Feature Importance %)",
                )
                fig_fi.update_layout(coloraxis_showscale=False, height=420)
                st.plotly_chart(fig_fi, use_container_width=True)
            with ml_right:
                st.dataframe(
                    feat_df.style.format(
                        {
                            "Tầm quan trọng RF (%)": "{:.2f}%",
                            "Hệ số Logit (β)": "{:+.3f}",
                            "Odds Ratio (e^β)": "{:.2f}x",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )

    st.divider()
    st.subheader("🚨 Trung tâm Cảnh báo Đơn hàng Thua lỗ Nghiêm trọng")
    loss_thresh = st.slider(
        "Ngưỡng cảnh báo lỗ tối thiểu (Profit ≤ USD)",
        min_value=-1000,
        max_value=-20,
        value=-150,
        step=10,
    )
    anomalies_df = detect_loss_anomalies(df_filtered, profit_threshold=float(loss_thresh))
    if anomalies_df.empty:
        st.success("Không có đơn hàng nào lỗ vượt ngưỡng cảnh báo trong bộ lọc hiện tại.")
    else:
        st.warning(f"Phát hiện **{len(anomalies_df):,}** giao dịch có lợi nhuận ≤ {format_usd(loss_thresh)}.")
        st.dataframe(
            anomalies_df.style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}", "Discount": "{:.0%}"}),
            use_container_width=True,
            hide_index=True,
        )

with tab_lookup:
    st.subheader("🔎 Tra cứu Hồ sơ 360° (Đơn hàng, Khách hàng & Sản phẩm)")
    q1, q2 = st.columns([1.2, 2.8])
    with q1:
        lookup_type = st.selectbox(
            "Tìm kiếm theo trường",
            options=["Customer Name", "Order ID", "Product Name", "City", "State"],
        )
    with q2:
        search_text = st.text_input(
            "Nhập từ khóa tra cứu",
            placeholder="Ví dụ: Sean Miller, CA-2016-152156, Canon, Texas...",
        ).strip()

    if not search_text:
        st.info("Nhập từ khóa để tra cứu nhanh chi tiết giao dịch và tổng hợp KPI tương ứng.")
    else:
        matched = df_filtered[
            df_filtered[lookup_type].astype(str).str.contains(search_text, case=False, na=False)
        ]
        if matched.empty:
            st.warning(f"Không tìm thấy kết quả nào cho `{search_text}` trong `{lookup_type}`.")
        else:
            lk1, lk2, lk3, lk4 = st.columns(4)
            lk1.metric("Số dòng khớp", format_int(len(matched)))
            lk2.metric("Số đơn hàng", format_int(matched["Order ID"].nunique()))
            lk3.metric("Tổng doanh thu", format_usd(matched["Sales"].sum()))
            lk4.metric("Tổng lợi nhuận", format_usd(matched["Profit"].sum()))

            display_cols = [
                "Order ID",
                "Order Date",
                "Customer Name",
                "State",
                "Category",
                "Sub-Category",
                "Product Name",
                "Sales",
                "Quantity",
                "Discount",
                "Profit",
            ]
            st.dataframe(
                matched[display_cols].style.format(
                    {"Sales": "${:,.2f}", "Profit": "${:,.2f}", "Discount": "{:.0%}"}
                ),
                use_container_width=True,
                hide_index=True,
            )

with tab_disc:
    st.subheader("Mối quan hệ giữa Chiết khấu (Discount) và Lợi nhuận (Profit)")

    # Biểu đồ kép tổng hợp — ĐỨNG RIÊNG, KHÔNG dùng columns
    st.subheader("📊 Tác động Chiết khấu đến Lợi nhuận — Phân tích theo Ngưỡng")
    st.caption(
        "Trục trái (cột xanh-đỏ): Lợi nhuận trung bình (USD) theo từng ngưỡng Discount. "
        "Trục phải (đường cam): Tỷ lệ giao dịch lỗ (%). "
        "Ngưỡng >20% là điểm bùng phát: lợi nhuận âm và tỷ lệ lỗ vọt lên gần 100%."
    )
    discount_bands = summarize_discount_bands(df_filtered)
    if not discount_bands.empty:
        # Màu cho cột: xanh lá nếu LN > 0, đỏ nếu LN < 0
        bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in discount_bands["Loi_nhuan_TB"]]

        fig_disc_main = go.Figure()
        # Cột lợi nhuận TB (trục Y trái)
        fig_disc_main.add_trace(go.Bar(
            x=discount_bands["Nguong_chiet_khau"].astype(str),
            y=discount_bands["Loi_nhuan_TB"],
            name="Lợi nhuận TB (USD)",
            marker_color=bar_colors,
            text=[f"${v:,.1f}" for v in discount_bands["Loi_nhuan_TB"]],
            textposition="outside",
            yaxis="y1",
        ))
        # Đường tỷ lệ lỗ (trục Y phải)
        fig_disc_main.add_trace(go.Scatter(
            x=discount_bands["Nguong_chiet_khau"].astype(str),
            y=discount_bands["Ty_le_lo"],
            name="Tỷ lệ lỗ (%)",
            mode="lines+markers+text",
            marker=dict(size=10, color="#e67e22"),
            line=dict(color="#e67e22", width=3),
            text=[f"{v:.1f}%" for v in discount_bands["Ty_le_lo"]],
            textposition="top center",
            yaxis="y2",
        ))
        fig_disc_main.add_hline(y=0, line_dash="dash", line_color="black", line_width=1.5, yref="y1")
        fig_disc_main.update_layout(
            title="📉 Chiết khấu > 20%: Điểm bùng phát thua lỗ — Lợi nhuận TB & Tỷ lệ lỗ theo Ngưỡng Discount",
            xaxis_title="Ngưỡng chiết khấu",
            yaxis=dict(title="Lợi nhuận TB (USD)", tickformat="$,.0f"),
            yaxis2=dict(title="Tỷ lệ lỗ (%)", overlaying="y", side="right", range=[0, 110], ticksuffix="%"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=460,
            hovermode="x unified",
            bargap=0.3,
        )
        st.plotly_chart(fig_disc_main, use_container_width=True)

        # Callout metrics dưới biểu đồ
        no_disc = discount_bands[discount_bands["Nguong_chiet_khau"] == "0%"]
        over30 = discount_bands[discount_bands["Nguong_chiet_khau"] == ">30%"]
        d1, d2, d3 = st.columns(3)
        if not no_disc.empty:
            d1.metric("Không chiết khấu (0%): LN TB", f"${float(no_disc['Loi_nhuan_TB'].iloc[0]):,.2f}", "0% giao dịch lỗ")
        if not over30.empty:
            d2.metric("Chiết khấu >30%: LN TB", f"${float(over30['Loi_nhuan_TB'].iloc[0]):,.2f}",
                      f"{float(over30['Ty_le_lo'].iloc[0]):.1f}% giao dịch lỗ", delta_color="inverse")

        above_20 = discount_bands[discount_bands["Nguong_chiet_khau"].isin([">20–30%", ">30%"])]
        if not above_20.empty and above_20["So_dong"].sum() > 0:
            loss_above = float((above_20["Ty_le_lo"] * above_20["So_dong"]).sum() / above_20["So_dong"].sum())
            d3.metric("Gộp >20%: Tỷ lệ lỗ TB", f"{loss_above:.1f}%", "Nên đặt trần 20%", delta_color="inverse")

        st.info("💡 **Khuyến nghị:** Đặt trần chiết khấu tối đa 20% cho tất cả đơn hàng. Mọi đơn vượt ngưỡng này cần phê duyệt quản lý.")

    fig_scatter = px.scatter(
        df_filtered,
        x="Discount",
        y="Profit",
        color="Category",
        size="Sales",
        hover_data=["State", "Sub-Category", "Region", "Sales", "Quantity"],
        title="Biểu đồ phân tán: Chiết khấu vs. Lợi nhuận theo Ngành hàng",
        labels={"Discount": "Tỷ lệ chiết khấu", "Profit": "Lợi nhuận (USD)"},
    )
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Điểm hòa vốn (0 USD)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    with st.expander("💡 Giải thích Insight về chính sách chiết khấu"):
        loss_count = int((df_filtered["Profit"] < 0).sum())
        loss_pct = (loss_count / len(df_filtered) * 100.0) if len(df_filtered) else 0.0
        st.markdown(
            f"""
            - Tập dữ liệu đang lọc có **{format_int(loss_count)}** giao dịch lỗ ({loss_pct:.2f}%).
            - Khi chiết khấu vượt quá **20%**, phần lớn giao dịch rơi vào vùng lợi nhuận âm.
            """
        )

with tab_pivot:
    st.subheader("Tổng hợp đa chiều với Pivot Table (đã được @st.cache_data)")
    pivot_reg_cat, pivot_yr_seg = build_cached_pivots(df_filtered)

    with st.expander("1️⃣ Pivot Table: Doanh thu theo Khu vực (Region) × Danh mục (Category)", expanded=True):
        st.dataframe(pivot_reg_cat.style.format("${:,.2f}"), use_container_width=True)

    with st.expander("2️⃣ Pivot Table: Doanh thu & Số đơn hàng theo Năm × Phân khúc (Segment)", expanded=True):
        st.dataframe(pivot_yr_seg.round(2), use_container_width=True)

    with st.expander("3️⃣ GroupBy: Tổng hợp doanh thu và đơn hàng theo Năm"):
        yr_summary = summarize_by_time(df_filtered, freq="Y")
        st.dataframe(
            yr_summary.style.format(
                {
                    "Total_Sales": "${:,.2f}",
                    "Total_Profit": "${:,.2f}",
                    "Order_Count": "{:,}",
                    "Total_Quantity": "{:,}",
                    "Profit_Margin_Pct": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

with tab_data:
    st.subheader("Xem trước dữ liệu sau khi làm sạch và xuất báo cáo nhanh")
    st.dataframe(df_filtered.head(100), use_container_width=True, hide_index=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        csv_bytes = df_filtered.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="⬇️ Tải dữ liệu đã lọc (.CSV)",
            data=csv_bytes,
            file_name="superstore_cleaned_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        tot_sales = float(df_filtered["Sales"].sum())
        tot_profit = float(df_filtered["Profit"].sum())
        margin_pct = (tot_profit / tot_sales * 100.0) if tot_sales else 0.0
        top_cat = df_filtered.groupby("Category")["Profit"].sum().idxmax()
        worst_subcat = df_filtered.groupby("Sub-Category")["Profit"].sum().idxmin()
        summary_md = (
            f"# BÁO CÁO TÓM TẮT ĐIỀU HÀNH SUPERSTORE\n\n"
            f"- **Phạm vi dữ liệu:** {df_filtered['Order Date'].min().strftime('%Y-%m-%d')} → "
            f"{df_filtered['Order Date'].max().strftime('%Y-%m-%d')} ({len(df_filtered):,} dòng)\n"
            f"- **Tổng doanh thu:** {format_usd(tot_sales)}\n"
            f"- **Tổng lợi nhuận:** {format_usd(tot_profit)} (Biên lợi nhuận: {margin_pct:.2f}%)\n"
            f"- **Tổng số đơn hàng:** {format_int(df_filtered['Order ID'].nunique())}\n"
            f"- **Ngành hàng sinh lời cao nhất:** {top_cat}\n"
            f"- **Nhóm sản phẩm cần tối ưu nhất:** {worst_subcat}\n"
        ).encode("utf-8")
        st.download_button(
            label="📝 Tải Báo cáo Tóm tắt (.MD)",
            data=summary_md,
            file_name="executive_summary_superstore.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with dl3:
        html_report = build_executive_html_report(df_filtered).encode("utf-8")
        st.download_button(
            label="🖨️ Xuất Báo cáo HTML (In PDF Ctrl+P)",
            data=html_report,
            file_name="executive_report_superstore.html",
            mime="text/html",
            use_container_width=True,
        )

with st.expander("🛠️ Chi tiết Kiến trúc Kỹ thuật: Cache, Session State & Chất lượng Dữ liệu"):
    st.markdown(
        f"""
        - **Clean Architecture:** Tách biệt hoàn toàn thành 4 tầng `data/`, `components/`, `pages/`, `utils/`.
        - **Đồng bộ `st.session_state`:** Lưu trữ các bộ lọc toàn cục (`ss_date_range`, `ss_regions`, `ss_categories`, `ss_segments`, `ss_cap_outliers`) giúp giữ nguyên bộ lọc khi chuyển giữa Page 1–5.
        - **Tối ưu hiệu năng (`@st.cache_data`):** Dữ liệu CSV ({audit['rows']:,} dòng × {audit['columns']} cột, đã loại {audit['business_duplicates']} trùng lặp) chỉ nạp vào RAM một lần duy nhất.
        """
    )
