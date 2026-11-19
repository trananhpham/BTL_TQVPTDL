"""Module giao diện CSS hiện đại (SaaS BI Theme — tương thích Dark/Light Mode), Bảng màu Ngữ nghĩa và Banner Auto-Insight."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data.analytics import generate_dynamic_insight

# Bảng màu ngữ nghĩa nhất quán xuyên suốt toàn bộ 6 trang Dashboard
CATEGORY_COLORS: dict[str, str] = {
    "Technology": "#2563eb",       # Xanh Navy đậm (Ngành lãi cao nhất)
    "Office Supplies": "#0d9488",  # Xanh Ngọc Teal (Ngành ổn định)
    "Furniture": "#f59e0b",        # Cam Hổ Phách (Ngành biên mỏng cần cảnh báo)
}

REGION_COLORS: dict[str, str] = {
    "West": "#1d4ed8",
    "East": "#0f766e",
    "South": "#7c3aed",
    "Central": "#dc2626",          # Đỏ cảnh báo cho vùng biên lợi nhuận thấp nhất
}


def apply_custom_styles() -> None:
    """Nhúng CSS nâng cấp giao diện KPI Cards, Tabs, Filter Breadcrumb và Banner tương thích cả Light Mode & Dark Mode."""
    st.markdown(
        """
        <style>
        /* Nâng cấp thẻ st.metric tương thích tự động với cả Light Mode & Dark Mode */
        div[data-testid="stMetric"] {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.15);
            border-left-color: #2563eb;
        }
        div[data-testid="stMetricLabel"] > div {
            font-weight: 600 !important;
            opacity: 0.88;
        }
        /* Thanh trạng thái bộ lọc toàn cục (Active Filter Breadcrumb) */
        .filter-breadcrumb-bar {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
            margin: 4px 0 10px 0;
            font-size: 0.82rem;
        }
        .filter-chip {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 10px;
            border-radius: 999px;
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid rgba(148, 163, 184, 0.3);
            font-weight: 500;
        }
        /* Hộp Auto-Insight động tương thích hoàn toàn Dark & Light Mode */
        .dynamic-insight-box {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid rgba(59, 130, 246, 0.32);
            border-left: 5px solid #3b82f6;
            border-radius: 12px;
            padding: 14px 18px;
            margin: 6px 0 18px 0;
            font-size: 0.94rem;
            line-height: 1.55;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
        }
        .dynamic-insight-box b {
            color: #60a5fa;
        }
        @media (prefers-color-scheme: light) {
            .dynamic-insight-box b {
                color: #1d4ed8;
            }
        }
        .badge-pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-blue { background: rgba(59, 130, 246, 0.18); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.35); }
        .badge-red { background: rgba(239, 68, 68, 0.18); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.35); }
        .badge-green { background: rgba(34, 197, 94, 0.18); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.35); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dynamic_insight_banner(df: pd.DataFrame) -> None:
    """Hiển thị thanh trạng thái bộ lọc đồng bộ (Filter Breadcrumb) và hộp Auto-Insight tự động nhảy số."""
    if df.empty:
        return
    ins = generate_dynamic_insight(df)
    if not ins:
        return

    # 1. Tạo thanh trạng thái bộ lọc (Active Filter Breadcrumb)
    dt_min = df["Order Date"].min().strftime("%d/%m/%Y")
    dt_max = df["Order Date"].max().strftime("%d/%m/%Y")
    regs = sorted(df["Region"].unique().tolist())
    cats = sorted(df["Category"].unique().tolist())
    segs = sorted(df["Segment"].unique().tolist())
    iqr_mode = "Đang bật IQR Winsorize" if st.session_state.get("ss_cap_outliers", False) else "Giữ nguyên đơn B2B gốc"

    reg_str = "Tất cả (4/4)" if len(regs) == 4 else ", ".join(regs)
    cat_str = "Tất cả (3/3)" if len(cats) == 3 else ", ".join(cats)
    seg_str = "Tất cả (3/3)" if len(segs) == 3 else ", ".join(segs)

    yoy_str = (
        f"Tăng trưởng năm gần nhất ({ins['latest_year']}): <b>{ins['yoy_sales_pct']:+.1f}%</b>"
        if ins["yoy_sales_pct"] != 0
        else f"Năm quan sát: <b>{ins['latest_year']}</b>"
    )
    st.markdown(
        f"""
        <div class="filter-breadcrumb-bar">
          <span class="filter-chip">🗓️ <b>Thời gian:</b> {dt_min} → {dt_max}</span>
          <span class="filter-chip">🌎 <b>Khu vực:</b> {reg_str}</span>
          <span class="filter-chip">📦 <b>Danh mục:</b> {cat_str}</span>
          <span class="filter-chip">👥 <b>Phân khúc:</b> {seg_str}</span>
          <span class="filter-chip">🛡️ <b>Ngoại lệ:</b> {iqr_mode}</span>
        </div>
        <div class="dynamic-insight-box">
          <span class="badge-pill badge-blue">⚡ AUTO-INSIGHT THEO BỘ LỌC</span>
          <span class="badge-pill badge-green">Biên LN: {ins['margin_pct']:.2f}%</span>
          <span class="badge-pill badge-red">Đơn chiết khấu &gt;20% lỗ: {ins['high_disc_loss_rate']:.1f}%</span>
          <div style="margin-top:7px;">
            • <b>Hiệu suất:</b> Doanh thu đạt <b>${ins['total_sales']:,.0f}</b> ({ins['total_orders']:,} đơn) — {yoy_str}.
            Ngành dẫn đầu lợi nhuận: <b>{ins['best_cat']}</b> (<b>+${ins['best_cat_profit']:,.0f}</b>).<br>
            • <b>Điểm nghẽn cần xử lý:</b> Nhóm hàng lỗ nặng nhất là <b>{ins['worst_sub']}</b> (<b>${ins['worst_sub_profit']:,.0f}</b>)
            tại bang <b>{ins['worst_state']}</b> (<b>${ins['worst_state_profit']:,.0f}</b>).
            Có <b>{ins['loss_rows']:,}</b> dòng giao dịch âm lợi nhuận ({ins['loss_rate_pct']:.1f}%).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
