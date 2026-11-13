"""Module giao diện CSS hiện đại (SaaS BI Theme) và Banner Auto-Insight động theo bộ lọc."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from data.analytics import generate_dynamic_insight


def apply_custom_styles() -> None:
    """Nhúng CSS nâng cấp giao diện KPI Cards, Tabs và Banner theo chuẩn SaaS Dashboard."""
    st.markdown(
        """
        <style>
        /* Nâng cấp thẻ st.metric */
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.10);
        }
        div[data-testid="stMetricLabel"] > div {
            font-weight: 600 !important;
            color: #475569 !important;
        }
        /* Hộp Auto-Insight động */
        .dynamic-insight-box {
            background: linear-gradient(90deg, #eff6ff 0%, #f8fafc 100%);
            border: 1px solid #bfdbfe;
            border-left: 5px solid #1d4ed8;
            border-radius: 10px;
            padding: 12px 18px;
            margin: 8px 0 16px 0;
            font-size: 0.94rem;
            color: #1e293b;
        }
        .dynamic-insight-box b {
            color: #1e3a8a;
        }
        .badge-pill {
            display: inline-block;
            padding: 2px 9px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            margin-right: 6px;
        }
        .badge-blue { background: #dbeafe; color: #1e40af; }
        .badge-red { background: #fee2e2; color: #991b1b; }
        .badge-green { background: #dcfce7; color: #166534; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dynamic_insight_banner(df: pd.DataFrame) -> None:
    """Hiển thị hộp Auto-Insight tự động nhảy số và nhận xét theo bộ lọc hiện tại."""
    if df.empty:
        return
    ins = generate_dynamic_insight(df)
    if not ins:
        return

    yoy_str = (
        f"Tăng trưởng năm gần nhất ({ins['latest_year']}): <b>{ins['yoy_sales_pct']:+.1f}%</b>"
        if ins["yoy_sales_pct"] != 0
        else f"Năm quan sát: <b>{ins['latest_year']}</b>"
    )
    st.markdown(
        f"""
        <div class="dynamic-insight-box">
          <span class="badge-pill badge-blue">⚡ AUTO-INSIGHT THEO BỘ LỌC</span>
          <span class="badge-pill badge-green">Biên LN: {ins['margin_pct']:.2f}%</span>
          <span class="badge-pill badge-red">Đơn chiết khấu &gt;20% lỗ: {ins['high_disc_loss_rate']:.1f}%</span>
          <div style="margin-top:6px;">
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
