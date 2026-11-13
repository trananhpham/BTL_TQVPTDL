"""Thành phần Sidebar dùng chung với đồng bộ st.session_state xuyên suốt 5 trang."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from data.loader import filter_dataframe, get_cached_superstore_data
from utils.session_state import (
    get_date_range_timestamps,
    init_global_state,
    reset_global_state,
)
from utils.ui_styles import apply_custom_styles


def init_page(title: str, icon: str = "📊") -> None:
    """Thiết lập cấu hình chuẩn và giao diện CSS hiện đại cho từng trang Dashboard."""
    st.set_page_config(
        page_title=f"{title} | Superstore BI",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()


def _sync_key(widget_key: str, state_key: str) -> None:
    """Callback sao chép giá trị từ widget sang global session_state."""
    st.session_state[state_key] = st.session_state[widget_key]


def render_global_sidebar(
    include_date: bool = True,
    include_region: bool = True,
    include_category: bool = True,
    include_segment: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Hiển thị Sidebar toàn cục được đồng bộ bằng st.session_state và trả về (df_clean, audit, df_filtered)."""
    # Đọc tạm dữ liệu theo trạng thái cap_outliers & tệp tải lên (nếu có)
    cap_flag = bool(st.session_state.get("ss_cap_outliers", False))
    custom_bytes = st.session_state.get("ss_uploaded_csv_bytes", None)
    df_clean, audit = get_cached_superstore_data(cap_outliers=cap_flag, uploaded_bytes=custom_bytes)
    init_global_state(df_clean)

    min_dt = df_clean["Order Date"].min().date()
    max_dt = df_clean["Order Date"].max().date()
    all_regions = sorted(df_clean["Region"].unique().tolist())
    all_categories = sorted(df_clean["Category"].unique().tolist())
    all_segments = sorted(df_clean["Segment"].unique().tolist())

    with st.sidebar:
        st.header("🎛️ Bộ lọc đồng bộ toàn cục")
        st.caption("Mọi lựa chọn được lưu vào `st.session_state` và giữ nguyên khi chuyển trang.")

        with st.expander("📂 Nạp tệp CSV tùy chỉnh (Tùy chọn)"):
            up_file = st.file_uploader(
                "Tải lên CSV cùng cấu trúc Superstore",
                type=["csv"],
                key="_w_csv_uploader",
                help="Nếu để trống, hệ thống sử dụng Sample - Superstore.csv chuẩn (9.994 dòng).",
            )
            if up_file is not None:
                st.session_state["ss_uploaded_csv_bytes"] = up_file.getvalue()
                st.success(f"Đang dùng: `{up_file.name}`")
            elif st.session_state.get("ss_uploaded_csv_bytes") is not None:
                if st.button("↩️ Khôi phục dữ liệu gốc", use_container_width=True):
                    st.session_state["ss_uploaded_csv_bytes"] = None
                    st.rerun()

        if st.button("🔄 Đặt lại bộ lọc mặc định", use_container_width=True):
            reset_global_state(df_clean)
            st.rerun()

        iqr_enabled = st.checkbox(
            "Chặn trần ngoại lệ (IQR Winsorize)",
            value=bool(st.session_state["ss_cap_outliers"]),
            key="_w_cap_outliers",
            on_change=_sync_key,
            args=("_w_cap_outliers", "ss_cap_outliers"),
            help=(
                "Giới hạn Sales/Profit vượt 1.5×IQR về đúng ngưỡng Q3+1.5×IQR. "
                "Tắt (mặc định): giữ nguyên ~1.167 đơn Sales lớn — đây là đơn B2B thực tế, "
                "KHÔNG phải lỗi nhập liệu. Bật khi muốn xem xu hướng không bị lệch bởi đơn hàng lớn."
            ),
        )
        if iqr_enabled:
            st.caption(
                "⚠️ IQR Winsorize đang BẬT. ~1.167 đơn B2B lớn bị chặn trần. "
                "Khi bảo vệ, nên tắt để phân tích dữ liệu đầy đủ."
            )

        if include_date:
            saved_dates = st.session_state["ss_date_range"]
            st.date_input(
                "Khoảng ngày đặt hàng",
                value=saved_dates,
                min_value=min_dt,
                max_value=max_dt,
                key="_w_date_range",
                on_change=_sync_key,
                args=("_w_date_range", "ss_date_range"),
            )

        if include_region:
            st.multiselect(
                "Khu vực (Region)",
                options=all_regions,
                default=st.session_state["ss_regions"],
                key="_w_regions",
                on_change=_sync_key,
                args=("_w_regions", "ss_regions"),
            )

        if include_category:
            st.multiselect(
                "Danh mục (Category)",
                options=all_categories,
                default=st.session_state["ss_categories"],
                key="_w_categories",
                on_change=_sync_key,
                args=("_w_categories", "ss_categories"),
            )

        if include_segment:
            st.multiselect(
                "Phân khúc (Segment)",
                options=all_segments,
                default=st.session_state["ss_segments"],
                key="_w_segments",
                on_change=_sync_key,
                args=("_w_segments", "ss_segments"),
            )

        # Tóm tắt state đang hoạt động
        active_dates = get_date_range_timestamps()
        date_text = (
            f"{active_dates[0].strftime('%d/%m/%Y')} → {active_dates[1].strftime('%d/%m/%Y')}"
            if active_dates
            else "Toàn thời gian"
        )
        st.divider()
        st.markdown(
            f"**State đang giữ:**\n"
            f"- 📅 `{date_text}`\n"
            f"- 🗺️ Khu vực: `{len(st.session_state['ss_regions'])}/{len(all_regions)}`\n"
            f"- 📦 Danh mục: `{len(st.session_state['ss_categories'])}/{len(all_categories)}`"
        )

    df_filtered = filter_dataframe(
        df_clean,
        date_range=get_date_range_timestamps() if include_date else None,
        regions=st.session_state["ss_regions"] if include_region else None,
        categories=st.session_state["ss_categories"] if include_category else None,
        segments=st.session_state["ss_segments"] if include_segment else None,
    )
    return df_clean, audit, df_filtered

