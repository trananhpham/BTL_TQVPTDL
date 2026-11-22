"""Module đọc CSV, kiểm tra chất lượng, làm sạch, lọc và cache dữ liệu với @st.cache_data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent


def resolve_data_path(csv_path: str | Path | None = None) -> Path:
    """Tìm đường dẫn tệp Sample - Superstore.csv trong thư mục data/ hoặc thư mục gốc."""
    if csv_path is not None:
        return Path(csv_path)
    in_data = ROOT_DIR / "data" / "Sample - Superstore.csv"
    if in_data.exists():
        return in_data
    return ROOT_DIR / "Sample - Superstore.csv"


def load_raw_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Đọc tệp Sample - Superstore.csv với cơ chế tự động dò bảng mã phù hợp."""
    path = resolve_data_path(csv_path)
    last_err: Exception | None = None
    for encoding in ("utf-8", "windows-1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as err:
            last_err = err
    raise ValueError(f"Không thể đọc tệp {path} với các bảng mã hỗ trợ.") from last_err


def _iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """Tính cận dưới và cận trên theo quy tắc 1.5 * IQR."""
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def audit_raw_data(df: pd.DataFrame) -> dict[str, Any]:
    """Kiểm tra chất lượng dữ liệu thô: thiếu dữ liệu, trùng lặp, ngày hợp lệ và ngoại lệ IQR."""
    missing_by_col = df.isna().sum()
    missing_dict = {col: int(cnt) for col, cnt in missing_by_col.items() if cnt > 0}

    full_duplicates = int(df.duplicated().sum())
    business_cols = [col for col in df.columns if col != "Row ID"]
    business_duplicates = int(df.duplicated(subset=business_cols).sum())

    order_dt = pd.to_datetime(df["Order Date"], format="%m/%d/%Y", errors="coerce")
    ship_dt = pd.to_datetime(df["Ship Date"], format="%m/%d/%Y", errors="coerce")
    invalid_dates = int((order_dt.isna() | ship_dt.isna()).sum())
    invalid_ship_sequence = int((ship_dt < order_dt).sum())

    sales_low, sales_high = _iqr_bounds(pd.to_numeric(df["Sales"], errors="coerce"))
    profit_low, profit_high = _iqr_bounds(pd.to_numeric(df["Profit"], errors="coerce"))

    sales_outliers = int(((df["Sales"] < sales_low) | (df["Sales"] > sales_high)).sum())
    profit_outliers = int(((df["Profit"] < profit_low) | (df["Profit"] > profit_high)).sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(missing_by_col.sum()),
        "missing_by_column": missing_dict,
        "full_duplicates": full_duplicates,
        "business_duplicates": business_duplicates,
        "invalid_dates": invalid_dates,
        "invalid_ship_sequence": invalid_ship_sequence,
        "sales_iqr_bounds": (round(sales_low, 4), round(sales_high, 4)),
        "sales_outliers_count": sales_outliers,
        "profit_iqr_bounds": (round(profit_low, 4), round(profit_high, 4)),
        "profit_outliers_count": profit_outliers,
    }


def clean_superstore_data(
    df: pd.DataFrame,
    cap_outliers: bool = False,
) -> pd.DataFrame:
    """Làm sạch, chuẩn hóa kiểu dữ liệu, xử lý trùng lặp/ngoại lệ và tạo biến mới."""
    cleaned = df.copy()

    # Loại bỏ bản ghi trùng lặp nghiệp vụ (bỏ qua Row ID chỉ mang tính số thứ tự)
    business_cols = [col for col in cleaned.columns if col != "Row ID"]
    cleaned = cleaned.drop_duplicates(subset=business_cols, keep="first").reset_index(drop=True)

    cleaned["Order Date"] = pd.to_datetime(cleaned["Order Date"], format="%m/%d/%Y")
    cleaned["Ship Date"] = pd.to_datetime(cleaned["Ship Date"], format="%m/%d/%Y")
    cleaned["Postal Code"] = cleaned["Postal Code"].astype(str).str.strip().str.zfill(5)

    for num_col in ("Sales", "Discount", "Profit"):
        cleaned[num_col] = pd.to_numeric(cleaned[num_col], errors="coerce").astype("float64")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce").astype("int64")

    obj_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    sales_low, sales_high = _iqr_bounds(cleaned["Sales"])
    profit_low, profit_high = _iqr_bounds(cleaned["Profit"])
    cleaned["Is_Sales_Outlier"] = (cleaned["Sales"] < sales_low) | (cleaned["Sales"] > sales_high)
    cleaned["Is_Profit_Outlier"] = (cleaned["Profit"] < profit_low) | (cleaned["Profit"] > profit_high)

    if cap_outliers:
        cleaned["Sales"] = np.clip(cleaned["Sales"], max(0.0, sales_low), sales_high)
        cleaned["Profit"] = np.clip(cleaned["Profit"], profit_low, profit_high)

    cleaned["Order_Year"] = cleaned["Order Date"].dt.year
    cleaned["Order_Month"] = cleaned["Order Date"].dt.month
    cleaned["Order_Quarter"] = cleaned["Order Date"].dt.quarter
    cleaned["Order_YearMonth"] = cleaned["Order Date"].dt.strftime("%Y-%m")
    cleaned["Order_YearQuarter"] = (
        cleaned["Order_Year"].astype(str) + "-Q" + cleaned["Order_Quarter"].astype(str)
    )
    cleaned["Shipping_Days"] = (cleaned["Ship Date"] - cleaned["Order Date"]).dt.days
    cleaned["Profit_Margin_Pct"] = np.where(
        cleaned["Sales"] != 0,
        (cleaned["Profit"] / cleaned["Sales"]) * 100.0,
        0.0,
    )

    return cleaned


def filter_dataframe(
    df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    regions: list[str] | None = None,
    categories: list[str] | None = None,
    segments: list[str] | None = None,
) -> pd.DataFrame:
    """Lọc tập dữ liệu theo khoảng thời gian, khu vực, danh mục và phân khúc bằng mặt nạ boolean vector hóa."""
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)
    if date_range is not None:
        start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        mask &= (df["Order Date"] >= start_dt) & (df["Order Date"] <= end_dt)
    if regions is not None and len(regions) < 4:
        mask &= df["Region"].isin(regions)
    if categories is not None and len(categories) < 3:
        mask &= df["Category"].isin(categories)
    if segments is not None and len(segments) < 3:
        mask &= df["Segment"].isin(segments)

    if mask.all():
        return df
    return df[mask]


def get_cached_superstore_data(
    csv_path: str | Path | None = None,
    cap_outliers: bool = False,
    uploaded_bytes: bytes | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Tải và làm sạch dữ liệu đúng 1 lần vào RAM nhờ @st.cache_data."""
    resolved = str(resolve_data_path(csv_path))
    try:
        import io
        import streamlit as st

        @st.cache_data(show_spinner="Đang nạp và tối ưu bộ nhớ dữ liệu Superstore...")
        def _cached_loader(
            path_str: str,
            cap: bool,
            raw_bytes: bytes | None = None,
        ) -> tuple[pd.DataFrame, dict[str, Any]]:
            if raw_bytes is not None:
                raw_df = None
                for enc in ("utf-8", "windows-1252", "latin1"):
                    try:
                        raw_df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                if raw_df is None:
                    raise ValueError("Không thể giải mã tệp CSV tải lên.")
            else:
                raw_df = load_raw_data(path_str)
            audit = audit_raw_data(raw_df)
            cleaned_df = clean_superstore_data(raw_df, cap_outliers=cap)
            return cleaned_df, audit

        return _cached_loader(resolved, cap_outliers, uploaded_bytes)
    except ImportError:
        raw_df = load_raw_data(resolved)
        audit = audit_raw_data(raw_df)
        cleaned_df = clean_superstore_data(raw_df, cap_outliers=cap_outliers)
        return cleaned_df, audit


def get_data_dictionary() -> pd.DataFrame:
    """Trả về bảng từ điển dữ liệu 21 cột chuẩn phục vụ Phần 1 (Mô tả dữ liệu)."""
    rows = [
        ("Row ID", "int64", "Đơn hàng", "Số thứ tự định danh dòng bản ghi gốc"),
        ("Order ID", "object (str)", "Đơn hàng", "Mã định danh đơn hàng (5.009 đơn)"),
        ("Order Date", "datetime64[ns]", "Đơn hàng", "Ngày khách đặt hàng (2014–2017)"),
        ("Ship Date", "datetime64[ns]", "Giao vận", "Ngày xuất kho giao hàng"),
        ("Ship Mode", "object (str)", "Giao vận", "Hình thức giao (Standard, Second, First, Same Day)"),
        ("Customer ID", "object (str)", "Khách hàng", "Mã định danh khách hàng (793 mã)"),
        ("Customer Name", "object (str)", "Khách hàng", "Họ và tên khách hàng"),
        ("Segment", "object (str)", "Khách hàng", "Phân khúc (Consumer, Corporate, Home Office)"),
        ("Country", "object (str)", "Địa lý", "Quốc gia (United States)"),
        ("City", "object (str)", "Địa lý", "Thành phố nhận hàng (531 thành phố)"),
        ("State", "object (str)", "Địa lý", "Tiểu bang nhận hàng (49 bang)"),
        ("Postal Code", "object (str 5 ký tự)", "Địa lý", "Mã bưu chính chuẩn hóa 5 chữ số"),
        ("Region", "object (str)", "Địa lý", "Khu vực địa lý (West, East, Central, South)"),
        ("Product ID", "object (str)", "Sản phẩm", "Mã định danh sản phẩm (1.862 mã)"),
        ("Category", "object (str)", "Sản phẩm", "Ngành hàng chính (Technology, Furniture, Office Supplies)"),
        ("Sub-Category", "object (str)", "Sản phẩm", "Nhóm hàng chi tiết (17 nhóm)"),
        ("Product Name", "object (str)", "Sản phẩm", "Tên thương mại đầy đủ của sản phẩm"),
        ("Sales", "float64", "Tài chính", "Doanh thu thuần của dòng giao dịch (USD)"),
        ("Quantity", "int64", "Tài chính", "Số lượng đơn vị sản phẩm bán ra"),
        ("Discount", "float64", "Tài chính", "Tỷ lệ chiết khấu áp dụng (0.00 đến 0.80)"),
        ("Profit", "float64", "Tài chính", "Lợi nhuận ròng thu được (USD; âm là lỗ)"),
    ]
    return pd.DataFrame(rows, columns=["Tên cột (Column)", "Kiểu dữ liệu", "Nhóm nghiệp vụ", "Ý nghĩa & Mô tả chi tiết"])


