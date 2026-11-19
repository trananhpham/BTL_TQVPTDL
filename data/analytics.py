"""Module phân tích thống kê (GroupBy, Pivot Table) và phân tích nâng cao (Forecasting, RFM, Apriori)."""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from utils.formatters import US_STATE_TO_CODE


def summarize_by_time(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Tổng hợp doanh thu, lợi nhuận và số đơn hàng theo mốc thời gian (M, Q, Y)."""
    freq_col_map = {
        "M": "Order_YearMonth",
        "Q": "Order_YearQuarter",
        "Y": "Order_Year",
    }
    if freq not in freq_col_map:
        raise ValueError("freq phải là 'M', 'Q' hoặc 'Y'.")

    time_col = freq_col_map[freq]
    summary = (
        df.groupby(time_col, as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Order_Count=("Order ID", "nunique"),
            Total_Quantity=("Quantity", "sum"),
        )
        .rename(columns={time_col: "Period"})
        .sort_values("Period")
    )
    summary["Profit_Margin_Pct"] = np.where(
        summary["Total_Sales"] != 0,
        (summary["Total_Profit"] / summary["Total_Sales"]) * 100.0,
        0.0,
    )
    return summary


def summarize_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp doanh thu, lợi nhuận và số đơn hàng theo khu vực (Region)."""
    summary = (
        df.groupby("Region", as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Order_Count=("Order ID", "nunique"),
            Total_Quantity=("Quantity", "sum"),
        )
        .sort_values("Total_Sales", ascending=False)
    )
    summary["Profit_Margin_Pct"] = np.where(
        summary["Total_Sales"] != 0,
        (summary["Total_Profit"] / summary["Total_Sales"]) * 100.0,
        0.0,
    )
    return summary


def summarize_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp doanh thu, lợi nhuận theo tiểu bang kèm mã bang 2 ký tự cho Choropleth Map."""
    summary = (
        df.groupby(["State", "Region"], as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Order_Count=("Order ID", "nunique"),
            Total_Quantity=("Quantity", "sum"),
        )
        .sort_values("Total_Sales", ascending=False)
    )
    summary["State_Code"] = summary["State"].map(US_STATE_TO_CODE)
    summary["Profit_Margin_Pct"] = np.where(
        summary["Total_Sales"] != 0,
        (summary["Total_Profit"] / summary["Total_Sales"]) * 100.0,
        0.0,
    )
    return summary


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp doanh thu, lợi nhuận theo Category và Sub-Category."""
    summary = (
        df.groupby(["Category", "Sub-Category"], as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Order_Count=("Order ID", "nunique"),
            Total_Quantity=("Quantity", "sum"),
        )
        .sort_values("Total_Sales", ascending=False)
    )
    summary["Profit_Margin_Pct"] = np.where(
        summary["Total_Sales"] != 0,
        (summary["Total_Profit"] / summary["Total_Sales"]) * 100.0,
        0.0,
    )
    return summary


def summarize_top_products(
    df: pd.DataFrame,
    top_n: int = 10,
    rank_by: str = "Sales",
) -> pd.DataFrame:
    """Lọc Top N sản phẩm theo Doanh thu cao nhất (Sales), Lãi cao nhất (Profit_Max) hoặc Lỗ nặng nhất (Profit_Min)."""
    grouped = (
        df.groupby(["Product ID", "Product Name", "Category", "Sub-Category"], as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
            Order_Count=("Order ID", "nunique"),
            Total_Quantity=("Quantity", "sum"),
        )
    )
    grouped["Profit_Margin_Pct"] = np.where(
        grouped["Total_Sales"] != 0,
        (grouped["Total_Profit"] / grouped["Total_Sales"]) * 100.0,
        0.0,
    )
    if rank_by == "Profit_Max":
        return grouped.sort_values("Total_Profit", ascending=False).head(top_n).reset_index(drop=True)
    if rank_by == "Profit_Min":
        return grouped.sort_values("Total_Profit", ascending=True).head(top_n).reset_index(drop=True)
    return grouped.sort_values("Total_Sales", ascending=False).head(top_n).reset_index(drop=True)


def pivot_region_category(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot Table doanh thu theo Khu vực (hàng) × Danh mục sản phẩm (cột)."""
    return pd.pivot_table(
        df,
        values="Sales",
        index="Region",
        columns="Category",
        aggfunc="sum",
        fill_value=0.0,
        margins=True,
        margins_name="Tổng cộng",
    )


def pivot_year_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot Table doanh thu và số đơn hàng theo Năm × Phân khúc."""
    return pd.pivot_table(
        df,
        values=["Sales", "Order ID"],
        index="Order_Year",
        columns="Segment",
        aggfunc={"Sales": "sum", "Order ID": "nunique"},
        fill_value=0.0,
    )


def forecast_monthly_sales(
    df: pd.DataFrame,
    periods: int = 6,
    method: str = "Prophet (Meta)",
    allow_fallback: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Dự báo doanh thu theo tháng cho 3–6 tháng tiếp theo bằng Prophet hoặc ARIMA.

    allow_fallback=True giữ đường dự báo trên dashboard khi thư viện ưu tiên không cài được.
    allow_fallback=False dùng khi chấm sai số từng mô hình riêng trên tập hold-out.
    """
    monthly = (
        df.set_index("Order Date")["Sales"]
        .resample("MS")
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "ds", "Sales": "y"})
    )
    if len(monthly) < 6:
        raise ValueError("Cần tối thiểu 6 tháng dữ liệu lịch sử để thực hiện dự báo.")

    last_date = monthly["ds"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")

    hist_df = monthly.rename(columns={"y": "yhat"}).copy()
    hist_df["yhat_lower"] = hist_df["yhat"]
    hist_df["yhat_upper"] = hist_df["yhat"]
    hist_df["Loại dữ liệu"] = "Lịch sử thực tế"

    seasonal_only = "Seasonal" in method
    prophet_error: Exception | None = None
    arima_error: Exception | None = None

    # 1. Thử Prophet nếu người dùng chọn mô hình này
    if not seasonal_only and "Prophet" in method:
        try:
            from prophet import Prophet  # type: ignore

            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
            model.fit(monthly)
            future = model.make_future_dataframe(periods=periods, freq="MS")
            pred = model.predict(future).tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            pred["yhat"] = pred["yhat"].clip(lower=0.0)
            pred["yhat_lower"] = pred["yhat_lower"].clip(lower=0.0)
            pred["Loại dữ liệu"] = "Dự báo (Prophet)"
            return pd.concat([hist_df, pred], ignore_index=True), "Prophet (Meta)"
        except Exception as err:
            prophet_error = err
            if not allow_fallback:
                raise RuntimeError(f"Prophet không chạy được: {err}") from err

    # 2. Thử ARIMA / SARIMAX (statsmodels) khi được chọn, hoặc khi Prophet lỗi và còn được phép đổi mô hình
    if not seasonal_only:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

            seasonal_order = (1, 1, 0, 12) if len(monthly) >= 24 else (0, 0, 0, 0)
            model = SARIMAX(
                monthly["y"],
                order=(1, 1, 0),
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            try:
                fit_res = model.fit(method="powell", disp=False)
            except Exception:
                fit_res = model.fit(method="bfgs", disp=False)
            fc = fit_res.get_forecast(steps=periods)
            ci = fc.conf_int(alpha=0.05)
            pred = pd.DataFrame(
                {
                    "ds": future_dates,
                    "yhat": np.clip(fc.predicted_mean.to_numpy(), 0.0, None),
                    "yhat_lower": np.clip(ci.iloc[:, 0].to_numpy(), 0.0, None),
                    "yhat_upper": np.clip(ci.iloc[:, 1].to_numpy(), 0.0, None),
                    "Loại dữ liệu": "Dự báo (ARIMA/SARIMAX)",
                }
            )
            return pd.concat([hist_df, pred], ignore_index=True), "ARIMA / SARIMAX (statsmodels)"
        except Exception as err:
            arima_error = err
            if not allow_fallback:
                raise RuntimeError(f"ARIMA không chạy được: {err}") from err

    if not allow_fallback and (prophet_error or arima_error) and not seasonal_only:
        raise RuntimeError(prophet_error or arima_error)

    # 3. Xu hướng tuyến tính + mùa vụ tháng (không phụ thuộc Prophet)
    y = monthly["y"].to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    trend = intercept + slope * x
    residuals = y - trend

    monthly["month_num"] = monthly["ds"].dt.month
    seasonal_map = (
        pd.DataFrame({"m": monthly["month_num"], "res": residuals})
        .groupby("m")["res"]
        .mean()
        .to_dict()
    )
    resid_std = float(np.std(residuals))

    future_x = np.arange(len(y), len(y) + periods, dtype=float)
    future_yhat = []
    for idx_step, dt_val in enumerate(future_dates):
        val = (intercept + slope * future_x[idx_step]) + seasonal_map.get(dt_val.month, 0.0)
        future_yhat.append(max(0.0, float(val)))

    future_yhat_arr = np.array(future_yhat)
    pred = pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": future_yhat_arr,
            "yhat_lower": np.clip(future_yhat_arr - 1.64 * resid_std, 0.0, None),
            "yhat_upper": future_yhat_arr + 1.64 * resid_std,
            "Loại dữ liệu": "Dự báo (Seasonal Trend)",
        }
    )
    return pd.concat([hist_df, pred], ignore_index=True), "Seasonal Linear Trend"


def _score_quintiles(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Chia điểm 1-5 ổn định ngay cả khi phân phối có nhiều giá trị trùng nhau."""
    ranked = series.rank(method="first", ascending=True)
    labels = [1, 2, 3, 4, 5] if ascending else [5, 4, 3, 2, 1]
    bins = pd.qcut(ranked, q=min(5, len(series)), labels=False, duplicates="drop")
    max_bin = int(bins.max()) if len(bins) else 0
    if max_bin == 4:
        return bins.map(lambda b: labels[int(b)]).astype(int)
    # Chuẩn hóa về thang 1..5 nếu số lượng khách hàng quá ít
    scaled = np.ceil(((bins + 1) / (max_bin + 1)) * 5).astype(int)
    return scaled if ascending else (6 - scaled)


def compute_rfm_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Tính chỉ số RFM (Recency, Frequency, Monetary) và phân nhóm khách hàng tự động."""
    snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby(["Customer ID", "Customer Name", "Segment"], as_index=False)
        .agg(
            Last_Order_Date=("Order Date", "max"),
            Frequency=("Order ID", "nunique"),
            Monetary=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
        )
    )
    rfm["Recency"] = (snapshot_date - rfm["Last_Order_Date"]).dt.days

    if len(rfm) < 5:
        rfm["R_Score"] = 3
        rfm["F_Score"] = 3
        rfm["M_Score"] = 3
    else:
        rfm["R_Score"] = _score_quintiles(rfm["Recency"], ascending=False)
        rfm["F_Score"] = _score_quintiles(rfm["Frequency"], ascending=True)
        rfm["M_Score"] = _score_quintiles(rfm["Monetary"], ascending=True)

    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    def _assign_label(row: pd.Series) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "VIP"
        if f >= 4 and r >= 3:
            return "Khách hàng trung thành"
        if r >= 4 and f <= 2:
            return "Khách hàng mới"
        if r <= 2 and (f >= 3 or m >= 3):
            return "Nguy cơ rời bỏ"
        if r <= 2 and f <= 2:
            return "Ngủ đông / Cần kích hoạt"
        return "Khách hàng tiềm năng"

    rfm["RFM_Segment"] = rfm.apply(_assign_label, axis=1)
    return rfm.sort_values("Monetary", ascending=False).reset_index(drop=True)


def run_market_basket_apriori(
    df: pd.DataFrame,
    item_col: str = "Sub-Category",
    min_support: float = 0.015,
    min_confidence: float = 0.2,
) -> pd.DataFrame:
    """Phân tích giỏ hàng (Market Basket Analysis) tìm luật kết hợp bán chéo (Apriori)."""
    baskets = df.groupby("Order ID")[item_col].apply(lambda s: sorted(set(s.astype(str)))).tolist()
    total_orders = len(baskets)
    if total_orders == 0:
        return pd.DataFrame()

    # Thử dùng mlxtend trước nếu môi trường có cài đặt
    try:
        from mlxtend.frequent_patterns import apriori, association_rules  # type: ignore
        from mlxtend.preprocessing import TransactionEncoder  # type: ignore

        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_)
        freq_items = apriori(basket_df, min_support=min_support, use_colnames=True)
        if not freq_items.empty:
            rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
            if not rules.empty:
                rules = rules.sort_values(["lift", "confidence"], ascending=False).copy()
                rules["Antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
                rules["Consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
                rules["Pair_Orders"] = (rules["support"] * total_orders).round().astype(int)
                return rules[
                    ["Antecedents", "Consequents", "Pair_Orders", "support", "confidence", "lift"]
                ].rename(
                    columns={
                        "support": "Support",
                        "confidence": "Confidence",
                        "lift": "Lift",
                    }
                ).reset_index(drop=True)
    except Exception:
        pass

    # Thuật toán Apriori thuần Python (tìm cặp 2 sản phẩm thường mua cùng nhau)
    single_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    for items in baskets:
        for item in items:
            single_counts[item] += 1
        if len(items) >= 2:
            for a, b in combinations(items, 2):
                pair_counts[(a, b)] += 1

    rows = []
    for (a, b), pair_cnt in pair_counts.items():
        support = pair_cnt / total_orders
        if support < min_support:
            continue
        sup_a = single_counts[a] / total_orders
        sup_b = single_counts[b] / total_orders

        conf_a_b = pair_cnt / single_counts[a]
        lift_a_b = conf_a_b / sup_b if sup_b else 0.0
        if conf_a_b >= min_confidence:
            rows.append(
                {
                    "Antecedents": a,
                    "Consequents": b,
                    "Pair_Orders": pair_cnt,
                    "Support": round(support, 4),
                    "Confidence": round(conf_a_b, 4),
                    "Lift": round(lift_a_b, 4),
                }
            )

        conf_b_a = pair_cnt / single_counts[b]
        lift_b_a = conf_b_a / sup_a if sup_a else 0.0
        if conf_b_a >= min_confidence:
            rows.append(
                {
                    "Antecedents": b,
                    "Consequents": a,
                    "Pair_Orders": pair_cnt,
                    "Support": round(support, 4),
                    "Confidence": round(conf_b_a, 4),
                    "Lift": round(lift_b_a, 4),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)


def compute_apriori_lift_matrix(df: pd.DataFrame, item_col: str = "Sub-Category") -> pd.DataFrame:
    """Tính ma trận hệ số Lift hai chiều giữa các nhóm sản phẩm thường mua chung trong cùng đơn hàng."""
    rules = run_market_basket_apriori(df, item_col=item_col, min_support=0.002, min_confidence=0.05)
    if rules.empty:
        return pd.DataFrame()
    # Chỉ giữ các cặp đơn (1 nhóm -> 1 nhóm) để vẽ ma trận nhiệt trực quan
    simple = rules[
        (~rules["Antecedents"].str.contains(",")) & (~rules["Consequents"].str.contains(","))
    ].copy()
    if simple.empty:
        return pd.DataFrame()
    mat = simple.pivot_table(
        index="Antecedents",
        columns="Consequents",
        values="Lift",
        aggfunc="max",
        fill_value=0.0,
    )
    return mat.round(2)


def compute_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """Tính ma trận tỷ lệ giữ chân khách hàng theo năm gia nhập (Cohort Retention %)."""
    if df.empty:
        return pd.DataFrame()

    cohort_df = df[["Customer ID", "Order_Year"]].drop_duplicates().copy()
    cohort_df["Cohort_Year"] = cohort_df.groupby("Customer ID")["Order_Year"].transform("min")
    cohort_df["Cohort_Index"] = cohort_df["Order_Year"] - cohort_df["Cohort_Year"]

    counts = (
        cohort_df.groupby(["Cohort_Year", "Cohort_Index"])["Customer ID"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    if 0 not in counts.columns:
        return pd.DataFrame()

    cohort_sizes = counts[0].replace(0, np.nan)
    retention = counts.div(cohort_sizes, axis=0) * 100.0
    retention.columns = [f"Năm +{int(c)}" if c > 0 else "Năm 0 (Gốc)" for c in retention.columns]
    retention.index = [f"Cohort {int(y)}" for y in retention.index]
    return retention.round(1)


def compute_seasonality_matrix(df: pd.DataFrame, metric: str = "Sales") -> pd.DataFrame:
    """Tạo ma trận mùa vụ Năm × Tháng cho Heatmap phân tích chu kỳ."""
    if df.empty:
        return pd.DataFrame()
    pivot = pd.pivot_table(
        df,
        values=metric,
        index="Order_Year",
        columns="Order_Month",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot.columns = [f"Tháng {int(m)}" for m in pivot.columns]
    return pivot


def compute_pareto_abc(df: pd.DataFrame, entity_col: str = "Product Name") -> pd.DataFrame:
    """Phân tích Pareto 80/20 (ABC Classification) theo doanh thu cho sản phẩm hoặc khách hàng."""
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(entity_col, as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Order_Count=("Order ID", "nunique"),
        )
        .sort_values("Total_Sales", ascending=False)
        .reset_index(drop=True)
    )
    total_sales = float(grouped["Total_Sales"].sum())
    if total_sales <= 0:
        return pd.DataFrame()

    grouped["Sales_Share_Pct"] = (grouped["Total_Sales"] / total_sales) * 100.0
    grouped["Cumulative_Sales_Pct"] = grouped["Sales_Share_Pct"].cumsum()
    grouped["Rank"] = np.arange(1, len(grouped) + 1)

    def _abc_class(cum_pct: float) -> str:
        if cum_pct <= 80.0:
            return "Nhóm A (Top 80% doanh thu)"
        if cum_pct <= 95.0:
            return "Nhóm B (80% – 95%)"
        return "Nhóm C (5% cuối)"

    grouped["ABC_Class"] = grouped["Cumulative_Sales_Pct"].apply(_abc_class)
    return grouped


def simulate_discount_what_if(
    df: pd.DataFrame,
    max_discount_cap: float = 0.20,
    loss_subcat_uplift_pct: float = 0.0,
    price_elasticity: float = 0.0,
) -> dict[str, float | int]:
    """Mô phỏng kịch bản What-If: đặt trần chiết khấu tối đa, điều chỉnh giá nhóm lỗ và độ co giãn cầu theo giá (Price Elasticity)."""
    if df.empty:
        return {
            "actual_sales": 0.0,
            "sim_sales": 0.0,
            "actual_profit": 0.0,
            "sim_profit": 0.0,
            "profit_gain": 0.0,
            "actual_loss_rows": 0,
            "sim_loss_rows": 0,
            "actual_margin_pct": 0.0,
            "sim_margin_pct": 0.0,
            "volume_retention_pct": 100.0,
        }

    sim = df[["Sales", "Discount", "Profit", "Sub-Category"]].copy()
    # Suy ra giá vốn giao dịch và giá niêm yết gốc trước chiết khấu
    implied_cost = sim["Sales"] - sim["Profit"]
    safe_disc = np.clip(sim["Discount"], 0.0, 0.99)
    gross_price = sim["Sales"] / (1.0 - safe_disc)

    new_disc = np.minimum(safe_disc, max_discount_cap)
    uplift_mask = sim["Sub-Category"].isin(["Tables", "Bookcases", "Supplies"])
    uplift_mult = np.where(uplift_mask, 1.0 + (loss_subcat_uplift_pct / 100.0), 1.0)

    # Tỷ lệ thay đổi giá bán thực tế: %ΔP = (P_new - P_old) / P_old
    rel_price_change = ((1.0 - new_disc) * uplift_mult - (1.0 - safe_disc)) / (1.0 - safe_disc)
    if price_elasticity != 0.0:
        q_mult = np.clip(1.0 + price_elasticity * rel_price_change, 0.10, 2.00)
    else:
        q_mult = np.ones(len(sim), dtype=float)

    sim_sales = gross_price * (1.0 - new_disc) * uplift_mult * q_mult
    sim_cost = implied_cost * q_mult
    sim_profit = sim_sales - sim_cost

    actual_sales = float(sim["Sales"].sum())
    actual_profit = float(sim["Profit"].sum())
    total_sim_sales = float(sim_sales.sum())
    total_sim_profit = float(sim_profit.sum())

    return {
        "actual_sales": actual_sales,
        "sim_sales": total_sim_sales,
        "actual_profit": actual_profit,
        "sim_profit": total_sim_profit,
        "profit_gain": total_sim_profit - actual_profit,
        "actual_loss_rows": int((sim["Profit"] < 0).sum()),
        "sim_loss_rows": int((sim_profit < 0).sum()),
        "actual_margin_pct": (actual_profit / actual_sales * 100.0) if actual_sales else 0.0,
        "sim_margin_pct": (total_sim_profit / total_sim_sales * 100.0) if total_sim_sales else 0.0,
        "volume_retention_pct": float(np.mean(q_mult) * 100.0),
    }


def detect_loss_anomalies(df: pd.DataFrame, profit_threshold: float = -100.0) -> pd.DataFrame:
    """Phát hiện các dòng giao dịch thua lỗ nghiêm trọng kèm phân loại nguyên nhân gốc rễ."""
    anomalies = df[df["Profit"] <= profit_threshold].copy()
    if anomalies.empty:
        return anomalies

    def _root_cause(row: pd.Series) -> str:
        high_disc = row["Discount"] > 0.20
        loss_subcat = row["Sub-Category"] in ("Tables", "Bookcases", "Supplies")
        if high_disc and loss_subcat:
            return "Chiết khấu >20% & Nhóm hàng lỗ"
        if high_disc:
            return "Chiết khấu sâu (>20%)"
        if loss_subcat:
            return "Nhóm hàng biên âm (Tables/Bookcases/Supplies)"
        return "Giá vốn/Chi phí vận chuyển cao"

    anomalies["Nguyên_Nhân_Gợi_Ý"] = anomalies.apply(_root_cause, axis=1)
    cols = [
        "Order ID",
        "Order Date",
        "State",
        "Region",
        "Category",
        "Sub-Category",
        "Product Name",
        "Sales",
        "Discount",
        "Profit",
        "Nguyên_Nhân_Gợi_Ý",
    ]
    return anomalies[cols].sort_values("Profit", ascending=True).reset_index(drop=True)


def compute_kmeans_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Phân cụm khách hàng không giám sát (K-Means) trên không gian RFM chuẩn hóa."""
    rfm = compute_rfm_segments(df)
    if len(rfm) < max(3, n_clusters):
        return rfm.assign(Cluster_Label="Cluster 0")

    scaled = _scaled_rfm_features(rfm)

    labels: np.ndarray
    try:
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
    except Exception:
        # Thuật toán K-Means thuần NumPy dự phòng khi chưa cài scikit-learn
        rng = np.random.default_rng(42)
        init_idx = rng.choice(len(scaled), size=n_clusters, replace=False)
        centroids = scaled[init_idx].copy()
        labels = np.zeros(len(scaled), dtype=int)
        for _ in range(50):
            dists = np.linalg.norm(scaled[:, None, :] - centroids[None, :, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c_id in range(n_clusters):
                mask = labels == c_id
                if mask.any():
                    centroids[c_id] = scaled[mask].mean(axis=0)

    rfm = rfm.copy()
    rfm["Cluster_ID"] = labels
    # Sắp xếp nhãn cụm theo Monetary trung bình giảm dần để đặt tên nhất quán
    order = (
        rfm.groupby("Cluster_ID")["Monetary"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    rank_map = {cid: f"Cụm {idx + 1}" for idx, cid in enumerate(order)}
    rfm["Cluster_Label"] = rfm["Cluster_ID"].map(rank_map)
    return rfm


def _scaled_rfm_features(rfm: pd.DataFrame) -> np.ndarray:
    """Chuẩn hóa Z-score không gian RFM (Recency, log-Frequency, log-Monetary)."""
    features = np.column_stack(
        [
            rfm["Recency"].to_numpy(dtype=float),
            np.log1p(rfm["Frequency"].to_numpy(dtype=float)),
            np.log1p(np.clip(rfm["Monetary"].to_numpy(dtype=float), 0.0, None)),
        ]
    )
    means = features.mean(axis=0)
    stds = np.where(features.std(axis=0) == 0, 1.0, features.std(axis=0))
    return (features - means) / stds


def score_kmeans_range(df: pd.DataFrame, k_min: int = 2, k_max: int = 6) -> pd.DataFrame:
    """Tính Inertia và Silhouette cho từng k để chọn số cụm, không chọn k bằng cảm tính."""
    rfm = compute_rfm_segments(df)
    if len(rfm) < k_min + 1:
        return pd.DataFrame(columns=["k", "Inertia", "Silhouette"])

    scaled = _scaled_rfm_features(rfm)
    upper = min(k_max, len(rfm) - 1)
    rows: list[dict[str, float | int]] = []
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception:
        return pd.DataFrame(columns=["k", "Inertia", "Silhouette"])

    for k in range(k_min, upper + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        silhouette = float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 else float("nan")
        rows.append({"k": k, "Inertia": float(model.inertia_), "Silhouette": round(silhouette, 4)})
    return pd.DataFrame(rows)


def evaluate_forecast_holdout(df: pd.DataFrame, holdout_months: int = 6) -> pd.DataFrame:
    """Chấm MAE và MAPE bằng cách giữ lại các tháng cuối, rồi dự báo lại đúng các tháng đó."""
    if df.empty or holdout_months < 1:
        return pd.DataFrame()

    monthly = (
        df.set_index("Order Date")["Sales"]
        .resample("MS")
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "ds", "Sales": "y"})
        .dropna(subset=["y"])
    )
    if len(monthly) < holdout_months + 6:
        return pd.DataFrame()

    cutoff = monthly["ds"].iloc[-holdout_months]
    actual = monthly.tail(holdout_months)[["ds", "y"]].copy()
    train_df = df[df["Order Date"] < cutoff].copy()
    methods = [
        "ARIMA / SARIMAX (statsmodels)",
        "Seasonal Linear Trend",
        "Prophet (Meta)",
    ]
    rows: list[dict[str, object]] = []
    for method in methods:
        try:
            forecast_df, used_name = forecast_monthly_sales(
                train_df,
                periods=holdout_months,
                method=method,
                allow_fallback=False,
            )
        except Exception as err:
            rows.append(
                {
                    "Mô hình": method,
                    "Trạng thái": f"Không chạy ({type(err).__name__})",
                    "MAE (USD)": np.nan,
                    "MAPE (%)": np.nan,
                    "Số tháng hold-out": holdout_months,
                }
            )
            continue

        pred = forecast_df.loc[
            forecast_df["Loại dữ liệu"] != "Lịch sử thực tế",
            ["ds", "yhat"],
        ]
        merged = actual.merge(pred, on="ds", how="inner")
        if merged.empty:
            continue
        abs_err = (merged["y"] - merged["yhat"]).abs()
        denom = merged["y"].replace(0, np.nan)
        rows.append(
            {
                "Mô hình": used_name,
                "Trạng thái": "Đã đánh giá",
                "MAE (USD)": round(float(abs_err.mean()), 2),
                "MAPE (%)": round(float((abs_err / denom).mean() * 100.0), 2),
                "Số tháng hold-out": holdout_months,
            }
        )
    return pd.DataFrame(rows)


def summarize_discount_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Tổng hợp lợi nhuận và tỷ lệ lỗ theo từng ngưỡng chiết khấu."""
    if df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["Nguong_chiet_khau"] = pd.cut(
        work["Discount"],
        bins=[-0.001, 0.0, 0.10, 0.20, 0.30, 1.01],
        labels=["0%", ">0–10%", ">10–20%", ">20–30%", ">30%"],
        include_lowest=True,
    )
    summary = (
        work.groupby("Nguong_chiet_khau", observed=False)
        .agg(
            So_dong=("Profit", "size"),
            Doanh_thu=("Sales", "sum"),
            Loi_nhuan=("Profit", "sum"),
            Loi_nhuan_TB=("Profit", "mean"),
            Ty_le_lo=("Profit", lambda s: float((s < 0).mean() * 100.0)),
        )
        .reset_index()
    )
    summary["Bien_LN_pct"] = np.where(
        summary["Doanh_thu"] != 0,
        summary["Loi_nhuan"] / summary["Doanh_thu"] * 100.0,
        0.0,
    )
    return summary


def summarize_state_performance(df: pd.DataFrame, region: str | None = None) -> pd.DataFrame:
    """Tổng hợp doanh thu, lợi nhuận, chiết khấu và tỷ lệ lỗ theo tiểu bang."""
    scoped = df if region is None else df[df["Region"] == region]
    if scoped.empty:
        return pd.DataFrame()

    summary = (
        scoped.groupby(["State", "Region"], as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
            Order_Count=("Order ID", "nunique"),
            Loss_Rate_Pct=("Profit", lambda s: float((s < 0).mean() * 100.0)),
        )
        .sort_values("Total_Profit", ascending=True)
    )
    summary["Profit_Margin_Pct"] = np.where(
        summary["Total_Sales"] != 0,
        summary["Total_Profit"] / summary["Total_Sales"] * 100.0,
        0.0,
    )
    return summary.reset_index(drop=True)


def summarize_state_subcategory(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Khoan sâu lợi nhuận và chiết khấu theo nhóm hàng của một tiểu bang."""
    scoped = df[df["State"] == state]
    if scoped.empty:
        return pd.DataFrame()

    summary = (
        scoped.groupby(["Category", "Sub-Category"], as_index=False)
        .agg(
            Total_Sales=("Sales", "sum"),
            Total_Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
            Order_Count=("Order ID", "nunique"),
        )
        .sort_values("Total_Profit", ascending=True)
    )
    summary["Profit_Margin_Pct"] = np.where(
        summary["Total_Sales"] != 0,
        summary["Total_Profit"] / summary["Total_Sales"] * 100.0,
        0.0,
    )
    return summary.reset_index(drop=True)


def compute_yoy_waterfall(df: pd.DataFrame, metric: str = "Sales") -> pd.DataFrame:
    """Tính biến động tăng/giảm tuyệt đối qua từng năm (YoY Delta) cho biểu đồ Waterfall."""
    if df.empty or metric not in ("Sales", "Profit"):
        return pd.DataFrame()

    yearly = (
        df.groupby("Order_Year", as_index=False)[metric]
        .sum()
        .sort_values("Order_Year")
        .reset_index(drop=True)
    )
    if yearly.empty:
        return pd.DataFrame()

    rows = []
    for idx, row in yearly.iterrows():
        yr = int(row["Order_Year"])
        val = float(row[metric])
        if idx == 0:
            rows.append({"Label": f"Năm {yr} (Gốc)", "Value": val, "Measure": "absolute", "Total": val})
        else:
            prev_val = float(yearly.loc[idx - 1, metric])
            delta = val - prev_val
            rows.append({"Label": f"Thay đổi {yr}", "Value": delta, "Measure": "relative", "Total": val})

    if len(yearly) > 1:
        last_yr = int(yearly.iloc[-1]["Order_Year"])
        last_val = float(yearly.iloc[-1][metric])
        rows.append({"Label": f"Tổng năm {last_yr}", "Value": last_val, "Measure": "total", "Total": last_val})

    return pd.DataFrame(rows)


def generate_dynamic_insight(df: pd.DataFrame) -> dict[str, object]:
    """Tạo tóm tắt Insight động theo thời gian thực dựa trên DataFrame đang lọc."""
    if df.empty:
        return {}

    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    margin_pct = (total_profit / total_sales * 100.0) if total_sales else 0.0
    total_orders = int(df["Order ID"].nunique())
    loss_rows = int((df["Profit"] < 0).sum())
    loss_rate_pct = (loss_rows / len(df) * 100.0) if len(df) else 0.0

    high_disc = df[df["Discount"] > 0.20]
    high_disc_loss_rate = (
        float((high_disc["Profit"] < 0).mean() * 100.0) if not high_disc.empty else 0.0
    )

    cat_profit = df.groupby("Category")["Profit"].sum()
    best_cat = str(cat_profit.idxmax()) if not cat_profit.empty else "N/A"
    best_cat_profit = float(cat_profit.max()) if not cat_profit.empty else 0.0

    sub_profit = df.groupby("Sub-Category")["Profit"].sum()
    worst_sub = str(sub_profit.idxmin()) if not sub_profit.empty else "N/A"
    worst_sub_profit = float(sub_profit.min()) if not sub_profit.empty else 0.0

    state_profit = df.groupby("State")["Profit"].sum()
    worst_state = str(state_profit.idxmin()) if not state_profit.empty else "N/A"
    worst_state_profit = float(state_profit.min()) if not state_profit.empty else 0.0

    # Tính tăng trưởng YoY của năm gần nhất so với năm liền trước
    yearly_sales = df.groupby("Order_Year")["Sales"].sum().sort_index()
    yoy_sales_pct = 0.0
    latest_year = int(yearly_sales.index[-1]) if not yearly_sales.empty else 0
    if len(yearly_sales) >= 2:
        prev_val = float(yearly_sales.iloc[-2])
        curr_val = float(yearly_sales.iloc[-1])
        if prev_val > 0:
            yoy_sales_pct = (curr_val - prev_val) / prev_val * 100.0

    return {
        "total_sales": total_sales,
        "total_profit": total_profit,
        "margin_pct": margin_pct,
        "total_orders": total_orders,
        "loss_rows": loss_rows,
        "loss_rate_pct": loss_rate_pct,
        "high_disc_loss_rate": high_disc_loss_rate,
        "best_cat": best_cat,
        "best_cat_profit": best_cat_profit,
        "worst_sub": worst_sub,
        "worst_sub_profit": worst_sub_profit,
        "worst_state": worst_state,
        "worst_state_profit": worst_state_profit,
        "latest_year": latest_year,
        "yoy_sales_pct": yoy_sales_pct,
    }


def compare_groups_ab(
    df: pd.DataFrame,
    dimension: str,
    group_a: str | int,
    group_b: str | int,
) -> pd.DataFrame:
    """So sánh đối chứng (A/B Benchmark) giữa 2 nhóm theo chiều Region, Order_Year, Segment hoặc Category."""
    if df.empty or dimension not in df.columns:
        return pd.DataFrame()

    def _metrics(sub: pd.DataFrame) -> dict[str, float]:
        sales = float(sub["Sales"].sum())
        profit = float(sub["Profit"].sum())
        orders = int(sub["Order ID"].nunique())
        customers = int(sub["Customer ID"].nunique())
        margin = (profit / sales * 100.0) if sales else 0.0
        aov = (sales / orders) if orders else 0.0
        avg_disc = float(sub["Discount"].mean() * 100.0) if len(sub) else 0.0
        loss_rate = float((sub["Profit"] < 0).mean() * 100.0) if len(sub) else 0.0
        return {
            "Tổng doanh thu ($)": sales,
            "Tổng lợi nhuận ($)": profit,
            "Biên lợi nhuận (%)": margin,
            "Số đơn hàng": float(orders),
            "Số khách hàng": float(customers),
            "Giá trị đơn TB - AOV ($)": aov,
            "Chiết khấu TB (%)": avg_disc,
            "Tỷ lệ giao dịch lỗ (%)": loss_rate,
        }

    m_a = _metrics(df[df[dimension].astype(str) == str(group_a)])
    m_b = _metrics(df[df[dimension].astype(str) == str(group_b)])

    rows = []
    for k in m_a:
        val_a = m_a[k]
        val_b = m_b[k]
        diff = val_a - val_b
        pct_diff = ((val_a - val_b) / abs(val_b) * 100.0) if val_b != 0 else 0.0
        rows.append(
            {
                "Chỉ số": k,
                f"{group_a}": round(val_a, 2),
                f"{group_b}": round(val_b, 2),
                "Chênh lệch (A - B)": round(diff, 2),
                "Tương đối (%)": round(pct_diff, 2),
            }
        )
    return pd.DataFrame(rows)


def build_executive_html_report(df: pd.DataFrame) -> str:
    """Sinh báo cáo HTML chuyên nghiệp chuẩn in PDF (Ctrl+P) cho Ban Giám đốc."""
    ins = generate_dynamic_insight(df)
    if not ins:
        return "<html><body><h2>Không có dữ liệu để xuất báo cáo.</h2></body></html>"

    reg_df = summarize_by_region(df).round(2)
    disc_df = summarize_discount_bands(df).round(2)
    date_min = df["Order Date"].min().strftime("%d/%m/%Y")
    date_max = df["Order Date"].max().strftime("%d/%m/%Y")

    reg_rows = "".join(
        f"<tr><td>{r['Region']}</td><td>${r['Total_Sales']:,.2f}</td>"
        f"<td>${r['Total_Profit']:,.2f}</td><td>{r['Order_Count']:,}</td>"
        f"<td>{r['Profit_Margin_Pct']:.2f}%</td></tr>"
        for _, r in reg_df.iterrows()
    )
    disc_rows = "".join(
        f"<tr><td>{r['Nguong_chiet_khau']}</td><td>{int(r['So_dong']):,}</td>"
        f"<td>${r['Doanh_thu']:,.2f}</td><td>${r['Loi_nhuan']:,.2f}</td>"
        f"<td>${r['Loi_nhuan_TB']:,.2f}</td><td>{r['Ty_le_lo']:.1f}%</td></tr>"
        for _, r in disc_df.iterrows()
    )

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Báo Cáo Điều Hành Superstore BI</title>
<style>
  body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; color: #0f172a; margin: 32px; line-height: 1.5; }}
  h1 {{ color: #1e3a8a; border-bottom: 3px solid #2563eb; padding-bottom: 8px; margin-bottom: 4px; }}
  h2 {{ color: #1e40af; margin-top: 24px; border-left: 4px solid #2563eb; padding-left: 10px; }}
  .meta {{ color: #475569; font-size: 14px; margin-bottom: 20px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 18px 0; }}
  .kpi-card {{ background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; }}
  .kpi-title {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }}
  .kpi-val {{ font-size: 22px; font-weight: 700; color: #0f172a; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
  th {{ background: #1e3a8a; color: white; text-align: left; padding: 9px 12px; }}
  td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 12px; }}
  tr:nth-child(even) {{ background: #f8fafc; }}
  .callout {{ background: #eff6ff; border-left: 4px solid #1d4ed8; padding: 12px 16px; margin-top: 18px; border-radius: 6px; }}
  @media print {{ body {{ margin: 12mm; }} }}
</style>
</head>
<body>
  <h1>📊 BÁO CÁO ĐIỀU HÀNH PHÂN TÍCH DỮ LIỆU SUPERSTORE</h1>
  <div class="meta">Phạm vi thời gian: <b>{date_min} → {date_max}</b> | Số dòng giao dịch: <b>{len(df):,}</b> | Xuất từ Hệ thống Superstore BI Dashboard</div>

  <div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Tổng Doanh Thu</div><div class="kpi-val">${ins['total_sales']:,.2f}</div></div>
    <div class="kpi-card"><div class="kpi-title">Tổng Lợi Nhuận</div><div class="kpi-val">${ins['total_profit']:,.2f}</div></div>
    <div class="kpi-card"><div class="kpi-title">Biên Lợi Nhuận</div><div class="kpi-val">{ins['margin_pct']:.2f}%</div></div>
    <div class="kpi-card"><div class="kpi-title">Tổng Đơn Hàng</div><div class="kpi-val">{ins['total_orders']:,}</div></div>
  </div>

  <h2>1. Hiệu suất theo Khu vực địa lý (Region)</h2>
  <table>
    <thead><tr><th>Khu vực</th><th>Doanh thu ($)</th><th>Lợi nhuận ($)</th><th>Số đơn hàng</th><th>Biên LN (%)</th></tr></thead>
    <tbody>{reg_rows}</tbody>
  </table>

  <h2>2. Phân tích Tác động Chiết khấu đến Lợi nhuận</h2>
  <table>
    <thead><tr><th>Ngưỡng chiết khấu</th><th>Số dòng</th><th>Doanh thu ($)</th><th>Lợi nhuận ($)</th><th>LN Trung bình ($)</th><th>Tỷ lệ lỗ (%)</th></tr></thead>
    <tbody>{disc_rows}</tbody>
  </table>

  <div class="callout">
    <b>💡 Insight & Khuyến nghị Chiến lược:</b><br>
    • Ngành hàng sinh lời cao nhất: <b>{ins['best_cat']}</b> (${ins['best_cat_profit']:,.2f}).<br>
    • Nhóm sản phẩm gây lỗ nặng nhất: <b>{ins['worst_sub']}</b> (${ins['worst_sub_profit']:,.2f}); Tiểu bang lỗ nặng nhất: <b>{ins['worst_state']}</b> (${ins['worst_state_profit']:,.2f}).<br>
    • Tỷ lệ lỗ ở các đơn chiết khấu &gt; 20% lên tới <b>{ins['high_disc_loss_rate']:.1f}%</b>. Đề xuất áp dụng trần chiết khấu tối đa 20% và tái cấu trúc giá nhóm Tables/Bookcases/Binders.
  </div>
</body>
</html>"""


def compute_elasticity_curve(
    df: pd.DataFrame,
    uplift_pct: float = 5.0,
    elasticity: float = -0.5,
) -> pd.DataFrame:
    """Quét đường cong tối ưu hóa lợi nhuận qua các mức trần chiết khấu (5% -> 50%) có xét độ co giãn cầu."""
    if df.empty:
        return pd.DataFrame()

    caps = [round(c, 2) for c in np.arange(0.05, 0.55, 0.05)]
    rows: list[dict[str, float]] = []
    for cap in caps:
        res_inelastic = simulate_discount_what_if(
            df, max_discount_cap=cap, loss_subcat_uplift_pct=uplift_pct, price_elasticity=0.0
        )
        res_elastic = simulate_discount_what_if(
            df, max_discount_cap=cap, loss_subcat_uplift_pct=uplift_pct, price_elasticity=elasticity
        )
        rows.append(
            {
                "Tran_chiet_khau_pct": round(cap * 100.0, 0),
                "Doanh_thu_co_co_gian": round(float(res_elastic["sim_sales"]), 2),
                "Loi_nhuan_khong_co_gian": round(float(res_inelastic["sim_profit"]), 2),
                "Loi_nhuan_co_co_gian": round(float(res_elastic["sim_profit"]), 2),
                "Bien_LN_co_co_gian_pct": round(float(res_elastic["sim_margin_pct"]), 2),
                "San_luong_giu_lai_pct": round(float(res_elastic["volume_retention_pct"]), 2),
            }
        )
    return pd.DataFrame(rows)


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Thực hiện bộ kiểm định giả thuyết thống kê tham số (Welch's t-test, ANOVA) & phi tham số (Mann-Whitney U, Kruskal-Wallis)."""
    if df.empty or len(df) < 10:
        return pd.DataFrame()

    from scipy import stats

    rows: list[dict[str, object]] = []

    # 1. Kiểm định lợi nhuận giữa nhóm chiết khấu <= 20% và > 20%
    g_safe = df.loc[df["Discount"] <= 0.20, "Profit"].dropna().to_numpy(dtype=float)
    g_deep = df.loc[df["Discount"] > 0.20, "Profit"].dropna().to_numpy(dtype=float)
    if len(g_safe) >= 2 and len(g_deep) >= 2:
        t_stat, p_val_t = stats.ttest_ind(g_safe, g_deep, equal_var=False)
        u_stat, p_val_u = stats.mannwhitneyu(g_safe, g_deep, alternative="two-sided")
        var_s = float(np.var(g_safe, ddof=1))
        var_d = float(np.var(g_deep, ddof=1))
        pooled_sd = np.sqrt(((len(g_safe) - 1) * var_s + (len(g_deep) - 1) * var_d) / (len(g_safe) + len(g_deep) - 2))
        cohen_d = float((np.mean(g_safe) - np.mean(g_deep)) / pooled_sd) if pooled_sd > 0 else 0.0

        rows.append(
            {
                "Giả thuyết kiểm định (H0)": "Profit TB nhóm CK ≤ 20% bằng nhóm CK > 20%",
                "Phương pháp": "Welch's t-test (Tham số) & Mann-Whitney U (Phi tham số)",
                "Thống kê kiểm định": f"t = {t_stat:.2f} | U = {u_stat:,.0f}",
                "p-value": f"{p_val_t:.2e} (U: {p_val_u:.2e})",
                "Độ lớn hiệu ứng (Effect Size)": f"Cohen's d = {cohen_d:.2f}",
                "Kết luận (α = 0.05)": (
                    "Bác bỏ H0 — CK > 20% làm sụt giảm lợi nhuận có ý nghĩa thống kê"
                    if p_val_t < 0.05
                    else "Chưa đủ bằng chứng bác bỏ H0"
                ),
            }
        )

    # 2. Kiểm định sự khác biệt lợi nhuận giữa các Khu vực (Region)
    reg_groups = [
        grp["Profit"].dropna().to_numpy(dtype=float)
        for _, grp in df.groupby("Region")
        if len(grp["Profit"].dropna()) >= 2
    ]
    if len(reg_groups) >= 2:
        f_stat, p_val_f = stats.f_oneway(*reg_groups)
        h_stat, p_val_h = stats.kruskal(*reg_groups)
        rows.append(
            {
                "Giả thuyết kiểm định (H0)": "Profit phân bổ đồng đều giữa các Khu vực (Region)",
                "Phương pháp": "One-way ANOVA (F) & Kruskal-Wallis (H)",
                "Thống kê kiểm định": f"F = {f_stat:.2f} | H = {h_stat:.2f}",
                "p-value": f"{p_val_f:.2e} (Kruskal: {p_val_h:.2e})",
                "Độ lớn hiệu ứng (Effect Size)": f"{len(reg_groups)} nhóm khu vực",
                "Kết luận (α = 0.05)": (
                    "Bác bỏ H0 — Khác biệt lợi nhuận giữa các vùng có ý nghĩa thống kê"
                    if (p_val_f < 0.05 or p_val_h < 0.05)
                    else "Chưa đủ bằng chứng bác bỏ H0"
                ),
            }
        )

    # 3. Kiểm định sự khác biệt lợi nhuận giữa các Ngành hàng (Category)
    cat_groups = [
        grp["Profit"].dropna().to_numpy(dtype=float)
        for _, grp in df.groupby("Category")
        if len(grp["Profit"].dropna()) >= 2
    ]
    if len(cat_groups) >= 2:
        f_stat_c, p_val_fc = stats.f_oneway(*cat_groups)
        h_stat_c, p_val_hc = stats.kruskal(*cat_groups)
        rows.append(
            {
                "Giả thuyết kiểm định (H0)": "Profit phân bổ đồng đều giữa 3 Ngành hàng (Category)",
                "Phương pháp": "One-way ANOVA (F) & Kruskal-Wallis (H)",
                "Thống kê kiểm định": f"F = {f_stat_c:.2f} | H = {h_stat_c:.2f}",
                "p-value": f"{p_val_fc:.2e} (Kruskal: {p_val_hc:.2e})",
                "Độ lớn hiệu ứng (Effect Size)": f"{len(cat_groups)} ngành hàng",
                "Kết luận (α = 0.05)": (
                    "Bác bỏ H0 — Khác biệt lợi nhuận giữa các ngành có ý nghĩa thống kê"
                    if (p_val_fc < 0.05 or p_val_hc < 0.05)
                    else "Chưa đủ bằng chứng bác bỏ H0"
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_correlation_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """So sánh hệ số tương quan tuyến tính Pearson (r) và tương quan hạng phi tham số Spearman (ρ) kèm p-value."""
    if df.empty or len(df) < 5:
        return pd.DataFrame()

    from scipy import stats

    pairs = [
        ("Discount", "Profit", "Chiết khấu ↔ Lợi nhuận"),
        ("Sales", "Profit", "Doanh thu ↔ Lợi nhuận"),
        ("Quantity", "Sales", "Số lượng ↔ Doanh thu"),
        ("Shipping_Days", "Profit", "Thời gian giao hàng ↔ Lợi nhuận"),
    ]
    rows: list[dict[str, object]] = []
    for col_x, col_y, label in pairs:
        if col_x not in df.columns or col_y not in df.columns:
            continue
        sub = df[[col_x, col_y]].dropna()
        if len(sub) < 5 or sub[col_x].nunique() < 2 or sub[col_y].nunique() < 2:
            continue
        r_val, p_r = stats.pearsonr(sub[col_x], sub[col_y])
        rho_val, p_rho = stats.spearmanr(sub[col_x], sub[col_y])
        rows.append(
            {
                "Cặp biến số": label,
                "Pearson r (Tuyến tính)": round(float(r_val), 4),
                "p-value (Pearson)": f"{p_r:.2e}",
                "Spearman ρ (Phi tham số)": round(float(rho_val), 4),
                "p-value (Spearman)": f"{p_rho:.2e}",
                "Đánh giá quan hệ": (
                    "Nghịch biến mạnh & có ý nghĩa thống kê"
                    if r_val < -0.15 and p_r < 0.05
                    else (
                        "Đồng biến có ý nghĩa thống kê"
                        if r_val > 0.15 and p_r < 0.05
                        else "Quan hệ yếu / Không tuyến tính"
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def train_loss_risk_model(df: pd.DataFrame) -> dict[str, object]:
    """Huấn luyện mô hình học máy dự báo xác suất đơn hàng bị lỗ P(Loss = 1 | X) và trích xuất Feature Importance / Odds Ratio."""
    if df.empty or len(df) < 50:
        return {"metrics": {}, "feature_importance": pd.DataFrame()}

    work = df.copy()
    work["Is_Loss"] = (work["Profit"] < 0).astype(int)
    if work["Is_Loss"].nunique() < 2:
        return {"metrics": {}, "feature_importance": pd.DataFrame()}

    cat_cols = [c for c in ["Category", "Sub-Category", "Region", "Segment", "Ship Mode"] if c in work.columns]
    num_cols = [c for c in ["Discount", "Quantity", "Shipping_Days"] if c in work.columns]

    X_df = pd.get_dummies(work[num_cols + cat_cols], columns=cat_cols, drop_first=False, dtype=float).fillna(0.0)
    y = work["Is_Loss"].to_numpy(dtype=int)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_prob = lr.predict_proba(X_test_sc)[:, 1]

    metrics = {
        "rf_roc_auc": round(float(roc_auc_score(y_test, rf_prob)), 4),
        "rf_f1": round(float(f1_score(y_test, rf_pred)), 4),
        "rf_accuracy": round(float(accuracy_score(y_test, rf_pred) * 100.0), 2),
        "lr_roc_auc": round(float(roc_auc_score(y_test, lr_prob)), 4),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "base_loss_rate_pct": round(float(y.mean() * 100.0), 2),
    }

    odds_ratios = np.exp(np.clip(lr.coef_[0], -5.0, 5.0))
    feat_df = (
        pd.DataFrame(
            {
                "Biến giải thích (Feature)": X_df.columns,
                "Tầm quan trọng RF (%)": np.round(rf.feature_importances_ * 100.0, 2),
                "Hệ số Logit (β)": np.round(lr.coef_[0], 3),
                "Odds Ratio (e^β)": np.round(odds_ratios, 2),
            }
        )
        .sort_values("Tầm quan trọng RF (%)", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    return {"metrics": metrics, "feature_importance": feat_df}


def _compute_adf_test(x: np.ndarray) -> tuple[float, float, float, int]:
    """Tính kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF) bằng OLS + MacKinnon p-value."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    from statsmodels.tsa.tsatools import add_trend, lagmat

    x = np.asarray(x, dtype=float)
    xdiff = np.diff(x)
    maxlag = min(4, max(1, int(np.ceil(12.0 * (len(x) / 100.0) ** 0.25))))
    xdall = lagmat(xdiff[:, None], maxlag, trim="both", original="in")
    nobs = xdall.shape[0]
    xdall[:, 0] = x[-nobs - 1 : -1]
    xdshort = xdiff[-nobs:]

    best_aic = float("inf")
    best_lag = 1
    for lag in range(0, maxlag + 1):
        rhs = add_trend(xdall[:, : lag + 1], "c", prepend=False)
        res = OLS(xdshort, rhs).fit()
        if res.aic < best_aic:
            best_aic = float(res.aic)
            best_lag = lag

    xdall_opt = lagmat(xdiff[:, None], best_lag, trim="both", original="in")
    nobs_opt = xdall_opt.shape[0]
    xdall_opt[:, 0] = x[-nobs_opt - 1 : -1]
    xdshort_opt = xdiff[-nobs_opt:]
    res_opt = OLS(xdshort_opt, add_trend(xdall_opt[:, : best_lag + 1], "c")).fit()

    adf_stat = float(res_opt.tvalues[0])
    p_val = float(mackinnonp(adf_stat, regression="c", N=1))
    crit_5 = float(mackinnoncrit(N=1, regression="c", nobs=nobs_opt)[1])
    return adf_stat, p_val, crit_5, best_lag


def analyze_timeseries_stationarity(df: pd.DataFrame) -> dict[str, object]:
    """Kiểm định tính dừng Augmented Dickey-Fuller (ADF) và phân rã chuỗi thời gian (Trend + Seasonal + Residual)."""
    if df.empty or "Order Date" not in df.columns:
        return {"adf_table": pd.DataFrame(), "decomposition": pd.DataFrame()}

    monthly = (
        df.set_index("Order Date")["Sales"]
        .resample("MS")
        .sum()
        .reset_index()
        .rename(columns={"Order Date": "ds", "Sales": "y"})
        .dropna()
    )
    if len(monthly) < 12:
        return {"adf_table": pd.DataFrame(), "decomposition": pd.DataFrame()}

    adf_rows: list[dict[str, object]] = []
    series_list = [
        ("Chuỗi gốc Y_t (Bậc tích hợp d = 0)", monthly["y"].to_numpy(dtype=float)),
        ("Sai phân bậc 1 ΔY_t (d = 1)", monthly["y"].diff().dropna().to_numpy(dtype=float)),
    ]
    for name, arr in series_list:
        if len(arr) < 8:
            continue
        stat_val, p_val, crit_5, best_lag = _compute_adf_test(arr)
        adf_rows.append(
            {
                "Chuỗi thời gian": name,
                "ADF Statistic": round(stat_val, 4),
                "p-value": round(p_val, 5),
                "Ngưỡng tới hạn (5%)": round(crit_5, 4),
                "Số độ trễ (Lags)": int(best_lag),
                "Kết luận tính dừng (α = 0.05)": (
                    "Chuỗi DỪNG (Stationary — Bác bỏ H0)"
                    if p_val < 0.05
                    else "Chuỗi KHÔNG DỪNG (Có nghiệm đơn vị)"
                ),
            }
        )

    y_vals = monthly["y"].to_numpy(dtype=float)
    period = 12 if len(monthly) >= 24 else max(2, len(monthly) // 2)
    # Phân rã cộng tính chuẩn (Centered Moving Average Trend + Chu kỳ mùa vụ trung bình)
    trend_series = pd.Series(y_vals).rolling(window=period, center=True, min_periods=max(2, period // 2)).mean()
    trend_series = trend_series.bfill().ffill()
    trend_vals = trend_series.to_numpy(dtype=float)

    detrended = y_vals - trend_vals
    month_indices = np.arange(len(y_vals)) % period
    seasonal_means = np.array([detrended[month_indices == m].mean() for m in range(period)], dtype=float)
    seasonal_means -= seasonal_means.mean()
    seasonal_vals = seasonal_means[month_indices]
    resid_vals = y_vals - trend_vals - seasonal_vals

    decomp_df = pd.DataFrame(
        {
            "ds": monthly["ds"].to_numpy(),
            "Observed": y_vals,
            "Trend": trend_vals,
            "Seasonal": seasonal_vals,
            "Residual": resid_vals,
        }
    )

    return {"adf_table": pd.DataFrame(adf_rows), "decomposition": decomp_df}


def compute_clv_and_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Ước lượng xác suất khách hàng còn hoạt động P(Alive), chu kỳ mua lặp lại (IPT), CLV dự phóng 12 tháng và Doanh thu có nguy cơ mất."""
    if df.empty:
        return pd.DataFrame()

    rfm = compute_rfm_segments(df)
    if rfm.empty:
        return pd.DataFrame()

    ref_date = df["Order Date"].max() + pd.Timedelta(days=1)
    tenure_df = (
        df.groupby("Customer ID", as_index=False)
        .agg(First_Order_Date=("Order Date", "min"))
    )
    merged = rfm.merge(tenure_df, on="Customer ID", how="left")
    merged["Lifespan_Days"] = np.maximum((ref_date - merged["First_Order_Date"]).dt.days, 30)
    active_span = np.maximum((merged["Last_Order_Date"] - merged["First_Order_Date"]).dt.days, 1)
    merged["IPT_Days"] = np.where(
        merged["Frequency"] > 1,
        active_span / np.maximum(merged["Frequency"] - 1, 1),
        merged["Lifespan_Days"],
    )
    merged["AOV"] = merged["Monetary"] / np.maximum(merged["Frequency"], 1)

    # Xác suất sống sót theo quy luật BG/NBD rút gọn: P(Alive) = (1 + Recency / Lifespan)^(-Frequency)
    merged["P_Alive"] = np.power(
        1.0 + (merged["Recency"] / np.maximum(merged["Lifespan_Days"], 30.0)),
        -np.clip(merged["Frequency"], 1, 15),
    )
    merged["P_Alive_Pct"] = np.round(merged["P_Alive"] * 100.0, 1)
    merged["Churn_Risk_Pct"] = np.round((1.0 - merged["P_Alive"]) * 100.0, 1)

    annual_order_rate = merged["Frequency"] / (merged["Lifespan_Days"] / 365.25)
    merged["Predicted_12M_CLV"] = np.round(merged["AOV"] * annual_order_rate * merged["P_Alive"], 2)
    merged["Revenue_At_Risk"] = np.round(
        merged["AOV"] * annual_order_rate * (1.0 - merged["P_Alive"]), 2
    )

    return merged.sort_values("Predicted_12M_CLV", ascending=False).reset_index(drop=True)





