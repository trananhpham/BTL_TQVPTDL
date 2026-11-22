"""Kiểm thử đơn vị (Unit Tests) cho tầng xử lý dữ liệu và phân tích nâng cao."""

from __future__ import annotations

import pandas as pd

from data.analytics import (
    analyze_timeseries_stationarity,
    build_apriori_network_elements,
    build_executive_html_report,
    build_sankey_flow_data,
    calculate_profit_goal_seek,
    compare_groups_ab,
    compute_apriori_lift_matrix,
    compute_clv_and_churn,
    compute_cohort_retention,
    compute_correlation_comparison,
    compute_customer_churn_scorecard,
    compute_elasticity_curve,
    compute_kmeans_clusters,
    compute_regional_radar_metrics,
    detect_multivariate_anomalies,
    evaluate_forecast_holdout,
    compute_pareto_abc,
    compute_rfm_segments,
    compute_seasonality_matrix,
    compute_yoy_waterfall,
    detect_loss_anomalies,
    forecast_monthly_sales,
    generate_dynamic_insight,
    generate_regional_action_playbook,
    generate_smart_cross_sell_bundles,
    pivot_region_category,
    pivot_year_segment,
    run_hypothesis_tests,
    run_market_basket_apriori,
    score_kmeans_range,
    simulate_discount_what_if,
    summarize_by_category,
    summarize_discount_bands,
    summarize_state_performance,
    summarize_state_subcategory,
    summarize_by_region,
    summarize_by_state,
    summarize_by_time,
    summarize_top_products,
    train_loss_risk_model,
)
from data.loader import (
    audit_raw_data,
    clean_superstore_data,
    filter_dataframe,
    get_data_dictionary,
    load_raw_data,
)


def _get_clean_df() -> pd.DataFrame:
    raw = load_raw_data()
    return clean_superstore_data(raw, cap_outliers=False)


def test_load_audit_and_clean() -> None:
    raw = load_raw_data()
    assert raw.shape == (9994, 21)

    audit = audit_raw_data(raw)
    assert audit["missing_total"] == 0
    assert audit["business_duplicates"] == 1
    assert audit["invalid_ship_sequence"] == 0

    clean = clean_superstore_data(raw, cap_outliers=False)
    assert clean.shape[0] == 9993
    assert pd.api.types.is_datetime64_any_dtype(clean["Order Date"])
    assert pd.api.types.is_datetime64_any_dtype(clean["Ship Date"])
    assert clean["Postal Code"].str.len().eq(5).all()


def test_filtering_and_aggregations() -> None:
    clean = _get_clean_df()
    west_tech = filter_dataframe(clean, regions=["West"], categories=["Technology"])
    assert not west_tech.empty
    assert set(west_tech["Region"].unique()) == {"West"}
    assert set(west_tech["Category"].unique()) == {"Technology"}

    assert not summarize_by_time(clean, freq="Y").empty
    assert not summarize_by_region(clean).empty
    assert not summarize_by_state(clean).empty
    assert not summarize_by_category(clean).empty
    assert len(summarize_top_products(clean, top_n=10)) == 10
    assert "Tổng cộng" in pivot_region_category(clean).index
    assert not pivot_year_segment(clean).empty


def test_advanced_analytics() -> None:
    clean = _get_clean_df()

    rfm = compute_rfm_segments(clean)
    assert len(rfm) == 793
    assert "RFM_Segment" in rfm.columns

    km = compute_kmeans_clusters(clean, n_clusters=4)
    assert km["Cluster_Label"].nunique() == 4

    cohort = compute_cohort_retention(clean)
    assert not cohort.empty

    season = compute_seasonality_matrix(clean, metric="Sales")
    assert season.shape[0] == 4

    wf = compute_yoy_waterfall(clean, metric="Sales")
    assert not wf.empty

    abc = compute_pareto_abc(clean, entity_col="Sub-Category")
    assert not abc.empty
    assert abc["Cumulative_Sales_Pct"].iloc[-1] <= 100.01

    sim = simulate_discount_what_if(clean, max_discount_cap=0.20, loss_subcat_uplift_pct=5.0)
    assert sim["sim_profit"] > sim["actual_profit"]
    assert sim["sim_loss_rows"] < sim["actual_loss_rows"]

    anomalies = detect_loss_anomalies(clean, profit_threshold=-100.0)
    assert not anomalies.empty
    assert (anomalies["Profit"] <= -100.0).all()

    rules = run_market_basket_apriori(clean, item_col="Sub-Category", min_support=0.015, min_confidence=0.2)
    assert not rules.empty

    fc, _ = forecast_monthly_sales(clean, periods=6, method="ARIMA")
    assert len(fc[fc["Loại dữ liệu"] != "Lịch sử thực tế"]) == 6


def test_grading_extensions() -> None:
    clean = _get_clean_df()

    bands = summarize_discount_bands(clean)
    assert list(bands["Nguong_chiet_khau"].astype(str)) == ["0%", ">0–10%", ">10–20%", ">20–30%", ">30%"]
    above = bands[bands["Nguong_chiet_khau"].astype(str).isin([">20–30%", ">30%"])]
    weighted_loss = float((above["Ty_le_lo"] * above["So_dong"]).sum() / above["So_dong"].sum())
    assert weighted_loss > 90

    central = summarize_state_performance(clean, region="Central")
    assert central.iloc[0]["State"] == "Texas"
    assert float(central.iloc[0]["Total_Profit"]) < 0
    texas = summarize_state_subcategory(clean, "Texas")
    assert not texas.empty
    assert float(texas.iloc[0]["Total_Profit"]) < 0

    scores = score_kmeans_range(clean, k_min=2, k_max=4)
    assert set(scores["k"]) == {2, 3, 4}
    assert scores["Silhouette"].notna().all()

    holdout = evaluate_forecast_holdout(clean, holdout_months=6)
    scored = holdout.dropna(subset=["MAPE (%)"])
    assert len(scored) >= 2
    assert (scored["MAPE (%)"] > 0).all()

    # Kiểm tra các tính năng nâng cấp Web (Dynamic Insight, A/B Benchmark, HTML Report, Data Dictionary)
    ins = generate_dynamic_insight(clean)
    assert ins["best_cat"] == "Technology"
    assert ins["worst_state"] == "Texas"
    assert ins["high_disc_loss_rate"] > 90.0

    ab_reg = compare_groups_ab(clean, "Region", "West", "Central")
    assert len(ab_reg) == 8
    assert "Chênh lệch (A - B)" in ab_reg.columns

    html_rep = build_executive_html_report(clean)
    assert "<table" in html_rep and "Superstore" in html_rep

    dd = get_data_dictionary()
    assert len(dd) == 21


def test_scientific_modules() -> None:
    clean = _get_clean_df()

    # 1. Hypothesis testing & correlation comparison
    hyp = run_hypothesis_tests(clean)
    assert len(hyp) == 3
    assert "Bác bỏ H0" in hyp.iloc[0]["Kết luận (α = 0.05)"]

    corr = compute_correlation_comparison(clean)
    assert len(corr) == 4
    assert corr.iloc[0]["Pearson r (Tuyến tính)"] < 0

    # 2. Price elasticity & profit optimization curve
    sim_el = simulate_discount_what_if(clean, max_discount_cap=0.20, loss_subcat_uplift_pct=5.0, price_elasticity=-0.5)
    assert sim_el["sim_profit"] > sim_el["actual_profit"]
    assert 80.0 < sim_el["volume_retention_pct"] < 100.0

    curve = compute_elasticity_curve(clean, uplift_pct=5.0, elasticity=-0.5)
    assert len(curve) == 10
    assert "Loi_nhuan_co_co_gian" in curve.columns

    # 3. Explainable ML for Order Loss Risk
    ml = train_loss_risk_model(clean)
    metrics = ml["metrics"]
    feat_df = ml["feature_importance"]
    assert isinstance(metrics, dict) and metrics["rf_roc_auc"] > 0.90
    assert isinstance(feat_df, pd.DataFrame) and len(feat_df) == 12
    assert feat_df.iloc[0]["Biến giải thích (Feature)"] == "Discount"

    # 4. Time-series ADF stationarity & Classical Decomposition
    ts = analyze_timeseries_stationarity(clean)
    adf_df = ts["adf_table"]
    decomp_df = ts["decomposition"]
    assert isinstance(adf_df, pd.DataFrame) and len(adf_df) == 2
    assert isinstance(decomp_df, pd.DataFrame) and len(decomp_df) == 48

    # 5. Customer Survival Probability P(Alive), CLV & Revenue at Risk
    clv = compute_clv_and_churn(clean)
    assert len(clv) == 793
    assert (clv["P_Alive"] > 0).all() and (clv["P_Alive"] <= 1.0).all()
    assert clv["Predicted_12M_CLV"].sum() > 0

    # 6. Top N Loss/Profit products & Apriori Lift Matrix
    top_gain = summarize_top_products(clean, top_n=5, rank_by="Profit_Max")
    top_loss = summarize_top_products(clean, top_n=5, rank_by="Profit_Min")
    assert len(top_gain) == 5 and float(top_gain.iloc[0]["Total_Profit"]) > 0
    assert len(top_loss) == 5 and float(top_loss.iloc[0]["Total_Profit"]) < 0

    lift_mat = compute_apriori_lift_matrix(clean, item_col="Sub-Category")
    assert not lift_mat.empty and lift_mat.shape[0] >= 5


def test_new_five_features() -> None:
    clean = _get_clean_df()

    # 1. Product Affinity Network Graph
    rules = run_market_basket_apriori(clean, item_col="Sub-Category", min_support=0.015, min_confidence=0.2)
    net_data = build_apriori_network_elements(rules, clean, min_lift=0.95)
    assert not net_data["nodes"].empty
    assert "node_size" in net_data["nodes"].columns
    assert "x" in net_data["nodes"].columns and "y" in net_data["nodes"].columns
    assert not net_data["edges"].empty

    # 2. Multi-stage Sankey Flow Data
    sankey = build_sankey_flow_data(clean)
    assert len(sankey["labels"]) > 0
    assert len(sankey["sources"]) == len(sankey["targets"]) == len(sankey["values"])
    assert any("Sinh lời" in lbl for lbl in sankey["labels"])

    # 3. Customer Churn Scorecard & Action Matrix
    churn_sc = compute_customer_churn_scorecard(clean)
    assert not churn_sc["customers"].empty
    assert "Action_Tier" in churn_sc["customers"].columns
    assert churn_sc["summary"]["total_customers"] == 793
    assert not churn_sc["at_risk_vip"].empty

    # 4. Multivariate Anomaly Detection (Isolation Forest)
    anom = detect_multivariate_anomalies(clean, contamination=0.02)
    assert not anom["anomalies"].empty
    assert "Nguyen_nhan_bat_thuong" in anom["anomalies"].columns
    assert anom["metrics"]["anomalies_count"] > 0

    # 5. Regional Operational Health Radar
    radar = compute_regional_radar_metrics(clean)
    assert len(radar["scores"]) == 4
    assert len(radar["dimensions"]) == 5
    assert "Diem_Tong_Hop" in radar["scores"].columns
    assert (radar["scores"]["Diem_Tong_Hop"] > 0).all()

    # 6. Regional Action Playbook
    playbook = generate_regional_action_playbook(radar["scores"])
    assert len(playbook) == 4
    assert all("actions" in p and len(p["actions"]) > 0 for p in playbook)

    # 7. Smart Cross-Sell Bundles
    bundles = generate_smart_cross_sell_bundles(clean, min_lift=1.1, top_n=5)
    assert not bundles.empty
    assert "Doanh thu gia tăng dự phóng ($)" in bundles.columns

    # 8. Profit Goal-Seeking Optimization
    goal = calculate_profit_goal_seek(clean, target_profit_increase_pct=15.0)
    assert goal["target_profit"] > goal["actual_profit"]
    assert 0.05 <= goal["recommended_cap"] <= 0.50





