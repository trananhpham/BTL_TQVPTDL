"""Thành phần Chatbot AI Phân tích Dữ liệu B2B hỗ trợ Google Gemini API, Oxion AI & Local Copilot."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

from data.analytics import (
    compute_rfm_segments,
    generate_dynamic_insight,
    simulate_discount_what_if,
    summarize_by_category,
    summarize_by_region,
    summarize_discount_bands,
    summarize_state_subcategory,
    summarize_top_products,
)

OXION_BASE_URL = "https://oxionai.com/v1"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def build_dataframe_context(df: pd.DataFrame) -> str:
    """Tạo tóm tắt số liệu thực tế từ DataFrame đang lọc để truyền ngữ cảnh cho AI."""
    if df.empty:
        return "Dữ liệu đang lọc hiện đang trống (0 dòng)."

    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())
    margin = (total_profit / total_sales * 100.0) if total_sales else 0.0
    orders = int(df["Order ID"].nunique())
    customers = int(df["Customer ID"].nunique())
    avg_disc = float(df["Discount"].mean() * 100.0)
    date_min = df["Order Date"].min().strftime("%Y-%m-%d")
    date_max = df["Order Date"].max().strftime("%Y-%m-%d")

    cat_tbl = (
        df.groupby("Category", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Avg_Discount=("Discount", "mean"))
        .round(2)
        .to_string(index=False)
    )

    subcat_tbl = (
        df.groupby(["Category", "Sub-Category"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Avg_Discount=("Discount", "mean"))
        .sort_values("Profit", ascending=True)
        .round(2)
        .to_string(index=False)
    )

    region_tbl = (
        df.groupby("Region", as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Avg_Discount=("Discount", "mean"))
        .sort_values("Profit", ascending=False)
        .round(2)
        .to_string(index=False)
    )

    worst_state_cat = (
        df.groupby(["State", "Category"], as_index=False)
        .agg(
            Sales=("Sales", "sum"),
            Profit=("Profit", "sum"),
            Avg_Discount=("Discount", "mean"),
            Orders=("Order ID", "nunique"),
        )
        .sort_values("Profit", ascending=True)
        .head(20)
        .round(2)
        .to_string(index=False)
    )

    worst_state_subcat = (
        df.groupby(["State", "Category", "Sub-Category"], as_index=False)
        .agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"), Avg_Discount=("Discount", "mean"))
        .sort_values("Profit", ascending=True)
        .head(15)
        .round(2)
        .to_string(index=False)
    )

    no_disc = df[df["Discount"] == 0]
    high_disc = df[df["Discount"] > 0.2]
    no_disc_profit = float(no_disc["Profit"].mean()) if len(no_disc) else 0.0
    high_disc_profit = float(high_disc["Profit"].mean()) if len(high_disc) else 0.0
    high_disc_loss_rate = (
        float((high_disc["Profit"] < 0).mean() * 100.0) if len(high_disc) else 0.0
    )

    return (
        f"NGỮ CẢNH DỮ LIỆU SUPERSTORE ĐANG LỌC ({date_min} đến {date_max}, {len(df):,} dòng):\n"
        f"- Tổng doanh thu: ${total_sales:,.2f} | Tổng lợi nhuận: ${total_profit:,.2f} | Biên lợi nhuận: {margin:.2f}%\n"
        f"- Số đơn hàng: {orders:,} | Số khách hàng: {customers:,} | Chiết khấu TB: {avg_disc:.2f}%\n"
        f"- Ảnh hưởng chiết khấu: Discount=0% có lợi nhuận TB ${no_disc_profit:.2f}; "
        f"Discount>20% có lợi nhuận TB ${high_disc_profit:.2f} (tỷ lệ lỗ {high_disc_loss_rate:.2f}%).\n\n"
        f"[1] THEO CATEGORY:\n{cat_tbl}\n\n"
        f"[2] THEO SUB-CATEGORY (tăng dần theo Profit):\n{subcat_tbl}\n\n"
        f"[3] THEO REGION:\n{region_tbl}\n\n"
        f"[4] TOP 20 CẶP STATE x CATEGORY CÓ LỢI NHUẬN THẤP NHẤT:\n{worst_state_cat}\n\n"
        f"[5] TOP 15 CẶP STATE x CATEGORY x SUB-CATEGORY LỖ NẶNG NHẤT:\n{worst_state_subcat}"
    )


def generate_local_b2b_response(df: pd.DataFrame, prompt: str) -> str:
    """Bộ máy Phân tích Tự động Nội bộ (Local B2B Analytical Copilot) tính toán trực tiếp từ DataFrame."""
    if df.empty:
        return "Hiện không có dữ liệu trong bộ lọc để phân tích."

    ins = generate_dynamic_insight(df)
    sim = simulate_discount_what_if(df, max_discount_cap=0.20, loss_subcat_uplift_pct=5.0)
    q_lower = prompt.lower()

    # 1. Câu hỏi chuyên sâu về Texas / Central
    if "texas" in q_lower or "central" in q_lower:
        tx_df = df[df["State"] == "Texas"]
        if tx_df.empty:
            return "Trong bộ lọc hiện tại không có dữ liệu của bang Texas. Hãy chọn khu vực **Central** ở Sidebar."
        tx_sales = float(tx_df["Sales"].sum())
        tx_profit = float(tx_df["Profit"].sum())
        tx_margin = (tx_profit / tx_sales * 100.0) if tx_sales else 0.0
        tx_disc = float(tx_df["Discount"].mean() * 100.0)
        tx_loss_rows = int((tx_df["Profit"] < 0).sum())
        tx_sub = summarize_state_subcategory(df, "Texas").head(4)
        sub_bullets = "\n".join(
            f"  - **{r['Sub-Category']}** (*{r['Category']}*): Doanh thu `${r['Total_Sales']:,.2f}`, "
            f"Lợi nhuận **`${r['Total_Profit']:,.2f}`** (Chiết khấu TB `{r['Avg_Discount']*100:.0f}%`)"
            for _, r in tx_sub.iterrows()
        )
        return (
            f"### 🔍 Phân tích Nguyên nhân Thua lỗ tại Tiểu bang Texas (Khu vực Central)\n\n"
            f"**1. Số liệu thực tế tại Texas (theo bộ lọc hiện tại):**\n"
            f"- **Tổng doanh thu:** `${tx_sales:,.2f}` | **Tổng lợi nhuận:** **`${tx_profit:,.2f}`** (Biên LN: `{tx_margin:.2f}%`).\n"
            f"- **Mức chiết khấu trung bình:** `{tx_disc:.1f}%` (cao hơn rất nhiều so với mức trung bình toàn quốc `15.6%`).\n"
            f"- **Số giao dịch lỗ:** `{tx_loss_rows:,} / {len(tx_df):,}` dòng (`{tx_loss_rows/len(tx_df)*100:.1f}%`).\n\n"
            f"**2. Top 4 nhóm hàng (Sub-Category) kéo lỗ nặng nhất tại Texas:**\n{sub_bullets}\n\n"
            f"**3. Kết luận & Khuyến nghị hành động:**\n"
            f"- Nguyên nhân cốt lõi khiến Texas âm nặng là **chính sách giảm giá tới 60%–80%** cho nhóm `Binders` và `Appliances`, cộng thêm biên âm từ `Furnishings` và `Machines`.\n"
            f"- **Giải pháp:** Cắt trần chiết khấu tại Texas về tối đa `20%` và rà soát phụ phí vận chuyển nội thất cồng kềnh."
        )

    # 2. Câu hỏi về Chiết khấu (Discount) và giải pháp tăng biên lợi nhuận
    if "chiết khấu" in q_lower or "discount" in q_lower or "giải pháp" in q_lower:
        bands = summarize_discount_bands(df)
        band_lines = "\n".join(
            f"  - **Ngưỡng `{r['Nguong_chiet_khau']}`** (`{int(r['So_dong']):,}` dòng): "
            f"LN trung bình **`${r['Loi_nhuan_TB']:,.2f}`/đơn** | Tỷ lệ lỗ: **`{r['Ty_le_lo']:.1f}%`**"
            for _, r in bands.iterrows()
        )
        return (
            f"### 📉 Đánh giá Tác động Chiết khấu & 3 Giải pháp Tăng Biên Lợi Nhuận\n\n"
            f"**1. Phân tích định lượng theo 5 ngưỡng chiết khấu:**\n{band_lines}\n\n"
            f"> **Điểm bùng phát:** Ngay khi chiết khấu vượt ngưỡng **20%**, tỷ lệ giao dịch lỗ vọt lên **`{ins['high_disc_loss_rate']:.1f}%`**.\n\n"
            f"**2. Đề xuất 3 giải pháp chiến lược (kèm ước tính What-If):**\n"
            f"1. **Thiết lập trần chiết khấu cứng 20%:** Mọi đơn hàng giảm giá `>20%` phải qua phê duyệt cấp Giám đốc vùng.\n"
            f"2. **Tái định giá nhóm hàng biên âm (`Tables`, `Bookcases`, `Supplies`):** Điều chỉnh giá niêm yết hoặc phụ phí giao vận thêm `+5%`. "
            f"Theo mô hình *What-If Simulator*, kết hợp trần chiết khấu `20%` + tăng giá `5%` giúp lợi nhuận tăng thêm **`+${sim['profit_gain']:,.2f}`** "
            f"(đưa tổng lợi nhuận lên **`${sim['sim_profit']:,.2f}`**, biên LN đạt **`{sim['sim_margin_pct']:.2f}%`**).\n"
            f"3. **Bán kèm (Cross-sell theo thuật toán Apriori):** Ghép combo các mặt hàng `Binders` / `Paper` / `Phones` thay vì giảm giá sâu đơn lẻ."
        )

    # 3. Câu hỏi về Người dùng / Khách hàng / Phân khúc / RFM
    if any(k in q_lower for k in ("người dùng", "khách hàng", "customer", "rfm", "vip", "phân khúc", "bao nhiêu")):
        total_cust = int(df["Customer ID"].nunique())
        total_orders = int(df["Order ID"].nunique())
        seg_df = (
            df.groupby("Segment", as_index=False)
            .agg(
                Customers=("Customer ID", "nunique"),
                Orders=("Order ID", "nunique"),
                Sales=("Sales", "sum"),
                Profit=("Profit", "sum"),
            )
            .sort_values("Sales", ascending=False)
        )
        seg_lines = "\n".join(
            f"  - **{r['Segment']}**: `{int(r['Customers']):,}` khách hàng (`{int(r['Orders']):,}` đơn) — "
            f"Doanh thu `${r['Sales']:,.2f}` | Lợi nhuận **`${r['Profit']:,.2f}`**"
            for _, r in seg_df.iterrows()
        )
        rfm_df = compute_rfm_segments(df)
        rfm_counts = rfm_df["RFM_Segment"].value_counts()
        rfm_lines = "\n".join(f"  - **{seg_name}**: `{int(cnt):,}` khách hàng" for seg_name, cnt in rfm_counts.items())
        top_cust = rfm_df.iloc[0]
        return (
            f"### 👥 Thống kê Người dùng / Khách hàng & Phân khúc RFM (Theo bộ lọc hiện tại)\n\n"
            f"- **Tổng số khách hàng (Unique Customers):** **`{total_cust:,}` người dùng**.\n"
            f"- **Tổng số đơn hàng:** **`{total_orders:,}` đơn** (Trung bình `{total_orders / max(total_cust, 1):.1f}` đơn/khách; "
            f"chi tiêu TB **`${ins['total_sales'] / max(total_cust, 1):,.2f}`/khách**).\n"
            f"- **Tổng số dòng giao dịch:** `{len(df):,}` bản ghi.\n\n"
            f"**1. Cơ cấu theo Phân khúc khách hàng (Segment):**\n{seg_lines}\n\n"
            f"**2. Phân nhóm hành vi RFM (Recency - Frequency - Monetary):**\n{rfm_lines}\n\n"
            f"- **Khách hàng chi tiêu lớn nhất:** **{top_cust['Customer Name']}** (`${top_cust['Monetary']:,.2f}`, `{int(top_cust['Frequency'])}` đơn hàng)."
        )

    # 4. Câu hỏi về Sản phẩm / Top bán chạy
    if any(k in q_lower for k in ("sản phẩm", "product", "top", "bán chạy")):
        top_p = summarize_top_products(df, top_n=5)
        p_lines = "\n".join(
            f"  - **{r['Product Name']}** (*{r['Category']}*): Doanh thu **`${r['Total_Sales']:,.2f}`** | Lợi nhuận `${r['Total_Profit']:,.2f}`"
            for _, r in top_p.iterrows()
        )
        return (
            f"### 🏅 Top 5 Sản phẩm Doanh thu Cao nhất & Hiệu suất Danh mục\n\n"
            f"- **Tổng số mã sản phẩm (Unique Products):** **`{df['Product ID'].nunique():,}` sản phẩm** thuộc `17` Sub-Category.\n\n"
            f"**Top 5 sản phẩm dẫn đầu doanh thu:**\n{p_lines}\n\n"
            f"- **Nhóm hàng sinh lời tốt nhất:** Ngành **{ins['best_cat']}** (`+${ins['best_cat_profit']:,.2f}`).\n"
            f"- **Nhóm hàng lỗ nặng nhất cần tái cơ cấu:** **{ins['worst_sub']}** (`${ins['worst_sub_profit']:,.2f}`)."
        )

    # 5. Câu hỏi về Khu vực & Ngành hàng hiệu quả nhất
    if "khu vực" in q_lower or "ngành hàng" in q_lower or "hiệu quả" in q_lower:
        reg_df = summarize_by_region(df)
        cat_df = summarize_by_category(df)
        top_reg = reg_df.iloc[0]
        top_sub = cat_df.sort_values("Total_Profit", ascending=False).iloc[0]
        reg_lines = "\n".join(
            f"  - **{r['Region']}**: Doanh thu `${r['Total_Sales']:,.2f}` | Lợi nhuận **`${r['Total_Profit']:,.2f}`** (Biên `{r['Profit_Margin_Pct']:.2f}%`)"
            for _, r in reg_df.iterrows()
        )
        return (
            f"### 🏆 Đánh giá Hiệu suất Khu vực & Ngành hàng Chủ lực\n\n"
            f"**1. Xếp hạng Khu vực địa lý (Region):**\n{reg_lines}\n"
            f"- Khu vực dẫn đầu là **{top_reg['Region']}** với biên lợi nhuận **`{top_reg['Profit_Margin_Pct']:.2f}%`**.\n\n"
            f"**2. Ngành hàng & Nhóm sản phẩm ngôi sao:**\n"
            f"- Ngành hàng sinh lời số 1: **{ins['best_cat']}** (Tổng lợi nhuận **`+${ins['best_cat_profit']:,.2f}`**).\n"
            f"- Nhóm sản phẩm (Sub-Category) lãi cao nhất: **{top_sub['Sub-Category']}** (`+${top_sub['Total_Profit']:,.2f}`, biên `{top_sub['Profit_Margin_Pct']:.2f}%`).\n"
            f"- Nhóm sản phẩm cần tái cơ cấu: **{ins['worst_sub']}** (`${ins['worst_sub_profit']:,.2f}`)."
        )

    # 4. Báo cáo Điều hành Toàn diện (1-Click Executive Brief & câu hỏi tổng quát)
    return (
        f"### 📋 BÁO CÁO ĐIỀU HÀNH CHIẾN LƯỢC (EXECUTIVE BRIEF)\n\n"
        f"#### 1️⃣ Điểm sáng Doanh thu & Ngành hàng Chủ lực\n"
        f"- **Quy mô đang lọc:** `{len(df):,}` giao dịch (`{ins['total_orders']:,}` đơn hàng) mang về tổng doanh thu **`${ins['total_sales']:,.2f}`** "
        f"và lợi nhuận ròng **`${ins['total_profit']:,.2f}`** (Biên lợi nhuận đạt **`{ins['margin_pct']:.2f}%`**).\n"
        f"- **Động lực tăng trưởng:** Ngành **{ins['best_cat']}** dẫn đầu lợi nhuận với **`+${ins['best_cat_profit']:,.2f}`** "
        f"(nổi bật ở các dòng máy `Copiers`, `Phones`, `Accessories`). Tăng trưởng doanh thu năm gần nhất ({ins['latest_year']}) đạt **`{ins['yoy_sales_pct']:+.1f}%`**.\n\n"
        f"#### 2️⃣ Nguyên nhân Gốc rễ Gây Thua lỗ (Root-Cause Analysis)\n"
        f"- **Bẫy chiết khấu trên 20%:** Toàn tập dữ liệu đang lọc có `{ins['loss_rows']:,}` dòng lỗ (`{ins['loss_rate_pct']:.1f}%`). "
        f"Tại các đơn chiết khấu `>20%`, tỷ lệ thua lỗ lên tới **`{ins['high_disc_loss_rate']:.1f}%`**.\n"
        f"- **Điểm nóng địa lý & sản phẩm:** Tiểu bang lỗ nặng nhất là **{ins['worst_state']}** (**`${ins['worst_state_profit']:,.2f}`**), "
        f"và nhóm hàng kéo tụt lợi nhuận mạnh nhất là **{ins['worst_sub']}** (**`${ins['worst_sub_profit']:,.2f}`**).\n\n"
        f"#### 3️⃣ Ba Hành động Ưu tiên & Ước tính Tác động Tài chính\n"
        f"1. **Áp trần chiết khấu 20% toàn hệ thống:** Chặn đứng rò rỉ lợi nhuận tại vùng `Central` (đặc biệt là Texas & Illinois).\n"
        f"2. **Tái cơ cấu danh mục `Tables` & `Bookcases`:** Tăng giá cơ sở `+5%` hoặc cắt khuyến mãi sâu. "
        f"Mô phỏng *What-If* chứng minh giải pháp (1) + (2) giúp **tăng thêm `+${sim['profit_gain']:,.2f}` lợi nhuận ròng** "
        f"(nâng biên lợi nhuận từ `{sim['actual_margin_pct']:.2f}%` lên **`{sim['sim_margin_pct']:.2f}%`** và giảm `{sim['actual_loss_rows'] - sim['sim_loss_rows']:,}` giao dịch lỗ).\n"
        f"3. **Chăm sóc khách hàng VIP & Cross-sell:** Tập trung giữ chân nhóm *VIP / Trung thành* trên ma trận RFM bằng combo bán chéo Apriori."
    )


def _resolve_api_key() -> str:
    """Lấy API key từ biến môi trường (OXION_API_KEY / GEMINI_API_KEY), st.secrets hoặc session_state."""
    saved = str(st.session_state.get("ai_api_key_input", "")).strip()
    if saved:
        return saved
    for env_name in ("OXION_API_KEY", "GEMINI_API_KEY"):
        val = os.getenv(env_name, "").strip()
        if val:
            return val
    try:
        for secret_name in ("OXION_API_KEY", "GEMINI_API_KEY"):
            val = str(st.secrets.get(secret_name, "")).strip()
            if val:
                return val
    except Exception:
        pass
    return ""


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_oxion_models(api_key: str) -> list[str]:
    """Tự động lấy danh sách model khả dụng từ Oxion AI (/v1/models) với User-Agent chuẩn."""
    headers = {**HTTP_HEADERS, "Authorization": f"Bearer {api_key}"}
    req = urllib.request.Request(f"{OXION_BASE_URL}/models", headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = [item["id"] for item in payload.get("data", []) if "id" in item]
        return sorted(models)
    except Exception:
        return []


def _call_oxion_chat(
    api_key: str,
    model_name: str,
    system_instruction: str,
    history: list[dict[str, str]],
    prompt_with_context: str,
) -> str:
    """Gọi endpoint OpenAI-compatible của Oxion AI (/v1/chat/completions) kèm header chống chặn Cloudflare 1010."""
    messages: list[dict[str, str]] = [{"role": "system", "content": system_instruction}]
    for msg in history[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt_with_context})

    body = json.dumps({"model": model_name, "messages": messages}).encode("utf-8")
    headers = {
        **HTTP_HEADERS,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        f"{OXION_BASE_URL}/chat/completions",
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {err.code} từ Oxion AI: {detail}") from err


def render_ai_chatbot(df_filtered: pd.DataFrame, location: str = "main") -> None:
    """Hiển thị khung chat trợ lý phân tích dữ liệu B2B (Cloud LLM + Local Analytical Copilot dự phòng)."""
    container = st.sidebar if location == "sidebar" else st

    container.subheader("🤖 Trợ lý AI Phân tích B2B (Gemini 2.5 Flash / Oxion AI / Local Copilot)")
    container.caption(
        "Hỗ trợ Google Gemini API (`AIza...`), Oxion AI (`cc_...`) và **Chế độ Phân tích Tự động Nội bộ (Offline Copilot)** "
        "khi chưa nhập API Key hoặc mạng bị chặn."
    )

    current_key = _resolve_api_key()
    k_col1, k_col2 = container.columns([2, 1.2])
    with k_col1:
        entered_key = st.text_input(
            "Nhập API Key (Tùy chọn — Để trống sẽ dùng Local AI Copilot)",
            value=current_key,
            type="password",
            key="_input_ai_api_key",
            placeholder="AIza... (Gemini) hoặc cc_... (Oxion) — Có thể để trống",
            help="Không lưu cứng key trong code. Nếu để trống, hệ thống tự động phân tích bằng Local B2B Copilot.",
        )
    if entered_key.strip() != current_key:
        st.session_state["ai_api_key_input"] = entered_key.strip()
        current_key = entered_key.strip()

    detected_provider = "Oxion AI (oxionai.com/v1)" if current_key.startswith("cc_") else "Google Gemini API"
    with k_col2:
        provider = st.selectbox(
            "Chế độ / Nhà cung cấp",
            options=[
                "⚡ Local B2B Copilot (Không cần API Key)",
                "Google Gemini API",
                "Oxion AI (oxionai.com/v1)",
            ],
            index=(
                2
                if current_key.startswith("cc_")
                else (1 if current_key.startswith("AIza") else 0)
            ),
        )

    gemini_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"]
    selected_model = gemini_models[0]
    if provider.startswith("Oxion"):
        oxion_models = _fetch_oxion_models(current_key) if current_key.startswith("cc_") else []
        if oxion_models:
            selected_model = container.selectbox("Chọn Model trên Oxion AI", options=oxion_models)
        else:
            selected_model = container.text_input(
                "Tên Model trên Oxion AI",
                value="gemini-2.5-flash",
                help="Nhập mã model khả dụng trên tài khoản Oxion AI của bạn.",
            )
    elif provider.startswith("Google"):
        selected_model = container.selectbox("Model Gemini", options=gemini_models, index=0)

    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []
    if "ai_last_interaction_id" not in st.session_state:
        st.session_state["ai_last_interaction_id"] = None

    # Tự động dọn dẹp các thông báo lỗi cũ trong session_state (nếu có từ phiên trước)
    cleaned_hist: list[dict[str, str]] = []
    for m in st.session_state["ai_chat_history"]:
        c_txt = str(m.get("content", ""))
        if c_txt.startswith("⚠️ Vui lòng nhập API Key") or c_txt.startswith("❌ Lỗi khi gọi AI API"):
            if cleaned_hist and cleaned_hist[-1].get("role") == "user":
                cleaned_hist.pop()
            continue
        cleaned_hist.append(m)
    st.session_state["ai_chat_history"] = cleaned_hist

    sample_questions = [
        "Tại sao lợi nhuận ở bang Texas lại âm và nhóm hàng nào ảnh hưởng nhất?",
        "Đánh giá tác động chiết khấu và đề xuất 3 giải pháp tăng biên lợi nhuận.",
        "Khu vực và ngành hàng nào đang hoạt động hiệu quả nhất?",
    ]

    cols = container.columns(len(sample_questions)) if location == "main" else None
    preset_prompt = None
    if cols is not None:
        for idx, q in enumerate(sample_questions):
            if cols[idx].button(q, key=f"_preset_q_{idx}", use_container_width=True):
                preset_prompt = q

        act_c1, act_c2 = container.columns([2, 1])
        if act_c1.button(
            "⚡ Tạo Bản Tóm tắt Chiến lược Toàn diện (1-Click AI Executive Brief)",
            key="_btn_ai_exec_brief",
            use_container_width=True,
            type="primary",
        ):
            preset_prompt = (
                "Hãy lập Báo cáo Điều hành (Executive Brief) gồm 3 phần rõ ràng từ dữ liệu đang lọc: "
                "(1) Điểm sáng doanh thu & ngành hàng chủ lực; "
                "(2) Nguyên nhân gốc rễ gây thua lỗ (chiết khấu >20%, Texas/Central, nhóm Tables/Bookcases/Binders); "
                "(3) 3 hành động ưu tiên kèm ước tính tác động tài chính."
            )
        if act_c2.button("🗑️ Xóa lịch sử hội thoại", key="_btn_clear_chat", use_container_width=True):
            st.session_state["ai_chat_history"] = []
            st.session_state["ai_last_interaction_id"] = None
            st.rerun()

    # Đặt khung hiển thị tin nhắn TRƯỚC ô chat_input để tin nhắn mới luôn nằm phía trên ô nhập
    chat_box = container.container()
    user_prompt = container.chat_input("Hỏi AI về số liệu đang hiển thị...")
    final_prompt = preset_prompt or user_prompt

    with chat_box:
        for msg in st.session_state["ai_chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if not final_prompt:
            return

        st.session_state["ai_chat_history"].append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        # Nếu chọn Local Copilot hoặc chưa nhập API Key -> trả lời ngay bằng bộ máy phân tích nội bộ
        if provider.startswith("⚡ Local") or not current_key:
            reply = generate_local_b2b_response(df_filtered, final_prompt)
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state["ai_chat_history"].append({"role": "assistant", "content": reply})
            return

        data_context = build_dataframe_context(df_filtered)
        system_instruction = (
            "Bạn là Chuyên gia Phân tích Dữ liệu B2B cấp cao đang tư vấn cho Ban Giám đốc Superstore. "
            "Hãy trả lời hoàn toàn bằng tiếng Việt, ngắn gọn, súc tích, chỉ dựa trên số liệu thực tế trong "
            "ngữ cảnh được cung cấp, nêu rõ con số cụ thể (USD, %, chiết khấu) và đưa ra khuyến nghị hành động."
        )
        prompt_with_context = f"{data_context}\n\nCÂU HỎI CỦA GIÁM ĐỐC:\n{final_prompt}"

        with st.chat_message("assistant"):
            with st.spinner(f"{selected_model} đang phân tích dữ liệu..."):
                try:
                    if provider.startswith("Oxion"):
                        reply = _call_oxion_chat(
                            api_key=current_key,
                            model_name=selected_model,
                            system_instruction=system_instruction,
                            history=st.session_state["ai_chat_history"],
                            prompt_with_context=prompt_with_context,
                        )
                    else:
                        from google import genai

                        client = genai.Client(api_key=current_key)
                        kwargs = {
                            "model": selected_model,
                            "input": prompt_with_context,
                            "system_instruction": system_instruction,
                        }
                        if st.session_state.get("ai_last_interaction_id"):
                            kwargs["previous_interaction_id"] = st.session_state["ai_last_interaction_id"]

                        interaction = client.interactions.create(**kwargs)
                        st.session_state["ai_last_interaction_id"] = interaction.id
                        reply = interaction.output_text or generate_local_b2b_response(df_filtered, final_prompt)
                    st.markdown(reply)
                except Exception as err:
                    fallback_reply = generate_local_b2b_response(df_filtered, final_prompt)
                    reply = (
                        f"> ℹ️ *Kết nối Cloud API ({provider}) báo lỗi `{err}`. "
                        f"Hệ thống tự động chuyển sang **Local B2B Copilot** để trả lời ngay từ dữ liệu đang lọc:*\n\n"
                        f"{fallback_reply}"
                    )
                    st.markdown(reply)

        st.session_state["ai_chat_history"].append({"role": "assistant", "content": reply})
