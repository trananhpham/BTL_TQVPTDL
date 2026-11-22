# 📊 Superstore Sales Analytics & AI BI Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![CI](https://github.com/trananhpham/BTL_TQVPTDL/actions/workflows/ci.yml/badge.svg)](https://github.com/trananhpham/BTL_TQVPTDL/actions/workflows/ci.yml)

Dự án **Bài tập lớn môn Trực quan & Phân tích Dữ liệu**: Phân tích kinh doanh, thống kê suy luận, kinh tế lượng và học máy toàn diện trên bộ dữ liệu bán lẻ **Kaggle Superstore Sales Dataset** (`Sample - Superstore.csv`, `9.994` dòng × `21` cột) và xây dựng hệ thống Dashboard tương tác **Streamlit** theo kiến trúc sạch (**Clean Architecture**), tích hợp **Dự báo chuỗi thời gian (SARIMAX / Prophet)**, **Kiểm định giả thuyết thống kê & tính dừng ADF**, **Mô hình độ co giãn cầu theo giá ($\epsilon$)**, **Học máy giải thích rủi ro đơn hàng lỗ (Explainable ML)**, **Phân khúc khách hàng RFM & Xác suất sống sót $P(\text{Alive})$ / CLV 12 tháng**, **Phân tích giỏ hàng Apriori**, **Bản đồ địa lý Hoa Kỳ**, **Biểu đồ phân cấp Sunburst** và **Trợ lý AI B2B (Gemini 2.5 Flash / Oxion AI + Local B2B Analytical Copilot)**.

---

## 🎯 Mục tiêu & Điểm nổi bật

* **Phần 1 – Mô tả dữ liệu (10%):** Giải thích đầy đủ nguồn dữ liệu, quy mô (`9.994 × 21`), ý nghĩa 21 biến (Data Dictionary tích hợp ngay trên trang chủ) và kiểu dữ liệu thực tế.
* **Phần 2 – Khám phá & Xử lý dữ liệu / EDA (25%):** Tự động kiểm tra giá trị thiếu (`0` Null), loại bỏ `1` bản ghi trùng lặp nghiệp vụ (`Row ID = 3407`), chuẩn hóa ngày tháng (`datetime64`) & mã bưu chính (`5` ký tự), phát hiện/gắn cờ ngoại lệ theo quy tắc `1.5 × IQR` (kèm tùy chọn *IQR Winsorize*), lọc dữ liệu đa chiều, tổng hợp `GroupBy` và `Pivot Table`.
* **Phần 3 – Trực quan hóa dữ liệu (30%):** Đáp ứng đầy đủ 4 nhóm biểu đồ bắt buộc (**So sánh** Bar/Line, **Phân phối** Histogram/Boxplot, **Xu hướng theo thời gian**, **Tổng hợp** Grouped/Stacked Bar) cùng **Bản đồ Choropleth Hoa Kỳ**, **Sunburst / Treemap**, **Biểu đồ kép Ngưỡng Chiết khấu** và **YoY Waterfall**.
* **Phần 4 – Dashboard tương tác Streamlit (25%):**
  * **Executive Command Center (`app.py`):** 6 thẻ KPI kèm tăng trưởng YoY, biểu đồ kết hợp Doanh thu & Biên lợi nhuận theo Quý, đối sánh Top 5 bang lãi cao nhất vs. lỗ nặng nhất, điều hướng nhanh 5 phân hệ và nạp tệp CSV tùy chỉnh (`st.file_uploader`).
  * **Kiến trúc chuẩn SaaS (`data/`, `components/`, `pages/`, `utils/`):** Sidebar đa điều kiện đồng bộ bằng `st.session_state` xuyên suốt các trang, `Dynamic Auto-Insight Banner` tự động cập nhật theo bộ lọc, so sánh đối chứng **A/B Benchmark** (Năm A vs. Năm B, Vùng A vs. Vùng B), xuất báo cáo **HTML/PDF Executive Report** và **Trợ lý AI B2B (Gemini 2.5 Flash / Oxion AI + Local B2B Copilot hoạt động ngay cả khi không cần API Key)**.
* **Phần 5 – Nhận xét, Insight & 5 Mô-đun Khoa học – Kinh tế lượng (10%+ Điểm cộng):**
  1. **Kiểm định giả thuyết thống kê & Tương quan phi tham số (Page 1):** Kiểm định tham số Welch's $t$-test ($t = 16.14, p = 2.44 \times 10^{-54}$, $\text{Cohen's } d = 0.64$), Mann-Whitney $U$, One-way ANOVA, Kruskal-Wallis ($H = 227.84, p = 4.06 \times 10^{-49}$) và đối sánh tương quan Pearson ($r = -0.2195$) vs. Spearman ($\rho = -0.5432$).
  2. **Độ co giãn của cầu theo giá ($\epsilon$) & Đường cong tối ưu lợi nhuận (Page 5):** Mô phỏng tác động sản lượng $Q_{\text{mới}} = Q_{\text{cũ}} \times (1 + \epsilon \cdot \%\Delta P)$ và vẽ đường cong tối ưu lợi nhuận qua các mức trần chiết khấu `5% → 50%`.
  3. **Mô hình Học máy Giải thích Rủi ro Đơn hàng Thua lỗ — Explainable ML (Page 5):** Huấn luyện Random Forest ($\text{ROC-AUC} = 0.9770, \text{F1} = 0.8300, \text{Accuracy} = 94.45\%$) và Logistic Regression ($\text{ROC-AUC} = 0.9846$), trích xuất Feature Importance & Odds Ratio (`Discount` chiếm `71.51%` quyền năng dự báo lỗ).
  4. **Kiểm định tính dừng chuỗi thời gian (ADF Test) & Phân rã cổ điển (Page 2):** Kiểm định nghiệm đơn vị Augmented Dickey-Fuller trên chuỗi gốc ($d=0$) và sai phân bậc 1 ($\Delta Y_t, d=1, \text{ADF} = -8.0522, p < 0.0001$), phân rã 4 tầng $Y_t = T_t + S_t + \varepsilon_t$ và chấm sai số Hold-out 6 tháng ($\text{MAPE} = 16.40\% - 17.50\%$).
  5. **Xác suất sống sót khách hàng $P(\text{Alive})$, Chu kỳ mua lặp lại (IPT) & CLV 12 tháng (Page 1):** Định lượng xác suất còn hoạt động trung bình (`62.6%`), tổng CLV dự báo 12 tháng (`$475,148.00`) và doanh thu có nguy cơ mất (`$287,793.41`).

---

## 🏗️ Cấu trúc thư mục (Clean Architecture)

```text
BTL_TQVPTDL/
├── Sample - Superstore.csv              # Tập dữ liệu gốc (9.994 dòng, 21 cột)
├── app.py                               # Trang chủ Executive Command Center (6 KPI YoY, Quý, Top/Bottom States, Data Dictionary)
├── eda_superstore.py                    # Script EDA: audit, làm sạch, lọc, GroupBy, Pivot, RFM, Apriori, Forecast
├── requirements.txt                     # Danh sách thư viện phụ thuộc (đã ghim phiên bản ==)
├── .gitignore                           # Quy tắc bỏ qua tệp tạm / môi trường ảo
├── .streamlit/
│   └── config.toml                      # Cấu hình theme giao diện Streamlit
├── data/
│   ├── __init__.py
│   ├── loader.py                        # Đọc CSV, nạp CSV tùy chỉnh, audit, làm sạch, Data Dictionary và @st.cache_data
│   └── analytics.py                     # GroupBy, Pivot, SARIMAX/Prophet, Kiểm định thống kê, Độ co giãn cầu, ML Rủi ro lỗ, ADF, CLV
├── components/
│   ├── __init__.py
│   ├── sidebar.py                       # Sidebar toàn cục đồng bộ st.session_state + Nạp CSV tùy chỉnh
│   └── ai_chatbot.py                    # Chatbot B2B Gemini 2.5 Flash / Oxion AI + Local B2B Analytical Copilot
├── utils/
│   ├── __init__.py
│   ├── formatters.py                    # Định dạng tiền tệ/số và ánh xạ 51 mã tiểu bang Hoa Kỳ
│   ├── session_state.py                 # Quản lý state toàn cục (ss_date_range, ss_regions, ...)
│   └── ui_styles.py                     # Giao diện SaaS CSS & Dynamic Auto-Insight Banner
├── pages/
│   ├── 1_Tong_Quan_Kinh_Doanh.py        # Page 1: KPI, Luồng Sankey, Histogram/Boxplot, Kiểm định Thống kê, RFM + CLV & Churn Scorecard
│   ├── 2_Xu_Huong_Doanh_Thu.py          # Page 2: Line + MA-3, Dự báo SARIMAX/Prophet, Hold-out MAE/MAPE, ADF + Decomposition & A/B Năm
│   ├── 3_Phan_Tich_Khu_Vuc.py           # Page 3: Radar Chart 4 Vùng, Bản đồ Mỹ, Khoan sâu Texas, A/B Benchmark & State Quadrant
│   ├── 4_Phan_Tich_San_Pham.py          # Page 4: Network Graph bán kèm, Top N, Sunburst/Treemap, Apriori & Pareto 80/20 (ABC)
│   └── 5_Trai_Nghiem_Nguoi_Dung.py      # Page 5: Isolation Forest bất thường đa biến, What-If (ε), ML Rủi ro lỗ & Trợ lý AI B2B
├── tests/
│   └── test_analytics.py                # 6 bộ Unit tests kiểm thử toàn diện (pytest)
├── .github/workflows/
│   └── ci.yml                           # GitHub Actions kiểm tra cú pháp & chạy unit tests tự động
├── notebooks/
│   └── 01_EDA_Superstore.ipynb          # Notebook EDA & Trực quan hóa đã chạy sẵn đầy đủ output (12 code cells)
├── powerbi/
│   ├── Huong_dan_PowerBI.md             # Hướng dẫn từng bước dựng file .pbix cho 4 trang
│   ├── power_query.m                    # Script Power Query làm sạch Sample - Superstore.csv
│   ├── measures.dax                     # Bộ công thức DAX Measures chuẩn
│   └── theme_superstore.json            # Theme màu sắc & bo góc dùng chung cho Power BI
└── docs/
    ├── Bao_cao_Du_an.md                 # Tài liệu báo cáo BTL đầy đủ 5 phần
    ├── Slide_Thuyet_Trinh.md            # Dàn ý & nội dung Slide thuyết trình (14 trang)
    └── Slide_Thuyet_Trinh.html          # Slide HTML trình chiếu / xuất PDF trực tiếp trên trình duyệt
```

---

## 👥 Phân công nhóm thực hiện (4 thành viên)

| Trang Dashboard | Tệp mã nguồn | Thành viên | Yêu cầu trực quan hóa & Tính năng chính |
| :--- | :--- | :--- | :--- |
| **Page 1** | [`pages/1_Tong_Quan_Kinh_Doanh.py`](pages/1_Tong_Quan_Kinh_Doanh.py) | *[Sinh viên 1]* | KPI (`st.metric`), **Luồng giá trị Sankey**, **Grouped/Stacked Bar**, **Kiểm định giả thuyết ($t$-test, ANOVA, Kruskal)**, **RFM + CLV 12M** & **Thẻ điểm Cảnh báo Churn Risk** |
| **Page 2** | [`pages/2_Xu_Huong_Doanh_Thu.py`](pages/2_Xu_Huong_Doanh_Thu.py) | *[Sinh viên 2]* | **Xu hướng thời gian** (Line + MA-3), **Dự báo SARIMAX/Prophet**, **Hold-out MAE/MAPE**, **Kiểm định tính dừng ADF & Phân rã 4 tầng**, **Seasonality**, **YoY Waterfall** & **A/B Year Benchmark** |
| **Page 3** | [`pages/3_Phan_Tich_Khu_Vuc.py`](pages/3_Phan_Tich_Khu_Vuc.py) | *[Sinh viên 3]* | **Radar Chart 4 Vùng**, **So sánh khu vực** (Bar), `st.multiselect`, **Choropleth Hoa Kỳ**, **Khoan sâu Central & Texas**, **A/B Region Benchmark**, **Shipping SLA** & **State Quadrant** |
| **Page 4 & 5** | [`pages/4_Phan_Tich_San_Pham.py`](pages/4_Phan_Tich_San_Pham.py)<br>[`pages/5_Trai_Nghiem_Nguoi_Dung.py`](pages/5_Trai_Nghiem_Nguoi_Dung.py) | *[Sinh viên 4]* *(cùng nhóm tích hợp)* | **Network Graph Bán kèm**, Top N, **Sunburst**, **Apriori**, **Isolation Forest Bất thường Đa biến**, **What-If + Độ co giãn cầu ($\epsilon$)**, **ML Dự báo Rủi ro Lỗ**, **Xuất Báo cáo HTML/PDF** & **Trợ lý AI B2B** |

---

## ⚙️ Hướng dẫn cài đặt và khởi chạy

### 1. Clone repository và tạo môi trường ảo
```bash
git clone https://github.com/trananhpham/BTL_TQVPTDL.git
cd BTL_TQVPTDL

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Cấu hình Gemini API Key (tùy chọn cho Chatbot AI trên Page 5)
```powershell
# Windows PowerShell:
$env:GEMINI_API_KEY="your_gemini_api_key_here"
```
*(Hoặc nhập trực tiếp API Key vào ô mật khẩu trên giao diện Tab **Trợ lý AI Phân tích B2B** tại Page 5).*

### 3. Chạy kiểm tra EDA & Khởi chạy Dashboard
```bash
# Chạy script EDA (kiểm tra chất lượng, làm sạch, GroupBy, Pivot, RFM, Apriori, Forecast)
python eda_superstore.py

# Khởi chạy ứng dụng Streamlit Dashboard
streamlit run app.py
```

### 4. Triển khai lên Streamlit Community Cloud (Tùy chọn)
Dashboard có thể triển khai trực tiếp lên **Streamlit Community Cloud** (miễn phí):
1. Đẩy mã nguồn lên kho lưu trữ GitHub cá nhân hoặc tổ chức.
2. Truy cập [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub.
3. Chọn Repository, Branch `main`, và Main file path là `app.py`.
4. Tại mục **Advanced settings** > **Secrets**, cấu hình API Key cho Chatbot Gemini (tuyệt đối không ghi key vào code):
   ```toml
   GEMINI_API_KEY = "your_gemini_api_key_here"
   ```
5. Bấm **Deploy!** Đường dẫn dashboard trực tuyến có thể gắn vào slide hoặc nộp cho giảng viên.

---

## 📚 Tài liệu & Sản phẩm nộp kèm
* **Báo cáo chi tiết 5 phần:** [`docs/Bao_cao_Du_an.md`](docs/Bao_cao_Du_an.md)
* **Notebook EDA tương tác:** [`notebooks/01_EDA_Superstore.ipynb`](notebooks/01_EDA_Superstore.ipynb)
* **Slide thuyết trình (14 trang):** [`docs/Slide_Thuyet_Trinh.md`](docs/Slide_Thuyet_Trinh.md) | Trình chiếu trực tiếp / Xuất PDF: [`docs/Slide_Thuyet_Trinh.html`](docs/Slide_Thuyet_Trinh.html)
