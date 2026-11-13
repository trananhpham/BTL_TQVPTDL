# 📊 DÀN Ý & NỘI DUNG SLIDE THUYẾT TRÌNH CHUYÊN SÂU (16 SLIDES)
**Đề tài:** Hệ thống Phân tích Dữ liệu Kinh doanh, Kinh tế lượng & Trí tuệ Nhân tạo (BI & AI Dashboard) trên Tập dữ liệu Bán lẻ Superstore Hoa Kỳ  
**Repository:** [https://github.com/trananhpham/BTL_TQVPTDL.git](https://github.com/trananhpham/BTL_TQVPTDL.git)

---

### Slide 1: Trang bìa (Title Slide)
* **Tên đề tài:** Phân tích Dữ liệu Kinh doanh, Kinh tế lượng & Xây dựng Hệ thống BI – AI Dashboard Tương tác trên Tập dữ liệu Superstore
* **Công nghệ cốt lõi:** Python, Pandas, SciPy, Statsmodels, Scikit-learn, Plotly, Streamlit, Gemini 2.5 Flash / Oxion AI
* **Quy mô triển khai:** Trang chủ **Executive Command Center** + **5 Phân hệ Chuyên sâu** + **5 Mô-đun Khoa học & Kinh tế lượng**
* **Nhóm thực hiện:** 4 Thành viên (Phân công phụ trách toàn diện từ Data Pipeline, EDA, Modeling đến UI/UX & AI)

---

### Slide 2: Bài toán Kinh doanh & 5 Câu hỏi Nghiên cứu Định lượng (Business Problem & RQs)
* **Bối cảnh thực tiễn:** Chuỗi bán lẻ đa kênh Superstore vận hành trên `49` tiểu bang Hoa Kỳ (2014–2017) với doanh thu tăng trưởng đều đặn nhưng **biên lợi nhuận bị bào mòn nghiêm trọng** ở nhiều khu vực và ngành hàng.
* **5 Câu hỏi Nghiên cứu cốt lõi (Research Questions):**
  1. **RQ1 (Thống kê suy luận):** Ngưỡng chiết khấu nào là "điểm gãy" gây thua lỗ, và sự khác biệt lợi nhuận giữa các nhóm chiết khấu/khu vực có ý nghĩa thống kê ($p\text{-value}$, $\text{Cohen's } d$) hay không?
  2. **RQ2 (Địa lý & Danh mục):** Tại sao vùng `Central` (đặc biệt bang **Texas**) và ngành hàng `Furniture` (`Tables`, `Bookcases`) có doanh thu cao nhưng lỗ nặng?
  3. **RQ3 (Chuỗi thời gian):** Doanh thu tháng có tính dừng (Stationarity) không, cấu trúc mùa vụ ra sao và mô hình nào dự báo chính xác nhất trên tập kiểm thử giữ lại (Hold-out 6 tháng)?
  4. **RQ4 (Khách hàng & Học máy):** Làm sao phân cụm khách hàng khách quan (Elbow/Silhouette) và định lượng xác suất sống sót $P(\text{Alive})$, giá trị vòng đời CLV 12 tháng cùng doanh thu có nguy cơ mất (Revenue at Risk)?
  5. **RQ5 (Kinh tế lượng & Tối ưu hóa):** Khi áp trần chiết khấu `20%` trong điều kiện cầu co giãn theo giá ($\epsilon < 0$), lợi nhuận ròng kỳ vọng thay đổi bao nhiêu USD?

---

### Slide 3: Mô tả Tập dữ liệu & Từ điển 21 Biến (Dataset Overview)
* **Nguồn dữ liệu:** Kaggle — *Sample - Superstore Dataset* (`9.994` dòng × `21` cột nguyên bản + `8` biến phái sinh).
* **Quy mô tổng thể (sau làm sạch `9.993` dòng):**
  * **Tổng doanh thu:** `$2,297,200.86` | **Tổng lợi nhuận:** `$286,397.02` | **Biên lợi nhuận ròng:** `12.47%`
  * **Tổng đơn hàng:** `5.009` đơn (`AOV = $458.61/đơn`) | **Khách hàng:** `793` | **Sản phẩm:** `1.862`
* **Cấu trúc Từ điển Dữ liệu (Data Dictionary tích hợp trực tiếp trên Trang chủ `app.py`):**
  * *Định danh & Thời gian:* `Row ID`, `Order ID`, `Order Date`, `Ship Date`, `Ship Mode` (+ phái sinh `Shipping_Days`, `Order_Year`, `Order_Quarter`).
  * *Khách hàng & Địa lý:* `Customer ID`, `Customer Name`, `Segment` (`Consumer`, `Corporate`, `Home Office`), `Country`, `Region`, `State`, `City`, `Postal Code` (+ `State_Code` chuẩn ANSI 2 ký tự).
  * *Sản phẩm & Tài chính:* `Product ID`, `Category`, `Sub-Category` (`17` nhóm), `Product Name`, `Sales`, `Quantity`, `Discount`, `Profit` (+ `Implied_Cost`, `Gross_Price`, `Profit_Margin_Pct`).

---

### Slide 4: Quy trình Kiểm toán Chất lượng & Tiền xử lý Dữ liệu (EDA — 25%)
* **Tự động nhận diện bảng mã & Kiểm tra Null:** Đọc tệp đa bảng mã (`utf-8` / `cp1252`), xác nhận **`0` giá trị khuyết thiếu (Missing/Null)** trên toàn bộ 21 cột.
* **Khử trùng lặp khóa nghiệp vụ (Deduplication):** Phát hiện và loại bỏ **`01` bản ghi trùng lặp nghiệp vụ** tại đơn hàng `US-2014-150119` (`Row ID = 3407`, ghế văn phòng tại bang Ohio ngày `23/04/2014`), giữ lại **`9.993` dòng chuẩn**.
* **Chuẩn hóa dữ liệu & Kiểm tra logic giao vận:**
  * Chuyển đổi `Order Date`, `Ship Date` sang `datetime64`; kiểm tra `100%` đơn hàng thỏa mãn `Ship Date ≥ Order Date` (thời gian giao hàng trung bình `3.96 ngày`).
  * Chuẩn hóa `Postal Code` đủ 5 chữ số (`zfill(5)`) cho các bang vùng Đông Bắc Hoa Kỳ.
* **Phát hiện Ngoại lệ ($1.5 \times \text{IQR}$) & Lý giải Nghiệp vụ B2B:**
  * Phát hiện `1.167` ngoại lệ `Sales` (`>$498.93`) và `1.881` ngoại lệ `Profit` (`<-$39.72` hoặc `>+$70.82`).
  * **Quyết định khoa học:** Không xóa ngoại lệ vì đây là các hợp đồng B2B quy mô lớn hoặc các đơn xả kho lỗ thật; trang bị nút bật/tắt **Chặn trần ngoại lệ (IQR Winsorize)** trên Sidebar để phân tích linh hoạt.

---

### Slide 5: Tổng hợp Đa chiều với GroupBy & Pivot Table
* **GroupBy theo Năm (2014–2017):**
  * `2014`: Doanh thu `$484.2K` | Lợi nhuận `$49.5K` (Biên LN `10.23%`)
  * `2015`: Doanh thu `$470.5K` (điều chỉnh nhẹ `-2.83%`) nhưng **Lợi nhuận tăng vọt `+24.3%` lên `$61.6K`** (Biên LN `13.10%`)
  * `2016`: Doanh thu `$609.2K` (`+29.5%`) | Lợi nhuận `$81.8K` (Biên LN `13.43%`)
  * `2017`: Doanh thu đạt đỉnh **`$733.2K`** (`+20.4%`) | Lợi nhuận **`$93.4K`** (`1.687` đơn hàng)
* **GroupBy theo Khu vực (`Region`) & Ngành hàng (`Category`):**
  * Hai vùng `West` (`$725.5K`, biên `14.94%`) và `East` (`$678.8K`, biên `13.48%`) tạo ra **69.7% tổng lợi nhuận**; vùng `Central` có biên lợi nhuận thấp nhất (`7.92%`).
  * Ngành `Technology` (`+$145.5K`) và `Office Supplies` (`+$122.5K`) sinh lời cao; riêng **`Furniture` đạt doanh thu `$742.0K` nhưng lợi nhuận chỉ `$18.5K` (biên `2.49%`)**.
* **Pivot Table 2 chiều (`Region × Category` & `Year × Segment`):** Được lưu đệm bằng `@st.cache_data`, cho phép đối chiếu nhanh ma trận đóng góp doanh thu và đơn hàng theo phân khúc.

---

### Slide 6: Kiến trúc Hệ thống Clean Architecture & Executive Command Center
* **Kiến trúc phần mềm tách biệt 4 tầng chuẩn SaaS:**
  1. `data/`: `loader.py` (Đọc CSV gốc & **Nạp tệp CSV tùy chỉnh qua `st.file_uploader`**, kiểm toán, làm sạch, `@st.cache_data`) & `analytics.py` (Hơn 25 hàm phân tích, kinh tế lượng, chuỗi thời gian và học máy).
  2. `components/`: `sidebar.py` (Bộ lọc toàn cục 5 chiều đồng bộ `st.session_state`) & `ai_chatbot.py` (Trợ lý AI B2B đa luồng).
  3. `utils/`: `formatters.py`, `session_state.py` & `ui_styles.py` (Giao diện SaaS hiện đại + **Dynamic Auto-Insight Banner** tự động tóm tắt điểm sáng/điểm nghẽn theo mọi bộ lọc).
  4. `app.py` & `pages/1..5`:
     * **Trang chủ `app.py` (Executive Command Center):** 6 thẻ KPI kèm tăng trưởng YoY, biểu đồ kết hợp Doanh thu & Biên lợi nhuận theo Quý, biểu đồ đối sánh Top 5 bang lãi cao nhất vs. Top 5 bang lỗ nặng nhất, điều hướng nhanh 5 trang và 3 Tab Kiểm toán / Từ điển 21 cột / Thống kê `.describe()`.

---

### Slide 7: Page 1 — Tổng quan Kinh doanh, Phân phối & Kiểm định Giả thuyết Thống kê
* **Trực quan hóa Tổng quan & Phân phối:**
  * 6 thẻ KPI cốt lõi (`st.metric`), biểu đồ Donut tỷ trọng ngành hàng, biểu đồ **Grouped / Stacked Bar** (`Segment × Category`) và cặp biểu đồ **Histogram + Boxplot** phân tích phân phối lệch phải & ngoại lệ.
* **Điểm nhấn Khoa học 1 — Bộ Kiểm định Giả thuyết Thống kê Tham số & Phi tham số:**
  * **Kiểm định lợi nhuận nhóm `Discount ≤ 20%` vs. `Discount > 20%`:**
    * Tham số (**Welch's $t$-test**): $t = \mathbf{16.14}$, $p = \mathbf{2.44 \times 10^{-54}}$.
    * Phi tham số (**Mann-Whitney $U$**): $U = \mathbf{11,473,490}$, $p < 10^{-100}$.
    * Độ lớn hiệu ứng (**Effect Size**): $\text{Cohen's } d = \mathbf{0.64}$ → **Bác bỏ hoàn toàn $H_0$**, chứng minh chiết khấu $>20\%$ hủy hoại lợi nhuận có ý nghĩa thống kê cực mạnh.
  * **Kiểm định khác biệt giữa 4 Khu vực (`Region`):** One-way ANOVA đạt $F = 2.62$ ($p = 0.0488$), nhưng kiểm định hạng **Kruskal-Wallis đạt $H = \mathbf{227.84}$ ($p = \mathbf{4.06 \times 10^{-49}}$)** — khẳng định tính ưu việt của kiểm định phi tham số trên dữ liệu B2B nhiều ngoại lệ.
  * **Đối sánh Tương quan Pearson ($r$) vs. Spearman ($\rho$):** Cặp `Discount ↔ Profit` có $r = -0.2195$ nhưng **$\rho = \mathbf{-0.5432}$ ($p < 10^{-100}$)** → Quan hệ đơn điệu phi tuyến (gãy khúc tại ngưỡng 20%).

---

### Slide 8: Page 1 (Tiếp) — Phân khúc RFM, K-Means 3D & Dự báo Xác suất Sống sót $P(\text{Alive})$ / CLV
* **Phân khúc RFM & Ma trận Giữ chân Cohort Retention:**
  * Phân loại `793` khách hàng thành 6 nhóm hành vi (*VIP / Trung thành, Khách hàng tiềm năng, Khách hàng mới, Cần kích hoạt lại, Nguy cơ rời bỏ*) và theo dõi tỷ lệ quay lại mua sắm qua từng năm gia nhập (Cohort Heatmap).
* **Phân cụm Học máy K-Means 3D có Biện luận Chọn số cụm $k$:**
  * Chuẩn hóa Z-score không gian 3 chiều $(\text{Recency}, \ln(1+\text{Frequency}), \ln(1+\text{Monetary}))$.
  * Vẽ cặp biểu đồ **Elbow (Inertia)** và **Silhouette Score** ($k = 2 \to 6$) để chứng minh cơ sở toán học khi chọn số cụm tối ưu trên biểu đồ 3D tương tác.
* **Điểm nhấn Khoa học 2 — Mô hình Xác suất Sống sót $P(\text{Alive})$, CLV 12 Tháng & Doanh thu Rủi ro:**
  * Ước lượng chu kỳ mua lặp lại trung bình (**Inter-Purchase Time — IPT**): **`128.8 ngày`**.
  * Xác suất khách hàng còn hoạt động trung bình **$P(\text{Alive}) = (1 + \text{Recency}/\text{Lifespan})^{-\text{Frequency}}$**: **`62.6%`**.
  * **Tổng CLV dự phóng 12 tháng tới:** **`$475,148.00`** | **Tổng Doanh thu có nguy cơ mất (Revenue at Risk):** **`$287,793.41`**.

---

### Slide 9: Page 2 — Phân tích Chuỗi thời gian, Kiểm định ADF & Dự báo Doanh thu (SARIMAX / Prophet)
* **Xu hướng Lịch sử, Mùa vụ & So sánh Đối chứng A/B Năm:**
  * Biểu đồ đường đa chuỗi kèm trung bình trượt `MA-3`, **Seasonality Heatmap (Năm × Tháng)** nhận diện đỉnh mua sắm Quý 4 (tháng 9, 11, 12), biểu đồ thác nước **YoY Waterfall** và bảng **A/B Year Benchmark** đối chiếu trực tiếp 2 năm bất kỳ.
* **Điểm nhấn Khoa học 3 — Kiểm định Tính dừng ADF & Phân rã Cổ điển 4 Tầng ($Y_t = T_t + S_t + \varepsilon_t$):**
  * Kiểm định nghiệm đơn vị **Augmented Dickey-Fuller (ADF)**: Chuỗi gốc ($d=0$) đạt $\text{ADF} = -4.4936$ ($p = 0.0002$); chuỗi sai phân bậc 1 ($\Delta Y_t, d=1$) đạt **$\text{ADF} = -8.0522$ ($p < 0.0001$, vượt xa ngưỡng tới hạn 5% là `-2.9283`)**, chứng minh cơ sở toán học của bậc sai phân $d=1$.
  * Biểu đồ 4 tầng tách biệt rõ **Observed – Trend – Seasonal – Residual**.
* **Đánh giá Sai số Dự báo Thực nghiệm trên Tập giữ lại (Out-of-sample Hold-out 6 Tháng):**
  * Tách 6 tháng cuối làm tập kiểm thử độc lập: mô hình **`SARIMAX(1,1,0)(1,1,0,12)`** tối ưu bằng thuật toán Powell đạt **$\text{MAE} = \$14,715.73, \text{MAPE} = 17.50\%$**; mô hình **`Seasonal Linear Trend`** đạt **$\text{MAE} = \$13,417.35, \text{MAPE} = 16.40\%$** với dải tin cậy 95% ổn định.

---

### Slide 10: Page 3 — Phân tích Khu vực, Bản đồ Choropleth Hoa Kỳ & Khoan sâu Central / Texas
* **Trực quan hóa Không gian Địa lý & Vận hành Giao vận SLA:**
  * Biểu đồ cột so sánh vùng + bộ lọc `st.multiselect` + **Bản đồ Choropleth Hoa Kỳ 49 tiểu bang** (`locationmode="USA-states"`).
  * Phân tích hiệu suất giao vận **Shipping SLA** theo 4 phương thức và biểu đồ **Ma trận Tứ phân vị Tiểu bang (State Quadrant: Sales vs. Profit Margin)** kèm bảng **A/B Region Benchmark**.
* **Khoan sâu Nguyên nhân Gốc rễ tại Khu vực `Central` & Tiểu bang `Texas`:**
  * Tại sao vùng `Central` có biên lợi nhuận thấp nhất (`7.92%`) dù các bang như **Michigan** (`+$24,463`, chiết khấu `0.7%`) và **Indiana** (`+$18,383`, chiết khấu `0%`) lãi rất cao?
  * **Thủ phạm chính là bang Texas:** Doanh thu đứng thứ 3 toàn quốc (`$170,188`) nhưng **lỗ ròng nặng nhất toàn quốc `-$25,729.36`** (biên LN `-15.12%`, tỷ lệ đơn lỗ `49.54%`) do mức chiết khấu trung bình tại Texas lên tới **`37.01%`**, tập trung vào nhóm `Binders` (lỗ `-$14,705`, CK `80%`), `Appliances` (lỗ `-$6,147`, CK `80%`) và `Furnishings` (lỗ `-$3,312`, CK `60%`).

---

### Slide 11: Page 4 — Danh mục Sản phẩm (Top N), Sunburst 3 Tầng, Pareto 80/20 & Giỏ hàng Apriori
* **Xếp hạng Top N & Biểu đồ Phân cấp Sunburst / Treemap:**
  * Lọc động Top N sản phẩm doanh thu cao nhất vs. lợi nhuận cao nhất; khoan sâu 3 cấp (`Category → Sub-Category → Product Name`) tô màu trực tiếp theo biên lợi nhuận.
  * Nhận diện 3 nhóm hàng lỗ ròng toàn cục: **`Tables` (`-$17,725.48`, biên `-8.56%`)**, **`Bookcases` (`-$3,472.56`, biên `-3.02%`)** và **`Supplies` (`-$1,189.10`)**; đối lập với nhóm siêu lợi nhuận **`Copiers` (`+$55,617.82`, biên `+37.20%`)** và **`Phones` (`+$44,515.73`)**.
* **Phân tích Pareto 80/20 (Phân hạng ABC):**
  * Tự động phân loại sản phẩm/nhóm hàng thành **Nhóm A** (tạo ra 80% doanh thu cốt lõi), **Nhóm B** (15% kế tiếp) và **Nhóm C** (5% cuối) để tối ưu vốn lưu động và tồn kho.
* **Khai phá Luật kết hợp Giỏ hàng (Market Basket Analysis — Apriori):**
  * Tính toán $\text{Support}, \text{Confidence}, \text{Lift}$ trên các hóa đơn mua từ 2 nhóm hàng trở lên và vẽ **Đồ thị Mạng lưới Bán chéo (Cross-selling Network Graph)**, gợi ý ghép gói combo giữa nhóm hàng chủ lực và phụ kiện văn phòng.

---

### Slide 12: Page 5 — Phát hiện Trọng tâm: Điểm gãy Chiết khấu 20% & Học máy Giải thích Rủi ro Lỗ
* **Phân tích Thực nghiệm theo 5 Ngưỡng Chiết khấu Độc lập (Biểu đồ Kép Bar – Line):**
  | Ngưỡng Chiết khấu | Số dòng | Doanh thu ($) | Lợi nhuận ($) | Lợi nhuận TB ($/đơn) | Tỷ lệ đơn lỗ (%) | Biên LN (%) |
  | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
  | **0% (Không giảm)** | 4,798 | $1,087,908 | +$320,988 | **+$66.90** | **0.00%** | +29.51% |
  | **>0–10%** | 94 | $54,369 | +$9,029 | **+$96.06** | 4.26% | +16.61% |
  | **>10–20%** | 3,709 | $792,153 | +$91,756 | **+$24.74** | 13.99% | +11.58% |
  | **>20–30%** | 226 | $102,945 | -$10,357 | **-$45.83** | **91.59%** | -10.06% |
  | **>30%** | 1,166 | $259,544 | -$125,007 | **-$107.21** | **97.77%** | -48.16% |
* **Điểm nhấn Khoa học 4 — Học máy Giải thích Rủi ro Đơn hàng Thua lỗ (Explainable ML):**
  * Huấn luyện mô hình phân loại nhị phân $P(\text{Loss}=1 \mid X)$ trên tập Train/Test phân tầng 80/20:
    * **Random Forest Classifier:** $\text{ROC-AUC} = \mathbf{0.9770}$, $\text{F1-Score} = \mathbf{0.8300}$, $\text{Test Accuracy} = \mathbf{94.45\%}$.
    * **Logistic Regression:** $\text{ROC-AUC} = \mathbf{0.9846}$.
  * **Trích xuất Feature Importance:** Biến **`Discount` chiếm tới `71.51%`** tổng quyền năng quyết định rủi ro thua lỗ của một đơn hàng!

---

### Slide 13: Page 5 (Tiếp) — Mô phỏng What-If Có Xét Độ Co Giãn Cầu ($\epsilon$) & Đường cong Tối ưu Lợi nhuận
* **Vượt qua hạn chế của mô phỏng tĩnh thông thường:**
  * Trong kinh tế học thực tế, khi doanh nghiệp cắt giảm chiết khấu (tức giá bán thực tế $P$ tăng lên), sản lượng tiêu thụ $Q$ và tổng giá vốn khả biến sẽ co giãn theo công thức:
    $$Q_{\text{mới}} = Q_{\text{cũ}} \times \left(1 + \epsilon \cdot \frac{\Delta P}{P}\right)$$
* **Điểm nhấn Khoa học 5 — Đường cong Tối ưu hóa Lợi nhuận (Profit Optimization Curve):**
  * Quét trần chiết khấu tối đa từ `5%` đến `50%` kết hợp tăng giá `+5%` nhóm hàng lỗ (`Tables`, `Bookcases`, `Supplies`) dưới các mức độ co giãn cầu $\epsilon \in [-1.5, 0.0]$.
  * **Kết quả định lượng:** Ngay cả khi cầu co giãn ở mức $\epsilon = -0.4$ đến $-0.5$ (sản lượng tiêu thụ giữ lại đạt `~92.6%`), việc áp dụng **trần chiết khấu `20%`** vẫn giúp doanh nghiệp phục hồi trên **`+$120,000 USD` lợi nhuận ròng** và nâng biên lợi nhuận toàn công ty từ `12.47%` lên trên **`18.0%`**.

---

### Slide 14: Page 5 (Tiếp) — Tối ưu Hiệu năng Cache, Xuất Báo cáo HTML/PDF & Trợ lý AI B2B
* **Tối ưu Trải nghiệm Người dùng (UX) & Hiệu năng:**
  * Đồng bộ xuyên suốt 5 trang bằng `st.session_state`; lưu đệm `@st.cache_data` giúp thời gian nạp dữ liệu đạt **`< 5 ms`**.
  * Tích hợp **Tra cứu Hồ sơ 360°** (tìm kiếm tức thì theo Khách hàng, Mã đơn, Sản phẩm, Tiểu bang) và **Trung tâm Cảnh báo Đơn lỗ Nghiêm trọng**.
* **Xuất Báo cáo Đa định dạng (1-Click Executive Export):**
  * Cho phép tải dữ liệu sạch `.CSV`, bản tóm tắt `.MD` và **Báo cáo Điều hành chuẩn HTML/PDF (In trực tiếp bằng `Ctrl+P`)**.
* **Trợ lý AI Phân tích B2B Đa chế độ (`Gemini 2.5 Flash` / `Oxion AI` + `Local B2B Analytical Copilot`):**
  * Tự động bơm ngữ cảnh số liệu thực tế từ DataFrame đang lọc vào `System Prompt` (Dynamic Context Injection).
  * Trang bị nút **⚡ Tạo Bản Tóm tắt Chiến lược Toàn diện (1-Click AI Executive Brief)** và cơ chế **Local B2B Analytical Copilot** trả lời chính xác mọi câu hỏi về Texas/Central, chiết khấu, khách hàng RFM, sản phẩm ngay cả khi không cần khóa API ngoài.

---

### Slide 15: Lộ trình Khuyến nghị Chiến lược 30 – 60 – 90 Ngày
1. **Giai đoạn 30 ngày (Chặn đứng rò rỉ lợi nhuận từ chiết khấu):**
   * Thiết lập quy tắc khóa trần chiết khấu tối đa **`20%`** trên hệ thống bán hàng; mọi đơn hàng chiết khấu `>20%` bắt buộc phải qua phê duyệt cấp Giám đốc Vùng kèm điều kiện giá trị đơn hàng tối thiểu.
2. **Giai đoạn 60 ngày (Tái cấu trúc thị trường vùng `Central` — trọng tâm bang `Texas` & `Illinois`):**
   * Chấm dứt ngay các chương trình xả hàng giảm giá `60% – 80%` đối với nhóm `Binders` và `Appliances` tại Texas; chuẩn hóa biểu giá theo mô hình thành công của bang Michigan và Indiana.
3. **Giai đoạn 60–90 ngày (Tái định giá ngành `Furniture` & Bán chéo theo Apriori):**
   * Điều chỉnh tăng giá cơ sở `+5%` hoặc tách phụ phí vận chuyển cồng kềnh đối với mặt hàng `Tables` và `Bookcases`; triển khai gói bán kèm (Bundle) với các sản phẩm `Technology` biên lợi nhuận cao (`Copiers`, `Phones`).
4. **Giai đoạn 90 ngày (Chiến dịch Giữ chân Khách hàng dựa trên $P(\text{Alive})$ & CLV):**
   * Ưu tiên phân bổ ngân sách Marketing chăm sóc các khách hàng thuộc nhóm có `Revenue at Risk` cao nhằm bảo vệ khoản doanh thu rủi ro **`$287,793.41`**.

---

### Slide 16: Tổng kết Đóng góp của Dự án & Q&A
* **Hoàn thành xuất sắc 100% yêu cầu Bài tập lớn & Mở rộng chiều sâu học thuật:**
  * ✅ **Quy trình EDA & Kiểm toán dữ liệu bài bản:** Xử lý chuẩn xác `9.993` bản ghi, giải quyết bài toán ngoại lệ B2B bằng cơ chế *IQR Winsorize* linh hoạt.
  * ✅ **Hệ thống Web BI & AI Dashboard 6 trang (`app.py` + `Page 1..5`):** Kiến trúc Clean Architecture 4 tầng, nạp CSV tùy chỉnh, Dynamic Insight Banner, A/B Benchmark và xuất báo cáo HTML/PDF.
  * ✅ **5 Mô-đun Khoa học & Kinh tế lượng:** Kiểm định giả thuyết thống kê ($t$-test, Kruskal-Wallis, Spearman), Độ co giãn cầu ($\epsilon$), Học máy giải thích rủi ro lỗ ($\text{ROC-AUC} = 0.9770$), Kiểm định tính dừng ADF + SARIMAX Hold-out, và Mô hình xác suất sống sót $P(\text{Alive})$ / CLV 12 tháng.
  * ✅ **Kiểm thử tự động & Đóng gói hoàn chỉnh:** Vượt qua `100%` bộ kiểm thử `pytest`, kèm Notebook EDA đã thực thi đầy đủ kết quả và Đề cương Báo cáo chi tiết 60 trang.
* **Cảm ơn Thầy/Cô và các bạn đã lắng nghe! Nhóm xin phép chuyển sang phần Demo trực tiếp trên Web Dashboard & Giải đáp câu hỏi (Q&A).**
