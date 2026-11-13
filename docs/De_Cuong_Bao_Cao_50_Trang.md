# 📘 ĐỀ CƯƠNG & HƯỚNG DẪN SOẠN THẢO BÁO CÁO BÀI TẬP LỚN (QUY MÔ 55 – 65 TRANG)
**Đề tài:** Hệ thống Phân tích Dữ liệu Kinh doanh, Kinh tế lượng & Trí tuệ Nhân tạo (BI & AI Dashboard) trên Tập dữ liệu Bán lẻ Superstore Hoa Kỳ  
**Quy tắc cốt lõi:**
1. **Tuyệt đối KHÔNG chụp ảnh màn hình code (No Code Screenshots)** và không dán các đoạn mã Python dài vào báo cáo.
2. **Thay thế hoàn toàn Code bằng 4 hình thức học thuật:**
   * **Sơ đồ khối / Lưu đồ thuật toán (Flowchart / Pipeline Diagram):** Vẽ bằng Draw.io, Visio hoặc Mermaid.
   * **Công thức Toán học & Thống kê (Equation Editor trong Word):** Viết tường minh công thức của IQR, Welch's $t$-test, ANOVA, Kruskal-Wallis, Pearson $r$, Spearman $\rho$, K-Means, SARIMAX, ADF Test, Độ co giãn cầu $\epsilon$, Random Forest/Logit, Apriori, RFM và $P(\text{Alive})$.
   * **Bảng số liệu định lượng (Data Tables):** Trích xuất trực tiếp từ Dashboard Streamlit và các bảng kết quả thống kê.
   * **Hình chụp Biểu đồ & Giao diện Dashboard (UI & Chart Figures):** Chụp ảnh sắc nét từng biểu đồ từ Streamlit (nền sáng hoặc tối đồng nhất), mỗi hình kèm **3 đoạn phân tích bắt buộc**: *(1) Mô tả cấu trúc biểu đồ; (2) Đọc số liệu cực trị & xu hướng; (3) Giải thích nguyên nhân gốc rễ & hàm ý quản trị*.

---

## 📏 QUY CHUẨN ĐỊNH DẠNG WORD ĐỂ ĐẠT 55 – 65 TRANG CHUẨN HỌC THUẬT
* **Khổ giấy:** A4 dọc; **Lề (Margins):** Trái `3.0 cm` (đóng gáy), Phải `2.0 cm`, Trên `2.5 cm`, Dưới `2.5 cm`.
* **Font chữ:** Times New Roman, cỡ chữ `13 pt` cho nội dung chính, giãn dòng **`1.3 – 1.5 lines`**, khoảng cách đoạn `After: 6 pt`.
* **Quy tắc trình bày Hình ảnh & Bảng biểu:**
  * Mỗi biểu đồ chụp từ Web Dashboard chiếm khoảng `40% – 50%` chiều cao trang A4, kèm chú thích `Hình X.Y: ...` ở dưới hình và bảng số liệu đi kèm ở trên/dưới.
  * Với 5 chương chính + Mở đầu + Kết luận + Phụ lục (gồm **28 hình biểu đồ/giao diện** và **24 bảng số liệu thống kê**), báo cáo tự nhiên đạt **56 – 64 trang** mà hoàn toàn không cần chèn code.

---

## 🗺️ BẢNG PHÂN BỔ SỐ TRANG & PHÂN CÔNG NHÓM (4 THÀNH VIÊN)

| Phần / Chương | Tên Chương / Nội dung | Số trang mục tiêu | Người phụ trách gợi ý |
| :--- | :--- | :---: | :---: |
| **Phần đầu** | Bìa, Lời cảm ơn, Tóm tắt (Abstract), Mục lục, Danh mục Bảng/Hình, Danh mục Từ viết tắt | **6 trang** | Cả nhóm (Nhóm trưởng ghép) |
| **Mở đầu** | Tính cấp thiết, Mục tiêu, Câu hỏi nghiên cứu, Đối tượng & Phạm vi, Đóng góp mới | **4 trang** | Sinh viên 1 |
| **Chương 1** | **Tổng quan Cơ sở Lý thuyết & Mô tả Tập dữ liệu Superstore** (Từ điển 21 biến, Kiến trúc BI, Lý thuyết Thống kê – Kinh tế lượng – ML) | **10 trang** | Sinh viên 1 |
| **Chương 2** | **Quy trình Tiền xử lý, Kiểm toán Chất lượng & Khám phá Dữ liệu (EDA)** (Kiểm toán, Làm sạch, Xử lý ngoại lệ IQR B2B, Kiểm định giả thuyết & Tương quan) | **11 trang** | Sinh viên 2 |
| **Chương 3** | **Phân tích Trực quan Đa chiều & Mô hình hóa Khoa học Dữ liệu** (Chuỗi thời gian ADF + SARIMAX, Địa lý Central/Texas, Sản phẩm Sunburst + Apriori + Pareto, Khách hàng RFM + K-Means + CLV) | **15 trang** | Sinh viên 2 & Sinh viên 3 |
| **Chương 4** | **Thiết kế Kiến trúc Hệ thống Web Dashboard, Mô phỏng Kinh tế lượng & Trợ lý AI** (Clean Architecture, UX/UI, What-If Độ co giãn cầu $\epsilon$, ML Dự báo đơn lỗ, AI Copilot) | **10 trang** | Sinh viên 4 |
| **Chương 5** | **Tổng hợp Phát hiện (Insights), Khuyến nghị Chiến lược & Kết luận** (Ma trận hành động 30-60-90 ngày, Ước tính tác động tài chính, Hạn chế & Hướng mở rộng) | **5 trang** | Cả nhóm |
| **Cuối bài** | Tài liệu tham khảo (IEEE/APA) & Phụ lục Bảng số liệu chi tiết (51 tiểu bang, 17 Sub-Category) | **4 trang** | Sinh viên 3 & 4 |
| **TỔNG CỘNG** | **Toàn bộ cuốn Báo cáo Bài tập lớn** | **65 trang** | **4 Sinh viên** |

---

## 📑 ĐỀ CƯƠNG CHI TIẾT TỪNG TRANG & HƯỚNG DẪN VIẾT (KHÔNG DÙNG CODE)

### PHẦN ĐẦU BÁO CÁO (6 TRANG — Đánh số La Mã: i, ii, iii, iv, v, vi)
* **Trang i – ii:** Trang bìa chính và Trang bìa phụ (Tên trường, Khoa, Tên môn học *Trực quan & Phân tích Dữ liệu*, Tên đề tài, Danh sách 4 thành viên + MSSV, Giảng viên hướng dẫn).
* **Trang iii:** Lời cam đoan & Lời cảm ơn + Tóm tắt đồ án (Executive Abstract khoảng 300 chữ bằng tiếng Việt và tiếng Anh nêu rõ quy mô `9.994` dòng, các phương pháp SARIMAX, K-Means, Apriori, Random Forest và kết quả tối ưu lợi nhuận).
* **Trang iv:** Mục lục tự động (Table of Contents — tạo bằng Heading 1, 2, 3 trong Word).
* **Trang v:** Danh mục các Bảng biểu (List of Tables — khoảng 24 bảng) & Danh mục các Hình vẽ (List of Figures — khoảng 28 hình).
* **Trang vi:** Danh mục các Từ viết tắt (BI, EDA, IQR, RFM, CLV, IPT, ADF, SARIMAX, MAE, MAPE, ROC-AUC, SLA, AOV, YoY).

---

### PHẦN MỞ ĐẦU (4 TRANG — Trang 1 đến Trang 4)
* **Mục 1. Tính cấp thiết của đề tài (1.5 trang):**
  * Phân tích bài toán thực tế của các tập đoàn bán lẻ đa kênh (Omnichannel B2B & B2C): dữ liệu giao dịch lớn nhưng bị phân mảnh, hiện tượng **"tăng trưởng doanh thu ảo nhưng lợi nhuận bị bào mòn"** do chính sách chiết khấu tràn lan và chi phí vận chuyển hàng cồng kềnh.
  * Nhu cầu chuyển dịch từ báo cáo tĩnh (Static Reporting) sang hệ thống **Decision Intelligence** kết hợp kiểm định thống kê, mô phỏng kinh tế lượng và trí tuệ nhân tạo.
* **Mục 2. Mục tiêu và 5 Câu hỏi Nghiên cứu cốt lõi (1 trang):**
  * Đặt ra 5 câu hỏi nghiên cứu định lượng (Research Questions — RQ1 đến RQ5):
    * *RQ1:* Ngưỡng chiết khấu nào là điểm gãy (tipping point) khiến biên lợi nhuận đảo chiều sang âm, và sự khác biệt đó có ý nghĩa thống kê không?
    * *RQ2:* Tại sao khu vực Central (đặc biệt là bang Texas) và ngành hàng Furniture lại có biên lợi nhuận thấp bất thường dù doanh số cao?
    * *RQ3:* Chuỗi thời gian doanh thu có tính dừng và tính mùa vụ như thế nào, mô hình dự báo nào cho sai số thực nghiệm (Hold-out MAPE) thấp nhất?
    * *RQ4:* Làm thế nào để phân cụm khách hàng khách quan bằng học máy và định lượng xác suất rời bỏ $P(\text{Alive})$ cùng doanh thu có nguy cơ mất (Revenue at Risk)?
    * *RQ5:* Nếu áp dụng trần chiết khấu 20% trong điều kiện cầu co giãn theo giá ($\epsilon < 0$), lợi nhuận ròng của doanh nghiệp sẽ thay đổi bao nhiêu USD?
* **Mục 3. Đối tượng, Phạm vi và Phương pháp nghiên cứu (1 trang):**
  * Vẽ **Sơ đồ quy trình nghiên cứu 6 bước** (Thu thập → Kiểm toán & Làm sạch → EDA & Kiểm định Thống kê → Mô hình hóa Chuỗi thời gian / Học máy → Xây dựng Web Dashboard → Đề xuất Chiến lược).
* **Mục 4. Bố cục của Báo cáo (0.5 trang):** Tóm tắt nội dung 5 chương.

---

### CHƯƠNG 1: TỔNG QUAN CƠ SỞ LÝ THUYẾT & MÔ TẢ TẬP DỮ LIỆU SUPERSTORE (10 TRANG — Trang 5 đến Trang 14)

#### 1.1. Tổng quan về Tập dữ liệu Bán lẻ US Superstore (2 trang)
* Giới thiệu nguồn gốc tập dữ liệu **Kaggle Sample - Superstore** (`9.994` bản ghi giao dịch, `21` thuộc tính, giai đoạn `03/01/2014 – 30/12/2017` trên toàn bộ `49` tiểu bang có phát sinh đơn tại Hoa Kỳ).
* Giải thích mô hình vận hành B2B/B2C của Superstore: 3 phân khúc khách hàng (`Consumer`, `Corporate`, `Home Office`), 4 khu vực địa lý (`West`, `East`, `Central`, `South`), 4 phương thức vận chuyển (`Standard Class`, `Second Class`, `First Class`, `Same Day`) và 3 ngành hàng chính (`Technology`, `Furniture`, `Office Supplies` chia thành `17` Sub-Category).

#### 1.2. Từ điển Dữ liệu Chi tiết — Data Dictionary 21 Biến (3 trang)
* *Lưu ý viết báo cáo:* Không chụp code `df.info()`. Thay vào đó, lập **Bảng 1.1: Từ điển dữ liệu 21 trường nguyên bản** và **Bảng 1.2: Các biến phái sinh (Feature Engineering)** với 6 cột: *STT, Tên trường, Kiểu dữ liệu chuẩn hóa, Miền giá trị / Đơn vị, Vai trò phân tích (Dimension / Measure / Key), Ý nghĩa nghiệp vụ chi tiết*.
* Phân tích chi tiết từng nhóm biến:
  1. Nhóm định danh giao dịch (`Row ID`, `Order ID`).
  2. Nhóm thời gian & vận chuyển (`Order Date`, `Ship Date`, `Ship Mode`, biến phái sinh `Shipping_Days`, `Order_Year`, `Order_Quarter`, `Order_Month`).
  3. Nhóm định danh khách hàng (`Customer ID`, `Customer Name`, `Segment`).
  4. Nhóm không gian địa lý (`Country`, `Region`, `State`, `City`, `Postal Code`, biến phái sinh `State_Code` chuẩn ANSI 2 ký tự).
  5. Nhóm danh mục sản phẩm (`Product ID`, `Category`, `Sub-Category`, `Product Name`).
  6. Nhóm chỉ số tài chính định lượng (`Sales`, `Quantity`, `Discount`, `Profit`, biến phái sinh `Profit_Margin_Pct`, `Implied_Unit_Cost`, `Gross_List_Price`).

#### 1.3. Cơ sở Lý thuyết về Thống kê Suy luận & Kinh tế lượng (2.5 trang)
* Trình bày công thức toán học (dùng Equation Editor) và điều kiện áp dụng của:
  * **Quy tắc phát hiện ngoại lệ Tukey's Fences ($1.5 \times \text{IQR}$)** và kỹ thuật **Winsorization (Chặn trần/sàn)**.
  * **Kiểm định giả thuyết tham số & phi tham số:** Công thức kiểm định Welch's $t$-test (khi phương sai hai nhóm không đồng nhất), Mann-Whitney $U$, phân tích phương sai một yếu tố One-way ANOVA ($F$-test), kiểm định phi tham số Kruskal-Wallis ($H$-test), và hệ số đo độ lớn hiệu ứng **Cohen's $d$**:
    $$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, \quad s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$
  * **Hệ số tương quan Pearson ($r$) và tương quan hạng Spearman ($\rho$):** Giải thích tại sao Spearman $\rho$ phản ánh đúng mối quan hệ đơn điệu phi tuyến trên dữ liệu tài chính lệch phải.
  * **Lý thuyết Độ co giãn của cầu theo giá (Price Elasticity of Demand — $\epsilon$):**
    $$\epsilon = \frac{\%\Delta Q}{\%\Delta P} = \frac{\Delta Q / Q_0}{\Delta P / P_0}$$

#### 1.4. Cơ sở Lý thuyết về Chuỗi thời gian & Khai phá Dữ liệu Học máy (2.5 trang)
* Trình bày mô hình toán học của:
  * **Kiểm định nghiệm đơn vị Augmented Dickey-Fuller (ADF Test)** và **Phân rã chuỗi thời gian cộng tính** ($Y_t = T_t + S_t + \varepsilon_t$).
  * **Mô hình SARIMAX$(p,d,q)(P,D,Q)_s$** và các thước đo đánh giá sai số dự báo trên tập giữ lại (Hold-out Validation): $\text{MAE}$ và $\text{MAPE}$.
  * **Mô hình RFM, Phân cụm K-Means với chỉ số Inertia (Elbow) & Silhouette Score**, và **Xác suất sống sót khách hàng $P(\text{Alive})$** theo quy luật BG/NBD rút gọn.
  * **Thuật toán khai phá luật kết hợp Apriori** ($\text{Support}$, $\text{Confidence}$, $\text{Lift}$) và **Mô hình phân loại rủi ro đơn lỗ (Random Forest & Logistic Regression Odds Ratio $e^{\beta}$)**.

---

### CHƯƠNG 2: QUY TRÌNH TIỀN XỬ LÝ, KIỂM TOÁN CHẤT LƯỢNG & KHÁM PHÁ DỮ LIỆU — EDA (11 TRANG — Trang 15 đến Trang 25)

#### 2.1. Quy trình Kiểm toán Chất lượng Dữ liệu (Data Quality Audit Pipeline) (2.5 trang)
* **Thay vì chụp code:** Vẽ **Sơ đồ lưu đồ (Flowchart) 5 bước Kiểm toán & Làm sạch dữ liệu** (Nạp tệp CSV với cơ chế tự động nhận diện bảng mã UTF-8/CP1252 → Kiểm tra Null/Missing → Kiểm tra trùng lặp khóa nghiệp vụ → Chuẩn hóa kiểu dữ liệu Thời gian & Mã bưu chính → Gắn cờ/Xử lý ngoại lệ IQR).
* Chèn **Bảng 2.1: Báo cáo kiểm toán chất lượng dữ liệu trước và sau làm sạch**:
  * Tổng số dòng gốc: `9.994` dòng; Giá trị khuyết thiếu (Missing/Null): `0` ô (`0,00%`).
  * Bản ghi trùng lặp toàn phần (bao gồm `Row ID`): `0` dòng; Bản ghi trùng lặp khóa nghiệp vụ (bỏ qua `Row ID`): phát hiện **01 cặp bản ghi trùng lặp** tại đơn hàng `US-2014-150119` (`Row ID = 3406` và `Row ID = 3407`, sản phẩm ghế văn phòng tại bang Ohio, ngày `23/04/2014`, doanh thu `$281.37`, chiết khấu `30%`, lỗ `-$12.06`) → Loại bỏ `Row ID = 3407`, giữ lại **`9.993` dòng sạch**.
  * Kiểm tra tính hợp lệ logic thời gian (`Ship Date >= Order Date`): `100%` hợp lệ (`0` đơn giao trước ngày đặt).
  * Chuẩn hóa `Postal Code` thành chuỗi 5 ký tự (bổ sung số `0` ở đầu cho các bang vùng Đông Bắc như Massachusetts `01841`, New Jersey `07090`).

#### 2.2. Thống kê Mô tả Đa biến & Phân tích Phân phối (2.5 trang)
* Chèn **Bảng 2.2: Thống kê mô tả chi tiết các biến định lượng (`Sales`, `Profit`, `Quantity`, `Discount`, `Shipping_Days`)** gồm: *Mean, Std, Min, Q1 (25%), Median (50%), Q3 (75%), Max, Skewness (Độ lệch), Kurtosis (Độ nhọn)*.
* Chèn **Hình 2.1 (Chụp từ Page 1 — Tab Phân phối):** Biểu đồ Histogram & Boxplot của `Sales` và `Profit` theo `Category`.
* **Phân tích học thuật (1 trang):** Giải thích hiện tượng phân phối lệch phải cực mạnh (Right-skewed) của `Sales` (trung bình `$229.86` lớn hơn nhiều so với trung vị `$54.49`) và phân phối có hai đuôi dày (Heavy-tailed / Leptokurtic) của `Profit` (dao động từ lỗ kỷ lục `-$6,599.98` đến lãi kỷ lục `+$8,399.98`).

#### 2.3. Phân tích Ngoại lệ (Outliers) theo Quy tắc $1.5 \times \text{IQR}$ và Lý giải Dưới góc độ B2B (2.5 trang)
* Lập **Bảng 2.3: Ngưỡng tứ phân vị $Q_1, Q_3, \text{IQR}$, cận dưới và cận trên cho `Sales` và `Profit`**:
  * Biến `Sales`: $Q_1 = \$17.28$, $Q_3 = \$209.94$, $\text{IQR} = \$192.66$, ngưỡng trên $Q_3 + 1.5 \times \text{IQR} = \$498.93$ → Phát hiện `1.167` giao dịch ngoại lệ (`11,68%`).
  * Biến `Profit`: $Q_1 = \$1.73$, $Q_3 = \$29.36$, $\text{IQR} = \$27.63$, ngưỡng dưới $-\$39.72$, ngưỡng trên $+\$70.82$ → Phát hiện `1.881` giao dịch ngoại lệ (`18,82%`).
* **Lập luận khoa học quan trọng (Điểm cộng lớn khi chấm báo cáo):** Giải thích tại sao **KHÔNG ĐƯỢC XÓA** các dòng ngoại lệ này trong bài toán bán lẻ B2B: đó là các hợp đồng mua sắm thiết bị văn phòng/máy photocopy quy mô doanh nghiệp (`Corporate`/`Home Office`) hoặc các đơn hàng xả kho chiết khấu 70–80%. Nếu xóa đi sẽ làm sai lệch nghiêm trọng tổng doanh thu thực tế (`$2,297M`) và che giấu các khoản lỗ lớn cần cảnh báo. Do đó, hệ thống thiết kế chế độ **Tùy chọn Chặn trần/sàn (IQR Winsorize)** trên Sidebar để phục vụ linh hoạt cả 2 mục đích: giữ nguyên khi báo cáo tài chính và bật Winsorize khi cần quan sát phân phối chuẩn hóa.

#### 2.4. Kiểm định Giả thuyết Thống kê & Đối sánh Tương quan Pearson vs. Spearman (3.5 trang)
* Chèn **Hình 2.2 & Bảng 2.4 (Chụp từ Page 1 — Tab `🧪 Kiểm định Thống kê & Tương quan`):**
  * Phân tích sâu kết quả kiểm định Giả thuyết 1 (`Discount ≤ 20%` vs `Discount > 20%`): Welch's $t = 16.14$ ($p = 2.44 \times 10^{-54}$), Mann-Whitney $U = 11,473,490$ ($p \approx 0.00$), độ lớn hiệu ứng **Cohen's $d = 0.64$**.
  * Phân tích sâu kết quả Giả thuyết 2 (So sánh lợi nhuận giữa 4 `Region`): Tại sao One-way ANOVA chỉ cho $F = 2.62$ ($p = 0.0488$) nhưng kiểm định phi tham số Kruskal-Wallis lại cho $H = 227.84$ ($p = 4.06 \times 10^{-49}$)? Viết 1 trang phân tích: ngoại lệ cực đoan làm phình phương sai trong công thức $F$-test, trong khi kiểm định dựa trên hạng (Rank-based Kruskal-Wallis) loại bỏ hoàn toàn nhiễu ngoại lệ và phơi bày sự chênh lệch bản chất giữa các vùng.
  * Phân tích **Bảng 2.5: Đối sánh tương quan Pearson ($r$) và Spearman ($\rho$)**: Giải thích sự chênh lệch giữa $r = -0.2195$ và $\rho = -0.5432$ của cặp `Discount ↔ Profit`.

---

### CHƯƠNG 3: PHÂN TÍCH TRỰC QUAN ĐA CHIỀU & MÔ HÌNH HÓA KHOA HỌC DỮ LIỆU (15 TRANG — Trang 26 đến Trang 40)

#### 3.1. Phân tích Cơ cấu Doanh thu – Lợi nhuận theo Danh mục & Phân khúc (2.5 trang)
* Chèn **Hình 3.1 & Hình 3.2 (Chụp từ Page 1 — Tab Cơ cấu):** Biểu đồ Donut tỷ trọng ngành hàng và Biểu đồ Grouped/Stacked Bar `Category × Segment` kèm **Bảng 3.1 (GroupBy Category)**.
* Phân tích nghịch lý ngành hàng **`Furniture`**: Doanh thu đạt `$741,999.80` (chiếm `32.3%` tổng doanh thu, gần ngang ngửa `Technology` và `Office Supplies`), nhưng tổng lợi nhuận chỉ vỏn vẹn `$18,451.27` (biên lợi nhuận `2.49%`, thấp hơn gần 7 lần so với `Technology` `17.40%` và `Office Supplies` `17.04%`).

#### 3.2. Phân tích Chuỗi thời gian, Kiểm định Tính dừng ADF & Dự báo Doanh thu (4 trang)
* Chèn **Hình 3.3 (Page 2):** Xu hướng doanh thu, lợi nhuận và đường trung bình trượt `MA-3` theo Tháng/Quý/Năm + Biểu đồ **YoY Waterfall** và **Seasonality Heatmap**.
* **Lưu ý chính xác số liệu 2014–2017:** Phân tích sự dịch chuyển năm `2015` (doanh thu giảm nhẹ `2.83%` từ `$484,247` xuống `$470,533` nhưng lợi nhuận tăng mạnh `24.3%` từ `$49,544` lên `$61,619` nhờ kiểm soát chiết khấu), sau đó bùng nổ liên tiếp vào năm `2016` (`$609,206`) và `2017` (`$733,215`).
* Chèn **Hình 3.4 & Bảng 3.2 (Page 2 — Mục `🔬 Kiểm định Tính dừng ADF & Phân rã Chuỗi thời gian`):**
  * Trình bày bảng kết quả kiểm định ADF trên chuỗi gốc ($d=0, \text{ADF} = -4.4936, p = 0.0002$) và sai phân bậc 1 ($\Delta Y_t, d=1, \text{ADF} = -8.0522, p < 0.0001$).
  * Phân tích 4 tầng biểu đồ phân rã $Y_t = T_t + S_t + \varepsilon_t$: chỉ rõ quy luật mùa vụ (Seasonal $S_t$) luôn tạo đỉnh vào tháng 9, tháng 11 và tháng 12 (mùa mua sắm cuối năm và giải ngân ngân sách doanh nghiệp Quý 4), và chạm đáy vào tháng 2 đầu năm.
* Chèn **Hình 3.5 & Bảng 3.3 (Page 2 — Dự báo & Đánh giá Hold-out 6 tháng):**
  * Trình bày kết quả tách 6 tháng cuối năm 2017 làm tập kiểm thử độc lập (Out-of-sample Hold-out Validation): đối sánh `SARIMAX(1,1,0)(1,1,0,12)` ($\text{MAE} = \$14,715.73$, $\text{MAPE} = 17.50\%$) và `Seasonal Linear Trend` ($\text{MAE} = \$13,417.35$, $\text{MAPE} = 16.40\%$).

#### 3.3. Phân tích Không gian Địa lý, Khoan sâu Vùng Central / Bang Texas & Vận hành Giao vận SLA (3.5 trang)
* Chèn **Hình 3.6 & Hình 3.7 (Page 3):** Bản đồ Choropleth Hoa Kỳ 49 tiểu bang, Biểu đồ Ma trận Tứ분 vị (State Quadrant: Doanh thu vs. Biên lợi nhuận) và Bảng so sánh đối chứng **A/B Region Benchmark** (`West` vs. `Central`).
* Chèn **Hình 3.8 & Bảng 3.4 (Page 3 — Khoan sâu Khu vực Central & Bang Texas):**
  * Giải thích bằng số liệu tại sao `Central` có biên lợi nhuận thấp nhất (`7.92%`): Tiểu bang **Texas** đứng thứ 3 toàn quốc về doanh thu (`$170,188.05`, `535` đơn hàng) nhưng đứng số 1 toàn quốc về thua lỗ (**`-$25,729.36`**, biên lợi nhuận `-15.12%`, tỷ lệ đơn lỗ `49.54%`) do mức chiết khấu trung bình tại Texas lên tới **`37.01%`** (đặc biệt ở nhóm `Binders` chiết khấu `80%`, `Appliances` chiết khấu `80%`, `Machines` và `Furnishings`).
  * Cùng với Texas, bang **Illinois** (`-$12,607.89`, chiết khấu TB `39.00%`) kéo toàn bộ lợi nhuận vùng Central đi xuống, dù các bang lân cận như **Michigan** (`+$24,463.19`, chiết khấu TB chỉ `0.7%`) và **Indiana** (`+$18,382.94`, chiết khấu `0%`) kinh doanh cực kỳ hiệu quả.

#### 3.4. Phân tích Danh mục Sản phẩm Chuyên sâu: Sunburst, Pareto 80/20 (ABC) & Giỏ hàng Apriori (2.5 trang)
* Chèn **Hình 3.9, 3.10, 3.11 (Page 4):**
  * Biểu đồ phân cấp **Sunburst / Treemap** 3 tầng (`Category → Sub-Category → Top Products`) tô màu theo biên lợi nhuận: chỉ rõ 3 "điểm đen" lỗ ròng là **`Tables` (`-$17,725.48`)**, **`Bookcases` (`-$3,472.56`)** và **`Supplies` (`-$1,189.10`)**.
  * Biểu đồ **Pareto 80/20 (Phân loại ABC)**: Nhóm A (chiếm 80% doanh thu), Nhóm B (15% kế tiếp) và Nhóm C (5% cuối).
  * Đồ thị mạng lưới & Bảng luật kết hợp **Market Basket Analysis (Apriori)**: phân tích các cặp ngành hàng thường được mua chung trong cùng một hóa đơn `Order ID` (ví dụ `Binders` đi kèm `Paper`, `Phones` đi kèm `Storage`) để làm cơ sở thiết kế gói bán kèm (Cross-selling Bundle).

#### 3.5. Phân tích Hành vi Khách hàng: RFM, Cohort Retention, K-Means 3D & Dự báo CLV / $P(\text{Alive})$ (2.5 trang)
* Chèn **Hình 3.12, 3.13, 3.14 (Page 1 — Tab RFM, Cohort, K-Means):**
  * Phân tích 6 phân khúc RFM trên `793` khách hàng độc nhất và ma trận giữ chân **Cohort Retention** theo năm gia nhập (2014–2017).
  * **Biện luận chọn số cụm $k$ trong K-Means:** Đưa vào báo cáo 2 biểu đồ **Elbow (Inertia)** và **Silhouette Score** ($k = 2 \to 6$), giải thích tại sao chọn $k = 4$ cụm khách hàng trên không gian 3D chuẩn hóa Z-score $(\text{Recency}, \ln(1+\text{Frequency}), \ln(1+\text{Monetary}))$.
  * Phân tích kết quả **Mô hình Xác suất Sống sót $P(\text{Alive})$ & CLV 12 tháng**: Chu kỳ mua lặp lại trung bình $\text{IPT} = 128.8$ ngày, $P(\text{Alive})$ trung bình `62.6%`, tổng CLV dự báo 12 tháng `$475,148.00` và tổng Doanh thu có nguy cơ mất (`Revenue at Risk`) là `$287,793.41`.

---

### CHƯƠNG 4: THIẾT KẾ KIẾN TRÚC HỆ THỐNG WEB DASHBOARD, MÔ PHỎNG KINH TẾ LƯỢNG & TRỢ LÝ AI (10 TRANG — Trang 41 đến Trang 50)

#### 4.1. Kiến trúc Phần mềm 4 Tầng (Clean Architecture) & Cơ chế Đồng bộ Trạng thái (2.5 trang)
* **Không chụp code:** Vẽ **Sơ đồ Kiến trúc 4 Tầng (Layered Architecture Diagram)**:
  1. **Tầng Dữ liệu (`data/loader.py`):** Đọc tệp gốc / tệp CSV tùy chỉnh từ người dùng, kiểm toán tự động, làm sạch và lưu đệm RAM bằng `@st.cache_data`.
  2. **Tầng Lõi Phân tích & Kinh tế lượng (`data/analytics.py`):** Chứa hơn 25 hàm tính toán độc lập (GroupBy, Pivot, RFM, K-Means, SARIMAX, ADF, Hypothesis Tests, Elasticity, Explainable ML, CLV).
  3. **Tầng Giao diện Dùng chung & AI (`components/` & `utils/`):** Quản lý trạng thái xuyên trang bằng `st.session_state`, hệ thống giao diện SaaS (`ui_styles.py`), `Dynamic Auto-Insight Banner` và mô-đun Chatbot AI (`ai_chatbot.py`).
  4. **Tầng Trình diễn Đa trang (`app.py` & `pages/1..5`):** Trang chủ Executive Command Center và 5 phân hệ chuyên sâu.
* Giải thích hiệu năng thực tế: Nhờ `@st.cache_data`, thời gian truy xuất dữ liệu khi chuyển trang giảm từ `~350 ms` (đọc đĩa cứng) xuống còn **`< 5 ms`** (truy xuất trực tiếp từ RAM).

#### 4.2. Phân tích Tác động Chiết khấu & Mô phỏng What-If Có Xét Độ Co Giãn Cầu ($\epsilon$) (3 trang)
* Chèn **Hình 4.1 & Bảng 4.1 (Page 5 — Tab Tác động Chiết khấu):** Biểu đồ kép (Dual-Axis Bar + Line Chart) thể hiện Lợi nhuận trung bình và Tỷ lệ giao dịch lỗ qua 5 ngưỡng chiết khấu (`0%`, `>0–10%`, `>10–20%`, `>20–30%`, `>30%`).
  * Phân tích định lượng: Tại ngưỡng `0%`, biên lợi nhuận đạt `+29.51%` (`0%` đơn lỗ); tại `>10–20%`, biên lợi nhuận đạt `+11.58%` (`13.99%` đơn lỗ); nhưng ngay khi vượt mốc `20%`, nhóm `>20–30%` lỗ tới **`91.59%`** số đơn (biên LN `-10.06%`) và nhóm `>30%` lỗ tới **`97.77%`** số đơn (biên LN `-48.16%`, tổng lỗ `-$125,007`).
* Chèn **Hình 4.2 & Hình 4.3 (Page 5 — Tab What-If Simulator & Đường cong Tối ưu hóa Lợi nhuận):**
  * Trình bày mô hình toán học suy ngược giá vốn giao dịch ($\text{Cost}_i = \text{Sales}_i - \text{Profit}_i$) và giá niêm yết gốc ($\text{GrossPrice}_i = \frac{\text{Sales}_i}{1 - d_i}$).
  * Lập **Bảng 4.2: Kết quả mô phỏng độ nhạy lợi nhuận theo các mức trần chiết khấu (`10%`, `15%`, `20%`, `25%`, `30%`) và các kịch bản hệ số co giãn cầu ($\epsilon = 0.0, -0.3, -0.5, -0.8$)**.
  * Chứng minh rằng ngay cả khi khách hàng phản ứng giảm sức mua ($\epsilon = -0.4$ đến $-0.5$, sản lượng giữ lại đạt `~92.6%`), việc áp trần chiết khấu `20%` kết hợp điều chỉnh giá `+5%` ở nhóm hàng lỗ vẫn giúp lợi nhuận toàn công ty tăng thêm trên **`+$120,000 USD`**.

#### 4.3. Mô hình Học máy Giải thích Rủi ro Đơn hàng Thua lỗ — Explainable ML (2.5 trang)
* Chèn **Hình 4.4 & Bảng 4.3 (Page 5 — Mục `🧠 Mô hình Học máy Giải thích Rủi ro Đơn hàng Thua lỗ`):**
  * Mô tả thiết kế bài toán phân loại nhị phân: Biến mục tiêu $Y = \mathbb{I}(\text{Profit} < 0)$ (tỷ lệ lớp dương cơ sở `18.71%`), chia tập dữ liệu `7.994` dòng Train / `1.999` dòng Test theo phương pháp phân tầng (Stratified Split).
  * Phân tích bảng chỉ số đánh giá: Random Forest đạt $\text{ROC-AUC} = 0.9770$, $\text{F1-Score} = 0.8300$, $\text{Accuracy} = 94.45\%$; Logistic Regression đạt $\text{ROC-AUC} = 0.9846$.
  * Phân tích biểu đồ **Top 12 Feature Importance (%) & Odds Ratio ($e^{\beta}$)**: Chứng minh bằng học máy rằng `Discount` chiếm **`71.51%`** tổng mức độ quyết định rủi ro thua lỗ của một đơn hàng.

#### 4.4. Tích hợp Trợ lý AI Phân tích B2B (Gemini 2.5 Flash / Oxion AI & Local B2B Copilot) và Xuất Báo cáo Điều hành (2 trang)
* Vẽ **Sơ đồ Luồng Ngữ cảnh Động (Dynamic Context Injection Flow)**: Khi người dùng lọc dữ liệu trên Sidebar → Hệ thống tự động tổng hợp các KPI, Top ngành hàng, điểm nghẽn Texas/Central, bảng ngưỡng chiết khấu thành một `System Prompt` có cấu trúc → Gửi tới LLM (`Gemini 2.5 Flash` / `Oxion AI`) hoặc kích hoạt `Local B2B Analytical Copilot` dự phòng → Sinh báo cáo phân tích tức thì (1-Click Executive Brief).
* Chèn **Hình 4.5 & Hình 4.6:** Giao diện Chatbot trả lời câu hỏi thực tế và mẫu **Báo cáo Điều hành HTML/PDF** xuất trực tiếp từ Page 5.

---

### CHƯƠNG 5: TỔNG HỢP PHÁT HIỆN (INSIGHTS), KHUYẾN NGHỊ CHIẾN LƯỢC & KẾT LUẬN (5 TRANG — Trang 51 đến Trang 55)

#### 5.1. Tổng hợp 5 Phát hiện Khoa học Cốt lõi (Key Empirical Findings) (1.5 trang)
* Tóm tắt hệ thống 5 phát hiện lớn nhất đã được chứng minh bằng cả kiểm định thống kê, kinh tế lượng và học máy ở Chương 2, 3, 4.

#### 5.2. Ma trận Khuyến nghị Chiến lược & Lộ trình Triển khai 30 – 60 – 90 Ngày (2 trang)
* Lập **Bảng 5.1: Ma trận Giải pháp Quản trị (Vấn đề gốc rễ → Giải pháp hành động → Phòng ban chịu trách nhiệm → KPI đo lường → Ước tính tác động tài chính USD)**:
  1. **Chính sách Giá & Chiết khấu (30 ngày đầu):** Thiết lập quy tắc phê duyệt tự động trên ERP — khóa trần chiết khấu tối đa `20%` cho nhân viên kinh doanh; mọi mức giảm trên `20%` bắt buộc phải đi kèm điều kiện giá trị đơn hàng tối thiểu (Minimum Order Value) hoặc bán kèm nhóm hàng biên lợi nhuận cao.
  2. **Tái cấu trúc thị trường vùng Central — trọng tâm bang Texas & Illinois (60 ngày):** Chấm dứt chương trình khuyến mãi giảm giá `60% – 80%` đối với nhóm `Binders` và `Appliances` tại Texas; áp dụng mô hình định giá chuẩn của bang Michigan và Indiana.
  3. **Tái cơ cấu danh mục `Furniture` (`Tables` & `Bookcases`) và Cross-selling (60–90 ngày):** Tăng giá cơ sở `5%` hoặc phụ thu phí vận chuyển cồng kềnh đối với mặt hàng `Tables`; triển khai gói bán kèm (Bundle) theo luật kết hợp Apriori.
  4. **Chiến dịch Giữ chân Khách hàng dựa trên $P(\text{Alive})$ & CLV (90 ngày):** Phân bổ ngân sách chăm sóc khách hàng tập trung vào nhóm *Nguy cơ rời bỏ* có `Revenue at Risk` cao để bảo vệ khoản doanh thu `$287,793`.

#### 5.3. Kết luận, Hạn chế của Đề tài và Hướng Phát triển Tương lai (1.5 trang)
* **Hạn chế khách quan của dữ liệu gốc:** Tập dữ liệu Superstore chưa công bố chi phí vận chuyển tách biệt trên từng đơn (Shipping Cost riêng) và chi phí thu hút khách hàng (CAC).
* **Hướng mở rộng:** Tích hợp dữ liệu luồng thời gian thực (Real-time Streaming từ cơ sở dữ liệu PostgreSQL/BigQuery), bổ sung thuật toán tối ưu hóa tồn kho (EOQ / Safety Stock) và triển khai mô hình dự báo nhu cầu cấp độ từng SKU.

---

### TÀI LIỆU THAM KHẢO & PHỤ LỤC (4–6 TRANG — Trang 56 đến Trang 60+)
* **Tài liệu tham khảo (1.5 trang):** Liệt kê 15–20 tài liệu học thuật chuẩn IEEE/APA (Sách thống kê kinh doanh, bài báo về mô hình RFM/CLV của Fader & Hardie, mô hình SARIMAX của Box-Jenkins, thuật toán Apriori của Agrawal & Srikant, tài liệu chính thức của Pandas, Scikit-learn, Statsmodels, Plotly, Streamlit).
* **Phụ lục A (2 trang):** Bảng tổng hợp hiệu suất kinh doanh đầy đủ của **49 tiểu bang Hoa Kỳ** (Doanh thu, Lợi nhuận, Biên LN %, Chiết khấu TB %, Số đơn hàng, Tỷ lệ đơn lỗ %).
* **Phụ lục B (1.5 trang):** Bảng tổng hợp hiệu suất đầy đủ của **17 nhóm sản phẩm con (`Sub-Category`)** kèm phân hạng Pareto ABC và kết quả kiểm toán từng bước.
* **Phụ lục C (1 trang):** Hướng dẫn sử dụng Hệ thống Dashboard cho Người dùng cuối (End-User Manual) và liên kết mã nguồn GitHub (`https://github.com/trananhpham/BTL_TQVPTDL.git`).

---

## 💡 MẸO VIẾT BÁO CÁO ĐỂ ĐẠT ĐIỂM TỐI ĐA MÀ KHÔNG CẦN CHỤP CODE
1. **Công thức 3 đoạn cho mỗi biểu đồ (1 Hình = 1 Trang hoàn chỉnh):**
   * *Đoạn 1 (Nhận diện & Thiết kế - 4 dòng):* Nêu lý do chọn loại biểu đồ (ví dụ: tại sao dùng biểu đồ kép Bar-Line cho ngưỡng chiết khấu hoặc tại sao dùng Boxplot cho phân phối lợi nhuận) và giải thích ý nghĩa các trục/màu sắc.
   * *Đoạn 2 (Đọc số liệu định lượng - 6 dòng):* Trích dẫn chính xác con số cao nhất, thấp nhất, trung bình, tỷ lệ phần trăm và điểm gãy từ bảng số liệu đi kèm.
   * *Đoạn 3 (Phân tích nguyên nhân & Hàm ý quản trị - 6 dòng):* Trả lời câu hỏi *"Tại sao lại xảy ra hiện tượng đó?"* và *"Nhà quản trị cần làm gì từ biểu đồ này?"*.
2. **Thay thế hoàn toàn Code bằng Bảng Tham Số Kỹ Thuật:**
   * Ví dụ khi viết về mô hình học máy Random Forest hoặc SARIMAX, thay vì chụp code Python, hãy kẻ **Bảng Cấu hình Siêu tham số (Hyperparameter Configuration Table)** gồm 3 cột: *Tên tham số, Giá trị thiết lập, Ý nghĩa toán học & Lý do lựa chọn*. Giảng viên chấm điểm rất cao cách trình bày chuẩn bài báo khoa học này!
