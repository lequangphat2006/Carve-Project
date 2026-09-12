# CARVE — implementation_notes.md

**Ngày tạo:** 2026-09-12
**Phiên bản tài liệu tham chiếu:** CARVE v5.1
**Trạng thái:** Đang chạy Stage 0 — đã xong 0.1 (unit tests), 0.2/0.3 (Coswara only); chờ DiCOVA cho 0.4/0.5
**Cập nhật lần cuối:** 2026-09-12

---

## A. Checklist chốt trước (từ Phần III + checklist cuối tài liệu)

| # | Mục | Nguồn | Trạng thái | Giá trị / ghi chú |
|---|---|---|---|---|
| A1 | Praat jitter/shimmer: local variant | Checklist Phần I | ☐ | local; báo cáo rap/ppq5/ddp như sensitivity |
| A2 | Time-stretch PretrainedAudio: phase vocoder | Checklist Phần I | ☐ | pitch-preserving |
| A3 | Fixed HPO protocol: 20 trial/model, random search | Checklist Phần I | ☐ | |
| A4 | SVM/RF ngoài Holm family (robustness) | Checklist Phần I | ☐ | báo cáo riêng |
| A5 | Danh sách baseline RQ2 | Checklist Phần I | ☐ | DiCOVA baseline + top-3 leaderboard + 1 SSL paper |
| A6 | covid_status: value_counts() + recovered_partial/negative | Checklist Phần I | ☑ | Hoàn thành 2026-09-12. recovered_partial KHÔNG tồn tại. NEGATIVE = {'healthy'}. Xem C.2, E.2 |
| A7 | Speaker DiCOVA vowel-e: 1 file/speaker | Checklist Phần I | ☐ | |
| B1 | N_B cho A1 (strict) và A2 (broad) + rule chọn | Mục III.1 | ☑ | A1 = 2024, A2 = 2114. Chốt A2 chính, A1 sensitivity. Xem B.1 |
| B2 | Power analysis TOST tiên nghiệm | Mục III.2 | ☐ | ghi ở mục D |
| B3 | Seed policy dùng SEED_UNIVERSE, L1 ⊂ L3 | Mục III.3 | ☐ | |
| B4 | Level 1.5 đã thêm vào pipeline | Mục III.3 | ☐ | |
| B5 | CAM stability check (Stage 6.0) trước Stage 6 chính | Mục III.4 | ☐ | ngưỡng mean_r đã chốt |
| B6 | Ngưỡng diễn giải correlation XAI đã chốt | Mục III.5 | ☐ | <0.1 / 0.1–0.3 / 0.3–0.5 / ≥0.5 |
| B7 | cuDNN deterministic theo hệ thống | Mục III.6 | ☐ | HandFeat=True, DeepFeat=True, PretrainedAudio=False |
| B8 | Unit test class weight pass ở Stage 0 | Mục III.7 | ☑ | 15 tests pass 2026-09-12. Xem ghi chú E.3 |
| B9 | SpecAugment preset "standard" là chính | Mục III.8 | ☐ | conservative/aggressive là sensitivity |
| B10 | Rule VIF → dominance (max VIF > 5 → tất cả) | Mục III.9 | ☐ | |
| B11 | HNR aggregation (HandFeat mean / XAI per-frame) | Mục III.10 | ☐ | cùng Harmonicity cc 50ms/20ms |
| B12 | Nested vs fixed trực tiếp trên PretrainedAudio | Mục III.11 | ☐ | 1 fold × 3 seed × 5 trial × 20 epoch |
| B13 | Privacy check HandFeat CSV release | Mục III.12 | ☐ | |

---

## B. Quyết định kỹ thuật chi tiết

### B.1. Định nghĩa "acute" (cập nhật sau Stage 0.3)

- ACUTE_STRICT (A1) = {'positive_mild', 'positive_moderate'}
- ACUTE_BROAD (A2) = A1 ∪ {'positive_asymp'}
- RECOVERED = {'recovered_full'}  # recovered_partial không tồn tại
- EXPOSED = {'no_resp_illness_exposed'}
- NEGATIVE = {'healthy'}  # tài liệu gọi là 'negative'; bản này dùng 'healthy'
- EXCLUDED = {'resp_illness_not_identified', 'under_validation'}
- Phân tích chính: A2. Sensitivity: A1.
- Quyết định: A2 (N_B = 2114). A1 (N_B = 2024) làm sensitivity.

### B.2. Ngưỡng thống kê chốt trước

- ROC-AUC là chỉ số chính duy nhất trong family Holm-Bonferroni
- 3 cặp so sánh Wilcoxon: HandFeat↔DeepFeat, HandFeat↔Pretrained, DeepFeat↔Pretrained
- TOST Δ = 0.05 (cố định, độc lập cỡ mẫu)
- VIF ngưỡng chính 5, sensitivity 10
- Holm-Bonferroni: 3 family × 3 nhánh = 9 lần hiệu chỉnh độc lập

### B.3. Ngân sách & seed

- SEED_UNIVERSE = list(range(20))
- L3 seeds = [0..9] (≥10 seed cho Wilcoxon)
- Budget khả dụng: ~240 GPU-hours (8 tuần × 30h)
- Phân bổ: HandFeat ~2.5h, DeepFeat ~83h, PretrainedAudio ~125h, contingency ~30h

### B.4. Ngưỡng diễn giải XAI (Mục III.5)

| \|r\| | Diễn giải |
|---|---|
| <0.1 | Không liên quan |
| 0.1–0.3 | Yếu |
| 0.3–0.5 | Trung bình |
| ≥0.5 | Mạnh |

### B.5. CAM stability ngưỡng (Mục III.4)

| mean_r | Diễn giải |
|---|---|
| ≥0.6 | Ổn định — diễn giải correlation |
| 0.3–0.6 | Tương đối — kèm cảnh báo |
| <0.3 | Không ổn định — không diễn giải, báo cáo như finding |

---

## C. Kết quả Stage 0 (ghi sau khi chạy)

### C.1. Data validation (Bước 0.2)

| Check | Kỳ vọng | Thực tế | Pass? |
|---|---|---|---|
| Coswara N | 2635 ±5 | **2746** | ⚠️ deviation E.2 |
| DiCOVA N | 1199 | pending (chờ email) | — |
| DiCOVA positive | 80 | pending | — |
| DiCOVA prevalence | ≈6.7% | pending | — |
| covid_status unique | 8 giá trị | **8 giá trị** (khác tên, xem E.2) | ✅ (khác nội dung) |
| test_status unique | 4 giá trị | **5 giá trị** (thêm '' empty) | ⚠️ |
| Speaker dup DiCOVA | 1 file/speaker | pending | — |
| Missing age/gender | <5% | **0.0% cả hai** | ✅ |

**Ghi chú:** missing rate age/gender = 0% (đầy đủ). Không cần xử lý missing cho covariate. Report JSON: `results/logs/validate_report_20260912_040554.json`.

### C.2. Metadata audit (Bước 0.3)

- N tổng: 2746 (không phải 2635)
- covid_status.value_counts():
  - healthy: 1433
  - positive_mild: 426
  - no_resp_illness_exposed: 248
  - positive_moderate: 165
  - resp_illness_not_identified: 157
  - recovered_full: 146
  - positive_asymp: 90
  - under_validation: 81
- test_status.value_counts():
  - '' (empty): 1413
  - p: 681
  - na: 314
  - n: 257
  - ut: 81
- recovered_partial: KHÔNG tồn tại (xem E.2)
- N_B (A1 strict) = 2024 (591 acute + 1433 healthy)
- N_B (A2 broad) = 2114 (681 acute + 1433 healthy)
- Định nghĩa acute chốt: **A2 (broad)**
- Lý do: cả hai đều ≥ 500 → theo bảng Mục III.1, dùng A2 chính, A1 sensitivity.

### C.3. DiCOVA split (Bước 0.4)

- Speaker leakage: [yes/no] — pending
- Tỉ lệ positive/fold: [min __%, max __%] — pending
- N/fold: [~240] — pending

---

## D. Power analysis TOST (Bước 0.5)

| ES | power |
|---|---|
| 0.2 | |
| 0.3 | |
| 0.4 | |
| 0.5 | |

Diễn giải: [theo ngưỡng Mục III.2] — pending (block bởi DiCOVA)

---

## E. Deviations log

### E.1. DiCOVA Access Status (2026-09-12)

- **Registration form:** Closed (không còn nhận phản hồi)
- **Action:** Đã gửi email đến `dicova2021@gmail.com` — chờ phản hồi
- **Timeline dự kiến:** 5–7 ngày làm việc
- **Fallback nếu không nhận được:** Replicate DiCOVA curation từ Coswara gốc:
  - Loại file < 500ms
  - Giữ `vowel-e` recordings
  - Giữ ACUTE_BROAD + healthy
  - Ghi rõ trong Limitations là "curated proxy, không phải official DiCOVA Track-2"
- **Ảnh hưởng tạm thời:** Stage 0.2 (C.3), 0.4, 0.5 bị block cho đến khi có DiCOVA
- **Quyết định:** Chờ phản hồi (Option A). Nếu quá 7 ngày không có, chuyển sang Option B (fallback)

### E.2. Coswara Dataset Deviations (2026-09-12)

**1. N tổng:** 2746 (không phải 2635). Chênh +111.
- Lý do: Coswara đã cập nhật vài lần kể từ 2020
- Ảnh hưởng: N_B tăng tương ứng; không ảnh hưởng logic phân tích

**2. `recovered_partial` không tồn tại** trong bản này
- Xử lý: bỏ nhánh xử lý `recovered_partial` (đúng như tài liệu yêu cầu)
- `recovered_full` (146) vẫn bị loại khỏi nhánh B

**3. Hai giá trị `covid_status` mới:**
- `resp_illness_not_identified` (157) — bệnh hô hấp không xác định
- `under_validation` (81) — chưa rõ nhãn
- Xử lý: **loại khỏi tất cả các nhánh**

**4. `healthy` thay cho `negative`:**
- Tài liệu gọi nhóm khoẻ mạnh là `negative`; bản này dùng `healthy`
- Ánh xạ: `NEGATIVE = {'healthy'}`

**5. `test_status` có 5 giá trị** (thêm `''` empty = 1413):
- Xác nhận quyết định không dùng `test_status` cho phân nhánh

**N_B tính được:**
- A1 (strict) = 591 acute + 1433 healthy = 2024
- A2 (broad) = 681 acute + 1433 healthy = 2114
- Quyết định: A2 làm chính, A1 làm sensitivity (cả hai ≥ 500)

### E.3. Unit Test Environment (2026-09-12)

- **Môi trường chạy:** Kaggle Notebook, Python 3.12.13, pytest 8.4.2
- **Kết quả:** 15/15 tests pass (3.53s)
- **Vấn đề gặp và fix:**
  1. sklearn không expose `class_weight_` sau fit → test lại qua `compute_class_weight` utility + acceptance của param
  2. Grad-CAM cần `retain_grad()` trên feature map (non-leaf tensor) mới giữ được `.grad`
  3. Praat command đúng là `"To PointProcess (periodic, cc)"`, không phải `"To PointProcess (cc)"`
- **numpy conflict trên Kaggle:** `pip install numpy==1.26.4` gây binary incompatibility với pandas mặc định (numpy 2.x) → quyết định dùng numpy mặc định Kaggle (2.x) thay vì pin 1.26.4, vì 4 đặc trưng HandFeat không phụ thuộc version numpy
- **Chi tiết:** xem commit `Fix unit tests: sklearn API, retain_grad, Praat PointProcess command`

### E.4. Coswara Audio Quality (phát hiện Stage 1)

**Ngày:** 2026-09-12 (test ngày 20200413)

- Tổng file vowel-e: 76
- OK: 67 (88.2%)
- Partial (silent, RMS=0): 7 (9.2%)
- Failed (0 samples): 2 (2.6%)

**Nguyên nhân:** File silent (RMS=0) do mic fail khi ghi âm;
file 0 samples do corrupt upload. Không recover được bằng
bất kỳ tham số Praat nào (đã test relaxed pitch range 50-600 Hz
và voicing threshold 0.30 — đều cho voiced=0).

**Xử lý:**
- Loại 9 file hỏng khỏi mọi phân tích
- QC report ghi rõ 2 tỉ lệ: 88.2% (tổng) và 100% (file hợp lệ)
- Sensitivity check: nếu tỉ lệ OK toàn Coswara < 85%, cân nhắc
  báo cáo như một limitation độc lập