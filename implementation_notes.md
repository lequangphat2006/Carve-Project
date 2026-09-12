# CARVE — implementation_notes.md

**Ngày tạo:** 2026-09-12
**Phiên bản tài liệu tham chiếu:** CARVE v5.1
**Trạng thái:** Stage 0 xong (0.1–0.3); Stage 1 xong nhánh A+B (Coswara full + acute). Chờ DiCOVA cho nhánh C (0.4/0.5 + RQ1 C + RQ2/RQ3)
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
| A8 | Branch B (acute-only) đã tách | Stage 1 | ☑ | N=2114, xem C.5 |
| A9 | Metadata đã join vào feature CSV | Stage 1 | ☑ | covid_status, a, g |
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
### B.1b. Branch A definition (chốt trước Stage 5a, 2026-09-12)

- **Branch A:** positive = ACUTE_BROAD ∪ RECOVERED; negative = NEGATIVE
  - N = 2260 (827 positive + 1433 negative)
  - Lý do: để scenario (i) của CARVE (population-dependence) kiểm chứng được,
    Branch A phải chứa recovered như "case" — A vs B khác đúng 1 biến
- **Branch B:** positive = ACUTE_BROAD; negative = NEGATIVE
  - N = 2114 (681 positive + 1433 negative) — không đổi
- **Exposed (248) + EXCLUDED (238):** loại khỏi cả A và B
  - Lý do: exposed là category thứ ba, không thuộc binary case/control

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

### C.4. Stage 1 — Coswara HandFeat extraction (2026-09-12)

- File: `data/processed/handfeat_coswara_full.csv`
- Rows: 2746 (khớp unique speakers metadata)
- Ngày: 43 (20200413 → 20220224)
- OK: 2617 (95.3%)
- Partial: 91 (3.3%)
- Failed: 38 (1.4%)
- Feature distribution (OK only):
  - F0_mean: 82.4–466.4 Hz (median 150.1)
  - jitter_local: 0.07%–13.6% (median 0.72%)
  - shimmer_local: 0.7%–34.5% (median 5.98%)
  - hnr_mean: -4.58 to 35.13 dB (median 18.98)
  - pct_voiced: 0.38%–99.1% (median 57.3%)
- HNR âm (0.31%): 8/2617 file, đều có pct_voiced < 32% (file nhiễu cao) → hợp lý sinh lý học
- Loại bỏ 1 file `._vowel-e.wav` (macOS metadata)

### C.5. Stage 1 — Branch split (2026-09-12)

- Branch A (full Coswara): `data/processed/handfeat_coswara_full.csv` (2746 rows, đã thêm covid_status + a + g)
- Branch B (acute-only): `data/processed/handfeat_coswara_branch_B.csv`
  - Rows: 2114
  - Positive (acute): 681
  - Negative (healthy): 1433
  - Prevalence: 32.21% (enriched, không phải dân số tự nhiên)
  - Extraction status: OK 2024 (95.7%), partial 61, failed 29
- Sensitivity:
  - A1 (strict): N = 2024
  - A2 (broad):  N = 2114
- Feature distribution (OK only, Branch B):
  - F0_mean: 82.4–466.4 Hz (median 147.4)
  - jitter_local: 0.07%–12.6% (median 0.70%)
  - shimmer_local: 0.9%–34.5% (median 5.74%)
  - hnr_mean: -4.58 to 35.13 dB (median 19.10)
  - pct_voiced: 0.38%–98.8% (median 58.2)

### C.6. Stage 5a — Audit RQ1 (Coswara A+B, 2026-09-12)

**Sample (complete-case analysis):** chỉ dùng `extraction_status == 'ok'`
- Branch A: N = 2163 (796 pos + 1367 neg)
- Branch B: N = 2024 (657 pos + 1367 neg)

**Gender distribution (raw):** male = 1810, female = 806, other = 1 (unknown khi encode)
**Age:** min = 1 (cần check), max = 87, missing = 0

---

#### Lớp 1 — Raw univariate (Mann-Whitney + rank-biserial)

Cả 4 features significant sau Holm-Bonferroni ở cả 2 branches.
Hướng hiệu ứng **ngược sinh lý học**:

| Feature | Raw r_rb | Hướng |
|---|---|---|
| F0_mean | −0.1532 | Case có F0 THẤP hơn |
| jitter_local | −0.2258 | Case có jitter THẤP hơn |
| shimmer_local | −0.2938 | Case có shimmer THẤP hơn |
| hnr_mean | +0.1372 | Case có HNR CAO hơn |

→ Case trông có "giọng tốt hơn" người khỏe — bất khả thi về mặt sinh lý.

#### Lớp 2 — Multivariate logistic regression (chuẩn cho y binary)

**Phát hiện ban đầu (partial correlation) là SAI** — partial correlation không
phù hợp cho outcome binary. Đã verify lại bằng logistic regression (chuẩn y sinh).

**Kết quả logreg — KHÔNG có sign flip:**

| Feature | Raw coef | Adj coef (age+gender) | Sign flip? |
|---|---|---|---|
| F0_mean | +0.2606 | +0.1674 | Không |
| jitter_local | −0.1738 | −0.1742 | Không |
| shimmer_local | +0.6229 | +0.6457 | Không |
| hnr_mean | +0.0966 | +0.1202 | Không |

→ **Confounding effect KHÔNG mạnh.** Raw coef và adjusted coef cùng dấu, magnitude
thay đổi nhẹ. Không có "confounder reversal".

Covariate coef (adjusted): age = +0.4265, gender_bin = −0.2157
→ Age effect mạnh, đúng sinh lý (giọng già có jitter/shimmer cao).

#### Lớp 3 — Suppression effect (univariate vs multivariate)

Phát hiện **thật** — có suppression giữa các features:

| Feature | r_rb (univariate) | logreg coef (multivariate) | Đồng dấu? |
|---|---|---|---|
| F0_mean | −0.1532 | +0.2606 | ❌ NGƯỢC |
| jitter_local | −0.2258 | −0.1738 | ✅ cùng |
| shimmer_local | −0.2938 | +0.6229 | ❌ NGƯỢC |
| hnr_mean | +0.1372 | +0.0966 | ✅ cùng |

**Nguyên nhân:** F0, jitter, shimmer, HNR tương quan với nhau (đặc biệt jitter
và shimmer share cycle-to-cycle perturbation variance). Khi đưa vào multivariate,
F0 và shimmer giữ lại variance khác → đảo dấu hệ số.

**Ý nghĩa cho CARVE:** Đây là finding quan trọng hơn sign flip — handcrafted
features không độc lập, "hiệu ứng đơn biến" và "hiệu ứng độc lập" khác dấu.
Điều này củng cố luận điểm CARVE: đọc trực tiếp hệ số đơn biến là misleading.

#### Lớp 4 — VIF: max = 3.109 < 5

| Feature | VIF |
|---|---|
| F0_mean | 1.106 |
| jitter_local | 2.168 |
| shimmer_local | 3.109 |
| hnr_mean | 2.691 |

**max VIF = 3.109 < 5** → theo rule Mục III.9, **không cần dominance analysis**.

#### A vs B: ổn định

Kết quả logreg raw và adjusted ổn định giữa Branch A và B → scenario (iii).
Không có population-dependence rõ rệt.

#### Sai methodology đã sửa

- Ban đầu dùng partial correlation (Pearson trên residuals) → cho sign flip giả
- Lỗi: partial corr giả định y liên tục, không phù hợp y binary
- Sửa: dùng logistic regression (chuẩn y sinh) → xác nhận KHÔNG có sign flip
- Bài học: verify bằng 2 phương pháp trước khi commit finding vào paper

#### Issues nhỏ

1. 1 sample `age = 1` — cần check và loại (đã loại trong verify logreg)
2. 1 sample `gender = 'other'` — encode NaN, bị loại ở lớp 2
3. N discrepancy giữa audit (2163/2024) và C.5 (2260/2114): do complete-case

#### File output

- `results/audit_rq1_branch_A.csv`
- `results/audit_rq1_branch_B.csv`
- `results/audit_rq1_summary.json`
- `results/audit_rq1_logreg_verify.csv`

### C.7. Stage 5a — Commonality analysis (2026-09-12)

**Mục đích:** định lượng suppression đã phát hiện ở C.6 (logreg verification).
Commonality analysis (Mood 1971, Pedhazur 1982) phân rã R²_full thành unique +
shared contributions từ mọi subset của 4 features.

**Phương pháp:** OLS regression với recursive commonality coefficients
C_S = R²(S) − Σ_{T ⊊ S} C_T. Kiểm tra Sum C_S = R²_full (diff = 0 → formula đúng).

#### Kết quả Branch A (N=2160, pos=794, neg=1366)

R²_full = 0.057758

| Subset | \|S\| | C_S |
|---|---|---|
| F0 | 1 | +0.013294 |
| jit | 1 | +0.013994 |
| shim | 1 | +0.040937 |
| hnr | 1 | +0.014045 |
| F0+jit | 2 | −0.003757 |
| F0+shim | 2 | −0.000677 |
| F0+hnr | 2 | **+0.003422** |
| jit+shim | 2 | **−0.012807** |
| jit+hnr | 2 | −0.010713 |
| shim+hnr | 2 | −0.010569 |
| F0+jit+shim | 3 | +0.006104 |
| F0+jit+hnr | 3 | +0.000889 |
| F0+shim+hnr | 3 | −0.005452 |
| jit+shim+hnr | 3 | +0.010155 |
| F0+jit+shim+hnr | 4 | −0.001106 |
| **Sum C_S** | | **+0.057758** = R²_full |

#### Kết quả Branch B (N=2021, pos=655, neg=1366)

R²_full = 0.054905

Pattern giống hệt Branch A:
- Sum unique: +0.0767 (139.8% R²_full)
- Sum shared: −0.0218 (−39.8% R²_full)
- 5/6 pairs âm (suppression)
- Pair mạnh nhất: jit+shim = −0.012716

#### Tổng kết suppression

| Chỉ số | Branch A | Branch B |
|---|---|---|
| Sum unique (% R²_full) | 142.4% | 139.8% |
| Sum shared (% R²_full) | −42.4% | −39.8% |
| Pairs âm | 5/6 | 5/6 |

**Diễn giải:**

- Nếu 4 features độc lập, kỳ vọng: unique ≈ 100%, shared ≈ 0
- Thực tế: unique = **142%**, shared = **−42%** (Branch A)
- → Các features "che khuất" lẫn nhau — tổng phương sai marginal > phương sai joint

**Top 3 pairs suppression mạnh (ổn định ở cả 2 branches):**

| Pair | Branch A | Branch B | Sinh lý học |
|---|---|---|---|
| jit+shim | −0.0128 | −0.0127 | Cùng đo cycle-to-cycle perturbation |
| jit+hnr | −0.0107 | −0.0099 | jitter↑ → HNR↓ |
| shim+hnr | −0.0106 | −0.0069 | shimmer↑ → HNR↓ |

**Pair duy nhất positive:** F0+hnr (+0.0034, +0.0028) — 2 features hài hòa.

#### Ý nghĩa cho CARVE

Ba tầng bằng chứng suppression:

| Tầng | Phương pháp | Kết quả |
|---|---|---|
| 1 | Univariate (r_rb) | Case giọng "tốt hơn" — ngược sinh lý |
| 2 | Multivariate logreg (C.6) | Suppression: hệ số đảo dấu |
| 3 | Commonality (C.7) | Shared variance = **−42%** — định lượng |

**Luận điểm:** handcrafted features không độc lập; hướng và độ lớn hiệu ứng
phụ thuộc mạnh vào việc các feature khác có mặt trong mô hình hay không.
Đây là lý do CARVE yêu cầu audit đầy đủ trước khi diễn giải bất kỳ hệ số nào.

**Ổn định:** pattern giống hệt Branch A và B → không có population-dependence.

### C.8. Stage 5a — Visualizations (2026-09-12)

3 figure đã tạo cho paper:

- **Fig 1** — `results/figures/fig1_effect_sizes.png`
  Forest plot: univariate (r_rb) vs multivariate (adj logreg coef), 4 features × 2 branches.
  Mục đích: cho thấy suppression effect (F0 và shimmer đổi dấu).

- **Fig 2** — `results/figures/fig2_commonality.png`
  Stacked bar: 15 commonality coefficients C_S mỗi branch. Đỏ = shared variance âm (suppression).
  Mục đích: định lượng suppression pattern.

- **Fig 3** — `results/figures/fig3_r2_progression.png`
  R² progression khi thêm từng feature (forward selection).
  Mục đích: cho thấy feature nào đóng góp R² khi vào mô hình.

Script: `scripts/04_viz_rq1.py`, output 300 DPI, sẵn sàng cho paper.

### C.9. Stage 2 — L0 Smoke Test PASS (2026-09-12)

**Môi trường:** Kaggle Notebook, GPU T4x2, Python 3.12.

**Kết quả:**
- Audio dir: 1 ngày Coswara (20200413)
- Pre-filter: 74 valid, 2 skipped (corrupt/silent)
- Train 59 / Val 15 (label fake random)
- Mel shape verify: `(8, 1, 128, 500)` ✅
- DeepFeat params: 520,897
- Train 2 epochs: không NaN, loss giảm nhẹ
- ROC-AUC ~0.39 (label fake → expected random)
- CAM shape: `(1, 1, 16, 62)` — hệ quả T=500 (spec ghi 16×25 với T=200)
- CAM uniform: expected vì model chưa học

**Kết luận:** pipeline chạy end-to-end. Sẵn sàng cho L1 pilot (data thật + label thật).

**Fix gặp:**
1. `spec_aug` → `spec_augment` (tên tham số)
2. `torchaudio.load` crash với file 0 samples → defensive load
3. Pre-filter với `soundfile.info` loại file corrupt trước khi vào DataLoader


### C.10. Stage 2 — L1 Pilot DeepFeat PASS (2026-09-12)

**Dataset:** Coswara A+B mel cache (N=2085, 677 pos / 1408 neg)
**Split:** speaker-aware 5-fold, fold 0, 3 seeds (SEED_UNIVERSE 0-2)
**Config:** 20 epochs, batch=32, lr=3e-4, weight_decay=1e-4, patience=10

**Kết quả:**

| Seed | Best epoch | Val ROC-AUC |
|---|---|---|
| 0 | 18 | 0.7533 |
| 1 | 13 | 0.7692 |
| 2 | 14 | 0.7517 |
| **Mean** | — | **0.7581 ± 0.0079** |

**So sánh với HandFeat (từ C.6):**
- HandFeat raw r_rb: 0.15-0.31 (yếu)
- DeepFeat: 0.758 → mạnh hơn hẳn

**Go/no-go:**
- Ngưỡng L1: ROC-AUC > 0.55 ✅ (vượt 0.2+)
- Std qua seed < 0.05 ✅ (thực tế 0.008)
- Time/epoch ổn định ✅ (~5.8s, không tăng)

**Kết luận:** L1 PASS → chuyển L1.5 (cross-fold sanity).

### C.11. Stage 2 — L1.5 Cross-fold Sanity PASS (2026-09-12)

**Config:** 5 folds × 1 seed (seed 0) × 20 epoch, config L1

| Fold | Val ROC-AUC |
|---|---|
| 0 | 0.7533 |
| 1 | 0.7306 |
| 2 | 0.7885 |
| 3 | 0.7642 |
| 4 | 0.7779 |
| **Mean** | **0.7629 ± 0.0201** |

**Go/no-go L1.5:** std < 0.08 ✅ (thực tế 0.0201)
→ Không có fold bất thường. Chuyển sang L2 HPO (20 trial random search).

### C.12. Stage 2 — L2 HPO PASS (2026-09-12)

**Config:** random search 20 trial, fold 0, seed 0, 100 epoch, early stop patience=10

**Search space:**
- lr ~ log_uniform(1e-5, 1e-3)
- weight_decay ~ log_uniform(1e-6, 1e-3)
- batch_size ∈ {16, 32, 64}
- dropout ~ uniform(0.3, 0.7)

**Top 5:**

| Trial | Val AUC | lr | weight_decay | batch_size | dropout |
|---|---|---|---|---|---|
| 18 | **0.7940** | 4.54e-4 | 1.18e-5 | 16 | 0.36 |
| 5 | 0.7793 | 9.65e-4 | 7.12e-5 | 32 | 0.51 |
| 4 | 0.7752 | 8.71e-4 | 3.14e-4 | 32 | 0.30 |
| 16 | 0.7660 | 2.85e-4 | 9.54e-6 | 32 | 0.38 |
| 3 | 0.7585 | 5.40e-4 | 6.36e-5 | 64 | 0.31 |

**Best config (trial 18):**
- lr = 4.544e-4
- weight_decay = 1.176e-5
- batch_size = 16
- dropout = 0.356
- Val AUC = 0.7940

**So sánh với L1 default:**

| | Config | Val AUC |
|---|---|---|
| L1 default | lr=3e-4, wd=1e-4, bs=32, do=0.5 | 0.7533 |
| L2 best | lr=4.54e-4, wd=1.18e-5, bs=16, do=0.36 | 0.7940 |
| Delta | — | **+0.041 (+5.4%)** |

**Pattern:**
- LR cao (>2.8e-4) tốt hơn; LR thấp (~1.2e-5) underfit (<0.71)
- weight_decay nhỏ (1e-6 – 3e-4) tốt hơn
- dropout < 0.5 ưu thế
- batch_size 16–32 > 64

**Thời gian:** 114.9 min (1.9h).

**Quyết định:** Dùng best config trial 18 cho L3 full run.

### C.13. Stage 2 — L3 Fold 0 (2026-09-12)

**Config:** best HPO (lr=4.544e-4, wd=1.176e-5, bs=16, dropout=0.356)
**Setup:** fold 0, 10 seeds (SEED_UNIVERSE 0-9), 100 epoch, patience=10

| Seed | Best epoch | Val ROC-AUC |
|---|---|---|
| 0 | 26 | 0.7588 |
| 1 | 11 | 0.7693 |
| 2 | 7 | 0.7464 |
| 3 | 17 | 0.7468 |
| 4 | 25 | 0.7822 |
| 5 | 26 | 0.7886 |
| 6 | 26 | 0.7503 |
| 7 | 28 | 0.7934 |
| 8 | 52 | 0.7889 |
| 9 | 19 | 0.8077 |
| **Mean** | — | **0.7732 ± 0.0208** |

**Đọc kết quả:**
- L3 fold 0 (0.7732) < HPO best (0.794) do HPO bị selection bias
  (best of 20 trials cùng fold). 0.7732 là honest estimate.
- Std 0.0208 — ổn định qua seeds
- Early stop tập trung 7-62 epoch → model không cần full 100 epoch
-----------------------------------------------------------------------------

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

### E.4. Coswara Audio Quality (Stage 1, full extraction)

**Ngày:** 2026-09-12

- Tổng file vowel-e: 2746
- OK: 2617 (95.3%)
- Partial: 91 (3.3%) — chủ yếu silent (RMS=0) do mic fail
- Failed: 38 (1.4%) — 37 file 0 samples (corrupt) + 1 file `._vowel-e.wav` (macOS metadata)

**Nguyên nhân:**
- Silent (RMS=0): mic fail khi ghi âm
- 0 samples: file corrupt upload
- Không recover được bằng Praat params (đã test relaxed pitch 50-600 Hz,
  voicing threshold 0.30 — tất cả silent files đều cho voiced=0)

**Xử lý:**
- Loại 38 file failed khỏi phân tích
- QC report ghi rõ 2 tỉ lệ: 95.3% (tổng) và 98.6% (file hợp lệ)
- Tỉ lệ < 2% và không có pattern rõ ràng → không cần báo cáo như limitation độc lập

### E.5. DeepFeat FC Head Extension (2026-09-12)

- **Quyết định:** Mở rộng FC head từ `Linear(256, 1)` thành
  `Linear(256, 512) + ReLU + Dropout + Linear(512, 1)`.
- **Lý do:** Tài liệu CARVE ước lượng ~1.35M params; conv-only architecture
  cho 389k. Mở rộng FC head là điều chỉnh defensible vì spec không cố định
  FC head (chỉ ghi "Linear(256→1)" như mô tả tối thiểu).
- **Params:** 520,897 (tăng 34% so với 389,057).
- **Không vi phạm spec:** Conv channels 32→64→128→256 giữ nguyên;
  Grad-CAM target vẫn là conv4 feature map.
- **Ảnh hưởng:** Regularization (dropout 0.5, weight_decay) giữ nguyên.

### E.5b. Vowel-e Duration Handling (2026-09-12)

- **Spec CARVE:** mel [128, 200] (T=200, ~2s)
- **Duration đo được (N=74, 20200413):**
  - median 10.84s, p25 6.42s, p75 15.19s, max 29.10s
  - Chỉ 5.4% audio ≤ 2s
- **Quyết định chính:** T=500 (5s), crop từ đầu (0–5s), pad zero nếu ngắn hơn
- **Lý do:**
  - Khớp PretrainedAudio input 5s (Mục 4.3) → RQ2 so sánh công bằng
  - Fit budget Kaggle (DeepFeat ~104h + PretrainedAudio ~125h = 229h < 240h)
  - Sustained phonation → 5s đầu đủ đại diện
  - XAI alignment: hop=10ms khớp Praat → Grad-CAM W=500 ↔ F0(t)/jitter(t)/HNR(t)
- **Sensitivity (Tier 2, nếu có compute):** T=1000, 1 fold × 3 seeds
  - Rule escalation: chênh ROC-AUC ≥ 0.05 → cân nhắc chuyển T=1000 main
- **Ảnh hưởng:** mel shape [1, 128, 500] thay vì [1, 128, 200]