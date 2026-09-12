# CARVE — implementation_notes.md



**Ngày tạo:** 2026-09-12

**Phiên bản tài liệu tham chiếu:** CARVE v5.1

**Trạng thái:** Pre-registration — chưa chạy Stage 0



---



## A. Checklist chốt trước (từ Phần III + checklist cuối tài liệu)



| # | Mục | Nguồn | Trạng thái | Giá trị / ghi chú |

|---|---|---|---|---|

| A1 | Praat jitter/shimmer: local variant | Checklist Phần I | ☐ | local; báo cáo rap/ppq5/ddp như sensitivity |

| A2 | Time-stretch PretrainedAudio: phase vocoder | Checklist Phần I | ☐ | pitch-preserving |

| A3 | Fixed HPO protocol: 20 trial/model, random search | Checklist Phần I | ☐ | |

| A4 | SVM/RF ngoài Holm family (robustness) | Checklist Phần I | ☐ | báo cáo riêng |

| A5 | Danh sách baseline RQ2 | Checklist Phần I | ☐ | DiCOVA baseline + top-3 leaderboard + 1 SSL paper |

| A6 | covid_status: value_counts() + recovered_partial/negative | Checklist Phần I | ☐ | ghi kết quả ở mục C |

| A7 | Speaker DiCOVA vowel-e: 1 file/speaker | Checklist Phần I | ☐ | |

| B1 | N_B cho A1 (strict) và A2 (broad) + rule chọn | Mục III.1 | ☐ | ghi ở mục C |

| B2 | Power analysis TOST tiên nghiệm | Mục III.2 | ☐ | ghi ở mục D |

| B3 | Seed policy dùng SEED_UNIVERSE, L1 ⊂ L3 | Mục III.3 | ☐ | |

| B4 | Level 1.5 đã thêm vào pipeline | Mục III.3 | ☐ | |

| B5 | CAM stability check (Stage 6.0) trước Stage 6 chính | Mục III.4 | ☐ | ngưỡng mean_r đã chốt |

| B6 | Ngưỡng diễn giải correlation XAI đã chốt | Mục III.5 | ☐ | <0.1 / 0.1–0.3 / 0.3–0.5 / ≥0.5 |

| B7 | cuDNN deterministic theo hệ thống | Mục III.6 | ☐ | HandFeat=True, DeepFeat=True, PretrainedAudio=False |

| B8 | Unit test class weight pass ở Stage 0 | Mục III.7 | ☐ | |

| B9 | SpecAugment preset "standard" là chính | Mục III.8 | ☐ | conservative/aggressive là sensitivity |

| B10 | Rule VIF → dominance (max VIF > 5 → tất cả) | Mục III.9 | ☐ | |

| B11 | HNR aggregation (HandFeat mean / XAI per-frame) | Mục III.10 | ☐ | cùng Harmonicity cc 50ms/20ms |

| B12 | Nested vs fixed trực tiếp trên PretrainedAudio | Mục III.11 | ☐ | 1 fold × 3 seed × 5 trial × 20 epoch |

| B13 | Privacy check HandFeat CSV release | Mục III.12 | ☐ | |



---



## B. Quyết định kỹ thuật chi tiết



### B.1. Định nghĩa "acute"

- ACUTE_STRICT (A1) = {positive_mild, positive_moderate}

- ACUTE_BROAD (A2) = A1 ∪ {positive_asymp}

- RECOVERED = {recovered_full, recovered_partial}

- EXPOSED = {no_resp_illness_exposed}

- NEGATIVE = {negative}

- Phân tích chính: A2. Sensitivity: A1.

- Rule chọn theo N_B (Mục III.1): [ghi quyết định sau khi đo ở Stage 0.3]



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

| |r| | Diễn giải |

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

| Coswara N | 2635 ±5 | | |

| DiCOVA N | 1199 | | |

| DiCOVA positive | 80 | | |

| DiCOVA prevalence | ≈6.7% | | |

| covid_status unique | 8 giá trị | | |

| test_status unique | 4 giá trị | | |

| Speaker dup DiCOVA | 1 file/speaker | | |

| Missing age/gender | <5% | | |



### C.2. Metadata audit (Bước 0.3)

- `covid_status.value_counts()`: [paste output]

- `test_status.value_counts()`: [paste output]

- N_B (A1 strict) = [__]

- N_B (A2 broad) = [__]

- Định nghĩa acute chốt: [A1 / A2], lý do: [theo bảng Mục III.1]



### C.3. DiCOVA split (Bước 0.4)

- Speaker leakage: [yes/no]

- Tỉ lệ positive/fold: [min __%, max __%]

- N/fold: [~240]



---



## D. Power analysis TOST (Bước 0.5)

| ES | power |

|---|---|

| 0.2 | |

| 0.3 | |

| 0.4 | |

| 0.5 | |



Diễn giải: [theo ngưỡng Mục III.2]



---



## E. Deviations log



| Ngày | Deviation | Lý do | Ảnh hưởng |

|---|---|---|---|

| | | | |

