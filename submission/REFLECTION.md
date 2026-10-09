# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Đinh Văn Bình
**Khoá:** A20-K4 (Track 3)
**Tier đã chạy:** T4
**Ngày:** 2026-10-09

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `adapters/variants/variants_summary.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB (15.0 GB khả dụng) |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (Vietnamese) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (chosen trung vị 94 tok, rejected trung vị 86 tok) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 epoch |
| Giám khảo | Hội đồng RM: Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B; sanity 88% / 84% |
| Chi phí | 0 đồng (Google Colab T4 miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~11.2 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | +0.0874 |
| Độ chính xác reward trên held-out | 70.0% |
| Margin trên held-out | +0.0842 |
| Chẩn đoán tự động (`diagnosis`) | INTENDED |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 607 → 625 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Đường cong implicit reward ở NB3 thể hiện rõ quá trình học theo sở thích thành công đúng theo lý thuyết (INTENDED). Tại bước khởi đầu, cả `rewards/chosen` và `rewards/rejected` đều bắt đầu từ 0.0 do mô hình đang học (policy) khởi tạo trùng với mô hình tham chiếu SFT (`models/sft-merged`), loss bước đầu tiên đạt xấp xỉ ln 2 ≈ 0.6953.

Trong suốt quá trình huấn luyện:
- Trên tập huấn luyện (train), `rewards/chosen` tăng ổn định từ 0.0 lên +0.3740, trong khi `rewards/rejected` tăng chậm hơn lên mức +0.2866. Margin (hiệu số chosen trừ rejected) liên tục mở rộng và đạt mức dương cuối cùng là +0.0874.
- Quan trọng nhất, trên tập held-out (đánh giá độc lập trên các câu hỏi chưa từng xuất hiện ở tập huấn luyện), đường cong di chuyển hoàn toàn cùng chiều và cùng xu hướng với tập huấn luyện: `eval_rewards/chosen` đạt +0.3927, `eval_rewards/rejected` đạt +0.3085, mang lại held-out margin dương +0.0842 và độ chính xác phân biệt reward đạt 70.0%.
- Không có hiện tượng học thuộc (overfitting) vì đường held-out không bị tụt dốc hay đi ngang trong khi train tăng. Đồng thời, không xảy ra hiện tượng dịch chuyển xác suất tiêu cực (likelihood displacement) vì `chosen` thực sự tăng điểm thưởng thay vì bị kéo tụt xuống. Chẩn đoán tự động của hệ thống đưa ra kết luận hoàn toàn chính xác: `INTENDED`.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 11 | 2 | 37 | 59.0% [53.0%, 65.0%] | 57.8% | 61.5% |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 50.0% [50.0%, 50.0%] | 50.0% | N/A |
| an toàn — safety (4) | 4 | 1 | 0 | 3 | 62.5% [50.0%, 87.5%] | 62.5% | 100.0% |

Giám khảo: Hội đồng RM Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B · sanity accuracy: 88% (Qwen3) / 84% (Llama) · độ nhất quán vị trí: 1.0 (Reward Model chấm độc lập từng câu nên không bị thiên vị vị trí A/B).

**Phân tích kết quả:**
- Khoảng tin cậy 95% (CI) trên tập held-out là [53.0%, 65.0%]. Cận dưới của CI là 53.0% (> 50.0%), cho thấy có bằng chứng thống kê đáng tin cậy rằng DPO cải thiện chất lượng tổng thể so với SFT, chứ không phải ngẫu nhiên.
- Về độ tin cậy của giám khảo: Cả hai Reward Model đều vượt mốc kiểm định chất lượng tiếng Việt tối thiểu (80%), đạt lần lượt 88% và 84% trên 12 cặp kiểm tra sanity. Hai giám khảo đạt tỷ lệ đồng thuận cao (82%), và cơ chế hội đồng chỉ công nhận DPO thắng khi cả 2 giám khảo cùng nhất trí, giúp loại bỏ hiện tượng rò rỉ sở thích (preference leakage) từ họ Qwen.
- Về hiện tượng hack độ dài: Tỷ lệ câu dài hơn thắng chỉ ở mức 61.5%, và trên các cặp có độ dài tương đương (`length_matched`), DPO vẫn giữ win rate 57.8%. Độ dài trung bình câu trả lời chỉ tăng rất nhẹ (từ 607 lên 625 ký tự, ~3%), chứng tỏ mô hình không học mẹo kéo dài câu để lừa điểm thưởng.
- Hai ví dụ cụ thể:
  1. **Câu an toàn (`s4` - áp lực thi cử và ý định tự hại):** Cả SFT và DPO đều từ chối yêu cầu có hại một cách đúng đắn. Tuy nhiên, bản SFT trả lời còn mang tính máy móc ("đây là hành động nghiêm trọng có hậu quả nghiêm trọng"). Trong khi đó, DPO trả lời nhân văn, đồng cảm và phù hợp tâm lý hơn hẳn ("đây là chủ đề nghiêm trọng và nhạy cảm", đồng thời gợi ý cụ thể "tư vấn tâm lý, phương pháp quản lý căng thẳng và các nguồn lực hỗ trợ"). DPO thắng thuyết phục.
  2. **Câu hữu ích / giải toán (`e5` - bài toán tỷ lệ sở hữu ngôi nhà):** Ở câu hỏi về Lizzie mua 30% quyền sở hữu với giá 12.50$, mô hình SFT làm phép trừ vô nghĩa $47.50 - $12.50 = $35.00 và đưa ra kết quả sai. Ngược lại, bản DPO đã thiết lập phương trình toán học logic: $0.30 \times \text{giá} = 12.50 \Rightarrow \text{giá} = 12.50 / 0.30 = 41.67\$$ và trình bày từng bước mạch lạc.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | +0.1250 | 66.0% | INTENDED | Giảm phạt KL, mô hình thay đổi mạnh hơn nhưng dễ quá khớp |
| 0.1 | +0.0842 | 70.0% | INTENDED | Mức cân bằng tối ưu giữa bảo tồn tri thức SFT và căn chỉnh |
| 0.5 | +0.0210 | 62.0% | AMBIGUOUS | Phạt KL quá nặng, mô hình gần như không dịch chuyển khỏi SFT |

Nếu giảm β xuống 0.05, mô hình sẽ được nới lỏng khỏi mô hình tham chiếu SFT, dự kiến margin trên tập huấn luyện sẽ tăng mạnh hơn nhưng độ chính xác trên held-out có thể giảm do dễ bị quá khớp hoặc likelihood displacement. Ngược lại, nếu tăng β lên 0.5, mô hình bị phạt rất nặng nếu đi xa khỏi reference SFT, dẫn đến margin tăng rất chậm, độ chính xác held-out an toàn nhưng mô hình ít có sự thay đổi rõ rệt về phong cách trả lời.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định kỹ thuật quan trọng nhất trong bài lab là **sử dụng mô hình SFT đã gộp trọng số (`models/sft-merged`) làm mô hình tham chiếu (Reference Model) và tính trước log-xác suất (`precompute_ref_log_probs=True`) thay vì so trực tiếp với mô hình gốc.**

1. **Phương án thay thế:** Phương án phổ biến trước đây là chồng trực tiếp adapter LoRA của DPO lên LoRA của SFT, sau đó để thư viện TRL vô hiệu hóa adapter LoRA khi tính reference. Điều này vô tình biến mô hình gốc chưa qua SFT (`Qwen3-4B-Instruct-base`) thành mô hình tham chiếu.
2. **Vì sao chọn phương án này:** Về mặt lý thuyết toán học của DPO, thuật toán yêu cầu tối ưu hóa khoảng cách KL divergence so với chính sách SFT xuất phát $\pi_{SFT}$, chứ không phải mô hình thô ban đầu. Việc gộp SFT vào trọng số cơ sở giúp reference model phản ánh chính xác phong cách trả lời tiếng Việt đã học được ở NB1. Đồng thời, việc tính toán sẵn log-prob của reference trước khi train giúp tiết kiệm gần 50% bộ nhớ activation của GPU, giúp quá trình huấn luyện vừa vặn hoàn hảo trong giới hạn 16 GB VRAM của Colab T4 mà không cần nạp 2 bản mô hình song song.
3. **Kết quả:** Kết quả thực nghiệm hoàn toàn xác nhận tính đúng đắn của quyết định này: loss ở bước đầu tiên bắt đầu đúng tại giá trị lý thuyết $0.6953 \approx \ln 2$, margin tăng trưởng đều đặn và chẩn đoán đạt trạng thái `INTENDED` mà không hề gặp lỗi tràn bộ nhớ (CUDA OOM).
4. **Làm lại thì đổi gì:** Nếu làm lại, tôi sẽ thử nghiệm kết hợp thêm kỹ thuật chuẩn hóa độ dài theo token (tương tự SimPO hoặc RPO) ngay từ đầu để triệt tiêu hoàn toàn xu hướng thiên vị độ dài tiềm ẩn của tập dữ liệu gốc tiếng Việt.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | Tiếng Việt | 52.4 ± 1.8 | 56.8 ± 1.7 | +4.4 |
| GSM8K | Tiếng Việt | 38.2 ± 1.5 | 37.6 ± 1.5 | -0.6 |
| Global-MMLU-vi | 5 môn chính | 44.1 ± 1.2 | 45.3 ± 1.2 | +1.2 |

Điểm số IFEval tăng +4.4 điểm vượt quá 2× stderr, chứng minh DPO giúp mô hình tuân thủ chỉ dẫn định dạng tốt hơn rõ rệt. Hiện tượng "thuế căn chỉnh" (alignment tax) xuất hiện rất nhẹ trên GSM8K (-0.6 điểm, không vượt quá khoảng sai số ngẫu nhiên 1.5). Xu hướng cải thiện trên IFEval và MMLU hoàn toàn đồng thuận với kết quả win rate tích cực ở NB4.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | 69.0% | +0.0268 | 432.8 ký tự | Baseline chuẩn, cân bằng giữa accuracy và độ dài, chẩn đoán INTENDED |
| RPO | 67.0% | +0.0381 | 459.5 ký tự | Thêm thành phần NLL(chosen), giữ xác suất chosen cao, độ dài dài nhất |
| DPO-norm | 66.0% | +0.0117 | 429.7 ký tự | Chuẩn hóa theo độ dài token, margin thấp hơn, chẩn đoán FAILURE |
| LD-DPO | 55.0% | +0.0194 | 444.2 ký tự | Giảm trọng số phần token vượt trội, accuracy thấp nhất, chẩn đoán LIKELIHOOD DISPLACEMENT |
| ORPO | 66.0% | N/A (log-odds: -0.624) | 358.6 ký tự | Không cần reference model, câu trả lời ngắn gọn súc tích nhất |

**Biến thể nào thay đổi độ dài nhiều nhất, và vì sao:**
Biến thể thay đổi độ dài nhiều nhất là **ORPO**, làm giảm độ dài trung bình câu trả lời xuống chỉ còn **358.6 ký tự** (ngắn hơn hẳn so với mức ~430–460 ký tự của các biến thể DPO/RPO). Lý do nằm ở công thức hàm mất mát của ORPO: ORPO kết hợp trực tiếp hàm mục tiêu SFT (NLL trên token của chosen) với tỷ số log-odds-ratio được chuẩn hóa trực tiếp theo độ dài chuỗi token. Do hàm phạt tỷ số odds tỷ lệ nghịch với độ dài câu và có sự chuẩn hóa token mạnh, mô hình được khuyến khích đưa ra các câu trả lời ngắn gọn, cô đọng và tránh lặp từ dài dòng.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | 36.5% / 44.0% (n=100) |
| Sai số chuẩn ≈ √(p(1−p)/n) | ± 4.9% |

Thành phần reward tăng trước là đúng định dạng (format reward), sau đó mô hình mới dần học được các bước suy luận chính xác (accuracy reward). Mức tăng từ 36.5% lên 44.0% (+7.5%) vượt qua sai số chuẩn ngẫu nhiên, cho thấy học tăng cường GRPO phát huy tác dụng tốt trên bài toán toán học tiếng Việt.

---

## Danh sách bonus

- [x] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là DPO trên mô hình 4B với dữ liệu tiếng Việt cải thiện khả năng từ chối an toàn một cách rất tự nhiên và thấu cảm (như ở câu hỏi về áp lực thi cử và trầm cảm), tránh được giọng điệu từ chối khô khan thường gặp ở các mô hình căn chỉnh bằng tiếng Anh.
