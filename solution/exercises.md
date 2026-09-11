# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi tăng temperature từ 0.0 lên 1.5, câu trả lời có xu hướng sáng tạo và đa dạng hơn; ở 0.0, câu trả lời sát thực tế và nhất quán hơn, trong khi ở 1.5, câu trả lời trở nên quá sáng tạo và dễ thiếu chính xác.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ đặt temperature = 0.2–0.3 để chatbot hỗ trợ khách hàng trả lời ổn định, chính xác, ít hallucination và vẫn giữ được cách diễn đạt tự nhiên, thân thiện.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> 10.000 user x 3 lần/ngày = 30.000 req/ngày; mỗi req 350 output tokens
tổng là 30.000 x 350 = 10.500.000 tokens/ngày = 10.500k tokens
chi phí:
GPT-4o: 10.500 x $0.010 = $105/ngày
GPT-40-mini: 10.500 x $0.0006 = $6,30/ ngày
Do đó, GPT-4o đắt hơn bản GPT-4o-mini khoảng 16,7 lần. (chỉ tính output, chưa tính input vì chưa có thông tin.) GPT-4o phù hợp với các tác vụ phức tạp, cần khả năng suy luận và độ chính xác cao như xử lý khiếu nại hoặc tư vấn chuyên sâu; trong khi GPT-4o-mini phù hợp với các tác vụ đơn giản, số lượng lớn như trả lời FAQ hoặc phân loại yêu cầu để tiết kiệm chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi sẽ khác nhau rõ rệt về độ dài, từ vựng và cách đưa ví dụ: prompt dành cho giáo viên tiểu học sẽ dùng từ ngữ đơn giản, câu ngắn và ví dụ gần gũi với trẻ em, trong khi prompt dành cho chuyên gia tài chính sẽ dài hơn, sử dụng nhiều thuật ngữ kỹ thuật và phân tích chuyên sâu. System prompt định hình vai trò, phong cách và mức độ chuyên môn mà model sử dụng khi trả lời. Vì vậy, cùng một câu hỏi nhưng thay đổi system prompt có thể tạo ra những câu trả lời rất khác nhau.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
"""
tham khảo code trong file cy.py
"""

> Số từ: 100
Số token (tiktoken): 124
Số token ước lượng (số từ / 0.75): 133.33
Chênh lệch: 7.00%
Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì cách tokenizer tách từ và ký tự tiếng Việt, đặc biệt với dấu và các từ có nhiều âm tiết, khiến một từ có thể bị chia thành nhiều token.
---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi chatbot tạo ra câu trả lời dài hoặc cần phản hồi nhanh, vì người dùng có thể thấy nội dung xuất hiện từng phần thay vì phải chờ toàn bộ câu trả lời. Ngược lại, non-streaming phù hợp hơn với các tác vụ ngắn, đơn giản hoặc khi cần xử lý toàn bộ kết quả trước khi hiển thị, chẳng hạn như phân loại, tính toán hoặc gọi API phía sau.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Giúp cho server giảm tải bằng cách tăng dần thời gian chờ cho lượt retry tiếp theo. Nếu hàng ngàn client cùng retry với delay cố định và giống nhau, thì server phải trả lời hàng ngàn request tại cùng một thời điểm nào đó dẫn đến quá tải.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: Bạn là trợ lý AI hỗ trợ học tập và lập trình. Hãy trả lời bằng tiếng Việt, ngắn gọn, dễ hiểu và đưa ví dụ khi cần thiết. Nếu không chắc chắn về thông tin, hãy nói rõ thay vì tự đoán.
Giải thích: Chọn yêu cầu “trả lời bằng tiếng Việt” để phù hợp với người dùng và giúp câu trả lời dễ hiểu hơn. Yêu cầu “ngắn gọn, dễ hiểu” giúp tránh câu trả lời quá dài, đặc biệt với các câu hỏi đơn giản, đồng thời vẫn có thể đưa ví dụ khi cần.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là history chỉ lưu tối đa 3 lượt hỏi–đáp, nên trợ lý có thể quên những thông tin được trao đổi ở các lượt trước. Một cải thiện cụ thể là tăng giới hạn history hoặc xây dựng bộ nhớ dài hạn, trong đó lưu các thông tin quan trọng của người dùng vào cơ sở dữ liệu và truy xuất lại khi cần. Cách này giúp trợ lý duy trì ngữ cảnh tốt hơn mà không phải gửi toàn bộ lịch sử vào mỗi request.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
