import tiktoken

OPENAI_MODEL = "gpt-4o"


def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


text = """
Trí tuệ nhân tạo đang ngày càng trở nên phổ biến trong cuộc sống hiện đại.
Công nghệ này được ứng dụng trong nhiều lĩnh vực như giáo dục, y tế, tài chính,
giao thông và chăm sóc khách hàng. Nhờ khả năng xử lý một lượng lớn dữ liệu,
các hệ thống trí tuệ nhân tạo có thể hỗ trợ con người phân tích thông tin,
đưa ra dự đoán và tự động hóa nhiều công việc. Tuy nhiên, việc sử dụng trí tuệ
nhân tạo cũng đặt ra nhiều vấn đề liên quan đến quyền riêng tư, bảo.
"""

# Tách từ
words = text.split()
word_count = len(words)

# Cách 1: tiktoken
token_count = count_tokens(text)

# Cách 2: ước lượng Part 1
estimated_tokens = word_count / 0.75

# Phần trăm chênh lệch
difference_percent = abs(token_count - estimated_tokens) / estimated_tokens * 100

print(f"Số từ: {word_count}")
print(f"Số token (tiktoken): {token_count}")
print(f"Số token ước lượng (số từ / 0.75): {estimated_tokens:.2f}")
print(f"Chênh lệch: {difference_percent:.2f}%")
