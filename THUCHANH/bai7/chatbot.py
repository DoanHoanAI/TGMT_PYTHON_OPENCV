import openai
import os

# Gán mã API vào biến
openai.api_key = "sk-...JtkA"  # Thay bằng mã API của bạn

def chat_with_bot():
    print("Chatbot đã sẵn sàng! Nhập 'quit' để thoát.")

    # Danh sách để lưu trữ các tin nhắn trong cuộc hội thoại
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        user_input = input("Bạn: ")
        if user_input.lower() == 'quit':
            break

        # Thêm tin nhắn người dùng vào danh sách
        messages.append({"role": "user", "content": user_input})

        try:
            # Gọi API OpenAI để lấy phản hồi
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu bạn có quyền
                messages=messages
            )

            # Lấy nội dung phản hồi từ chatbot
            bot_response = response['choices'][0]['message']['content']
            print("Chatbot:", bot_response)

            # Thêm phản hồi của chatbot vào danh sách để giữ bối cảnh
            messages.append({"role": "assistant", "content": bot_response})

        except openai.error.OpenAIError as e:
            print(f"Có lỗi xảy ra: {e}")

if __name__ == "__main__":
    chat_with_bot()
