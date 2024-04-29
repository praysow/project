from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT-2 모델 및 토크나이저 불러오기
model_name = "gpt2-medium"  # 사용할 GPT-2 모델 선택
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 대화 함수 정의
def chat():
    print("안녕하세요! 대화를 시작할까요? (종료하려면 '그만'을 입력하세요)")
    while True:
        user_input = input("사용자: ")
        if user_input == "그만":
            print("대화를 종료합니다.")
            break
        # 질문에 대한 답변 생성
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, temperature=0.7, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print("AI: ", generated_text)

# 대화 시작
chat()
