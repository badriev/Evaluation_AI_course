from ai_client import openai_chat_v3


def basic_llm_judge(question, answer):
    """
    Простая оценка ответа с помощью LLM-судьи.
    """
    prompt = f"""
Оцени качество ответа на вопрос от 1 до 10:

Вопрос: {question}
Ответ: {answer}

Критерии: точность, полнота, читаемость
Верни только число от 1 до 10 и краткое объяснение.
"""

    response = openai_chat_v3(prompt, model="gpt-4o-mini", temperature=0.1)
    return response


# Пример использования
question = "Как работает фотосинтез?"
answer = "Фотосинтез -- это процесс, при котором растения используют свет для создания глюкозы из CO2 и воды."

result = basic_llm_judge(question, answer)
print(f"Базовая оценка: {result}")
