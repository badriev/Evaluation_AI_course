# ai_proxy_client.py
from openai import OpenAI
import os


# OpenAI через прокси
client = OpenAI(
    # Личный ключ от прокси
    api_key=os.getenv(
        "PROXY_API_KEY", "sk-proxy-"),
    # прокси сервер
    base_url=os.getenv("PROXY_BASE_URL", "http://5.11.83.110:8000")
)


def openai_chat(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.choices[0].message.content
    print(answer)
    return answer


def openai_chat_v2(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content
    print(answer)
    return answer


def openai_chat_v3(prompt: str, model: str, temperature: float = 0.1) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )

    answer = response.choices[0].message.content
    return answer
