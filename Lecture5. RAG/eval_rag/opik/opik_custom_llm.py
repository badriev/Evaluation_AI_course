"""
Кастомная модель Opik для работы через прокси сервер
Аналог DeepEval ProxyLLM, но для Opik
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI


class OpikProxyLLM:
    """
    Кастомная модель для Opik, которая работает через прокси сервер
    Совместима с интерфейсом Opik metrics
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        """
        Инициализация прокси модели

        Args:
            model: Название модели (gpt-4o-mini, gpt-3.5-turbo и т.д.)
            api_key: API ключ прокси (по умолчанию из переменной окружения)
            base_url: URL прокси сервера (по умолчанию из переменной окружения)
            temperature: Температура генерации
        """
        self.model_name = model
        self.temperature = temperature

        # Получаем настройки из переменных окружения или используем переданные
        self.api_key = api_key or os.getenv(
            "PROXY_API_KEY",
            "sk-proxy-your-key"
        )
        self.base_url = base_url or os.getenv(
            "PROXY_BASE_URL",
            "http://5.11.83.110:8000"
        )

        # Создаем клиент OpenAI с прокси настройками
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def generate_provider_response(self, prompt: str) -> Dict[str, Any]:
        """
        Генерация ответа через прокси (для совместимости с Opik)

        Args:
            prompt: Текст промпта

        Returns:
            Словарь с ответом в формате, ожидаемом Opik
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature
            )

            # Возвращаем в формате, совместимом с Opik
            return {
                "choices": [{
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }]
            }

        except Exception as e:
            print(f"❌ Ошибка при генерации через прокси: {e}")
            raise

    def __call__(self, prompt: str) -> str:
        """
        Прямой вызов модели (для простоты использования)

        Args:
            prompt: Текст промпта

        Returns:
            Сгенерированный текст
        """
        response = self.generate_provider_response(prompt)
        return response["choices"][0]["message"]["content"]


def create_opik_proxy_model(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.1
) -> OpikProxyLLM:
    """
    Helper функция для создания прокси модели для Opik

    Args:
        model: Название модели
        api_key: API ключ прокси (ваш Bearer токен)
        base_url: URL прокси сервера
        temperature: Температура генерации

    Returns:
        Экземпляр OpikProxyLLM
    """
    return OpikProxyLLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature
    )


# # Тестирование модели при запуске файла напрямую
# if __name__ == "__main__":
#     # Тест прокси модели
#     print("🔄 Тестирование OpikProxyLLM...")

#     # Создаем модель
#     proxy_model = create_opik_proxy_model(
#         model="gpt-4o-mini",
#         api_key="sk-proxy-your-key",
#         base_url="http://5.11.83.110:8000"
#     )

#     # Тестовый промпт
#     test_prompt = "Explain what is RAG in one sentence."

#     print(f"\n📝 Промпт: {test_prompt}")
#     print(f"🤖 Модель: {proxy_model.model_name}")
#     print(f"🌐 Прокси: {proxy_model.base_url}")
#     print(f"🔑 API Key: {proxy_model.api_key[:20]}...")

#     # Генерируем ответ
#     try:
#         response = proxy_model(test_prompt)
#         print(f"\n✅ Ответ: {response}")
#     except Exception as e:
#         print(f"\n❌ Ошибка: {e}")
