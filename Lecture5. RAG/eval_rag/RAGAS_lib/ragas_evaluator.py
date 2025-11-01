"""
RAGAS Evaluator с поддержкой прокси (упрощенная версия)
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import asyncio

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
    Faithfulness
)
from ragas_proxy import create_proxy_llm, create_proxy_embeddings


class RagasEvaluator:
    """RAGAS Evaluator с поддержкой прокси"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        use_proxy: bool = False,
        proxy_api_key: Optional[str] = None,
        proxy_base_url: Optional[str] = None
    ):
        """
        Инициализация

        Args:
            model: Название модели
            temperature: Температура
            use_proxy: Использовать прокси
            proxy_api_key: API ключ прокси
            proxy_base_url: URL прокси
        """
        self.model = model
        self.temperature = temperature
        self.metric_configs = {}

        # Создаем LLM и embeddings
        if use_proxy:
            print(f"🌐 Используется прокси: {proxy_base_url}")
            self.evaluator_llm = create_proxy_llm(
                model=model,
                api_key=proxy_api_key or "sk-proxy-your-key",
                base_url=proxy_base_url or "http://5.11.83.110:8000",
                temperature=temperature
            )
            self.evaluator_embeddings = create_proxy_embeddings(
                api_key=proxy_api_key or "sk-proxy-your-key",
                base_url=proxy_base_url or "http://5.11.83.110:8000"
            )
        else:
            # Обычный OpenAI
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            llm = ChatOpenAI(model=model, temperature=temperature)
            self.evaluator_llm = LangchainLLMWrapper(llm)

            embeddings = OpenAIEmbeddings()
            self.evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

        print(f"✅ RAGAS настроен: {model}")

    def configure_metrics(self, metrics_config: Dict[str, Dict[str, Any]]):
        """Настройка метрик"""
        self.metric_configs = metrics_config

    def _create_metric(self, metric_name: str):
        """Создать метрику"""
        if metric_name == 'context_precision':
            return LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        elif metric_name == 'context_recall':
            return LLMContextRecall(llm=self.evaluator_llm)
        elif metric_name == 'response_relevancy':
            return ResponseRelevancy(
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )
        elif metric_name == 'faithfulness':
            return Faithfulness(llm=self.evaluator_llm)

    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Оценить один случай

        Args:
            question: Вопрос пользователя
            answer: Ответ RAG системы
            contexts: Список извлеченных контекстов
            ground_truth: Правильный ответ

        Returns:
            Словарь с оценками метрик
        """
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            reference=ground_truth,
            retrieved_contexts=contexts
        )

        results = {}

        for metric_name, config in self.metric_configs.items():
            if not config.get('enabled', True):
                continue

            print(f"   Оценка: {metric_name}")
            metric = self._create_metric(metric_name)
            score = await metric.single_turn_ascore(sample)
            results[metric_name] = float(score)

        return results

    def evaluate_batch(self, test_cases: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Пакетная оценка

        Args:
            test_cases: Список словарей с ключами:
                - question: str
                - answer: str
                - contexts: List[str]
                - ground_truth: str

        Returns:
            DataFrame с результатами
        """
        async def evaluate_all():
            all_results = []
            for i, case in enumerate(test_cases, 1):
                print(f"\n📝 {i}/{len(test_cases)}: {case['question'][:50]}...")

                scores = await self.evaluate_single(
                    question=case['question'],
                    answer=case['answer'],
                    contexts=case['contexts'],
                    ground_truth=case['ground_truth']
                )

                scores['question'] = case['question']
                all_results.append(scores)

            return all_results

        results = asyncio.run(evaluate_all())
        return pd.DataFrame(results)


# ==================== ПРИМЕР ====================

if __name__ == "__main__":

    # Просто список словарей!
    test_cases = [
        {
            'question': "What is the capital of France?",
            'answer': "Paris is the capital of France.",
            'contexts': ["Paris is the capital and largest city of France."],
            'ground_truth': "Paris"
        },

        # Пример 2: Хороший ответ с дополнительной информацией
        {
            'question': "Who invented the telephone?",
            'answer': "Alexander Graham Bell invented the telephone in 1876.",
            'contexts': [
                "Alexander Graham Bell was a Scottish-born inventor who is credited with inventing the telephone.",
                "The first successful telephone call was made on March 10, 1876."
            ],
            'ground_truth': "Alexander Graham Bell"
        }
    ]

    # Пример 1: С обычной OpenAI
    print("="*60)
    print("📊 Пример 1: Обычная OpenAI")

    evaluator = RagasEvaluator(model="gpt-4o-mini", use_proxy=False)

    evaluator.configure_metrics({
        'faithfulness': {'enabled': True, 'threshold': 0.7},
        'response_relevancy': {'enabled': True, 'threshold': 0.7}
    })

    df = evaluator.evaluate_batch(test_cases)
    print("\n✅ Результаты:")
    print(df)

    # Пример 2: С прокси
    print("\n" + "="*60)
    print("📊 Пример 2: Через прокси")

    evaluator_proxy = RagasEvaluator(
        model="gpt-4o-mini",
        use_proxy=True,
        proxy_api_key="sk-proxy-your-key",
        proxy_base_url="http://5.11.83.110:8000"
    )

    evaluator_proxy.configure_metrics({
        'faithfulness': {'enabled': True},
        'response_relevancy': {'enabled': True, 'threshold': 0.7}
    })

    df = evaluator_proxy.evaluate_batch(test_cases)
    print("\n✅ Результаты:")
    print(df)

    # Пример 3: Оценка одного случая напрямую
    print("\n" + "="*60)
    print("📊 Пример 3: Один случай")

    async def test_single():
        scores = await evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            contexts=[
                "RAG is a technique that combines retrieval and generation."],
            ground_truth="RAG is Retrieval-Augmented Generation"
        )
        print(f"\n✅ Оценки: {scores}")

    asyncio.run(test_single())
