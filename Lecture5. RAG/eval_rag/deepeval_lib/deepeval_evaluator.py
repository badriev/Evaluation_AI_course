from typing import List, Union, Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric
)
from deepeval import evaluate
from deepeval_custom_llm import create_proxy_model, ProxyLLM


def create_metrics(
    model: Union[str, ProxyLLM],
    metrics_list: List[str] = None,
    threshold: float = 0.7
):
    if metrics_list is None:
        metrics_list = ['answer_relevancy', 'faithfulness']

    METRICS = {
        'answer_relevancy': AnswerRelevancyMetric,
        'faithfulness': FaithfulnessMetric,
        'contextual_relevancy': ContextualRelevancyMetric,
        'contextual_recall': ContextualRecallMetric,
        'contextual_precision': ContextualPrecisionMetric
    }

    metrics = []
    for metric_name in metrics_list:
        if metric_name in METRICS:
            metric = METRICS[metric_name](
                threshold=threshold,
                model=model,
                include_reason=True
            )
            metrics.append(metric)

    return metrics


if __name__ == "__main__":

    # ============= ПРИМЕР 1: С обычной OpenAI =============
    print("📊 Пример 1: Обычная OpenAI модель")

    # Создаем метрики с обычной моделью
    metrics = create_metrics(
        model="gpt-4o-mini",  # ← Обычная OpenAI
        metrics_list=['answer_relevancy', 'faithfulness'],
        threshold=0.7
    )

    # Создаем тест кейс
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
        expected_output="You are eligible for a 30 day full refund at no extra cost.",
        retrieval_context=[
            "All customers are eligible for a 30 day full refund at no extra cost."]
    )

    # Оцениваем (DeepEval сам выведет результаты)
    evaluate(test_cases=[test_case], metrics=metrics)

    # ============= ПРИМЕР 2: С ПРОКСИ =============
    print("\n" + "="*60)
    print("📊 Пример 2: Кастомная прокси модель")

    # Создаем прокси модель
    proxy_model = create_proxy_model(
        model="gpt-4o-mini",
        api_key="sk-proxy-your-key",
        base_url="http://5.11.83.110:8000"
    )

    # Создаем метрики с прокси моделью
    metrics = create_metrics(
        model=proxy_model,  # ← Кастомная прокси модель
        metrics_list=['answer_relevancy', 'faithfulness'],
        threshold=0.7
    )

    # Оцениваем
    evaluate(test_cases=[test_case], metrics=metrics)

    # ============= ПРИМЕР 3: Без wrapper функции =============
    print("\n" + "="*60)
    print("📊 Пример 3: Прямое использование DeepEval (вообще без обертки)")

    # Прямо создаем метрику
    metric = AnswerRelevancyMetric(
        model=proxy_model,  # Можно proxy_model или "gpt-4o-mini"
        threshold=0.7,
        include_reason=True
    )

    # Измеряем
    metric.measure(test_case)

    # Смотрим результаты
    print(f"\nScore: {metric.score:.3f}")
    print(f"Reason: {metric.reason}")
