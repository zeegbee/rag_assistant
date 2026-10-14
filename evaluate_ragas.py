"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import math
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Относительные пути (data/docs.txt, ./chroma_db, кеш) считаются от папки проекта,
# поэтому скрипт можно запускать из любой директории
os.chdir(BASE_DIR)

# Загрузка переменных окружения из .env файла (лежит рядом со скриптом)
load_dotenv(BASE_DIR / '.env')

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference

from rag_pipeline import RAGPipeline


# Тестовые вопросы и эталонные ответы (составлены по data/docs.txt).
# Типы вопросов:
#   определение    - прямой вопрос по одному абзацу
#   факт           - конкретная деталь (число, год, название)
#   перефразировка - вопрос другими словами, без ключевых терминов из документа
#   сравнение      - ответ требует сопоставить несколько понятий
#   составной      - ответ собирается из нескольких абзацев
#   вне базы       - ответа в документах нет, система должна это признать
EVALUATION_DATA = [
    {
        "type": "определение",
        "question": "Что такое машинное обучение?",
        "ground_truth": (
            "Машинное обучение - это раздел искусственного интеллекта, который "
            "позволяет компьютерам обучаться на данных без явного программирования."
        ),
    },
    {
        "type": "определение",
        "question": "Что такое RAG и как он работает?",
        "ground_truth": (
            "RAG (Retrieval-Augmented Generation) комбинирует извлечение информации из "
            "базы знаний с генерацией текста: система сначала находит релевантные "
            "документы, затем использует их как контекст для генерации ответа."
        ),
    },
    {
        "type": "факт",
        "question": "В каком году и в какой статье была представлена архитектура трансформеров?",
        "ground_truth": (
            "Трансформеры были представлены в 2017 году в статье "
            "\"Attention is All You Need\"."
        ),
    },
    {
        "type": "факт",
        "question": "Какой размер чанка считается типичным при разбиении с фиксированным размером?",
        "ground_truth": "При фиксированном размере чанка обычно используют 300-500 символов.",
    },
    {
        "type": "перефразировка",
        "question": "Как сделать так, чтобы модель отвечала менее случайно и более предсказуемо?",
        "ground_truth": (
            "Нужно снизить температуру генерации до 0.1-0.3: низкая температура делает "
            "ответы более детерминированными и сфокусированными. Для фактических "
            "вопросов в RAG обычно используют низкую температуру."
        ),
    },
    {
        "type": "перефразировка",
        "question": "Как найти тексты, близкие по смыслу, если в них нет одинаковых слов?",
        "ground_truth": (
            "Использовать семантический поиск: тексты представляются векторами "
            "(embeddings), и близость измеряется между векторами, а не по совпадению "
            "ключевых слов. Для хранения и быстрого поиска ближайших соседей "
            "применяются векторные базы данных."
        ),
    },
    {
        "type": "сравнение",
        "question": "Чем обучение с учителем отличается от других типов машинного обучения?",
        "ground_truth": (
            "Обучение с учителем использует размеченные данные для предсказания "
            "результатов. Кроме него, существуют обучение без учителя и обучение "
            "с подкреплением."
        ),
    },
    {
        "type": "сравнение",
        "question": "Чем контекстуальные embeddings отличаются от Word2Vec и GloVe?",
        "ground_truth": (
            "Word2Vec и GloVe - ранние методы, которые дают слову одно фиксированное "
            "представление. В контекстуальных embeddings современных моделей "
            "представление слова зависит от контекста."
        ),
    },
    {
        "type": "составной",
        "question": "Какие способы помогают снизить затраты на API в RAG-системе?",
        "ground_truth": (
            "Кеширование ответов на частые вопросы снижает затраты на API и уменьшает "
            "latency; используются стратегии LRU, TTL и семантическое кеширование. "
            "Также важно эффективно управлять контекстом с учётом лимита токенов, "
            "который включает промпт, контекст и ответ."
        ),
    },
    {
        "type": "составной",
        "question": "Как адаптировать языковую модель под свою задачу без обучения с нуля?",
        "ground_truth": (
            "Можно использовать fine-tuning - дообучение предобученной модели на "
            "специфических данных; методы PEFT, такие как LoRA, делают его ещё "
            "эффективнее. Другой вариант - RAG, который позволяет отвечать на основе "
            "актуальной информации без дополнительного обучения. Также помогает "
            "prompt engineering: few-shot learning и chain-of-thought prompting."
        ),
    },
    {
        "type": "вне базы",
        "question": "Сколько параметров у модели GPT-4?",
        "ground_truth": "В документах нет информации о количестве параметров GPT-4.",
    },
    {
        "type": "вне базы",
        "question": "Как установить ChromaDB на Windows?",
        "ground_truth": "В документах нет информации об установке ChromaDB.",
    },
]

# Названия колонок метрик в результатах RAGAS
FAITHFULNESS_KEY = "faithfulness"
CONTEXT_PRECISION_KEY = "llm_context_precision_with_reference"


def prepare_dataset(pipeline: RAGPipeline, evaluation_data: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.

    Args:
        pipeline: RAG pipeline для получения ответов
        evaluation_data: список словарей с полями question и ground_truth

    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, item in enumerate(evaluation_data, 1):
        question = item["question"]
        print(f"  {i}/{len(evaluation_data)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - заранее подготовленный эталонный ответ
        ground_truths_list.append(item["ground_truth"])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data/docs.txt",
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_DATA)
    print("=" * 70)
    
    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision")
    print("   (Answer Relevancy отключена из-за несовместимости embeddings API)")
    print(f"   Вопросов: {len(EVALUATION_DATA)}")
    print("   (это займёт 1-2 минуты, так как RAGAS использует OpenAI для оценки)\n")
    
    # Используем только метрики, которые работают без проблем с embeddings
    print("   [+] Используем базовые метрики RAGAS\n")
    metrics_to_use = [Faithfulness(), LLMContextPrecisionWithReference()]
    
    # Запускаем оценку RAGAS
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)
    
    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    # Вычисляем средние значения метрик (игнорируя NaN)
    faithfulness_scores = result[FAITHFULNESS_KEY]
    context_precision_scores = result[CONTEXT_PRECISION_KEY]

    faithfulness_values = [
        v for v in faithfulness_scores
        if not (isinstance(v, float) and math.isnan(v))
    ]
    context_precision_values = [
        v for v in context_precision_scores
        if not (isinstance(v, float) and math.isnan(v))
    ]
    
    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values) 
        if faithfulness_values else 0
    )
    avg_context_precision = (
        sum(context_precision_values) / len(context_precision_values) 
        if context_precision_values else 0
    )
    
    # Выводим общие метрики
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    
    # Вычисляем и выводим средний балл
    avg_score = (avg_faithfulness + avg_context_precision) / 2
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    # Оценка качества системы
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")
    
    # Средние значения по типам вопросов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПО ТИПАМ ВОПРОСОВ")
    print("=" * 70)
    print(f"\n   {'Тип':<16}{'Faithfulness':>14}{'Context Precision':>20}")

    question_types = list(dict.fromkeys(item["type"] for item in EVALUATION_DATA))
    for q_type in question_types:
        indices = [i for i, item in enumerate(EVALUATION_DATA) if item["type"] == q_type]
        type_avgs = []
        for scores in (faithfulness_scores, context_precision_scores):
            values = [
                scores[i] for i in indices
                if not (isinstance(scores[i], float) and math.isnan(scores[i]))
            ]
            type_avgs.append(f"{sum(values) / len(values):.4f}" if values else "—")
        print(f"   {q_type:<16}{type_avgs[0]:>14}{type_avgs[1]:>20}")

    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)

    answers = dataset["answer"]
    for i, item in enumerate(EVALUATION_DATA):
        print(f"\n{i+1}. [{item['type']}] {item['question']}")

        # Для вопросов вне базы показываем ответ: система должна признать,
        # что информации нет, а не выдумывать
        if item["type"] == "вне базы":
            print(f"   Ответ: {answers[i][:200]}")
        
        # Faithfulness
        faith_val = faithfulness_scores[i]
        if not (isinstance(faith_val, float) and math.isnan(faith_val)):
            print(f"   Faithfulness:       {faith_val:.4f}")
        else:
            print(f"   Faithfulness:       не удалось вычислить")
        
        # Context Precision
        cp_val = context_precision_scores[i]
        if not (isinstance(cp_val, float) and math.isnan(cp_val)):
            print(f"   Context Precision:  {cp_val:.4f}")
        else:
            print(f"   Context Precision:  не удалось вычислить")
    
    # Пояснения к метрикам
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

ПРИМЕЧАНИЕ:
  Answer Relevancy временно отключена из-за несовместимости embeddings API
  в текущей версии RAGAS с langchain-openai.
    """)
    
    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()

