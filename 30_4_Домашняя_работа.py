#pip install gradio wikipedia transformers torch accelerate sentence-transformers

import gradio as gr
import wikipedia
import re
import torch
from typing import List, Dict, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SberDevicesModel:
    """Класс для работы с текстовой моделью от SberDevices/AI-Forever"""
    
    def __init__(self, model_name: str = "ai-forever/FRED-T5-1.7B"):
        """
        Инициализация модели SberDevices.
        FRED-T5-1.7B лучше подходит для инструкций и борьбы с галлюцинациями.
        Альтернатива: 'ai-forever/rugpt3large_based_on_gpt2'
        """
        logger.info(f"🚀 Загрузка модели SberDevices: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Используемое устройство: {self.device}")
        
        try:
            # Загружаем модель и токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Создаем пайплайн для генерации текста
            self.generator = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"✅ Модель {model_name} успешно загружена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
            logger.info("🔄 Используем резервный режим без нейросети...")
            self.generator = None
    
    def generate_with_context(self, question: str, context: str, max_length: int = 300) -> str:
        """Генерация ответа с использованием контекста для борьбы с галлюцинациями"""
        if self.generator is None:
            # Резервный режим: просто возвращаем факты
            return self._fallback_answer(context)
        
        try:
            # Формируем строгий промпт для борьбы с галлюцинациями
            prompt = self._create_strict_prompt(question, context)
            
            # Генерируем ответ с консервативными параметрами
            response = self.generator(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.2,  # Очень низкая температура для минимизации выдумок
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
                num_beams=3  # Поиск по лучам для лучшего качества
            )
            
            generated_text = response[0]['generated_text']
            
            # Извлекаем и очищаем ответ
            answer = self._extract_answer(generated_text, prompt)
            
            # Проверяем, не содержит ли ответ отказ
            if self._is_refusal(answer):
                return self._fallback_answer(context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            return self._fallback_answer(context)
    
    def _create_strict_prompt(self, question: str, context: str) -> str:
        """Создание строгого промпта для ограничения галлюцинаций"""
        return f"""Ты — ассистент, который отвечает на вопросы ТОЛЬКО на основе предоставленных фактов.
        
ФАКТЫ ИЗ WIKIPEDIA:
{context}

ВОПРОС: {question}

ИНСТРУКЦИИ:
1. Ответь ТОЛЬКО используя информацию из фактов выше.
2. Если в фактах нет ответа, скажи: "В предоставленных данных нет информации для ответа на этот вопрос."
3. Не добавляй информацию, которой нет в фактах.
4. Будь точным и кратким.

ОТВЕТ:"""
    
    def _extract_answer(self, generated_text: str, prompt: str) -> str:
        """Извлечение ответа из сгенерированного текста"""
        if "ОТВЕТ:" in generated_text:
            return generated_text.split("ОТВЕТ:")[-1].strip()
        elif prompt in generated_text:
            return generated_text.replace(prompt, "").strip()
        else:
            return generated_text.strip()
    
    def _is_refusal(self, answer: str) -> bool:
        """Проверяет, является ли ответ отказом из-за отсутствия информации"""
        refusal_phrases = [
            "нет информации",
            "не могу ответить",
            "не указано",
            "не найдено",
            "данных нет"
        ]
        return any(phrase in answer.lower() for phrase in refusal_phrases)
    
    def _fallback_answer(self, context: str) -> str:
        """Резервный ответ на основе фактов (если модель не загрузилась)"""
        facts = context.split('\n')
        if len(facts) > 3:
            return "На основе загруженных статей:\n\n" + "\n".join(facts[:3])
        return "Ответ на основе фактов из Wikipedia."

class WikipediaParser:
    """Парсер Wikipedia с обработкой ошибок"""
    
    def __init__(self, llm_model=None):
        # Устанавливаем русский язык
        wikipedia.set_lang("ru")
        self.llm = llm_model  # Модель для генерации ответов
        logger.info("Инициализирован парсер Wikipedia (русский язык)")
    
    def parse_topic(self, topic: str, num_articles: int = 3) -> tuple[str, Dict]:
        """Парсинг статей по теме, возвращает результат и данные"""
        logger.info(f"Начинаем парсинг темы: '{topic}'")
        
        if not topic or not topic.strip():
            return "❌ Пожалуйста, введите тему для поиска", {}
        
        articles = {}  # Создаем новый словарь для каждой темы
        
        try:
            # Ищем статьи
            logger.info(f"Поиск статей по теме: {topic}")
            search_results = wikipedia.search(topic, results=num_articles)
            
            if not search_results:
                logger.warning(f"По теме '{topic}' ничего не найдено")
                return self._get_search_suggestions(topic), {}
            
            logger.info(f"Найдено статей: {len(search_results)}")
            
            result_text = f"## 📚 Результаты поиска по теме: '{topic}'\n\n"
            result_text += f"✅ Найдено статей: {len(search_results)}\n\n"
            successful = 0
            
            for i, title in enumerate(search_results, 1):
                try:
                    logger.info(f"Загрузка статьи {i}/{len(search_results)}: '{title}'")
                    
                    # Получаем статью
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    # Очищаем содержимое
                    cleaned_content = self._clean_content(page.content)
                    
                    # Извлекаем факты
                    facts = self._extract_facts(cleaned_content)
                    
                    # Сохраняем статью
                    article_data = {
                        'title': title,
                        'content': cleaned_content[:5000],
                        'summary': page.summary,
                        'url': page.url,
                        'facts': facts,
                        'original_topic': topic  # Сохраняем исходную тему
                    }
                    
                    articles[title] = article_data
                    successful += 1
                    
                    # Формируем результат
                    result_text += f"### 📖 {title}\n"
                    result_text += f"**Кратко:** {page.summary[:300]}...\n"
                    result_text += f"**Ссылка:** [Открыть статью]({page.url})\n"
                    result_text += f"**Извлечено фактов:** {len(facts)}\n\n"
                    
                    logger.info(f"Статья '{title}' успешно загружена")
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    logger.warning(f"Неоднозначность для '{title}': {e.options[:3]}")
                    result_text += f"⚠️ **Неоднозначный запрос:** '{title}'\n"
                    result_text += f"   *Варианты:* {', '.join(e.options[:3])}\n\n"
                    
                except wikipedia.exceptions.PageError:
                    logger.warning(f"Страница '{title}' не найдена")
                    result_text += f"⚠️ **Страница не найдена:** '{title}'\n\n"
                    
                except Exception as e:
                    logger.error(f"Ошибка при загрузке '{title}': {str(e)}")
                    result_text += f"⚠️ **Ошибка загрузки:** '{title}'\n\n"
            
            if successful == 0:
                return "❌ Не удалось загрузить ни одной статьи. Попробуйте другую тему.", {}
            
            result_text += f"---\n✨ **Готово!** Загружено {successful} статей. Теперь можете задавать вопросы по теме **'{topic}'**."
            return result_text, articles
            
        except Exception as e:
            logger.error(f"Критическая ошибка при парсинге: {str(e)}")
            return f"❌ Произошла ошибка: {str(e)}\n\nПроверьте подключение к интернету.", {}
    
    def _clean_content(self, content: str) -> str:
        """Очистка содержимого статьи"""
        # Удаляем разделы "См. также", "Примечания" и т.д.
        sections_to_remove = [
            "== См. также ==", "== Примечания ==", "== Ссылки ==",
            "== Литература ==", "== Источники ==", "== Примечания и ссылки ==",
            "== Внешние ссылки ==", "== Библиография =="
        ]
        
        for section in sections_to_remove:
            if section in content:
                content = content.split(section)[0]
        
        # Удаляем вики-разметку
        content = re.sub(r'\{\{.*?\}\}', '', content)
        content = re.sub(r'\[\[.*?\|', '', content)
        content = re.sub(r'\[\[|\]\]', '', content)
        
        # Удаляем лишние пробелы и переносы
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _extract_facts(self, content: str, max_facts: int = 12) -> List[str]:
        """Извлечение фактов из текста"""
        # Разделяем на предложения
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        facts = []
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            words = sentence.split()
            # Берем предложения от 4 до 35 слов
            if 4 <= len(words) <= 35:
                clean_sentence = sentence.strip()
                # Убедимся, что предложение заканчивается точкой
                if not clean_sentence.endswith(('.', '!', '?')):
                    clean_sentence += '.'
                facts.append(clean_sentence)
                
                if len(facts) >= max_facts:
                    break
        
        return facts
    
    def _get_search_suggestions(self, topic: str) -> str:
        """Предложения по поиску"""
        suggestions = [
            "## ❌ По запросу ничего не найдено",
            "",
            "**Возможные причины:**",
            "1. Слишком узкая или специфичная тема",
            "2. Опечатка в запросе",
            "3. Тема на английском языке",
            "",
            "**Попробуйте эти темы (гарантированно работают):**",
            "- `Искусственный интеллект`",
            "- `Python`",
            "- `Москва`",
            "- `Солнце`",
            "- `История России`",
            "- `Физика`",
            "- `Математика`"
        ]
        
        return "\n".join(suggestions)
    
    def search_in_articles(self, question: str, articles: Dict) -> Dict:
        """Поиск информации в загруженных статьях"""
        if not articles:
            return {'found': False, 'facts': [], 'sources': [], 'topic': None}
        
        # Получаем тему из первой статьи
        first_article = next(iter(articles.values()))
        current_topic = first_article.get('original_topic', 'неизвестная тема')
        
        logger.info(f"Поиск ответа на вопрос: '{question}' (тема: {current_topic})")
        
        question_lower = question.lower().strip()
        # Удаляем знаки вопроса в конце
        question_lower = question_lower.rstrip('?')
        
        # Разбиваем на ключевые слова (игнорируем предлоги и союзы)
        stop_words = {'что', 'как', 'где', 'когда', 'почему', 'зачем', 'кто', 'чей', 'чье', 'чем', 'на', 'в', 'с', 'по', 'о', 'об', 'от', 'до', 'из', 'у', 'и', 'или', 'но', 'а', 'же', 'ли', 'бы', 'б', 'не', 'ни', 'да', 'то', 'же', 'вот', 'вон', 'это', 'эти', 'этого'}
        question_words = [w for w in question_lower.split() if w not in stop_words and len(w) > 2]
        
        relevant_facts = []
        sources = []
        
        for title, article in articles.items():
            content_lower = article['content'].lower()
            
            # Проверяем, есть ли совпадения ключевых слов
            matches = 0
            for word in question_words:
                if word in content_lower:
                    matches += 1
            
            if matches > 0:
                logger.info(f"Найдены совпадения ({matches}) в статье: '{title}'")
                
                # Ищем релевантные факты
                for fact in article.get('facts', []):
                    fact_lower = fact.lower()
                    fact_matches = 0
                    for word in question_words:
                        if word in fact_lower:
                            fact_matches += 1
                    
                    if fact_matches > 0:
                        # Добавляем с оценкой релевантности
                        relevant_facts.append({
                            'text': fact,
                            'relevance': fact_matches,
                            'source': title
                        })
                
                sources.append({
                    'title': title,
                    'url': article['url'],
                    'preview': article['summary'][:200] + '...'
                })
        
        # Сортируем факты по релевантности
        relevant_facts.sort(key=lambda x: x['relevance'], reverse=True)
        
        logger.info(f"Найдено фактов: {len(relevant_facts)}, источников: {len(sources)}")
        
        return {
            'found': len(relevant_facts) > 0 or len(sources) > 0,
            'facts': [f['text'] for f in relevant_facts[:5]],
            'sources': sources[:3],
            'topic': current_topic
        }
    
    def generate_answer(self, question: str, articles: Dict) -> str:
        """Генерация ответа на основе фактов с использованием модели SberDevices"""
        if not articles:
            return "⚠️ **Сначала загрузите статьи!**\n\nНажмите 'Распарсить данные' для выбора темы."
        
        # Ищем релевантную информацию
        info = self.search_in_articles(question, articles)
        
        if not info['found']:
            topic = info.get('topic', 'текущей')
            return (
                f"## ❌ Информация не найдена\n\n"
                f"В загруженных статьях по теме **'{topic}'** нет информации по вопросу:\n"
                f"**«{question}»**\n\n"
                f"**Что можно сделать:**\n"
                f"1. Перезагрузите статьи по другой теме\n"
                f"2. Переформулируйте вопрос\n"
                f"3. Используйте более общие ключевые слова\n"
                f"4. Задайте вопрос, связанный с текущей темой"
            )
        
        # Подготавливаем контекст для модели
        context = "Факты из Wikipedia:\n"
        for i, fact in enumerate(info['facts'], 1):
            context += f"{i}. {fact}\n"
        
        # Используем модель SberDevices для генерации ответа
        if self.llm and self.llm.generator is not None:
            logger.info(f"Используем модель SberDevices для генерации ответа")
            answer = self.llm.generate_with_context(question, context)
            
            # Добавляем источники к сгенерированному ответу
            answer_with_sources = f"## 🤖 Ответ модели SberDevices\n\n"
            answer_with_sources += f"**Вопрос:** {question}\n\n"
            answer_with_sources += f"**Ответ:** {answer}\n\n"
            
        else:
            # Если модель не загрузилась, используем факты напрямую
            logger.info("Используем ответ на основе фактов (без нейросети)")
            answer_with_sources = f"## 📊 Ответ на основе фактов\n\n"
            answer_with_sources += f"**Вопрос:** {question}\n\n"
            answer_with_sources += "**Факты из статей:**\n"
            for i, fact in enumerate(info['facts'], 1):
                answer_with_sources += f"{i}. {fact}\n"
        
        # Добавляем источники
        answer_with_sources += "\n### 📚 Источники информации:\n"
        for source in info['sources']:
            answer_with_sources += f"• [{source['title']}]({source['url']})\n"
        
        answer_with_sources += "\n---\n"
        answer_with_sources += "⚠️ **Борьба с галлюцинациями:** Этот ответ основан ТОЛЬКО на фактах из Wikipedia."
        
        return answer_with_sources

# Функции для Gradio с сохранением состояния
def parse_data(topic: str, current_articles: gr.State) -> tuple[str, gr.State]:
    """Обработчик парсинга с сохранением состояния"""
    # Инициализируем парсер с моделью
    parser = WikipediaParser(sber_model)
    result, new_articles = parser.parse_topic(topic)
    
    # Отображаем, какая тема была загружена
    if new_articles:
        display_result = f"## 📥 Загружена новая тема\n\n{result}"
    else:
        display_result = result
    
    return display_result, new_articles

def ask_question(question: str, current_articles: gr.State) -> str:
    """Обработчик вопросов с использованием текущих статей"""
    if not question or not question.strip():
        return "❌ Пожалуйста, введите вопрос"
    
    # Создаем парсер с моделью
    parser = WikipediaParser(sber_model)
    return parser.generate_answer(question, current_articles)

def update_topic_display(articles):
    """Обновление отображения текущей темы"""
    if articles:
        first_article = next(iter(articles.values()))
        topic = first_article.get('original_topic', 'неизвестная тема')
        return f"**Текущая тема:** {topic}\n\n**Загружено статей:** {len(articles)}"
    return "**Текущая тема:** не загружена"

def clear_data():
    """Очистка данных"""
    return {}, "✅ Данные очищены. Введите новую тему.", "**Текущая тема:** не загружена"

# Создаем глобальный экземпляр модели SberDevices
sber_model = SberDevicesModel("ai-forever/FRED-T5-1.7B")

# Создаем интерфейс Gradio
with gr.Blocks(title="Wikipedia QA - SberDevices Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 Wikipedia QA System с моделью SberDevices
    ### Домашнее задание: Борьба с галлюцинациями в LLM
    """)
    
    # Состояние для хранения текущих статей
    current_articles = gr.State(value={})
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Шаг 1: Загрузите данные")
            topic_input = gr.Textbox(
                label="Тема для поиска в Wikipedia",
                placeholder="Введите тему на русском языке...",
                value="Искусственный интеллект"
            )
            
            with gr.Row():
                parse_btn = gr.Button("🚀 Распарсить данные", variant="primary", scale=2)
                clear_btn = gr.Button("🔄 Очистить", variant="secondary", scale=1)
            
            current_topic_display = gr.Markdown(
                label="Текущая тема",
                value="**Текущая тема:** не загружена"
            )
            
            parse_output = gr.Markdown(
                label="Результат загрузки",
                value="Введите тему и нажмите 'Распарсить данные'..."
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🤖 Шаг 2: Задайте вопрос")
            question_input = gr.Textbox(
                label="Ваш вопрос",
                placeholder="Задайте вопрос по загруженной теме...",
                lines=4,
                value="Что такое искусственный интеллект?"
            )
            
            ask_btn = gr.Button("🔍 Получить ответ от модели SberDevices", variant="secondary", size="lg")
            
            answer_output = gr.Markdown(
                label="Ответ системы",
                value="Сначала загрузите статьи, затем задайте вопрос..."
            )
    
    # Примеры
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 💡 Примеры тем:")
            gr.Examples(
                examples=[
                    ["Искусственный интеллект"],
                    ["Солнце"],
                    ["Python"],
                    ["Москва"],
                    ["История России"]
                ],
                inputs=[topic_input],
                label="Нажмите для загрузки темы"
            )
        
        with gr.Column():
            gr.Markdown("### 💡 Примеры вопросов:")
            gr.Examples(
                examples=[
                    ["Что такое искусственный интеллект?"],
                    ["Из чего состоит Солнце?"],
                    ["Что такое Python?"],
                    ["Что такое Москва?"],
                    ["Когда началась история России?"]
                ],
                inputs=[question_input],
                label="Нажмите для быстрой вставки"
            )
    
    # Обработчики событий
    # При парсинге обновляем статьи и тему
    parse_btn.click(
        fn=parse_data,
        inputs=[topic_input, current_articles],
        outputs=[parse_output, current_articles]
    ).then(
        fn=update_topic_display,
        inputs=[current_articles],
        outputs=[current_topic_display]
    )
    
    # При вопросе используем текущие статьи
    ask_btn.click(
        fn=ask_question,
        inputs=[question_input, current_articles],
        outputs=[answer_output]
    )
    
    # Кнопка очистки
    clear_btn.click(
        fn=clear_data,
        outputs=[current_articles, parse_output, current_topic_display]
    )
    
    # Информация о системе
    gr.Markdown("""
    ---
    
    ### 🛡️ Методы борьбы с галлюцинациями:
    
    | Метод | Описание | Реализация в системе |
    |-------|----------|---------------------|
    | **Модель SberDevices** | Использование русскоязычной LLM | FRED-T5-1.7B для генерации ответов |
    | **Факт-чекинг** | Проверка информации по авторитетным источникам | Использование Wikipedia как источника данных |
    | **Строгий промптинг** | Жесткие инструкции для модели | Промпты ограничивают ответы только фактами |
    | **Контекстуализация** | Привязка ответа к конкретной теме | Отслеживание текущей темы |
    | **Указание источников** | Прозрачность происхождения информации | Ссылки на конкретные статьи Wikipedia |
    
    ### 🔧 Технические детали:
    - **Модель:** FRED-T5-1.7B от SberDevices/AI-Forever
    - **Тип:** Seq2Seq (T5 архитектура)
    - **Параметры генерации:** temperature=0.2, num_beams=3 (для минимизации галлюцинаций)
    - **Источник данных:** Русская Wikipedia
    - **Фреймворк:** Transformers + Gradio
    
    ### 📊 Как это работает:
    1. **Парсинг Wikipedia** → Загрузка статей по выбранной теме
    2. **Извлечение фактов** → Автоматическое выделение ключевых утверждений
    3. **Подготовка контекста** → Формирование строгого промпта для модели
    4. **Генерация ответа** → Модель SberDevices создает ответ на основе фактов
    5. **Проверка и вывод** → Добавление источников и валидация ответа
    
    ---
    
    **Разработано для домашнего задания 'Галлюцинации в LLM' | Курс: Разработчик нейросетей**
    """)

# Запуск приложения
if __name__ == "__main__":
    print("=" * 70)
    print("🚀 СИСТЕМА ДЛЯ БОРЬБЫ С ГАЛЛЮЦИНАЦИЯМИ В LLM")
    print(f"📊 Используемая модель: FRED-T5-1.7B от SberDevices/AI-Forever")
    print("=" * 70)
    print("\n📋 КАК РАБОТАЕТ:")
    print("1. Введите тему (например: 'Искусственный интеллект')")
    print("2. Нажмите 'Распарсить данные' для загрузки статей из Wikipedia")
    print("3. Задайте вопрос по теме")
    print("4. Модель SberDevices сгенерирует ответ на основе фактов из статей")
    print("\n💡 ПОДСКАЗКА: Используйте примеры тем и вопросов для быстрого тестирования")
    print("=" * 70)
    
    # Тестируем подключение к Wikipedia
    print("\n🔍 Тестирование подключения к Wikipedia...")
    try:
        wikipedia.set_lang("ru")
        test_search = wikipedia.search("тест", results=1)
        if test_search:
            print("✅ Подключение к Wikipedia работает!")
        else:
            print("⚠️ Проверьте подключение к интернету")
    except:
        print("❌ Не удалось подключиться к Wikipedia")
    
    print(f"\n💻 Запуск Gradio интерфейса на http://127.0.0.1:7860")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )