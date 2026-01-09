import getpass
import os
import requests
import re
import gradio as gr
from openai import OpenAI

# Запрос ввода ключа от OpenAI
os.environ["OPENAI_API_KEY"] = getpass.getpass("Введите OpenAI API Key:")

# Словарь с вашим нейро-сотрудником
models = [
    {
        "doc": "https://docs.google.com/document/d/1DuV1jehC8uiKITD7Gb95nkbmnIz70284fJmt77Ejqs8/edit",
        "prompt": '''Ты — нейро-ассистент по SQL и дашбордам компании DataShop.
Твоя целевая аудитория: сотрудники отделов маркетинга, продаж, поддержки и начинающие аналитики.
Твоя задача — помогать коллегам:
1. Писать корректные SQL-запросы к базам данных компании
2. Понимать, как рассчитываются ключевые метрики (TR, AOV, CR, LTV, Refund Rate)
3. Работать с дашбордом "Продажи и трафик"
4. Решать типовые ошибки при работе с запросами

ОТВЕЧАЙ СТРОГО НА ОСНОВЕ ПРЕДОСТАВЛЕННОЙ ДОКУМЕНТАЦИИ.
Твои ответы должны быть технически точными, лаконичными и практичными.
Используй код SQL только если он есть в документации.
Если в документации нет точного ответа на вопрос, скажи: "В документации этой информации нет. Обратитесь к старшему аналитику."
Не придумывай названия таблиц, полей, формул или процессов.
Используй только факты из документации.

Контекст из документации:''',
        "name": "Нейро-ассистент по SQL и дашбордам",
        "query": "Как посчитать дневную выручку? Напиши SQL-запрос."
    }
]

class SimpleAssistant:
    def __init__(self):
        self.log = ''
        self.document_text = ''
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.vsegpt.ru/v1")
        # Автоматически загружаем документ при инициализации
        self.auto_load_document()
    
    def auto_load_document(self):
        """Автоматически загружает документ при запуске"""
        try:
            print("🔄 Автоматическая загрузка документа...")
            url = models[0]['doc']
            
            # Извлекаем document ID из URL Google Docs
            match_ = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
            if match_ is None:
                self.log = "❌ Неверная ссылка на Google Docs"
                print(self.log)
                return
            
            doc_id = match_.group(1)
            
            # Скачиваем документ
            response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
            response.raise_for_status()
            
            self.document_text = response.text
            self.log = f"✅ Документ автоматически загружен при запуске\nРазмер: {len(self.document_text)} символов\n"
            print("✅ Документ успешно загружен!")
            
        except requests.exceptions.RequestException as e:
            self.log = f"❌ Ошибка сети при загрузке документа: {str(e)}"
            print(self.log)
        except Exception as e:
            self.log = f"❌ Ошибка при автоматической загрузке документа: {str(e)}"
            print(self.log)
    
    def load_document(self, url):
        """Ручная загрузка документа (если нужно перезагрузить)"""
        try:
            # Извлекаем document ID из URL Google Docs
            match_ = re.search(r'/document/d/([a-zA-Z0-9-_]+)', url)
            if match_ is None:
                return "❌ Неверная ссылка на Google Docs"
            
            doc_id = match_.group(1)
            
            # Скачиваем документ
            response = requests.get(f'https://docs.google.com/document/d/{doc_id}/export?format=txt')
            response.raise_for_status()
            
            self.document_text = response.text
            self.log = f"✅ Документ загружен успешно\nРазмер: {len(self.document_text)} символов\n"
            return self.log
            
        except Exception as e:
            return f"❌ Ошибка при загрузке документа: {str(e)}"
    
    def get_document_preview(self):
        """Возвращает предпросмотр документа"""
        if not self.document_text:
            return "Документ не загружен"
        
        preview = self.document_text[:500]  # Первые 500 символов
        lines = preview.split('\n')[:10]    # Первые 10 строк
        return "📄 Предпросмотр документа:\n" + "\n".join(lines) + "\n..."
    
    def ask_question(self, system_prompt, user_question):
        if not self.document_text:
            return ["❌ Документ не загружен. Нажмите 'Загрузить документацию'", "Документ не загружен"]
        
        try:
            # Разбиваем документ на фрагменты для лучшего контекста
            # Ищем релевантные части документа по ключевым словам
            relevant_parts = self.find_relevant_parts(user_question)
            
            # Если нашли релевантные части, используем их, иначе берем начало документа
            if relevant_parts:
                context = relevant_parts
            else:
                # Берем первые 4000 символов документа
                context = self.document_text[:4000]
            
            messages = [
                {"role": "system", "content": system_prompt + f"\n\n{context}"},
                {"role": "user", "content": user_question}
            ]
            
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )
            
            answer = completion.choices[0].message.content
            self.log += f"\n✅ Запрос выполнен: '{user_question[:50]}...'\nТокенов использовано: {completion.usage.total_tokens}\n"
            
            return [answer, self.log]
            
        except Exception as e:
            return [f"❌ Ошибка при выполнении запроса: {str(e)}", self.log]
    
    def find_relevant_parts(self, question):
        """Находит релевантные части документа по ключевым словам из вопроса"""
        if not self.document_text:
            return ""
        
        # Ключевые слова для поиска
        keywords = []
        question_lower = question.lower()
        
        # Определяем ключевые слова в зависимости от вопроса
        if any(word in question_lower for word in ['выручк', 'revenue', 'tr', 'доход']):
            keywords.extend(['выручк', 'revenue', 'tr', 'доход'])
        if any(word in question_lower for word in ['чек', 'aov', 'средн', 'average']):
            keywords.extend(['чек', 'aov', 'средн', 'average'])
        if any(word in question_lower for word in ['конверс', 'cr', 'convers']):
            keywords.extend(['конверс', 'cr', 'convers'])
        if any(word in question_lower for word in ['sql', 'запрос', 'select', 'query']):
            keywords.extend(['sql', 'запрос', 'select', 'query'])
        if any(word in question_lower for word in ['дашборд', 'dashboard', 'виджет']):
            keywords.extend(['дашборд', 'dashboard', 'виджет'])
        if any(word in question_lower for word in ['ошибк', 'error', 'проблем', 'debug']):
            keywords.extend(['ошибк', 'error', 'проблем', 'debug'])
        
        if not keywords:
            return ""
        
        # Ищем строки, содержащие ключевые слова
        lines = self.document_text.split('\n')
        relevant_lines = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in keywords):
                relevant_lines.append(line)
                if len(relevant_lines) >= 20:  # Ограничиваем количество строк
                    break
        
        return "\n".join(relevant_lines[:20]) if relevant_lines else ""

# Создаем ассистента (автоматически загрузит документ)
print("🚀 Инициализация Нейро-ассистента...")
assistant = SimpleAssistant()

# Создаем интерфейс
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Нейро-ассистент по SQL и дашбордам")
    gr.Markdown("### Помощник для работы с SQL-запросами и аналитикой DataShop")
    
    # Отображаем статус загрузки
    with gr.Row():
        status_box = gr.Textbox(
            label="Статус загрузки документа",
            value=assistant.log if assistant.log else "Загрузка...",
            lines=3
        )
    
    # Предпросмотр документа
    with gr.Row():
        preview_btn = gr.Button("📋 Показать предпросмотр документа")
        preview_box = gr.Textbox(
            label="Предпросмотр документа",
            lines=6,
            interactive=False
        )
    
    with gr.Row():
        train_btn = gr.Button("🔄 Перезагрузить документацию", variant="primary")
    
    with gr.Row():
        question = gr.Textbox(
            label="Ваш вопрос к ассистенту",
            value="Как посчитать дневную выручку?",
            lines=3
        )
    
    with gr.Row():
        ask_btn = gr.Button("💭 Задать вопрос", variant="secondary")
    
    with gr.Row():
        with gr.Column(scale=2):
            answer_box = gr.Textbox(
                label="Ответ ассистента",
                lines=10
            )
        with gr.Column(scale=1):
            log_box = gr.Textbox(
                label="Логи работы",
                lines=10
            )
    
    # Обработчики
    def train():
        result = assistant.load_document(models[0]['doc'])
        return result, ""
    
    def ask(q):
        return assistant.ask_question(models[0]['prompt'], q)
    
    def show_preview():
        return assistant.get_document_preview()
    
    def update_status():
        return assistant.log
    
    # Привязываем события
    train_btn.click(
        fn=train,
        outputs=[status_box, preview_box]
    )
    
    ask_btn.click(
        fn=ask,
        inputs=[question],
        outputs=[answer_box, log_box]
    )
    
    preview_btn.click(
        fn=show_preview,
        outputs=preview_box
    )
    
    # Автоматически обновляем статус при загрузке
    demo.load(
        fn=update_status,
        outputs=status_box
    )

# Запускаем
if __name__ == "__main__":
    print("=" * 50)
    print("Нейро-ассистент по SQL и дашбордам")
    print("=" * 50)
    print(f"Документ: {models[0]['name']}")
    print(f"Ссылка: {models[0]['doc']}")
    print("\n📊 После запуска откройте: http://localhost:7860")
    print("=" * 50)
    
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7860,
        show_error=True,
        share=False
    )