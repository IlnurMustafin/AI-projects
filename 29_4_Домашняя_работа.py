#!/usr/bin/env python3
# good_medical_jokes_with_token.py
from huggingface_hub import login
from llama_cpp import Llama
import time

print("ГЕНЕРАТОР НАСТОЯЩИХ МЕДИЦИНСКИХ ШУТОК")
print("=" * 60)

# 1. АВТОРИЗАЦИЯ С ТОКЕНОМ
print("Авторизация в Hugging Face...")
try:
    # Вставьте свой токен здесь
    HF_TOKEN = ""
    login(token=HF_TOKEN)
    print("Авторизация успешна")
except Exception as e:
    print(f"Авторизация не удалась: {e}")
    print("Продолжаем без токена...")

# 2. ЗАГРУЗКА МОДЕЛИ
print("\nЗагружаю модель Saiga Mistral...")
model = Llama(
    model_path="model-q4_K.gguf",
    n_ctx=2048,
    n_threads=6,
    verbose=False
)
print("Модель загружена!")

# 3. ШАБЛОН ПРОМПТА ДЛЯ ШУТОК
prompt_template = """<s>system
Ты - профессиональный комик, специализирующийся на медицинском юморе.
Твоя задача - придумывать смешные, короткие шутки на медицинские темы.
user
Придумай смешную медицинскую шутку на тему "{theme}".
Шутка должна быть:
1. Короткой (1-2 предложения)
2. Остроумной и действительно смешной
3. Завершенной (не обрываться на полуслове)
4. В формате: "Шутка: [текст шутки]"
5. Без черного юмора

Пример хорошей шутки: "Почему врач носит маску? Чтобы пациенты не видели, как он улыбается, выписывая счет!"

Теперь придумай шутку на тему "{theme}":
assistant
Шутка:"""

# 4. ТЕМЫ ДЛЯ ШУТОК
themes = [
    "врачи и пациенты",
    "больничная жизнь", 
    "посещение поликлиники",
    "стоматологи",
    "аптека и лекарства"
]

print(f"\n🎪 Всего тем для шуток: {len(themes)}")
print("=" * 60)

# 5. ГЕНЕРАЦИЯ ШУТОК
all_jokes = []

for i, theme in enumerate(themes, 1):
    print(f"\n Шутка #{i}: {theme}")
    
    # Заполняем шаблон
    prompt = prompt_template.format(theme=theme)
    
    print("Генерирую...")
    start_time = time.time()
    
    # Генерация с хорошими параметрами
    result = model(
        prompt,
        max_tokens=100,
        temperature=0.9,        # Высокая креативность
        top_p=0.95,             # Лучшее качество
        top_k=40,
        repeat_penalty=1.15,    # Сильный штраф за повторения
        stop=["</s>", "user\n", "User:", "Тема:", "###"],
        echo=False
    )
    
    generation_time = time.time() - start_time
    
    # Получаем результат
    joke = result['choices'][0]['text'].strip()
    
    # Очищаем
    joke = joke.replace("assistant", "").replace("Assistant:", "").strip()
    
    # Если шутка не начинается с "Шутка:", добавляем
    if not joke.startswith("Шутка:"):
        joke = f"Шутка: {joke}"
    
    print(f"{generation_time:.1f} сек")
    print(f"{joke}")
    print("-" * 60)
    
    all_jokes.append((theme, joke))

# 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
print("\nСохраняю все шутки в файл...")

timestamp = time.strftime("%Y%m%d_%H%M%S")
filename = f"медицинские_шутки_{timestamp}.txt"

with open(filename, "w", encoding="utf-8") as f:
    f.write("=" * 70 + "\n")
    f.write("СБОРНИК МЕДИЦИНСКИХ ШУТОК\n")
    f.write(f"Сгенерировано: {time.strftime('%d.%m.%Y %H:%M:%S')}\n")
    f.write(f"Модель: Saiga Mistral 7B GGUF\n")
    f.write(f"Токен HF: {'использован' if 'HF_TOKEN' in locals() else 'не использован'}\n")
    f.write("=" * 70 + "\n\n")
    
    for i, (theme, joke) in enumerate(all_jokes, 1):
        f.write(f"ШУТКА #{i}\n")
        f.write(f"Тема: {theme}\n")
        f.write(f"{joke}\n")
        f.write("-" * 70 + "\n\n")

print(f"Все шутки сохранены в файл: {filename}")

# 7. ИТОГИ
print("ИТОГОВЫЙ ОТЧЕТ:")
print(f"Всего сгенерировано: {len(all_jokes)} шуток")
print(f"Среднее время генерации: {sum([t for _, t in [(j, 0) for j in all_jokes]])/len(all_jokes):.1f} сек")
print("\nГЕНЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО! ")