from config import DB_NAME
import aiosqlite

async def create_table():
    # Создаем соединение с базой данных (если она не существует, она будет создана)
    async with aiosqlite.connect(DB_NAME) as db:
        # Создаем таблицу
        await db.execute('''CREATE TABLE IF NOT EXISTS quiz_state (user_id INTEGER PRIMARY KEY, question_index INTEGER)''')
        # Сохраняем изменения
        await db.commit()

async def get_quiz_index(user_id):
     # Подключаемся к базе данных
     async with aiosqlite.connect(DB_NAME) as db:
        # Получаем запись для заданного пользователя
        async with db.execute('SELECT question_index FROM quiz_state WHERE user_id = (?)', (user_id, )) as cursor:
            # Возвращаем результат
            results = await cursor.fetchone()
            if results is not None:
                return results[0]
            else:
                return 0


async def update_quiz_index(user_id, index):
    # Создаем соединение с базой данных (если она не существует, она будет создана)
    async with aiosqlite.connect(DB_NAME) as db:
        # Вставляем новую запись или заменяем ее, если с данным user_id уже существует
        await db.execute('INSERT OR REPLACE INTO quiz_state (user_id, question_index) VALUES (?, ?)', (user_id, index))
        # Сохраняем изменения
        await db.commit()


async def update_score(user_id, is_correct):
    async with aiosqlite.connect(DB_NAME) as db:
        if is_correct:
            await db.execute('INSERT OR REPLACE INTO quiz_results (user_id, correct_answers, total_questions) VALUES (?, COALESCE((SELECT correct_answers FROM quiz_results WHERE user_id = ?), 0) + 1, COALESCE((SELECT total_questions FROM quiz_results WHERE user_id = ?), 0) + 1)', (user_id, user_id, user_id))
        else:
            await db.execute('INSERT OR REPLACE INTO quiz_results (user_id, correct_answers, total_questions) VALUES (?, COALESCE((SELECT correct_answers FROM quiz_results WHERE user_id = ?), 0), COALESCE((SELECT total_questions FROM quiz_results WHERE user_id = ?), 0) + 1)', (user_id, user_id, user_id))
        await db.commit()


async def reset_score(user_id):
    async with aiosqlite.connect(DB_NAME) as db:
        await db.execute('INSERT OR REPLACE INTO quiz_results (user_id, correct_answers, total_questions) VALUES (?, 0, 0)', (user_id,))
        await db.commit()


async def get_score(user_id):
    async with aiosqlite.connect(DB_NAME) as db:
        async with db.execute('SELECT correct_answers, total_questions FROM quiz_results WHERE user_id = ?', (user_id,)) as cursor:
            result = await cursor.fetchone()
            return {'correct': result[0], 'total': result[1]} if result else {'correct': 0, 'total': 0}
