from  database import pool, execute_update_query, execute_select_query
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from aiogram import types
from database import quiz_data




def generate_options_keyboard(answer_options, right_answer):
    builder = InlineKeyboardBuilder()

    for option in answer_options:
        builder.add(types.InlineKeyboardButton(
            text=option,
            callback_data="right_answer" if option == right_answer else "wrong_answer")
        )

    builder.adjust(1)
    return builder.as_markup()


async def new_quiz(message):
    user_id = message.from_user.id
    current_question_index = 0
    await update_quiz_index(user_id, current_question_index, 0)  # Обнуляем счет
    await get_question(message, user_id)

async def get_quiz_index(user_id):
    get_user_index = f"""
        DECLARE $user_id AS Uint64;

        SELECT question_index, score
        FROM `quiz_state`
        WHERE user_id == $user_id;
    """
    results = execute_select_query(pool, get_user_index, user_id=user_id)
    
    if len(results) == 0:
        return 0, 0  # Возвращаем 0 для индекса и очков
    if results[0]["question_index"] is None:
        return 0, results[0]["score"]
    return results[0]["question_index"], results[0]["score"]
    
    

async def update_quiz_index(user_id, question_index, score):
    set_quiz_state = f"""
        DECLARE $user_id AS Uint64;
        DECLARE $question_index AS Uint64;
        DECLARE $score AS Uint64;

        UPSERT INTO `quiz_state` (`user_id`, `question_index`, `score`)
        VALUES ($user_id, $question_index, $score);
    """

    execute_update_query(
        pool,
        set_quiz_state,
        user_id=user_id,
        question_index=question_index,
        score=score,
    )


async def get_question(message, user_id):

    current_question_index, _ = await get_quiz_index(user_id)  # Извлекаем только индекс, игнорируя очки
    # Извлекаем только индекс, игнорируя очки
    
    
    
    if current_question_index >= get_len_quiz():  # Добавлено условие для проверки диапазона
        await message.answer("Квиз завершен! Вы ответили на все вопросы.")
        return
    



    get_question_query = f"""
        DECLARE $question_index AS Uint64;

        SELECT id, question, options, correct_option 
        FROM `questions`
        WHERE id = $question_index

    """

    result = execute_select_query(
        pool,
        get_question_query,
        question_index=current_question_index,
        )

     

    if len(result) == 0:
        return 0, 0  # Возвращаем 0 для индекса и очков   
    
    correct_index = result[0]['correct_option'] #quiz_data[current_question_index]['correct_option']
    opts = result[0]['options'].split(',') #quiz_data[current_question_index]['options']
    question = result[0]['question']
    kb = generate_options_keyboard(opts, opts[correct_index])

    

    await message.answer(f"{question}", reply_markup=kb)   


def get_len_quiz():

    get_len = "SELECT COUNT(*) as cnt FROM `questions`"
    result = execute_select_query(pool,get_len)
    

    return result[0]['cnt']

async def get_correct_answer(user_id):

    current_question_index, _ = await get_quiz_index(user_id)  # Извлекаем только индекс, игнорируя очки
    # Извлекаем только индекс, игнорируя очки
    
    
    
    if current_question_index >= get_len_quiz():  # Добавлено условие для проверки диапазона
        await message.answer("Квиз завершен! Вы ответили на все вопросы.")
        return
    



    get_question_query = f"""
        DECLARE $question_index AS Uint64;

        SELECT options, correct_option 
        FROM `questions`
        WHERE id = $question_index

    """

    result = execute_select_query(
        pool,
        get_question_query,
        question_index=current_question_index,
        )

    if len(result) == 0:
        return 0, 0  # Возвращаем 0 для индекса и очков   
        
    
    correct_index = result[0]['correct_option'] #quiz_data[current_question_index]['correct_option']
    
    opts = result[0]['options'].split(',')

    print("get_correct_answer, opts: ", opts)

    return opts[correct_index]