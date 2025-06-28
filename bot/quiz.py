from config import dp
from aiogram import F
from aiogram import Bot, types
from db import get_quiz_index
from questions import *
from db import update_quiz_index, update_score, get_score
from handler_start import get_question



@dp.callback_query(F.data == "right_answer")
async def right_answer(callback: types.CallbackQuery):
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )
    await callback.message.answer("Верно!")
    user_id = callback.from_user.id
    await update_score(user_id, True)  # Увеличиваем счетчик правильных ответов
    await handle_next_question(callback, user_id)


@dp.callback_query(F.data == "wrong_answer")
async def wrong_answer(callback: types.CallbackQuery):
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )
    user_id = callback.from_user.id
    await update_score(user_id, False)  # Увеличиваем счетчик неправильных ответов
    correct_option = quiz_data[await get_quiz_index(user_id)]['correct_option']
    await callback.message.answer(f"Неправильно. Правильный ответ: {quiz_data[await get_quiz_index(user_id)]['options'][correct_option]}")
    await handle_next_question(callback, user_id)


async def handle_next_question(callback, user_id):
    current_question_index = await get_quiz_index(user_id)
    current_question_index += 1
    await update_quiz_index(user_id, current_question_index)

    if current_question_index < len(quiz_data):
        await get_question(callback.message, user_id)
    else:
        await finish_quiz(callback.message, user_id)


async def finish_quiz(message, user_id):
    score = await get_score(user_id)
    await message.answer(f"Это был последний вопрос. Квиз завершен! Ваш результат: {score['correct']} из {score['total']}.")
