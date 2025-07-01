from aiogram import types, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart, StateFilter, CommandObject, CREATOR
from aiogram.fsm.context import FSMContext
from aiogram.filters.command import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder, ReplyKeyboardBuilder
from database import quiz_data
from service import generate_options_keyboard, get_question, new_quiz, get_quiz_index, update_quiz_index, get_len_quiz, get_correct_answer

router = Router()

len_quiz = get_len_quiz()

@router.callback_query(F.data == "right_answer")
async def right_answer(callback: types.CallbackQuery):
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )

    await callback.message.answer("Верно!")
    current_question_index, current_score = await get_quiz_index(callback.from_user.id)

    # Увеличиваем очки за правильный ответ
    current_score += 1
    current_question_index += 1
    await update_quiz_index(callback.from_user.id, current_question_index, current_score)



    if current_question_index < len_quiz:
        await get_question(callback.message, callback.from_user.id)
    else:
        await callback.message.answer(f"Это был последний вопрос. Квиз завершен! Ваш результат: {current_score} из {len_quiz}!")
  
@router.callback_query(F.data == "wrong_answer")
async def wrong_answer(callback: types.CallbackQuery):
    await callback.bot.edit_message_reply_markup(
        chat_id=callback.from_user.id,
        message_id=callback.message.message_id,
        reply_markup=None
    )

    current_question_index, current_score = await get_quiz_index(callback.from_user.id)
    correct_option = await get_correct_answer(callback.from_user.id) #quiz_data[current_question_index]['correct_option']



    
    await callback.message.answer(f"Неправильно. Правильный ответ: {correct_option}")

    # Увеличиваем индекс вопроса, очки не увеличиваем
    current_question_index += 1
    await update_quiz_index(callback.from_user.id, current_question_index, current_score)

    if current_question_index < len_quiz:
        await get_question(callback.message, callback.from_user.id)
    else:
        await callback.message.answer(f"Это был последний вопрос. Квиз завершен! Ваш результат: {current_score} из {len_quiz}!")

# Хэндлер на команду /start
@router.message(Command("start"))
async def cmd_start(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Начать игру"))
    await message.answer("Добро пожаловать в квиз!", reply_markup=builder.as_markup(resize_keyboard=True))


# Хэндлер на команду /quiz
@router.message(F.text=="Начать игру")
@router.message(Command("quiz"))
async def cmd_quiz(message: types.Message):
    
    await message.answer(f"Давайте начнем квиз!")
    await new_quiz(message)
    

