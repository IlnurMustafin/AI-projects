from config import dp
from aiogram import F
from aiogram.filters.command import Command
from aiogram import types
from handler_start import new_quiz

# Хэндлер на команду /quiz
@dp.message(F.text=="Начать игру")
@dp.message(Command("quiz"))
async def cmd_quiz(message: types.Message):

    await message.answer(f"Давайте начнем квиз!")
    await new_quiz(message)
