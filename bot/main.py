import asyncio
import logging
from config import *
from questions import *
from db import *
from handler_start import *
from handler_quiz import *
from quiz import *

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)


# Объект бота
bot = Bot(token=API_TOKEN)
# Диспетчер



# Запуск процесса поллинга новых апдейтов
async def main():

    # Запускаем создание таблицы базы данных
    await create_table()

    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())