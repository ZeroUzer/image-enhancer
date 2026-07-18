# Image Enhancer

Автоматическое улучшение изображений с помощью нейросети.

## Датасет для обучения

Модель обучалась на датасете **Landscape Pictures** с Kaggle.

- Источник: [Landscape Pictures | Kaggle](https://www.kaggle.com/datasets/arnaud58/landscape-pictures)
- Формат: JPG
- Количество изображений: 4319
- Размер: 655 МБ
- Содержание: фотографии природных ландшафтов (горы, леса, моря, пустыни, пляжи, острова)

**Важно:** Для работы веб-приложения скачивать датасет не требуется. Модель уже обучена и встроена в проект.

## Возможности

- Улучшение яркости, контраста и цветности
- Пакетная обработка до 10 изображений
- Поддержка JPG, PNG, BMP, TIFF, WEBP, JFIF
- Асинхронная обработка с отображением прогресса
- REST API для интеграции

## Установка и запуск

1. Клонируй репозиторий:
```bash
git clone https://github.com/ZeroUzer/image-enhancer.git
cd image-enhancer

2. Создай виртуальное окружение и установи зависимости:

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

3. Запусти сервер:

cd webapp
python app.py

4. Открой в браузере: http://127.0.0.1:5000

