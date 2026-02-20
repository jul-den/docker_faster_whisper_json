FROM python:3.10-slim

# Устанавливаем ffmpeg (нужен для работы с видео/аудио)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Устанавливаем faster-whisper и requests
RUN pip install --no-cache-dir faster-whisper requests

# Отключаем буферизацию вывода Python
ENV PYTHONUNBUFFERED=1

# Создаём рабочую директорию
WORKDIR /app

# Копируем скрипт
COPY process_webinar.py .

# Точка входа
CMD ["python", "process_webinar.py"]
