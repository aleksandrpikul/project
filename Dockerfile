FROM python:3.9-slim

WORKDIR /app

# Копируем весь код приложения
COPY . .

# Запуск приложения
CMD ["python", "test_heart_attack.py"]

# Для безопасности: не используем root-пользователя
USER nonrootuser
