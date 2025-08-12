# LogBot — Telegram-бот для анализа логов через Loki, Prometheus и OpenAI

LogBot — это Telegram-бот и backend API на FastAPI для:
- поиска и агрегирования логов из **Loki**;
- получения метрик из **Prometheus**;
- интерактивной навигации по логам (хост → unit → окно) через inline-кнопки;
- получения сводок, причин, рекомендаций с помощью **OpenAI API**;
- быстрых выборок последних событий по уровням (**Error / Warning / Info**).

---

## Архитектура

- `backend/` — FastAPI шлюз к Loki/Prometheus + AI-анализ
- `appbot/` — Telegram-бот на aiogram v3
- `caddy/` — обратный прокси (опционально)
- `docker-compose.yml` — для локального запуска

```
opt/logbot
├── appbot/         # бот
│   └── bot.py
├── backend/        # backend API
│   └── main.py
├── caddy/
│   └── Caddyfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Быстрый старт

1. Скопируйте шаблон окружения:
   ```bash
   cp .env.example .env
   ```
   Отредактируйте значения токенов, URL и прочих параметров.

2. Запустите:
   ```bash
   docker compose up -d --build
   ```

3. Проверьте:
   ```bash
   curl -f http://localhost/healthz
   curl -f http://localhost:8000/healthz
   ```

4. Откройте Telegram и отправьте боту `/start`.

---

## Переменные окружения

| Переменная              | Описание |
|-------------------------|----------|
| TELEGRAM_BOT_TOKEN      | Токен Telegram-бота |
| BACKEND_URL             | URL backend API (по умолчанию http://backend:8000) |
| GRAFANA_URL             | URL Grafana для генерации ссылок Explore |
| ALLOWED_TELEGRAM_IDS    | Список ID пользователей, которым разрешён доступ (через запятую) |
| OPENAI_API_KEY          | API ключ OpenAI (для AI анализа) |

---

## Основные команды бота

- `/start` — главное меню
- `/menu` — главное меню
- `/analyze <host> <окно> <вопрос>` — анализ логов с AI
- Inline-кнопки:
  - **Error / Warning / Info** — последние события по уровню
  - **Сводка, RCA, Рекомендации, Аномалии** — AI-анализ

---

## Пример использования

1. Выберите хост
2. Выберите unit (или все)
3. Выберите временное окно (15m, 1h, 6h, 24h)
4. Просмотрите статистику по уровням и топ ошибок
5. Получите AI-анализ или примеры событий

---

## Лицензия

MIT
