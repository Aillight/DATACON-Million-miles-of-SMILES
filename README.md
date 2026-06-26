# Будущий агент

Модульный MVP ИИ-агента для датакона ИТМО 2026

## Структура проекта

```text
.
├── agents/
│   ├── core.py        # Основная логика агентов
│   ├── prompts.py     # Промпты / инструкции
│   └── tools.py       # Инструменты агента
├── backend/
│   └── api.py         # чё то под капотом
├── ui/
│   └── app.py         # Streamlit UI наверное
├── notebooks/         # Jupyter-ноутбуки ?
├── .env.example       # Шаблон переменных окружения
├── requirements.txt   # зависимости
└── README.md          # читай меня
```

## Роли в команде

- ML Engineer: `agent/`
- Backend/Data Engineer: `backend/`
- Frontend Engineer: `ui/`
- Research/QA: `notebooks/`

## Цель MVP

Быстро собрать рабочий прототип ИИ-агента с разделением ответственности по папкам, чтобы минимизировать конфликты при параллельной разработке.
