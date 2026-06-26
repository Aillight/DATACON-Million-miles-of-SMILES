# Hackathon AI Agent MVP

Модульный MVP ИИ-агента для хакатона.

## Структура проекта

```text
.
├── agent/
│   ├── core.py        # Основная логика агента
│   ├── prompts.py     # Промпты
│   └── tools.py       # Инструменты агента
├── backend/
│   ├── api.py         # FastAPI backend
│   └── parsing/       # PDF parsing pipeline
├── ui/
│   └── app.py         # Streamlit UI
├── notebooks/         # Jupyter-ноутбуки и эксперименты
├── .env.example       # Шаблон переменных окружения
├── requirements.txt
└── README.md
```

## Установка

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Для Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Настройка окружения

Скопируйте шаблон переменных окружения:

```bash
cp .env.example .env
```

Заполните `.env` локальными значениями, например `OPENAI_API_KEY`.

## Запуск backend

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

После запуска API будет доступен по адресу:

```text
http://localhost:8000
```

Проверка статуса API:

```bash
curl http://localhost:8000/health
```

Тестовый парсинг PDF:

```bash
curl -X POST http://localhost:8000/parse/pdf \
  -F "file=@sample.pdf"
```

Текущая тестовая реализация использует Docling для извлечения текста и Camelot для таблиц. Camelot сначала пробует `lattice`, затем `stream`.

Для режима `lattice` у Camelot на локальной машине могут потребоваться системные зависимости вроде Ghostscript. Если они не установлены, backend вернет предупреждение и попробует режим `stream`.

## Запуск UI

В отдельном терминале:

```bash
streamlit run ui/app.py
```

Интерфейс будет доступен по адресу:

```text
http://localhost:8501
```

## Роли в команде

- ML Engineer: `agent/`
- Backend/Data Engineer: `backend/`
- Frontend Engineer: `ui/`
- Research/QA: `notebooks/`

## Цель MVP

Быстро собрать рабочий прототип ИИ-агента с разделением ответственности по папкам, чтобы минимизировать конфликты при параллельной разработке.
