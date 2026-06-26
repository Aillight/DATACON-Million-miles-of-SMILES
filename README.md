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

## Локальный запуск

Ветка с тестовой реализацией:

```bash
git checkout deployment
```

Быстрая настройка окружения:

```bash
bash scripts/setup_local.sh
```

Для Windows PowerShell:

```powershell
.\scripts\setup_local.ps1
```

Или двойным кликом / из `cmd.exe`:

```cmd
scripts\setup_local.bat
```

Запуск backend:

```bash
bash scripts/run_backend.sh
```

Для Windows PowerShell:

```powershell
.\scripts\run_backend.ps1
```

Или через `cmd.exe`:

```cmd
scripts\run_backend.bat
```

Backend будет доступен по адресам:

```text
http://localhost:8000
http://localhost:8000/docs
```

Запуск UI во втором терминале:

```bash
bash scripts/run_ui.sh
```

Для Windows PowerShell:

```powershell
.\scripts\run_ui.ps1
```

Или через `cmd.exe`:

```cmd
scripts\run_ui.bat
```

Streamlit UI будет доступен по адресу:

```text
http://localhost:8501
```

Первичная установка может занять время из-за `docling` и `camelot-py[cv]`. Для Camelot в режиме `lattice` на системе может потребоваться Ghostscript.

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

## Чанкование текста

Backend умеет делить статью по научным разделам и выбирать только релевантные блоки, например `Experimental Section` или `Results and Discussion`:

```bash
curl -X POST http://localhost:8000/chunk/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Experimental Section\n\nExample text.","target_sections":["experimental section"],"max_chars":1000,"overlap_chars":100}'
```

## MAS orchestration

Легковесный MAS-оркестратор реализован на чистом Python в `agent/`. Тестовый endpoint прогоняет цепочку `planner -> chunker -> synthesizer` и возвращает публичный trace выполнения без скрытых рассуждений модели:

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"task":"Prepare article context","text":"Results and Discussion\n\nImportant result."}'
```

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
