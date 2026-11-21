# DataCon PDF Chemistry Extractor

Исходное название проекта — Million miles of SMILES.

Локальный пайплайн с несколькими агентами для извлечения химических данных из PDF-статей.

Проект умеет:

- парсить PDF в Markdown и таблицы через Docling/Camelot;
- выбирать релевантные чанки статьи через TF-IDF или HF embeddings;
- извлекать small-molecule данные, например `SMILES / pMIC`;
- извлекать nanozyme/nanocatalyst данные: формулы, размеры, `Km`, `Vmax`, yield, conversion, selectivity;
- валидировать SMILES через RDKit, а формулы и физические величины через Python validators;
- анализировать изображения/CV для scale bar и размеров частиц;
- показывать trace агентов, rejected rows, conflicts, evidence и quality flags;
- экспортировать clean/rejected/conflicts/vision/agent trace CSV/JSON.

## Установка и быстрый запуск

Рекомендуется Python 3.12: эту версию выбирают скрипты установки при наличии
`uv`. Без `uv` используется установленный `python`; скрипты предупреждают
о возможных проблемах установки RDKit на Python 3.14. Зависимости перечислены
в [requirements.txt](requirements.txt).

Сначала скачайте проект (нужен Git):

```sh
git clone https://github.com/Aillight/datacon-pdf-chemistry-extractor.git
cd datacon-pdf-chemistry-extractor
```

Windows PowerShell:

```powershell
.\scripts\setup_local.ps1
.\scripts\run_ui.ps1
```

macOS / Linux (из корня репозитория):

```sh
bash scripts/setup_local.sh
bash scripts/run_ui.sh
```

Если команда `python` отсутствует, создайте окружение вручную с Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/run_ui.sh
```

Или напрямую в Windows:

```powershell
.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Откройте [localhost:8501](http://localhost:8501).

## API-ключи

Откройте Streamlit, раскройте `API keys for this run` и вставьте ключ
только на текущий запуск. Поддерживаются:

- `HF_TOKEN` для Hugging Face;
- `OPENROUTER_API_KEY` для OpenRouter;
- `OPENAI_API_KEY` для OpenAI.

Более постоянный вариант: создайте локальный `.env` рядом с `README.md`:

```powershell
Copy-Item .env.example .env
```

В macOS / Linux: `cp .env.example .env`. Заполните нужные поля:

```env
HF_TOKEN=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4.1-mini
OPENAI_API_KEY=
```

`.env` исключён через `.gitignore`. Не добавляйте ключи в исходники,
коммиты, скриншоты и примеры результатов. При выборе внешнего extractor
фрагменты статьи отправляются соответствующему провайдеру.
Для проверки разбора без LLM выберите `None` в UI или `--extractor empty`
в CLI; это не выполняет извлечение моделью.

## Как пользоваться UI

1. Выберите `Domain`: `Oxazolidinones`, `Benzimidazoles` или `Nanozymes`.
2. Загрузите PDF.
3. Выберите `Extractor`: `Hugging Face`, `OpenRouter`, `OpenAI` или `None`.
4. При необходимости вставьте API-ключ в `API keys for this run`.
5. Нажмите `Run`.

Основные вкладки:

- `Results`: чистая итоговая таблица;
- `Evidence`: полный текстовый фрагмент для выбранной строки;
- `Rejected`: строки, которые не прошли схему или валидаторы;
- `Conflicts`: конфликтующие значения;
- `Vision`: результаты CV;
- `Agents`: trace multi-agent pipeline;
- `Chunks`: какие чанки были выбраны retrieval-модулем;
- `Log`: полный лог запуска.

## CLI

Пример запуска одной статьи в Windows:

```powershell
.venv\Scripts\python.exe scripts\run_article_pipeline.py path\to\article.pdf --domain Nanozymes --extractor openrouter --max-chunks 3
```

macOS / Linux:

```sh
.venv/bin/python scripts/run_article_pipeline.py path/to/article.pdf --domain Nanozymes --extractor openrouter --max-chunks 3
```

Доступные extractor backend: `auto | empty | hf | openrouter | openai`.

Артефакты сохраняются в `outputs/articles/<pdf-name>/`:

- `clean.csv`;
- `rejected.csv`;
- `conflicts.csv`;
- `prepared.md`;
- `chunks.json`;
- `retrieval.json`;
- `agent_trace.json`;
- `extraction_states.json`;
- `manifest.json`.

## Локальный API

API запускается отдельно от UI. Из корня репозитория с установленными
зависимостями:

```sh
.venv/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
```

В Windows замените `.venv/bin/python` на `.venv\Scripts\python.exe`.
Справка API: [localhost:8000/docs](http://localhost:8000/docs),
проверка доступности: [localhost:8000/health](http://localhost:8000/health).
В API есть маршруты `/parse/pdf`, `/chunk/text` и `/agent/run`;
полный пайплайн статьи доступен через UI и CLI.

## Проверки

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
```

В macOS / Linux:

```sh
.venv/bin/python -m unittest discover -s tests
```

## Архитектура

```text
agent/
  extraction_graph.py       small molecules: extractor -> RDKit critic -> retry
  nano_extraction_graph.py  nano/catalyst: extractor -> formula/sanity critic -> retry
  article_supervisor.py     article-level multi-agent trace

backend/
  parsing/                  PDF -> Markdown/chunks
  retrieval.py              TF-IDF / HF embedding chunk ranking
  aggregation.py            small-molecule aggregation
  nano_aggregation.py       nano aggregation + rejected/conflicts
  vision/                   CV panel/scale/particle analysis

ui/
  app.py                    Streamlit UI
  pipeline.py               full article orchestration
```

Дополнительные материалы: [тестовый набор](data/test_set/README.md),
[презентация](presentation/DataCon_Extraction_Agent_Presentation.pptx).
Скрипты установки, запуска и оценки находятся в `scripts/`, проверки — в
`tests/`. Локальные результаты в `outputs/` исключены из Git.

## Ограничения

- Camelot lattice может предупреждать про Ghostscript. Это не блокирует весь pipeline: используется fallback.
- OpenAI extractor пока подключён только для small molecules. Для Nanozymes используйте Hugging Face или OpenRouter.
- Для воспроизводимости сохраняйте `manifest.json`, `agent_trace.json` и `clean.csv`.
