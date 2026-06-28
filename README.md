# DataCon ChemX Agent

Локальный multi-agent пайплайн для извлечения химических данных из PDF-статей.

Проект умеет:
- парсить PDF в Markdown и таблицы через Docling/Camelot;
- выбирать релевантные чанки статьи через TF-IDF или HF embeddings;
- извлекать small-molecule данные, например `SMILES / pMIC`;
- извлекать nanozyme/nanocatalyst данные: формулы, размеры, `Km`, `Vmax`, yield, conversion, selectivity;
- валидировать SMILES через RDKit, а формулы и физические величины через Python validators;
- анализировать изображения/CV для scale bar и размеров частиц;
- показывать trace агентов, rejected rows, conflicts, evidence и quality flags;
- экспортировать clean/rejected/conflicts/vision/agent trace CSV/JSON.

## Быстрый Запуск

Windows PowerShell:

```powershell
.\scripts\setup_local.ps1
.\scripts\run_ui.ps1
```

Или напрямую:

```powershell
.venv\Scripts\python.exe -m streamlit run ui/app.py
```

Откройте:

```text
http://localhost:8501
```

## API-Ключи

Самый простой способ: откройте Streamlit, раскройте `API keys for this run` и вставьте ключ только на текущий запуск.

Поддерживаются:
- `HF_TOKEN` для Hugging Face;
- `OPENROUTER_API_KEY` для OpenRouter;
- `OPENAI_API_KEY` для OpenAI.

Более постоянный вариант: создайте локальный `.env` рядом с `README.md`:

```powershell
Copy-Item .env.example .env
```

Заполните нужные поля:

```env
HF_TOKEN=
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openai/gpt-4.1-mini
OPENAI_API_KEY=
```

`.env` не коммитится в git.

## Как Пользоваться UI

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

Пример запуска одной статьи:

```powershell
.venv\Scripts\python.exe scripts\run_article_pipeline.py path\to\article.pdf --domain Nanozymes --extractor openrouter --max-chunks 3
```

Доступные extractor backend:

```text
auto | empty | hf | openrouter | openai
```

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

## Проверки

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests
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

## Заметки

- Camelot lattice может предупреждать про Ghostscript. Это не блокирует весь pipeline: используется fallback.
- OpenAI extractor пока подключён только для small molecules. Для Nanozymes используйте Hugging Face или OpenRouter.
- Для воспроизводимости сохраняйте `manifest.json`, `agent_trace.json` и `clean.csv`.
