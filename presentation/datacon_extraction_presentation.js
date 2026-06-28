const pptxgen = require("pptxgenjs");
const {
  warnIfSlideHasOverlaps,
  warnIfSlideElementsOutOfBounds,
} = require("./pptxgenjs_helpers");

const pptx = new pptxgen();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "DataCon Extraction Agent";
pptx.company = "DataCon 2026";
pptx.subject = "Multi-agent PDF extraction pipeline for chemistry article data";
pptx.title = "DataCon Extraction Agent";
pptx.lang = "ru-RU";
pptx.theme = {
  headFontFace: "Aptos Display",
  bodyFontFace: "Aptos",
  lang: "ru-RU",
};
pptx.defineLayout({ name: "LAYOUT_WIDE", width: 13.333, height: 7.5 });

const W = 13.333;
const H = 7.5;
const C = {
  ink: "111827",
  muted: "5B6472",
  light: "EEF2F7",
  line: "D6DDE8",
  blue: "2563EB",
  blue2: "DBEAFE",
  green: "16A34A",
  green2: "DCFCE7",
  amber: "D97706",
  amber2: "FEF3C7",
  red: "DC2626",
  red2: "FEE2E2",
  white: "FFFFFF",
};

const slides = [];

function addSlide(title, kicker = "") {
  const slide = pptx.addSlide();
  slide.background = { color: C.white };
  slide.addText(kicker, {
    x: 0.55,
    y: 0.22,
    w: 3.2,
    h: 0.22,
    fontFace: "Aptos",
    fontSize: 8.5,
    color: C.blue,
    bold: true,
    margin: 0,
    breakLine: false,
  });
  slide.addText(title, {
    x: 0.55,
    y: 0.5,
    w: 10.8,
    h: 0.48,
    fontFace: "Aptos Display",
    fontSize: 23,
    color: C.ink,
    bold: true,
    margin: 0,
    breakLine: false,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.55,
    y: 1.15,
    w: 12.25,
    h: 0,
    line: { color: C.line, width: 1 },
  });
  slide.addText("DataCon Extraction Agent", {
    x: 9.75,
    y: 7.05,
    w: 2.15,
    h: 0.16,
    fontSize: 7.5,
    color: C.muted,
    align: "right",
    margin: 0,
  });
  slides.push(slide);
  return slide;
}

function addFooterNumber(slide, number) {
  slide.addText(String(number).padStart(2, "0"), {
    x: 12.55,
    y: 7.02,
    w: 0.42,
    h: 0.18,
    fontSize: 7.5,
    color: C.muted,
    align: "right",
    margin: 0,
  });
}

function text(slide, value, x, y, w, h, opts = {}) {
  slide.addText(value, {
    x,
    y,
    w,
    h,
    fontFace: opts.fontFace || "Aptos",
    fontSize: opts.fontSize || 13,
    color: opts.color || C.ink,
    bold: opts.bold || false,
    margin: opts.margin ?? 0.03,
    breakLine: false,
    valign: opts.valign || "top",
    align: opts.align || "left",
    ...opts,
  });
}

function pill(slide, value, x, y, w, color = C.blue2, textColor = C.blue) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h: 0.28,
    rectRadius: 0.05,
    fill: { color },
    line: { color, transparency: 100 },
  });
  text(slide, value, x + 0.08, y + 0.055, w - 0.16, 0.12, {
    fontSize: 7.8,
    color: textColor,
    bold: true,
    align: "center",
  });
}

function card(slide, title, body, x, y, w, h, opts = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.04,
    fill: { color: opts.fill || C.white },
    line: { color: opts.line || C.line, width: 1 },
  });
  if (opts.accent) {
    slide.addShape(pptx.ShapeType.rect, {
      x,
      y,
      w: 0.08,
      h,
      fill: { color: opts.accent },
      line: { color: opts.accent, transparency: 100 },
    });
  }
  text(slide, title, x + 0.22, y + 0.18, w - 0.44, 0.28, {
    fontSize: opts.titleSize || 13,
    bold: true,
    color: opts.titleColor || C.ink,
  });
  text(slide, body, x + 0.22, y + 0.58, w - 0.44, h - 0.78, {
    fontSize: opts.bodySize || 10.5,
    color: opts.bodyColor || C.muted,
    breakLine: true,
    valign: "mid",
  });
}

function bulletList(slide, items, x, y, w, h, opts = {}) {
  const runs = [];
  items.forEach((item, idx) => {
    runs.push({
      text: item,
      options: {
        bullet: { indent: 10 },
        hanging: 3,
        breakLine: idx < items.length - 1,
      },
    });
  });
  slide.addText(runs, {
    x,
    y,
    w,
    h,
    fontFace: "Aptos",
    fontSize: opts.fontSize || 13,
    color: opts.color || C.ink,
    margin: 0.02,
    breakLine: false,
    paraSpaceAfterPt: opts.paraSpaceAfterPt || 5,
  });
}

function flow(slide, labels, x, y, w, h, colors = []) {
  const gap = 0.16;
  const boxW = (w - gap * (labels.length - 1)) / labels.length;
  labels.forEach((label, i) => {
    const bx = x + i * (boxW + gap);
    const color = colors[i] || C.blue2;
    const textColor = color === C.blue2 ? C.blue : C.ink;
    slide.addShape(pptx.ShapeType.roundRect, {
      x: bx,
      y,
      w: boxW,
      h,
      rectRadius: 0.04,
      fill: { color },
      line: { color: C.line, width: 1 },
    });
    text(slide, label, bx + 0.08, y + 0.18, boxW - 0.16, h - 0.28, {
      fontSize: 10,
      color: textColor,
      bold: true,
      align: "center",
      valign: "mid",
    });
    if (i < labels.length - 1) {
      // Intentional connector placed inside the gap between boxes.
      slide.addShape(pptx.ShapeType.rightArrow, {
        x: bx + boxW + 0.04,
        y: y + h / 2 - 0.1,
        w: Math.max(0.06, gap - 0.08),
        h: 0.2,
        fill: { color: C.line },
        line: { color: C.line, transparency: 100 },
      });
    }
  });
}

function metric(slide, value, label, x, y, w, color = C.blue) {
  text(slide, value, x, y, w, 0.55, {
    fontFace: "Aptos Display",
    fontSize: 26,
    bold: true,
    color,
    align: "center",
  });
  text(slide, label, x, y + 0.55, w, 0.35, {
    fontSize: 9,
    color: C.muted,
    align: "center",
  });
}

// 1
{
  const s = pptx.addSlide();
  s.background = { color: C.white };
  s.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.18, h: H, fill: { color: C.blue }, line: { color: C.blue } });
  text(s, "DataCon Extraction Agent", 0.75, 1.05, 9.8, 0.55, {
    fontFace: "Aptos Display",
    fontSize: 32,
    bold: true,
  });
  text(s, "Multi-agent extraction pipeline for chemistry PDFs", 0.78, 1.82, 8.6, 0.35, {
    fontSize: 15,
    color: C.muted,
  });
  flow(s, ["PDF article", "Agents", "Validation", "Clean CSV"], 0.78, 3.05, 8.8, 0.72);
  card(s, "Что делает система", "От PDF-статьи до проверяемой таблицы: Results, Evidence, Rejected, Conflicts, Vision и agent trace.", 0.78, 4.55, 5.4, 1.25, { accent: C.blue });
  card(s, "Главная идея", "LLM извлекает кандидатов, Python-валидаторы проверяют факты, UI показывает причину каждого решения.", 6.55, 4.55, 5.75, 1.25, { accent: C.green });
  text(s, "DataCon 2026", 0.78, 6.75, 3.2, 0.25, { fontSize: 10, color: C.muted });
  slides.push(s);
}

// 2
{
  const s = addSlide("Один слайд: что построено", "PROJECT SUMMARY");
  card(s, "Цель", "Сделать локальный инструмент, который превращает научную PDF-статью в проверяемый CSV для DataCon-задач извлечения данных.", 0.7, 1.55, 3.9, 1.5, { accent: C.blue });
  card(s, "Подход", "Multi-agent pipeline: parsing, retrieval, extraction, validation, vision, aggregation и quality scoring.", 4.85, 1.55, 3.9, 1.5, { accent: C.green });
  card(s, "Результат", "Streamlit UI + CLI + CSV/JSON артефакты + документация + воспроизводимый локальный запуск.", 9.0, 1.55, 3.5, 1.5, { accent: C.amber });
  metric(s, "93", "unit tests", 1.1, 4.0, 2.0, C.blue);
  metric(s, "3", "LLM providers", 4.1, 4.0, 2.0, C.green);
  metric(s, "2", "data domains", 7.1, 4.0, 2.0, C.amber);
  metric(s, "1", "simple UI", 10.1, 4.0, 2.0, C.red);
}

// 3
{
  const s = addSlide("Проблема", "WHY IT MATTERS");
  bulletList(s, [
    "PDF-статьи содержат нужные данные в тексте, таблицах, формулах и картинках.",
    "LLM без валидаторов часто даёт синтаксически красивый, но химически неверный JSON.",
    "Для соревнования важны не только строки CSV, но и воспроизводимая проверка качества.",
    "Пользователю нужно видеть, откуда взялась строка и почему часть строк отклонена.",
  ], 0.85, 1.55, 5.25, 3.35, { fontSize: 14 });
  flow(s, ["PDF", "noise", "missing units", "bad formulas", "CSV risk"], 6.7, 2.0, 5.4, 0.62, [C.light, C.red2, C.red2, C.red2, C.amber2]);
  card(s, "Ключевой риск", "Если не показывать evidence и rejected rows, невозможно быстро понять: ошибка в модели, в PDF-парсинге или в правилах валидации.", 6.7, 3.25, 5.4, 1.65, { accent: C.red });
}

// 4
{
  const s = addSlide("Что извлекаем", "DOMAINS");
  card(s, "Small molecules", "Oxazolidinones / Benzimidazoles\n\nПоля: compound_id, SMILES, canonical SMILES, property_name, value, unit.\n\nВалидатор: RDKit.", 0.75, 1.55, 5.7, 3.25, { accent: C.blue });
  card(s, "Nanozymes / Nanocatalysts", "Формулы и свойства материалов: size, diameter, Km, Vmax, yield, conversion, selectivity.\n\nВалидаторы: formula sanity, physical sanity, property policy.", 6.9, 1.55, 5.7, 3.25, { accent: C.green });
  pill(s, "SMILES -> canonical", 1.0, 5.35, 2.3);
  pill(s, "formula -> normalized", 4.0, 5.35, 2.3, C.green2, C.green);
  pill(s, "text + table + vision", 7.0, 5.35, 2.6, C.amber2, C.amber);
  pill(s, "CSV + evidence", 10.1, 5.35, 2.2);
}

// 5
{
  const s = addSlide("Пользовательский сценарий", "USER FLOW");
  flow(s, ["1. Select domain", "2. Upload PDF", "3. Choose extractor", "4. Add API key", "5. Run"], 0.85, 1.6, 11.65, 0.75);
  card(s, "Минимум действий", "Обычный сценарий укладывается в четыре шага: домен, PDF, extractor, Run. Дополнительные параметры спрятаны в Advanced.", 0.95, 3.0, 5.45, 1.5, { accent: C.blue });
  card(s, "API без боли", "Ключ можно вставить прямо в UI на один запуск или положить в `.env`. Секреты не попадают в git.", 6.9, 3.0, 5.45, 1.5, { accent: C.green });
  card(s, "Проверяемость", "После запуска сразу видны clean rows, rejected rows, conflicts, evidence, agent trace и логи.", 0.95, 5.05, 11.4, 0.9, { accent: C.amber });
}

// 6
{
  const s = addSlide("Архитектура пайплайна", "PIPELINE");
  flow(s, ["PDF parser", "Text cleanup", "Chunking", "Retrieval", "Extractor", "Validator", "Aggregation"], 0.65, 1.55, 12.0, 0.7);
  card(s, "Parser", "Docling/Camelot превращают PDF в Markdown, таблицы и warnings.", 0.8, 2.75, 2.7, 1.2, { accent: C.blue });
  card(s, "Retrieval", "TF-IDF или HF embeddings выбирают наиболее полезные чанки.", 3.75, 2.75, 2.7, 1.2, { accent: C.green });
  card(s, "Extraction", "LLM возвращает structured rows по Pydantic schema.", 6.7, 2.75, 2.7, 1.2, { accent: C.amber });
  card(s, "Aggregation", "Pandas дедуплицирует, решает конфликты и готовит CSV.", 9.65, 2.75, 2.7, 1.2, { accent: C.red });
  text(s, "Идея: LLM не является единственным источником истины. Каждый шаг пишет trace и отдаёт данные Python-валидаторам.", 0.9, 5.0, 11.55, 0.75, { fontSize: 17, bold: true, color: C.ink, align: "center" });
}

// 7
{
  const s = addSlide("Multi-agent система", "AGENTS");
  const agents = [
    ["ParserAgent", "PDF -> Markdown, tables, chunks"],
    ["RouterAgent", "выбор домена и extraction graph"],
    ["RetrievalAgent", "ranking релевантных чанков"],
    ["ExtractorValidatorAgent", "LLM extraction + retry loop"],
    ["VisionAgent", "панели, scale bar, частицы"],
    ["AggregatorAgent", "clean / rejected / conflicts"],
  ];
  agents.forEach((a, i) => {
    const x = 0.85 + (i % 3) * 4.05;
    const y = 1.55 + Math.floor(i / 3) * 1.75;
    card(s, a[0], a[1], x, y, 3.45, 1.1, { accent: [C.blue, C.green, C.amber, C.blue, C.green, C.red][i], bodySize: 10 });
  });
  text(s, "В UI это видно во вкладке Agents: агент, статус, summary, metrics и warnings.", 1.0, 5.6, 11.4, 0.45, { fontSize: 16, bold: true, align: "center" });
}

// 8
{
  const s = addSlide("Цикл извлечения и валидации", "LANGGRAPH LOOP");
  flow(s, ["Chunk", "LLM extractor", "Pydantic schema", "Python critic", "Retry or Done"], 0.85, 1.55, 11.65, 0.75, [C.light, C.blue2, C.green2, C.amber2, C.light]);
  card(s, "Small molecules", "Критик: RDKit\n\nПроверяет SMILES, канонизирует, ловит ошибки колец/валентности, исключает дубли.", 1.0, 3.0, 5.1, 1.95, { accent: C.blue });
  card(s, "Nano / catalyst", "Критик: formula + sanity\n\nПроверяет формулы, физические величины, допустимость endpoint и отправляет плохие строки в Rejected.", 7.0, 3.0, 5.1, 1.95, { accent: C.green });
  text(s, "Retry limit: до 3 попыток на чанк. Ошибки валидации добавляются в следующий prompt.", 1.1, 5.65, 11.0, 0.35, { fontSize: 14, color: C.muted, align: "center" });
}

// 9
{
  const s = addSlide("Retrieval: меньше шума для модели", "CONTEXT SELECTION");
  card(s, "TF-IDF", "Быстро и локально. Хороший default для демо и воспроизводимости.", 0.9, 1.65, 3.6, 1.5, { accent: C.blue });
  card(s, "Hybrid HF embeddings", "Использует HF feature extraction, если есть HF_TOKEN. Даёт semantic ranking.", 4.85, 1.65, 3.6, 1.5, { accent: C.green });
  card(s, "UI trace", "Во вкладке Chunks видно selected, rank, score, method и preview.", 8.8, 1.65, 3.6, 1.5, { accent: C.amber });
  flow(s, ["All chunks", "ranked chunks", "top-k", "LLM input"], 2.0, 4.25, 9.2, 0.7);
  text(s, "Зачем: LLM получает не всю статью, а наиболее вероятные разделы с результатами, таблицами и экспериментом.", 1.25, 5.65, 10.8, 0.42, { fontSize: 15, bold: true, align: "center" });
}

// 10
{
  const s = addSlide("Computer Vision слой", "VISION");
  card(s, "Что делает", "Рендерит страницы, выделяет панели, ищет scale bar и оценивает размеры частиц по контурам.", 0.9, 1.55, 5.25, 1.55, { accent: C.blue });
  card(s, "Как попадает в данные", "Если найден материал из текста, CV-измерение превращается в строку `particle diameter` с source_type=vision.", 6.7, 1.55, 5.25, 1.55, { accent: C.green });
  flow(s, ["PDF page", "panel crop", "scale bar", "particles", "vision row"], 1.0, 3.9, 11.2, 0.7, [C.light, C.blue2, C.green2, C.amber2, C.light]);
  text(s, "Если scale bar или material hint не найден, строка не теряется молча: она уходит в Rejected с причиной.", 1.15, 5.55, 10.9, 0.45, { fontSize: 15, bold: true, align: "center" });
}

// 11
{
  const s = addSlide("Aggregation: чистый CSV", "OUTPUT LOGIC");
  card(s, "Deduplication", "Группировка по canonical SMILES/property или normalized formula/property/unit/assay/condition.", 0.9, 1.55, 3.7, 1.45, { accent: C.blue });
  card(s, "Source priority", "Таблицы имеют приоритет над текстом; vision расположен между table и text.", 4.85, 1.55, 3.7, 1.45, { accent: C.green });
  card(s, "Conflicts", "Если значения расходятся, выбранная строка идёт в Results, альтернативы попадают в Conflicts.", 8.8, 1.55, 3.7, 1.45, { accent: C.amber });
  flow(s, ["validated rows", "normalize", "sort priority", "deduplicate", "clean.csv"], 1.25, 4.15, 10.7, 0.7);
}

// 12
{
  const s = addSlide("Quality score и evidence", "TRUST");
  card(s, "quality_score", "Численный индикатор 0-100. Учитывает источник, валидный идентификатор, unit, evidence и condition.", 0.9, 1.55, 5.45, 1.65, { accent: C.blue });
  card(s, "quality_flags", "`from_table`, `from_text`, `from_vision`, `valid_formula`, `valid_smiles`, `has_unit`, `has_evidence`.", 6.9, 1.55, 5.45, 1.65, { accent: C.green });
  card(s, "Evidence tab", "Для каждой строки Results можно открыть полный фрагмент текста, из которого взято значение.", 0.9, 4.0, 5.45, 1.45, { accent: C.amber });
  card(s, "Rejected tab", "Плохие строки не исчезают: сохраняются причина, валидатор, source_id и evidence.", 6.9, 4.0, 5.45, 1.45, { accent: C.red });
}

// 13
{
  const s = addSlide("UI: что видит пользователь", "STREAMLIT");
  bulletList(s, [
    "Sidebar: домен, PDF, extractor, Run, временные API-ключи.",
    "Advanced: chunks, attempts, retrieval, model ids, vision pages, scale label.",
    "Main tabs: Results, Evidence, Rejected, Conflicts, Vision, Agents, Chunks, Log, Markdown.",
    "Download-кнопки и локальные CSV-копии одинаковы на всех вкладках.",
  ], 0.95, 1.55, 6.0, 3.55, { fontSize: 13.5 });
  card(s, "Design principle", "Максимум пользы на первом экране: таблица, статус и диагностика. Подробности доступны, но не мешают основному flow.", 7.25, 2.0, 4.8, 1.8, { accent: C.blue });
  flow(s, ["Run", "Results", "Evidence", "Download CSV"], 7.25, 4.55, 4.8, 0.65);
}

// 14
{
  const s = addSlide("API-провайдеры", "LLM CONNECTIVITY");
  card(s, "Hugging Face", "`HF_TOKEN`\n\nБесплатные/open-weight модели. Используется и для embeddings.", 0.9, 1.55, 3.5, 2.2, { accent: C.blue });
  card(s, "OpenRouter", "`OPENROUTER_API_KEY`\n\nOpenAI-compatible `chat.completions.create`. Работает для small molecules и Nanozymes.", 4.9, 1.55, 3.5, 2.2, { accent: C.green });
  card(s, "OpenAI", "`OPENAI_API_KEY`\n\nПодключён через structured `responses.parse`. Сейчас для small molecules.", 8.9, 1.55, 3.5, 2.2, { accent: C.amber });
  card(s, "Удобство", "Ключи можно вставить прямо в UI на один запуск, положить в `.env`, Streamlit secrets или системные переменные.", 1.05, 4.65, 11.1, 1.1, { accent: C.blue });
}

// 15
{
  const s = addSlide("CLI и артефакты", "REPRODUCIBILITY");
  text(s, ".venv\\Scripts\\python.exe scripts\\run_article_pipeline.py article.pdf --domain Nanozymes --extractor openrouter", 0.9, 1.6, 11.8, 0.4, {
    fontFace: "Cascadia Mono",
    fontSize: 11,
    color: C.ink,
  });
  const artifacts = [
    "clean.csv",
    "rejected.csv",
    "conflicts.csv",
    "prepared.md",
    "chunks.json",
    "retrieval.json",
    "agent_trace.json",
    "manifest.json",
  ];
  artifacts.forEach((a, i) => {
    const x = 0.95 + (i % 4) * 3.0;
    const y = 2.75 + Math.floor(i / 4) * 0.85;
    pill(s, a, x, y, 2.35, i % 2 ? C.green2 : C.blue2, i % 2 ? C.green : C.blue);
  });
  card(s, "Почему это важно", "Можно воспроизвести запуск, посмотреть параметры, выбранные чанки, ошибки extractors и итоговые таблицы.", 1.05, 5.2, 11.1, 0.9, { accent: C.green });
}

// 16
{
  const s = addSlide("Текущее качество инженерной части", "VALIDATION");
  metric(s, "93", "tests OK", 1.1, 1.8, 2.2, C.green);
  metric(s, "0", "known secret leaks", 4.0, 1.8, 2.5, C.blue);
  metric(s, "20", "deck slides", 7.25, 1.8, 2.2, C.amber);
  metric(s, "200", "local UI OK", 10.0, 1.8, 2.2, C.green);
  bulletList(s, [
    "Unit tests покрывают extraction graph, nano validation, aggregation, retrieval, UI pipeline и CLI artifacts.",
    "Секреты не коммитятся: `.env` в `.gitignore`, проверка regex перед push.",
    "README описывает запуск, API-ключи, UI, CLI, архитектуру и ограничения.",
  ], 1.05, 4.0, 11.0, 1.55, { fontSize: 13.5 });
}

// 17
{
  const s = addSlide("Как читать результат", "RESULT INTERPRETATION");
  flow(s, ["Results", "Evidence", "Rejected", "Conflicts", "Agents"], 0.85, 1.55, 11.65, 0.7);
  card(s, "Results", "Основная таблица: материал или SMILES, свойство, значение, единица измерения, source_type и quality_score.", 0.9, 2.85, 3.65, 1.45, { accent: C.blue });
  card(s, "Evidence", "Проверочный слой: полный фрагмент текста, из которого была получена выбранная строка.", 4.85, 2.85, 3.65, 1.45, { accent: C.green });
  card(s, "Rejected", "Контроль качества: строки, которые модель нашла, но схема, формула, unit или sanity-check их отклонили.", 8.8, 2.85, 3.65, 1.45, { accent: C.red });
  card(s, "Conflicts + Agents", "Conflicts показывают расхождения значений, Agents объясняет путь данных через этапы pipeline.", 2.1, 5.05, 9.1, 0.95, { accent: C.amber });
}

// 18
{
  const s = addSlide("Ограничения и честные риски", "RISKS");
  card(s, "PDF parsing", "Сложные PDF и отсутствие Ghostscript могут снижать качество таблиц. Есть fallback, но не магия.", 0.9, 1.55, 3.7, 1.55, { accent: C.amber });
  card(s, "LLM variability", "HF/OpenRouter модели могут возвращать неполные поля. Сейчас такие строки попадают в Rejected.", 4.85, 1.55, 3.7, 1.55, { accent: C.red });
  card(s, "Vision", "CV зависит от качества страниц, scale bar и material hint. Ошибки явно логируются.", 8.8, 1.55, 3.7, 1.55, { accent: C.blue });
  card(s, "Как управляем риском", "Evidence, rejected rows, conflicts, agent trace, tests и воспроизводимые artifacts делают ошибки видимыми.", 1.05, 4.25, 11.1, 1.15, { accent: C.green });
}

// 19
{
  const s = addSlide("Следующие улучшения", "ROADMAP");
  card(s, "Benchmark tab", "Загрузка эталонного CSV, precision/recall/F1, missed/extra rows.", 0.9, 1.55, 3.7, 1.55, { accent: C.blue });
  card(s, "Better repair", "Автоматически восстанавливать пустые formula/unit из evidence и соседних строк.", 4.85, 1.55, 3.7, 1.55, { accent: C.green });
  card(s, "Vision fallback", "Добавить альтернативный рендер PDF, если PDFium падает.", 8.8, 1.55, 3.7, 1.55, { accent: C.amber });
  card(s, "Provider presets", "Готовые профили моделей: free/fast/accurate, чтобы пользователю не вводить model id руками.", 0.9, 3.8, 3.7, 1.55, { accent: C.blue });
  card(s, "Report export", "Единый HTML/Markdown отчет: Results + Evidence + Agents + Manifest.", 4.85, 3.8, 3.7, 1.55, { accent: C.green });
  card(s, "More domains", "Расширить правила и схемы под другие предметные домены и типы статей.", 8.8, 3.8, 3.7, 1.55, { accent: C.red });
}

// 20
{
  const s = addSlide("Финальный вывод", "TAKEAWAY");
  text(s, "Проект превращает PDF-статью в проверяемый CSV, а не просто в ответ LLM.", 1.0, 1.65, 11.3, 0.65, {
    fontFace: "Aptos Display",
    fontSize: 25,
    bold: true,
    align: "center",
    color: C.ink,
  });
  flow(s, ["Simple UI", "Multi-agent trace", "Validators", "Evidence", "Clean CSV"], 1.15, 3.1, 11.0, 0.75, [C.blue2, C.green2, C.amber2, C.light, C.blue2]);
  card(s, "Главная ценность", "Система делает извлечение данных прозрачным: видно, что принято, что отклонено, почему и из какого фрагмента статьи.", 1.2, 4.85, 10.9, 1.2, { accent: C.blue });
}

slides.forEach((slide, index) => {
  addFooterNumber(slide, index + 1);
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
});

pptx.writeFile({ fileName: "DataCon_Extraction_Agent_Presentation.pptx" });
