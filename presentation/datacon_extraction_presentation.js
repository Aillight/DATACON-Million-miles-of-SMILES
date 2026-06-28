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
  text(s, "Помогает превратить научную PDF-статью в проверяемую таблицу", 0.78, 1.82, 8.6, 0.35, {
    fontSize: 15,
    color: C.muted,
  });
  flow(s, ["PDF-статья", "Агенты", "Проверка", "Чистый CSV"], 0.78, 3.05, 8.8, 0.72);
  card(s, "Что делает система", "Берёт статью, находит в ней данные, проверяет их и показывает, откуда взялась каждая строка.", 0.78, 4.55, 5.4, 1.25, { accent: C.blue });
  card(s, "Главная идея", "Модель предлагает кандидатов, а валидаторы и интерфейс помогают понять: строке можно доверять или её нужно отклонить.", 6.55, 4.55, 5.75, 1.25, { accent: C.green });
  text(s, "DataCon 2026", 0.78, 6.75, 3.2, 0.25, { fontSize: 10, color: C.muted });
  slides.push(s);
}

// 2
{
  const s = addSlide("Что получилось", "ИТОГ ПРОЕКТА");
  card(s, "Цель", "Локальный инструмент, который превращает сложную научную статью в понятный и проверяемый CSV.", 0.7, 1.55, 3.9, 1.5, { accent: C.blue });
  card(s, "Подход", "Несколько специализированных агентов: читают PDF, выбирают контекст, извлекают строки, проверяют и собирают результат.", 4.85, 1.55, 3.9, 1.5, { accent: C.green });
  card(s, "Результат", "Рабочий Streamlit-интерфейс, CLI-запуск, выгрузки CSV/JSON, документация и воспроизводимый локальный проект.", 9.0, 1.55, 3.5, 1.5, { accent: C.amber });
  metric(s, "93", "теста", 1.1, 4.0, 2.0, C.blue);
  metric(s, "3", "LLM-провайдера", 4.1, 4.0, 2.0, C.green);
  metric(s, "2", "типа данных", 7.1, 4.0, 2.0, C.amber);
  metric(s, "1", "простой интерфейс", 10.1, 4.0, 2.0, C.red);
}

// 3
{
  const s = addSlide("Проблема", "ЗАЧЕМ ЭТО НУЖНО");
  bulletList(s, [
    "В статьях нужные числа размазаны по тексту, таблицам, формулам и изображениям.",
    "Одна модель может красиво заполнить JSON, но ошибиться в SMILES, формуле или единицах измерения.",
    "Нужен не просто CSV, а результат, который можно быстро проверить и повторить.",
    "Важно видеть не только принятые строки, но и то, что система отклонила.",
  ], 0.85, 1.55, 5.25, 3.35, { fontSize: 14 });
  flow(s, ["PDF", "шум", "нет unit", "плохая формула", "риск CSV"], 6.7, 2.0, 5.4, 0.62, [C.light, C.red2, C.red2, C.red2, C.amber2]);
  card(s, "Ключевой риск", "Без evidence и rejected rows трудно понять, где проблема: в модели, в PDF-парсинге или в правилах проверки.", 6.7, 3.25, 5.4, 1.65, { accent: C.red });
}

// 4
{
  const s = addSlide("С какими данными работает", "ДАННЫЕ");
  card(s, "Малые молекулы", "Например Oxazolidinones / Benzimidazoles.\n\nСистема извлекает compound id, SMILES, свойство, значение и unit.\n\nSMILES проверяется через RDKit.", 0.75, 1.55, 5.7, 3.25, { accent: C.blue });
  card(s, "Наноматериалы и катализаторы", "Формулы материалов и экспериментальные свойства: размеры, Km, Vmax, yield, conversion, selectivity.\n\nПроверяются формулы и физический смысл значений.", 6.9, 1.55, 5.7, 3.25, { accent: C.green });
  pill(s, "SMILES -> canonical", 1.0, 5.35, 2.3);
  pill(s, "formula -> normalized", 4.0, 5.35, 2.3, C.green2, C.green);
  pill(s, "text + table + vision", 7.0, 5.35, 2.6, C.amber2, C.amber);
  pill(s, "CSV + evidence", 10.1, 5.35, 2.2);
}

// 5
{
  const s = addSlide("Как с этим работать", "СЦЕНАРИЙ");
  flow(s, ["Выбрать домен", "Загрузить PDF", "Выбрать модель", "Добавить ключ", "Нажать Run"], 0.85, 1.6, 11.65, 0.75);
  card(s, "Минимум действий", "Пользователь делает несколько понятных шагов. Тонкие настройки спрятаны в Advanced и не мешают основному сценарию.", 0.95, 3.0, 5.45, 1.5, { accent: C.blue });
  card(s, "Ключи без настройки системы", "API-ключ можно вставить прямо в интерфейс на один запуск или положить в `.env`. В репозиторий секреты не попадают.", 6.9, 3.0, 5.45, 1.5, { accent: C.green });
  card(s, "После запуска всё видно", "Итоговая таблица, отклонённые строки, конфликты, evidence, путь агентов и технический лог доступны в отдельных вкладках.", 0.95, 5.05, 11.4, 0.9, { accent: C.amber });
}

// 6
{
  const s = addSlide("Архитектура пайплайна", "КАК УСТРОЕНО");
  flow(s, ["Читаем PDF", "Чистим текст", "Режем на чанки", "Выбираем важное", "Извлекаем", "Проверяем", "Собираем CSV"], 0.65, 1.55, 12.0, 0.7);
  card(s, "Чтение PDF", "Docling/Camelot достают текст, таблицы и предупреждения о проблемах парсинга.", 0.8, 2.75, 2.7, 1.2, { accent: C.blue });
  card(s, "Выбор контекста", "Retrieval выбирает куски статьи, где с высокой вероятностью есть нужные значения.", 3.75, 2.75, 2.7, 1.2, { accent: C.green });
  card(s, "Извлечение", "Модель возвращает строки в строгом формате, чтобы их можно было проверить программно.", 6.7, 2.75, 2.7, 1.2, { accent: C.amber });
  card(s, "Сборка результата", "Pandas убирает дубли, решает конфликты и готовит итоговые таблицы.", 9.65, 2.75, 2.7, 1.2, { accent: C.red });
  text(s, "Главный принцип: модель не является последней инстанцией. Каждая строка проходит проверку и оставляет след.", 0.9, 5.0, 11.55, 0.75, { fontSize: 17, bold: true, color: C.ink, align: "center" });
}

// 7
{
  const s = addSlide("Система агентов", "КТО ЧТО ДЕЛАЕТ");
  const agents = [
    ["ParserAgent", "читает PDF и готовит текст"],
    ["RouterAgent", "выбирает нужный сценарий"],
    ["RetrievalAgent", "находит важные фрагменты"],
    ["ExtractorValidatorAgent", "извлекает и перепроверяет строки"],
    ["VisionAgent", "работает с картинками и scale bar"],
    ["AggregatorAgent", "собирает финальные таблицы"],
  ];
  agents.forEach((a, i) => {
    const x = 0.85 + (i % 3) * 4.05;
    const y = 1.55 + Math.floor(i / 3) * 1.75;
    card(s, a[0], a[1], x, y, 3.45, 1.1, { accent: [C.blue, C.green, C.amber, C.blue, C.green, C.red][i], bodySize: 10 });
  });
  text(s, "Вкладка Agents показывает, какой этап что сделал, где были предупреждения и сколько данных прошло дальше.", 1.0, 5.6, 11.4, 0.45, { fontSize: 16, bold: true, align: "center" });
}

// 8
{
  const s = addSlide("Цикл извлечения и валидации", "ЦИКЛ ПРОВЕРКИ");
  flow(s, ["Чанк", "Модель", "Строгая схема", "Python-проверка", "Повтор или готово"], 0.85, 1.55, 11.65, 0.75, [C.light, C.blue2, C.green2, C.amber2, C.light]);
  card(s, "Small molecules", "Критик: RDKit\n\nПроверяет SMILES, канонизирует, ловит ошибки колец/валентности, исключает дубли.", 1.0, 3.0, 5.1, 1.95, { accent: C.blue });
  card(s, "Nano / catalyst", "Критик: formula + sanity\n\nПроверяет формулы, физические величины, допустимость endpoint и отправляет плохие строки в Rejected.", 7.0, 3.0, 5.1, 1.95, { accent: C.green });
  text(s, "Если проверка нашла ошибку, система даёт модели ещё одну попытку и прямо сообщает, что нужно исправить.", 1.1, 5.65, 11.0, 0.35, { fontSize: 14, color: C.muted, align: "center" });
}

// 9
{
  const s = addSlide("Как выбирается контекст", "МЕНЬШЕ ШУМА");
  card(s, "TF-IDF", "Быстрый локальный вариант. Хорошо подходит как стабильный default.", 0.9, 1.65, 3.6, 1.5, { accent: C.blue });
  card(s, "HF embeddings", "Если есть HF_TOKEN, можно включить семантический поиск по смыслу, а не только по словам.", 4.85, 1.65, 3.6, 1.5, { accent: C.green });
  card(s, "Прозрачность", "Во вкладке Chunks видно, какие фрагменты выбраны и почему они попали в обработку.", 8.8, 1.65, 3.6, 1.5, { accent: C.amber });
  flow(s, ["Все чанки", "оценка", "top-k", "вход модели"], 2.0, 4.25, 9.2, 0.7);
  text(s, "Зачем: модель читает не всю статью подряд, а наиболее вероятные места с результатами, таблицами и экспериментом.", 1.25, 5.65, 10.8, 0.42, { fontSize: 15, bold: true, align: "center" });
}

// 10
{
  const s = addSlide("Что даёт компьютерное зрение", "ИЗОБРАЖЕНИЯ");
  card(s, "Что делает", "Система смотрит на страницы статьи как на изображения: выделяет панели, ищет scale bar и оценивает размеры частиц.", 0.9, 1.55, 5.25, 1.55, { accent: C.blue });
  card(s, "Как это становится данными", "Если найден материал из текста, измерение с картинки добавляется как отдельная строка `particle diameter`.", 6.7, 1.55, 5.25, 1.55, { accent: C.green });
  flow(s, ["страница", "панель", "scale bar", "частицы", "строка данных"], 1.0, 3.9, 11.2, 0.7, [C.light, C.blue2, C.green2, C.amber2, C.light]);
  text(s, "Если масштаба или материала не хватает, система не делает вид, что всё хорошо: строка уходит в Rejected с причиной.", 1.15, 5.55, 10.9, 0.45, { fontSize: 15, bold: true, align: "center" });
}

// 11
{
  const s = addSlide("Как получается чистый CSV", "ЛОГИКА ВЫХОДА");
  card(s, "Убираем дубли", "Одинаковые молекулы или материалы собираются вместе, чтобы в итоговой таблице не было повторов.", 0.9, 1.55, 3.7, 1.45, { accent: C.blue });
  card(s, "Выбираем надёжный источник", "Табличные значения обычно точнее текста, поэтому получают более высокий приоритет.", 4.85, 1.55, 3.7, 1.45, { accent: C.green });
  card(s, "Не прячем расхождения", "Если значения конфликтуют, выбранная строка идёт в Results, а альтернативы остаются в Conflicts.", 8.8, 1.55, 3.7, 1.45, { accent: C.amber });
  flow(s, ["проверенные строки", "нормализация", "приоритет", "дедупликация", "clean.csv"], 1.25, 4.15, 10.7, 0.7);
}

// 12
{
  const s = addSlide("Оценка качества и источник", "ДОВЕРИЕ");
  card(s, "quality_score", "Быстрый ориентир 0-100: насколько строка выглядит надёжной по источнику, unit, evidence и валидному идентификатору.", 0.9, 1.55, 5.45, 1.65, { accent: C.blue });
  card(s, "quality_flags", "Короткие метки вроде `from_table`, `valid_formula`, `has_unit`, `has_evidence` объясняют, из чего сложился score.", 6.9, 1.55, 5.45, 1.65, { accent: C.green });
  card(s, "Evidence", "Для каждой принятой строки можно открыть полный фрагмент статьи, из которого взято значение.", 0.9, 4.0, 5.45, 1.45, { accent: C.amber });
  card(s, "Rejected", "Неподходящие строки не исчезают: сохраняются причина, валидатор и исходный фрагмент.", 6.9, 4.0, 5.45, 1.45, { accent: C.red });
}

// 13
{
  const s = addSlide("Что видит пользователь", "ИНТЕРФЕЙС");
  bulletList(s, [
    "Слева: домен, PDF, модель, запуск и временные API-ключи.",
    "В Advanced: размер контекста, число попыток, retrieval, модели и параметры vision.",
    "В центре: Results, Evidence, Rejected, Conflicts, Vision, Agents, Chunks, Log, Markdown.",
    "На каждой вкладке с таблицей есть download-кнопка и локальная CSV-копия.",
  ], 0.95, 1.55, 6.0, 3.55, { fontSize: 13.5 });
  card(s, "Принцип интерфейса", "Пользователь сначала видит результат, а диагностика остаётся рядом: её легко открыть, но она не перегружает основной сценарий.", 7.25, 2.0, 4.8, 1.8, { accent: C.blue });
  flow(s, ["Run", "Results", "Evidence", "CSV"], 7.25, 4.55, 4.8, 0.65);
}

// 14
{
  const s = addSlide("API-провайдеры", "МОДЕЛИ");
  card(s, "Hugging Face", "`HF_TOKEN`\n\nПодходит для open-weight моделей и embedding retrieval.", 0.9, 1.55, 3.5, 2.2, { accent: C.blue });
  card(s, "OpenRouter", "`OPENROUTER_API_KEY`\n\nУдобный единый вход к разным моделям. Работает и для малых молекул, и для наноматериалов.", 4.9, 1.55, 3.5, 2.2, { accent: C.green });
  card(s, "OpenAI", "`OPENAI_API_KEY`\n\nИспользуется structured output. Сейчас включён для small-molecule сценария.", 8.9, 1.55, 3.5, 2.2, { accent: C.amber });
  card(s, "Удобство", "Ключ можно вставить в UI на один запуск или хранить в `.env`, Streamlit secrets либо системных переменных.", 1.05, 4.65, 11.1, 1.1, { accent: C.blue });
}

// 15
{
  const s = addSlide("CLI и артефакты", "ВОСПРОИЗВОДИМОСТЬ");
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
  card(s, "Почему это важно", "Результат можно не только скачать, но и восстановить: параметры запуска, выбранные чанки, ошибки и итоговые таблицы лежат рядом.", 1.05, 5.2, 11.1, 0.9, { accent: C.green });
}

// 16
{
  const s = addSlide("Текущее качество инженерной части", "ПРОВЕРКИ");
  metric(s, "93", "теста проходят", 1.1, 1.8, 2.2, C.green);
  metric(s, "0", "секретов в файлах", 4.0, 1.8, 2.5, C.blue);
  metric(s, "20", "слайдов", 7.25, 1.8, 2.2, C.amber);
  metric(s, "200", "UI отвечает", 10.0, 1.8, 2.2, C.green);
  bulletList(s, [
    "Тесты покрывают extraction graph, nano validation, aggregation, retrieval, UI pipeline и CLI artifacts.",
    "Секреты не коммитятся: `.env` в `.gitignore`, перед push проверяется отсутствие ключей в файлах.",
    "README описывает запуск, API-ключи, UI, CLI, архитектуру и ограничения.",
  ], 1.05, 4.0, 11.0, 1.55, { fontSize: 13.5 });
}

// 17
{
  const s = addSlide("Как читать результат", "РУЧНАЯ ПРОВЕРКА");
  flow(s, ["Results", "Evidence", "Rejected", "Conflicts", "Agents"], 0.85, 1.55, 11.65, 0.7);
  card(s, "Results", "То, что система считает готовым результатом: объект, свойство, значение, unit, источник и оценка качества.", 0.9, 2.85, 3.65, 1.45, { accent: C.blue });
  card(s, "Evidence", "Фрагмент статьи, который помогает быстро проверить выбранную строку руками.", 4.85, 2.85, 3.65, 1.45, { accent: C.green });
  card(s, "Rejected", "То, что модель предложила, но система не пропустила из-за схемы, формулы, unit или sanity-check.", 8.8, 2.85, 3.65, 1.45, { accent: C.red });
  card(s, "Conflicts + Agents", "Conflicts показывают спорные значения, Agents объясняет, через какие этапы прошла статья.", 2.1, 5.05, 9.1, 0.95, { accent: C.amber });
}

// 18
{
  const s = addSlide("Ограничения и честные риски", "РИСКИ");
  card(s, "PDF parsing", "Сложные PDF и отсутствие Ghostscript могут ухудшать таблицы. Есть fallback, но качество источника всё равно важно.", 0.9, 1.55, 3.7, 1.55, { accent: C.amber });
  card(s, "LLM variability", "Модели иногда возвращают неполные поля. Такие строки не попадают в clean CSV молча, а уходят в Rejected.", 4.85, 1.55, 3.7, 1.55, { accent: C.red });
  card(s, "Vision", "CV зависит от качества изображений, scale bar и связи картинки с материалом из текста.", 8.8, 1.55, 3.7, 1.55, { accent: C.blue });
  card(s, "Как управляем риском", "Evidence, отклонённые строки, конфликты, путь агентов и тесты делают ошибки видимыми и проверяемыми.", 1.05, 4.25, 11.1, 1.15, { accent: C.green });
}

// 19
{
  const s = addSlide("Что можно усилить дальше", "ДАЛЬШЕ");
  card(s, "Benchmark tab", "Загрузить эталонный CSV и сразу видеть precision, recall, F1, missed и extra rows.", 0.9, 1.55, 3.7, 1.55, { accent: C.blue });
  card(s, "Автовосстановление", "Пытаться восстанавливать пустые formula или unit из evidence и соседних строк, прежде чем отправлять строку в Rejected.", 4.85, 1.55, 3.7, 1.55, { accent: C.green });
  card(s, "Vision fallback", "Добавить второй способ рендера PDF, если основной CV-путь не смог открыть документ.", 8.8, 1.55, 3.7, 1.55, { accent: C.amber });
  card(s, "Профили моделей", "Сделать готовые режимы: free, fast, accurate, чтобы не вводить model id вручную.", 0.9, 3.8, 3.7, 1.55, { accent: C.blue });
  card(s, "Единый отчёт", "Экспортировать HTML/Markdown-отчёт: Results, Evidence, Agents, Manifest и предупреждения.", 4.85, 3.8, 3.7, 1.55, { accent: C.green });
  card(s, "Больше доменов", "Расширить правила и схемы под другие предметные области и типы статей.", 8.8, 3.8, 3.7, 1.55, { accent: C.red });
}

// 20
{
  const s = addSlide("Финальный вывод", "ВЫВОД");
  text(s, "Проект превращает PDF-статью в проверяемый CSV, а не просто в ответ LLM.", 1.0, 1.65, 11.3, 0.65, {
    fontFace: "Aptos Display",
    fontSize: 25,
    bold: true,
    align: "center",
    color: C.ink,
  });
  flow(s, ["Простой UI", "Путь агентов", "Проверки", "Evidence", "Чистый CSV"], 1.15, 3.1, 11.0, 0.75, [C.blue2, C.green2, C.amber2, C.light, C.blue2]);
  card(s, "Главная ценность", "Система делает извлечение данных прозрачным: видно, что принято, что отклонено, почему и из какого фрагмента статьи.", 1.2, 4.85, 10.9, 1.2, { accent: C.blue });
}

slides.forEach((slide, index) => {
  addFooterNumber(slide, index + 1);
  warnIfSlideHasOverlaps(slide, pptx);
  warnIfSlideElementsOutOfBounds(slide, pptx);
});

pptx.writeFile({ fileName: "DataCon_Extraction_Agent_Presentation.pptx" });
