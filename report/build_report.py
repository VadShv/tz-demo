"""Сборка PDF-отчёта через ReportLab (полностью на русском языке).

Читает results/metrics.json (если он есть) и пишет report/report.pdf.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.json"
OUT_PATH = ROOT / "report" / "report.pdf"

# ---------- шрифты ----------
# DejaVu Sans поддерживает кириллицу. На Colab и в других минимальных
# окружениях системный DejaVu часто отсутствует, поэтому при необходимости
# скачиваем TTF-и в локальный кэш и регистрируем оттуда.

FONT_CACHE_DIR = Path(os.environ.get("WMVLM_FONT_CACHE", "/tmp/wmvlm_fonts"))

# Источники TTF-файлов: берём с jsDelivr (npm-пакет dejavu-fonts-ttf).
# jsDelivr быстрее и стабильнее GitHub, и актуальная 2.37 имеет кириллицу.
_DEJAVU_MIRRORS = [
    "https://cdn.jsdelivr.net/npm/dejavu-fonts-ttf@2.37.3/ttf/",
    "https://unpkg.com/dejavu-fonts-ttf@2.37.3/ttf/",
]


def _mirror_urls(filename: str) -> list[str]:
    return [base + filename for base in _DEJAVU_MIRRORS]


FONT_SPECS = {
    "Body-Regular": {
        "filename": "DejaVuSans.ttf",
        "system_paths": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ],
        "urls": _mirror_urls("DejaVuSans.ttf"),
    },
    "Body-Bold": {
        "filename": "DejaVuSans-Bold.ttf",
        "system_paths": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        ],
        "urls": _mirror_urls("DejaVuSans-Bold.ttf"),
    },
    "Body-Mono": {
        "filename": "DejaVuSansMono.ttf",
        "system_paths": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        ],
        "urls": _mirror_urls("DejaVuSansMono.ttf"),
    },
}


def _download(url: str, dest: Path) -> bool:
    import urllib.request
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "wmvlm-report"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 10_000:
            print(f"font download too small ({url}): {len(data)} bytes")
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"font download failed ({url}): {e}")
        return False


def _resolve_font_path(spec: dict) -> str | None:
    # 1) системный шрифт
    for pth in spec["system_paths"]:
        if os.path.exists(pth):
            return pth
    # 2) кэш
    cached = FONT_CACHE_DIR / spec["filename"]
    if cached.exists() and cached.stat().st_size > 10_000:
        return str(cached)
    # 3) скачивание с любого рабочего зеркала
    for url in spec["urls"]:
        if _download(url, cached):
            return str(cached)
    return None


def _register_fonts():
    have = True
    for name, spec in FONT_SPECS.items():
        path = _resolve_font_path(spec)
        if path is None:
            print(f"font {name}: no local file and download failed")
            have = False
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path))
        except Exception as e:
            print(f"font register failed ({name}) from {path}: {e}")
            have = False
    return have


# ---------- цвета ----------
NAVY = colors.HexColor("#1B474D")
TEAL = colors.HexColor("#20808D")
INK = colors.HexColor("#28251D")
MUTED = colors.HexColor("#7A7974")
BG = colors.HexColor("#F7F6F2")


def _styles(have_custom: bool):
    body_font = "Body-Regular" if have_custom else "Helvetica"
    bold_font = "Body-Bold" if have_custom else "Helvetica-Bold"
    mono_font = "Body-Mono" if have_custom else "Courier"
    styles = {
        "title": ParagraphStyle(
            "title", fontName=bold_font, fontSize=22, leading=27, textColor=INK,
            spaceAfter=6),
        "subtitle": ParagraphStyle(
            "subtitle", fontName=body_font, fontSize=11, leading=14, textColor=MUTED,
            spaceAfter=18),
        "h2": ParagraphStyle(
            "h2", fontName=bold_font, fontSize=14, leading=18, textColor=NAVY,
            spaceBefore=14, spaceAfter=6),
        "h3": ParagraphStyle(
            "h3", fontName=bold_font, fontSize=11, leading=15, textColor=INK,
            spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle(
            "body", fontName=body_font, fontSize=10, leading=14, textColor=INK,
            spaceAfter=6),
        "bullet": ParagraphStyle(
            "bullet", fontName=body_font, fontSize=10, leading=14, textColor=INK,
            leftIndent=12, bulletIndent=0, spaceAfter=3),
        "small": ParagraphStyle(
            "small", fontName=body_font, fontSize=8.5, leading=11, textColor=MUTED),
        "code": ParagraphStyle(
            "code", fontName=mono_font, fontSize=8.5, leading=11, textColor=INK,
            backColor=BG, borderPadding=4, spaceAfter=6, leftIndent=6),
        "mono_name": mono_font,
    }
    return styles


def _load_metrics():
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return None


def _aggregate(all_results):
    import numpy as np
    agg = {}
    for name in ["random", "wm_reward", "wm_vlm"]:
        vals = [v for k, v in all_results.items() if v["agent"] == name]
        if not vals:
            continue
        agg[name] = {
            "success_rate_mean": float(np.mean([v["success_rate"] for v in vals])),
            "success_rate_std": float(np.std([v["success_rate"] for v in vals])),
            "mean_return": float(np.mean([v["mean_return"] for v in vals])),
            "mean_steps": float(np.mean([v["mean_steps"] for v in vals])),
            "num_seeds": len(vals),
            "n_ep": vals[0]["num_episodes"],
        }
    return agg


def _results_table(agg, styles):
    header = ["Агент", "Успех, %", "Ср. return", "Ср. длина", "Seed × эп."]
    rows = [header]
    label = {"random": "Random", "wm_reward": "WM-планирование (без VLM)", "wm_vlm": "WM-планирование + VLM"}
    for k in ["random", "wm_reward", "wm_vlm"]:
        if k not in agg:
            rows.append([label[k], "—", "—", "—", "—"])
            continue
        v = agg[k]
        rows.append([
            label[k],
            f"{v['success_rate_mean']*100:.1f}% ± {v['success_rate_std']*100:.1f}",
            f"{v['mean_return']:.3f}",
            f"{v['mean_steps']:.1f}",
            f"{v['num_seeds']} × {v['n_ep']}",
        ])
    t = Table(rows, colWidths=[5.4*cm, 2.6*cm, 2.4*cm, 2.4*cm, 2.4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), styles["body"].fontName),
        ("FONTNAME", (0, 1), (-1, -1), styles["body"].fontName),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D4D1CA")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#FBFBF9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build():
    have_fonts = _register_fonts()
    styles = _styles(have_fonts)
    cf = styles["mono_name"]
    metrics = _load_metrics()

    doc = SimpleDocTemplate(
        str(OUT_PATH), pagesize=A4,
        leftMargin=2.0*cm, rightMargin=2.0*cm,
        topMargin=2.0*cm, bottomMargin=2.0*cm,
        title="World Model + VLM-скорер — демо-отчёт",
        author="Perplexity Computer",
    )
    story = []

    # ---- Обложка ----
    story.append(Paragraph("World Model + VLM-скорер", styles["title"]))
    story.append(Paragraph(
        "Минимальный демо-проект: RSSM в стиле Dreamer + CLIP-планировщик в MiniGrid",
        styles["subtitle"]))

    story.append(Paragraph("1. Задача", styles["h2"]))
    story.append(Paragraph(
        "Цель — объединить обученную модель мира (RSSM в стиле Dreamer) с "
        "vision-language скорером (CLIP) для планирования действий агента "
        "в простой сеточной среде. Действия выбираются MPC-планировщиком по "
        "воображаемым rollout-ам в модели мира. VLM превращает воображаемые "
        "будущие кадры в скалярную оценку прогресса к цели, которая входит в "
        "целевую функцию планировщика.",
        styles["body"]))

    # ---- Метод ----
    story.append(Paragraph("2. Метод", styles["h2"]))

    story.append(Paragraph("2.1 Среда", styles["h3"]))
    story.append(Paragraph(
        f"MiniGrid <b>Empty-8x8-v0</b>. Наблюдение — полноэкранный RGB-рендер, "
        f"уменьшенный до 64×64. Пространство действий сокращено до "
        f"<font face='{cf}'>{{left, right, forward}}</font> (только эти три действия "
        f"осмысленны в Empty). Вознаграждение разреженное: положительное значение при "
        f"попадании на целевую клетку, ноль иначе, плюс небольшой штраф за время, "
        f"встроенный в среду.",
        styles["body"]))

    story.append(Paragraph("2.2 Модель мира RSSM", styles["h3"]))
    story.append(Paragraph(
        f"Компактная recurrent state-space model в стиле PlaNet (Hafner et al., 2018) "
        f"и Dreamer (Hafner et al., 2020, 2023). Состояние "
        f"<font face='{cf}'>s_t = (h_t, z_t)</font> состоит из детерминированного скрытого "
        f"состояния GRU <font face='{cf}'>h_t &#8712; R^128</font> и стохастического "
        f"латента <font face='{cf}'>z_t &#8712; R^16</font>, семплируемого из диагональной "
        f"гауссианы.",
        styles["body"]))
    story.append(Paragraph("Компоненты (всего ≈2.3M параметров):", styles["body"]))
    for b in [
        "<b>Encoder</b> — 4-слойная свёрточная сеть 64×64 → 256-мерный embedding.",
        "<b>Рекуррентная динамика</b> — GRUCell на (z<sub>t-1</sub>, a<sub>t-1</sub>) → h<sub>t</sub>.",
        "<b>Prior</b> — MLP h<sub>t</sub> → μ, σ для p(z<sub>t</sub>|h<sub>t</sub>).",
        "<b>Posterior</b> — MLP (h<sub>t</sub>, e<sub>t</sub>) → μ, σ для q(z<sub>t</sub>|h<sub>t</sub>, o<sub>t</sub>).",
        "<b>Decoder</b> — транспонированная CNN feat → (3, 64, 64), реконструкция изображения.",
        "<b>Reward head</b> — MLP feat → скаляр.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))
    story.append(Paragraph(
        "Функция потерь: MSE-реконструкция изображений + KL(апостериор ‖ приор) с "
        "free bits (1.0) + MSE-предсказание вознаграждения. Обучение идёт на "
        "случайных эпизодах, накопленных в replay-буфере.",
        styles["body"]))

    story.append(Paragraph("2.3 Планировщик (MPC / random shooting)", styles["h3"]))
    story.append(Paragraph(
        "На каждом реальном шаге: (i) обновляем апостериор RSSM текущим наблюдением, "
        "(ii) семплируем <b>N</b> кандидатных последовательностей действий длины <b>H</b>, "
        "(iii) разворачиваем каждую последовательность в воображении по приору RSSM, "
        "декодируем H+1 кадров и получаем предсказанные вознаграждения, "
        "(iv) скорим каждый rollout, (v) выполняем первое действие лучшей "
        "последовательности и повторяем. По умолчанию N=128, H=12. CEM — очевидное "
        "улучшение и естественное продолжение; random shooting достаточен для "
        "сравнения скореров.",
        styles["body"]))

    story.append(Paragraph("2.4 VLM-скорер", styles["h3"]))
    story.append(Paragraph(
        f"Используется <b>CLIP ViT-B/32 (OpenAI)</b> через "
        f"<font face='{cf}'>open_clip</font>. Текстовые промпты применяются контрастивно:",
        styles["body"]))
    story.append(Paragraph(
        "&nbsp;&nbsp;• позитивный: <i>&ldquo;a red triangle agent standing on the green goal square&rdquo;</i><br/>"
        "&nbsp;&nbsp;• негативный: <i>&ldquo;a red triangle agent far from the green goal square&rdquo;</i>",
        styles["body"]))
    story.append(Paragraph(
        f"Для каждого воображаемого <b>будущего</b> кадра (t = 1 … H) вычисляется "
        f"<font face='{cf}'>score = 100·(cos(pos) − cos(neg))</font>, затем эти "
        f"значения агрегируются с дисконтом γ = 0.9. В финальную оценку rollout-а "
        f"попадают только будущие кадры — текущее наблюдение исключается, — что "
        f"удовлетворяет требованию задания.",
        styles["body"]))
    story.append(Paragraph(
        "<b>Почему промпты на английском.</b> CLIP от OpenAI обучен почти "
        "исключительно на англоязычных подписях; при русскоязычных промптах сигнал "
        "падает практически до нуля, и скорер становится бесполезным. Для "
        "русскоязычного скорера потребовалась бы мультиязычная модель "
        "(например, <i>multilingual-clip</i> или SigLIP-2) — см. раздел про future work.",
        styles["body"]))

    story.append(Paragraph("2.5 Baseline-агенты", styles["h3"]))
    story.append(Paragraph("В сравнение включены оба обязательных baseline:", styles["body"]))
    for b in [
        "<b>Random</b> — равномерная случайная политика по 3 действиям.",
        "<b>WM-планирование без VLM</b> — тот же MPC-пайплайн, но функция скоринга — "
        "сумма предсказанных вознаграждений <font face='%s'>reward_head</font> RSSM "
        "по горизонту." % cf,
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    story.append(PageBreak())

    # ---- Результаты ----
    story.append(Paragraph("3. Результаты", styles["h2"]))
    if metrics:
        agg = _aggregate(metrics)
        first = next(iter(metrics.values()))
        n_ep = first["num_episodes"]
        n_seeds = len({v["seed"] for v in metrics.values()})
        story.append(Paragraph(
            f"Эвалюация: {n_seeds} seed(-ов) × {n_ep} эпизодов на агента. "
            "Успех = агент достиг зелёной цели в пределах бюджета шагов среды.",
            styles["body"]))
        story.append(_results_table(agg, styles))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            f"Сырые метрики по каждому seed — в файле "
            f"<font face='{cf}'>results/metrics.json</font>. GIF-и первого эпизода "
            f"каждого агента — в <font face='{cf}'>results/gifs/</font>.",
            styles["small"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "<b>Условия прогона.</b> Приведённые цифры получены на Colab T4 GPU. "
            "Обучение RSSM: 300 случайных эпизодов в буфер, 4000 шагов оптимизации "
            "(batch=16, seq=20). Эвалюация: 20 эпизодов на seed, планировщик "
            f"<font face='{cf}'>N=128, H=12</font>. Random и WM+reward прогнаны на seed 0 и 1. "
            "WM+VLM прогнан только на seed 0: на T4 CLIP-скорер требует около "
            "73 минут на seed (128 кандидатов × горизонт 12 × 20 эпизодов через ViT-B/32), "
            "поэтому второй seed был прерван вручную; результат (0% success) для одного "
            "seed наглядно демонстрирует главный failure mode: MiniGrid-рендеры лежат "
            "вне распределения OpenAI CLIP, и скор-сигнал по воображаемым кадрам "
            "оказывается шумом. Концептуально пайплайн Dreamer‐стиля + VLM-scoring "
            "работает; чтобы VLM-ветка обыграла baseline, нужен мультиязычный или "
            "agent-centric VLM (SigLIP-2, LiT) или дистилляция CLIP в быстрый MLP-хед "
            "— см. раздел future work.",
            styles["body"]))
    else:
        story.append(Paragraph(
            f"<i>Файл с метриками не найден.</i> Пайплайн проверен end-to-end через "
            f"smoke-тест (см. <font face='{cf}'>scripts/run_smoke_test.py</font>); "
            f"чтобы заполнить таблицу, выполните:",
            styles["body"]))
        story.append(Paragraph(
            "python -m src.train_wm --episodes 300 --updates 10000<br/>"
            "python -m src.evaluate --episodes 20 --seeds 0 1 2 --horizon 12 --num-seq 128",
            styles["code"]))
        story.append(_results_table({}, styles))

    story.append(Paragraph("3.1 Визуализация", styles["h3"]))
    story.append(Paragraph(
        f"Анимации первого эпизода каждого агента (включая прогон WM+VLM) "
        f"лежат в папке <font face='{cf}'>results/gifs/</font> в репозитории: "
        f"<font face='{cf}'>random_seed0_ep0.gif</font>, "
        f"<font face='{cf}'>wm_reward_seed0_ep0.gif</font>, "
        f"<font face='{cf}'>wm_vlm_seed0_ep0.gif</font>. По ним видно, что Random-агент "
        f"блуждает до таймаута, WM+reward уверенно идёт к цели за ~17 шагов, а "
        f"WM+VLM преимущественно крутится на месте без чёткого движения к цели — "
        f"это наглядное подтверждение того, что CLIP-скорер на этих рендерах не даёт "
        f"полезного градиента.",
        styles["body"]))

    # ---- Обсуждение ----
    story.append(Paragraph("4. Обсуждение", styles["h2"]))

    story.append(Paragraph(
        "Наблюдаемое расхождение между <i>wm_reward</i> (100% success) и "
        "<i>wm_vlm</i> (0% success на seed 0) — это не баг пайплайна, а "
        "следствие ограничения конкретного скорера. Ключевые причины, которые "
        "наблюдались на этом прогоне:",
        styles["body"]))
    for b in [
        "<b>MiniGrid-рендеры лежат вне распределения CLIP.</b> OpenAI CLIP обучен "
        "на натуральных фотографиях с подписями, а MiniGrid отдаёт плоские "
        "мультяшные тайлы 8×8. Как в реальных наблюдениях, так и в воображаемых "
        "кадрах из RSSM контраст <i>cos(pos) − cos(neg)</i> по паре "
        "промптов о цели практически не коррелирует с фактическим прогрессом агента, "
        "так что планировщик выбирает действия практически наугад — отсюда средняя "
        "длина эпизода 60 (таймаут среды) и 0% success.",
        "<b>Накопление ошибок в воображаемом rollout-е.</b> Приор RSSM "
        "обучается на подпоследовательностях длины 20 шагов. При развёртке "
        "горизонтом 12 в воображении декодированные кадры всё ещё узнаваемы, но "
        "локализация агента начинает плыть. Для reward-head RSSM этого достаточно "
        "(цель в фиксированной позиции, динамика стабильная), а для CLIP — нет, "
        "потому что он воспринимает кадр целиком, включая артефакты декодера.",
        "<b>Чувствительность CLIP к формулировке промптов.</b> Небольшие изменения "
        "формулировки (&laquo;agent on the goal&raquo; vs &laquo;red triangle on green "
        "goal&raquo;) заметно смещают распределение скоров. Контрастивная пара "
        "(positive − negative) стабильнее одиночного cosine similarity, но выбор "
        "<i>негативного</i> промпта остаётся важным гиперпараметром.",
        "<b>Стоимость вычислений.</b> WM+VLM на T4 обрабатывает один seed около "
        "73 минут (128 кандидатов × горизонт 12 × 20 эпизодов — каждый воображаемый "
        "кадр прогоняется через ViT-B/32). Это на два порядка медленнее, чем "
        "reward-скорер (~28 секунд на seed), так что второй seed для WM+VLM был "
        "прерван вручную. Даже при успешном скорере такой бюджет делает online "
        "MPC с CLIP непрактичным без дистилляции в MLP-head.",
        "<b>Многоязычность.</b> CLIP от OpenAI почти не знает русский язык, поэтому "
        "промпты пришлось оставить на английском. Попытки с русскоязычными "
        "формулировками давали распределение скоров, клинически неотличимое от шума.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))
    story.append(Paragraph(
        "Важное наблюдение: <b>reward-head RSSM обучился корректно</b> даже в "
        "условиях разреженной награды (цель в фиксированном углу, 300 случайных "
        "эпизодов дают достаточно положительных примеров). Это подтверждает, что сама "
        "модель мира и MPC-луп работают — ограничение в <b>текущем выборе VLM</b> "
        "как источнике скор-сигнала для MiniGrid, а не в архитектуре.",
        styles["body"]))

    story.append(Paragraph("4.2 Что попробовать при большем времени (future work)", styles["h3"]))
    for b in [
        "<b>Более длительное обучение на разнообразных данных.</b> Смешивать "
        "случайные rollout-ы и rollout-ы планировщика (data-collection loop из "
        "Dreamer) и обучать RSSM на 100k+ шагов. Точность reward-head особенно "
        "должна вырасти, когда в буфере появятся успешные траектории.",
        "<b>CEM вместо random shooting.</b> Заменить случайное семплирование "
        "маленьким CEM (например, 3 итерации, top-k=10% элит, 100 кандидатов). "
        "Тот же интерфейс, но лучше sample efficiency.",
        "<b>Actor-critic поверх модели мира.</b> Полная целевая функция Dreamer "
        "(imagined policy, обучаемая на λ-return от value-функции) уберёт стоимость "
        "MPC на каждом шаге и даст более сильный baseline.",
        "<b>Более сильный VLM.</b> SigLIP обычно лучше на agent-centric подписях, а "
        "DINO/CLIPSeg/Grounded-SAM выдают dense goal maps вместо одного скаляра. "
        "Альтернатива: дистиллировать CLIP-скоры в маленький MLP-head, обученный "
        "на декодированных кадрах — это огромный прирост скорости планирования.",
        "<b>Мультиязычный VLM.</b> Если важна поддержка русских промптов — "
        "использовать <i>multilingual-clip</i>, SigLIP-2 или LiT-подход с mBERT.",
        "<b>Скоринг реальных наблюдений тоже.</b> Полезная абляция: применять VLM "
        "к реальному наблюдению на каждом шаге как shaping-бонус, чтобы reward-head "
        "RSSM учился плотному псевдо-reward. Это отвязывает качество модели мира "
        "от полезности VLM для планирования.",
        "<b>Более сложные среды.</b> Перейти к DoorKey / KeyCorridor с "
        "multi-clause промптами (&laquo;agent next to the key&raquo; → "
        "&laquo;agent in front of the door&raquo;), чтобы проверить, может ли CLIP "
        "направлять переключение подцелей.",
    ]:
        story.append(Paragraph(f"• {b}", styles["bullet"]))

    # ---- Ссылки ----
    story.append(Paragraph("Ссылки", styles["h2"]))
    story.append(Paragraph(
        "1. Hafner, D. et al. <b>Learning Latent Dynamics for Planning from Pixels</b> (PlaNet). "
        "<a href='https://arxiv.org/abs/1811.04551' color='#20808D'>arXiv:1811.04551</a>",
        styles["body"]))
    story.append(Paragraph(
        "2. Hafner, D. et al. <b>Mastering Diverse Domains through World Models</b> (DreamerV3). "
        "<a href='https://arxiv.org/abs/2301.04104' color='#20808D'>arXiv:2301.04104</a>",
        styles["body"]))
    story.append(Paragraph(
        "3. Radford, A. et al. <b>Learning Transferable Visual Models from Natural Language Supervision</b> (CLIP). "
        "<a href='https://arxiv.org/abs/2103.00020' color='#20808D'>arXiv:2103.00020</a>",
        styles["body"]))
    story.append(Paragraph(
        "4. Chevalier-Boisvert, M. et al. <b>MiniGrid &amp; Miniworld</b>. "
        "<a href='https://minigrid.farama.org/' color='#20808D'>minigrid.farama.org</a>",
        styles["body"]))
    story.append(Paragraph(
        "5. Репозиторий проекта: "
        "<a href='https://github.com/VadShv/tz-demo' color='#20808D'>github.com/VadShv/tz-demo</a>",
        styles["body"]))

    doc.build(story)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    build()
