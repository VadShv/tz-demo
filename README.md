# World Model + VLM-скорер — демо

Минимальный демо-проект, который объединяет **модель мира в стиле Dreamer (RSSM)**
с **VLM-скорером на базе CLIP** для планирования в среде
[MiniGrid](https://minigrid.farama.org/).

```
                            ┌──────────────────┐
    реальное наблюдение ──▶ │   RSSM (модель   │  ─воображаемые rollout-ы─▶ декодированные кадры
                            │   мира: GRU +    │
                            │   стохастический │  ─reward head──▶           предсказанный return
                            │   латент)        │                                    │
                            └──────────────────┘                                    ▼
                                   ┌──────────────────────────┐    ┌─────────────────┐
                                   │  Планировщик             │◀──│  CLIP text-image│
                                   │  (random shooting):      │    │  скорер (VLM)   │
                                   │  выбирает первое действие│    └─────────────────┘
                                   └──────────────────────────┘
```

## Что внутри

| Компонент | Файл | Комментарий |
|---|---|---|
| MiniGrid pixel env (64×64 RGB) | `src/env.py` | `MiniGrid-Empty-8x8-v0`, 3 действия |
| Replay-буфер эпизодов | `src/replay.py` | последовательности для обучения RSSM |
| RSSM модель мира (~2.3M параметров) | `src/rssm.py` | CNN encoder/decoder, GRU, стохастический латент |
| CLIP-скорер по кадрам rollout-а | `src/vlm_scorer.py` | ViT-B/32 OpenAI, контрастивные промпты |
| Random-shooting MPC планировщик | `src/planner.py` | сменяемая функция скоринга |
| Агенты (Random, WM+reward, WM+VLM) | `src/agents.py` | |
| Скрипт обучения | `src/train_wm.py` | |
| Скрипт eval с метриками и GIF | `src/evaluate.py` | |

## Установка

```bash
pip install -r requirements.txt
```

## Использование

**1. Обучить модель мира** (собирает случайные эпизоды и оптимизирует RSSM):

```bash
python -m src.train_wm --episodes 200 --updates 2000 --batch 16 --seq 20 \
                      --out checkpoints/rssm.pt
```

Рекомендуемые настройки для GPU (Colab T4): `--episodes 500 --updates 10000`.

**2. Прогнать все три агента**:

```bash
python -m src.evaluate --ckpt checkpoints/rssm.pt \
                      --episodes 20 --seeds 0 1 2 \
                      --horizon 12 --num-seq 128
```

На выходе — `results/metrics.json` и GIF-и в `results/gifs/`.

## Метод

**Модель мира (RSSM).** Следуя recurrent state-space model из PlaNet
([Hafner et al., 2018](https://arxiv.org/abs/1811.04551)) и Dreamer
([Hafner et al., 2020](https://arxiv.org/abs/1912.01603) /
[Hafner et al., 2023](https://arxiv.org/abs/2301.04104)), состояние
`s_t = (h_t, z_t)` включает детерминированное скрытое состояние GRU `h_t`
и стохастический латент `z_t`. Обучение минимизирует MSE-реконструкцию
изображений + KL(апостериор ‖ приор) с free bits + MSE-предсказание
вознаграждения.

**Планировщик.** На каждом реальном шаге выполняется *random shooting*:
семплируется `N` последовательностей действий длины `H`, каждая
разворачивается в воображении моделью мира, скорится, и выполняется
первое действие лучшей последовательности. Всё работает на CPU; CEM
подключается как drop-in замена.

**VLM-скорер.** CLIP ViT-B/32 с контрастивными промптами:

- позитивный: `"a red triangle agent standing on the green goal square"`
- негативный: `"a red triangle agent far from the green goal square"`

Скор — это `100 · (cos_sim(pos) − cos_sim(neg))`, усреднённый (с
дисконтом) по *будущим* декодированным кадрам rollout-а. Это
удовлетворяет требованию задания, чтобы скоринг применялся к
воображаемым будущим состояниям, а не только к текущему наблюдению.

**Почему промпты на английском.** CLIP от OpenAI обучен почти
исключительно на англоязычных подписях; русскоязычные текстовые запросы
дают близкий к нулю сигнал и делают скорер бесполезным. Для
русскоязычного скорера потребовалась бы мультиязычная модель, например
`multilingual-clip` или SigLIP-2 — это возможное направление на
будущее.

**Baselines (обязательные).**

1. `random` — равномерная случайная политика
2. `wm_reward` — тот же MPC-планировщик, но rollout скорится собственной
   `reward_head` RSSM (без VLM)

## Результаты

Прогон на Colab T4 GPU, 20 эпизодов на seed:

| Агент | Seeds | Success rate | Средний return | Средняя длина эпизода |
|---|---|---|---|---|
| Random | 0, 1 | 5% ± 0% | 0.041 | 59.6 |
| WM + reward (без VLM) | 0, 1 | **100% ± 0%** | **0.940** | **17.0** |
| WM + VLM (CLIP ViT-B/32) | 0 | 0% | 0.000 | 60.0 |

- **WM + reward** — планировщик в модели мира со скорингом по `reward_head`
  RSSM — решает 8×8 Empty идеально и почти в 4 раза короче эпизодами,
  чем случайная политика. Это подтверждает, что RSSM выучил и динамику,
  и предсказание разреженной награды.
- **WM + VLM** — тот же MPC-пайплайн, но скоринг rollout-а — CLIP по
  воображаемым будущим кадрам. Результат наглядно демонстрирует
  ограничение: MiniGrid-рендеры лежат вне распределения OpenAI CLIP,
  и контрастивный скор становится шумом. Подробный разбор failure modes
  и future work — в `report/report.pdf`.

Подробности см. в `report/report.pdf`, сырые числа — в
`results/metrics.json`, GIF-и первого эпизода каждого агента — в
`results/gifs/`.

## Структура репозитория

```
tz-demo/
├── src/                  # библиотека
├── scripts/              # запускалки
├── configs/default.yaml  # гиперпараметры
├── checkpoints/          # обученные чекпоинты RSSM
├── results/              # метрики + GIF
├── report/report.pdf     # PDF-отчёт
└── notebooks/demo.ipynb  # Colab-ноутбук
```

## Запуск на CPU vs GPU

Приведённые в `report/report.pdf` и `results/metrics.json` числа
получены **на Colab T4 GPU**. Обучение RSSM — 300 случайных эпизодов
в буфер + 4000 шагов оптимизации (batch=16, seq=20, ≈5–10 минут).
Эвалюация 20 эпизодов на seed: Random и WM+reward прогнаны на seed 0
и 1 (несколько секунд и ≈30 с соответственно). WM+VLM прогнан на
seed 0 — один seed занимает ≈**73 минуты** на T4 (128 кандидатов ×
горизонт 12 × 20 эпизодов, CLIP ViT-B/32 на каждом воображаемом
кадре), поэтому второй seed был прерван вручную. Результат для
одного seed репрезентативен: он демонстрирует ключевой failure mode
(MiniGrid-рендеры лежат вне распределения OpenAI CLIP), подробности
— в `report/report.pdf`.

Смок-тест `scripts/run_smoke_test.py` (30 эпизодов / 200 шагов
обучения) нужен только чтобы за минуту убедиться, что pipeline
работает end-to-end на CPU — осмысленных policy-метрик он не даёт.

## Ссылки

- Hafner et al., **PlaNet** — Learning Latent Dynamics for Planning from Pixels. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
- Hafner et al., **Dreamer-V3** — Mastering Diverse Domains through World Models. [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)
- Radford et al., **CLIP** — Learning Transferable Visual Models. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- [MiniGrid](https://minigrid.farama.org/) — Chevalier-Boisvert et al., 2023

## Лицензия

MIT
