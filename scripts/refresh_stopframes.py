"""
Пересобирает stop-frames (start / end) для отчёта, отбирая нужный тип эпизода:
    random    → берём failure (агент НЕ на цели в конце)
    wm_reward → берём success (агент НА цели в конце)
    wm_vlm    → берём failure

Логика:
1. Идём по ВСЕМ GIF-ам агента в results/gifs/ (random_seed{0,1}_ep*.gif и т.д.).
2. Для каждого GIF читаем последний кадр, детектим success:
       success ⇔ красный пиксель агента совпадает по позиции с зелёной клеткой цели.
3. Берём первый подходящий эпизод и сохраняем 2 файла:
       results/gifs/<agent>_seed0_ep0_start.png   (первый кадр)
       results/gifs/<agent>_seed0_ep0_end.png     (последний кадр)
   (имена оставляем ep0, чтобы не менять build_report.py)

Запуск:  python scripts/refresh_stopframes.py
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
GIFS = ROOT / "results" / "gifs"

# что мы хотим для каждого агента: True = success, False = failure
TARGET = {
    "random":    False,
    "wm_reward": True,
    "wm_vlm":    False,
}


def _is_success(frame: Image.Image) -> bool:
    """Success ⇔ красный треугольник агента накладывается на зелёную клетку цели."""
    arr = np.array(frame.convert("RGB"))
    # маска "красноватых" пикселей (агент)
    red = (arr[..., 0] > 150) & (arr[..., 1] < 80) & (arr[..., 2] < 80)
    # маска "зелёных" пикселей (цель)
    green = (arr[..., 0] < 80) & (arr[..., 1] > 150) & (arr[..., 2] < 80)
    if not red.any() or not green.any():
        return False
    # центроид красного
    ry, rx = np.argwhere(red).mean(axis=0)
    # ближайшее зелёное расстояние
    gy, gx = np.argwhere(green).mean(axis=0)
    dist = ((ry - gy) ** 2 + (rx - gx) ** 2) ** 0.5
    # в MiniGrid-5x5 клетка ≈ H/5. Если расстояние меньше 0.7 клетки → на цели.
    cell = arr.shape[0] / 5.0
    return dist < 0.7 * cell


def _load_frames(gif_path: Path) -> tuple[Image.Image, Image.Image]:
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frames.append(img.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames[0], frames[-1]


def _pick_and_save(agent: str) -> None:
    want_success = TARGET[agent]
    candidates = sorted(glob.glob(str(GIFS / f"{agent}_seed*_ep*.gif")))
    print(f"\n[{agent}]  want_success={want_success}  candidates={len(candidates)}")

    picked = None
    for gp in candidates:
        gp = Path(gp)
        start, end = _load_frames(gp)
        ok = _is_success(end)
        print(f"  - {gp.name:<40s}  success={ok}")
        if ok == want_success and picked is None:
            picked = (gp, start, end)

    if picked is None:
        # fallback: если нужного типа нет, берём первый и печатаем warning
        gp = Path(candidates[0])
        start, end = _load_frames(gp)
        print(f"  ! нет эпизода с требуемым success={want_success}, беру {gp.name} как fallback")
        picked = (gp, start, end)

    gp, start, end = picked
    out_start = GIFS / f"{agent}_seed0_ep0_start.png"
    out_end   = GIFS / f"{agent}_seed0_ep0_end.png"
    # апскейл до 256×256 nearest-neighbor для читаемости в PDF
    start.convert("RGB").resize((256, 256), Image.NEAREST).save(out_start, "PNG")
    end.convert("RGB").resize((256, 256), Image.NEAREST).save(out_end, "PNG")
    print(f"  ✓ start → {out_start.name}")
    print(f"  ✓ end   → {out_end.name}   (source: {gp.name})")


def main() -> None:
    if not GIFS.exists():
        sys.exit(f"нет папки {GIFS}")
    for agent in TARGET:
        _pick_and_save(agent)
    print("\nГотово. Теперь можно пересобрать PDF:\n    python report/build_report.py")


if __name__ == "__main__":
    main()
