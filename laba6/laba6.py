# laba6.py

from pathlib import Path
import csv
import unicodedata
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator


# -----------------------------
# НАСТРОЙКИ
# -----------------------------
OUTPUT_DIR = Path("lab6_output")

INPUT_DIR = OUTPUT_DIR / "input"
VIS_DIR = OUTPUT_DIR / "visualizations"
PROFILES_DIR = OUTPUT_DIR / "profiles"
SEGMENTS_DIR = OUTPUT_DIR / "segmented_chars"
ALPHABET_PROFILES_DIR = OUTPUT_DIR / "alphabet_profiles"
ALPHABET_SYMBOLS_DIR = OUTPUT_DIR / "alphabet_symbols"

RECTS_CSV_PATH = OUTPUT_DIR / "rectangles.csv"
ALPHABET_PATH = OUTPUT_DIR / "alphabet.txt"

for folder in [
    OUTPUT_DIR,
    INPUT_DIR,
    VIS_DIR,
    PROFILES_DIR,
    SEGMENTS_DIR,
    ALPHABET_PROFILES_DIR,
    ALPHABET_SYMBOLS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

INPUT_IMAGE = None

PREFERRED_INPUT_NAMES = [
    "phrase.bmp",
    "text.bmp",
]

FONT_SIZE = 52
FONT_PATH = r"C:\Users\m9164\AppData\Local\Microsoft\Windows\Fonts\PonomarUnicode.ttf"

CANVAS_SIZE = (240, 240)
BINARIZE_THRESHOLD = 200
CROP_PADDING = 0

IMAGE_FRAME_PADDING = 8
IMAGE_FRAME_BORDER_WIDTH = 3
IMAGE_FRAME_BORDER_COLOR = (0, 160, 0)
IMAGE_FRAME_BACKGROUND_COLOR = (255, 255, 255)

ALPHABET = [
    "А", "Б", "В", "Г", "Д", "Є", "Ж", "Ѕ", "З", "И", "І", "Й",
    "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "Ѹ",
    "Ф", "Х", "Ц", "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Ѣ",
    "Ю", "Ѵ", "Ѯ", "Ѱ", "Ѡ", "Ѧ", "Ѩ"
]

LINE_PROFILE_THRESHOLD = 1
CHAR_PROFILE_THRESHOLD = 1

LINE_MIN_GAP = 6
LINE_MIN_RUN = 8

CHAR_MIN_GAP = 0
CHAR_MIN_RUN = 1

CHAR_BODY_TOP_RATIO = 0.15
CHAR_BODY_BOTTOM_RATIO = 0.92

WIDE_SEGMENT_FACTOR = 1.9

MERGE_GAP = 8
SMALL_PART_RATIO = 1.20

# -----------------------------
# ПОИСК ВХОДНОГО BMP
# -----------------------------
def find_input_image() -> Path:
    if INPUT_IMAGE is not None:
        if INPUT_IMAGE.exists():
            return INPUT_IMAGE
        raise FileNotFoundError(f"Входной файл не найден: {INPUT_IMAGE}")

    cwd = Path.cwd()

    for name in PREFERRED_INPUT_NAMES:
        candidate = cwd / name
        if candidate.exists():
            return candidate

    bmps = sorted(cwd.glob("*.bmp"))
    if bmps:
        return bmps[0]

    raise FileNotFoundError(
        "Не найден входной BMP-файл. "
        "Положите файл в папку со скриптом или укажите путь в INPUT_IMAGE."
    )


# -----------------------------
# ПОИСК ШРИФТА
# -----------------------------
def find_ponomar_font() -> str:
    if FONT_PATH is not None:
        path = Path(FONT_PATH)
        if path.exists():
            return str(path)
        raise FileNotFoundError(f"Шрифт не найден: {FONT_PATH}")

    candidates = [
        r"C:\Windows\Fonts\PonomarUnicode.ttf",
        r"C:\Windows\Fonts\PonomarUnicode.otf",
        r"C:\Windows\Fonts\Ponomar Unicode TT.ttf",
        r"C:\Windows\Fonts\Ponomar Unicode.ttf",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    windows_fonts = Path(r"C:\Windows\Fonts")
    if windows_fonts.exists():
        for path in windows_fonts.glob("*Ponomar*.*"):
            if path.suffix.lower() in [".ttf", ".otf"]:
                return str(path)

    raise FileNotFoundError(
        "Не найден шрифт Ponomar Unicode. "
        "Укажите путь к нему в переменной FONT_PATH."
    )


# -----------------------------
# РАБОТА С ИЗОБРАЖЕНИЯМИ
# -----------------------------
def rgb_to_grayscale_manual(rgb: np.ndarray) -> np.ndarray:
    rgb_f = rgb.astype(np.float32)
    r = rgb_f[:, :, 0]
    g = rgb_f[:, :, 1]
    b = rgb_f[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)


def load_image_as_binary(path: Path, threshold: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Возвращает:
    - gray: grayscale uint8
    - binary: black=1, white=0
    """
    img = Image.open(path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    gray = rgb_to_grayscale_manual(rgb)
    binary = np.where(gray < threshold, 1, 0).astype(np.uint8)
    return gray, binary


def add_green_frame(image: Image.Image) -> Image.Image:
    """
    Добавляет зелёную рамку и белый отступ вокруг изображения.
    """
    image_rgb = image.convert("RGB")

    out_w = image_rgb.width + 2 * (IMAGE_FRAME_PADDING + IMAGE_FRAME_BORDER_WIDTH)
    out_h = image_rgb.height + 2 * (IMAGE_FRAME_PADDING + IMAGE_FRAME_BORDER_WIDTH)

    out_img = Image.new("RGB", (out_w, out_h), IMAGE_FRAME_BACKGROUND_COLOR)
    draw = ImageDraw.Draw(out_img)

    for i in range(IMAGE_FRAME_BORDER_WIDTH):
        draw.rectangle(
            [i, i, out_w - 1 - i, out_h - 1 - i],
            outline=IMAGE_FRAME_BORDER_COLOR,
        )

    paste_xy = (
        IMAGE_FRAME_PADDING + IMAGE_FRAME_BORDER_WIDTH,
        IMAGE_FRAME_PADDING + IMAGE_FRAME_BORDER_WIDTH,
    )
    out_img.paste(image_rgb, paste_xy)
    return out_img


def save_gray_image_with_frame(gray: np.ndarray, path: Path) -> None:
    img = Image.fromarray(gray, mode="L")
    add_green_frame(img).save(path)


def save_binary_image(binary: np.ndarray, path: Path) -> None:
    img = np.where(binary == 1, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode="L")
    add_green_frame(pil_img).save(path)


def save_original_image_with_frame(input_path: Path, path: Path) -> None:
    img = Image.open(input_path).convert("RGB")
    add_green_frame(img).save(path)


def bounding_box_of_black(binary: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(binary == 1)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, binary.shape[1], binary.shape[0]

    x_min = int(xs.min())
    x_max = int(xs.max()) + 1
    y_min = int(ys.min())
    y_max = int(ys.max()) + 1
    return x_min, y_min, x_max, y_max


def crop_binary(binary: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    return binary[y1:y2, x1:x2]


# -----------------------------
# ПРОФИЛИ
# -----------------------------
def horizontal_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=1).astype(int)


def vertical_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=0).astype(int)


def bridge_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = mask.copy()
    n = len(result)
    i = 0

    while i < n:
        if result[i] == 0:
            start = i
            while i < n and result[i] == 0:
                i += 1
            end = i

            gap_len = end - start
            left_is_one = start > 0 and result[start - 1] == 1
            right_is_one = end < n and result[end] == 1

            if left_is_one and right_is_one and gap_len <= max_gap:
                result[start:end] = 1
        else:
            i += 1

    return result


def remove_small_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    result = mask.copy()
    n = len(result)
    i = 0

    while i < n:
        if result[i] == 1:
            start = i
            while i < n and result[i] == 1:
                i += 1
            end = i

            run_len = end - start
            if run_len < min_run:
                result[start:end] = 0
        else:
            i += 1

    return result


def thin_profile(
    profile: np.ndarray,
    threshold: int,
    min_gap: int,
    min_run: int,
    bridge_gaps: bool = True
) -> np.ndarray:
    """
    Прореживание профиля:
    1) порог;
    2) при необходимости склейка коротких разрывов;
    3) удаление коротких шумовых участков.
    """
    mask = (profile >= threshold).astype(np.uint8)

    if bridge_gaps and min_gap > 0:
        mask = bridge_small_gaps(mask, max_gap=min_gap)

    mask = remove_small_runs(mask, min_run=min_run)
    return mask


def mask_to_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges = []
    n = len(mask)
    i = 0

    while i < n:
        if mask[i] == 1:
            start = i
            while i < n and mask[i] == 1:
                i += 1
            end = i
            ranges.append((start, end))
        else:
            i += 1

    return ranges


# -----------------------------
# ДОПОЛНИТЕЛЬНАЯ ЛОГИКА ДЛЯ СИМВОЛОВ
# -----------------------------
def get_char_body(line_crop: np.ndarray) -> tuple[np.ndarray, int, int]:
    """
    Берём центральную часть строки для вертикального профиля,
    чтобы верхние декоративные элементы меньше мешали сегментации.
    """
    h = line_crop.shape[0]
    y1 = int(round(h * CHAR_BODY_TOP_RATIO))
    y2 = int(round(h * CHAR_BODY_BOTTOM_RATIO))

    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return line_crop[y1:y2, :], y1, y2


def estimate_symbol_widths_from_alphabet(font_path: str) -> float:
    """
    Оцениваем типичную ширину символа по алфавиту.
    """
    font = ImageFont.truetype(font_path, FONT_SIZE)
    widths = []

    for symbol in ALPHABET:
        gray = render_symbol(symbol, font)
        binary = np.where(gray < BINARIZE_THRESHOLD, 1, 0).astype(np.uint8)
        binary = crop_binary_image(binary, padding=CROP_PADDING)
        _, w = binary.shape
        widths.append(w)

    if not widths:
        return 20.0

    return float(np.median(widths))


def split_wide_segment_by_valleys(
    body_profile: np.ndarray,
    start: int,
    end: int,
    target_width: float
) -> list[tuple[int, int]]:
    """
    Если сегмент слишком широкий, пытаемся разрезать его по локальным минимумам профиля.
    """
    width = end - start
    if width <= WIDE_SEGMENT_FACTOR * target_width:
        return [(start, end)]

    local = body_profile[start:end]
    if len(local) < 4:
        return [(start, end)]

    valleys = []
    for i in range(1, len(local) - 1):
        if local[i] <= local[i - 1] and local[i] <= local[i + 1]:
            valleys.append((local[i], i))

    if not valleys:
        return [(start, end)]

    expected_parts = max(2, int(round(width / target_width)))
    expected_cuts = expected_parts - 1

    valleys_sorted = sorted(valleys, key=lambda t: (t[0], abs((start + t[1]) - (start + end) / 2)))
    candidate_positions = sorted([start + i for _, i in valleys_sorted[:expected_cuts]])

    if not candidate_positions:
        return [(start, end)]

    segments = []
    prev = start
    for cut in candidate_positions:
        if cut - prev >= 2:
            segments.append((prev, cut))
            prev = cut
    if end - prev >= 2:
        segments.append((prev, end))

    if len(segments) <= 1:
        return [(start, end)]

    return segments

def merge_adjacent_char_rects(
    char_rects: list[tuple[int, int, int, int]],
    median_symbol_width: float
) -> list[tuple[int, int, int, int]]:
    """
    Склеивает соседние сегменты, если:
    - между ними очень маленький промежуток;
    - один из сегментов слишком узкий относительно средней ширины символа.

    Это помогает для букв вроде Ы, Ю, а также декоративных старокириллических форм.
    """
    if not char_rects:
        return []

    merged = []
    i = 0

    while i < len(char_rects):
        current = char_rects[i]

        if i == len(char_rects) - 1:
            merged.append(current)
            break

        nxt = char_rects[i + 1]

        x1, y1, x2, y2 = current
        nx1, ny1, nx2, ny2 = nxt

        current_width = x2 - x1
        next_width = nx2 - nx1
        gap = nx1 - x2

        small_width = min(current_width, next_width)
        is_small_part = small_width <= median_symbol_width * SMALL_PART_RATIO

        if gap <= MERGE_GAP and is_small_part:
            new_rect = (
                min(x1, nx1),
                min(y1, ny1),
                max(x2, nx2),
                max(y2, ny2),
            )
            merged.append(new_rect)
            i += 2
        else:
            merged.append(current)
            i += 1

    return merged


# -----------------------------
# ВИЗУАЛИЗАЦИЯ ПРОФИЛЕЙ
# -----------------------------
def save_profile_plot(
    values: np.ndarray,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    horizontal: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    coords = np.arange(len(values))

    if horizontal:
        ax.barh(coords, values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.bar(coords, values)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# СЕГМЕНТАЦИЯ
# -----------------------------
def segment_lines(binary: np.ndarray, text_rect: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = text_rect
    text_crop = binary[y1:y2, x1:x2]

    hp = horizontal_profile(text_crop)
    hp_mask = thin_profile(
        hp,
        threshold=LINE_PROFILE_THRESHOLD,
        min_gap=LINE_MIN_GAP,
        min_run=LINE_MIN_RUN,
        bridge_gaps=True,
    )
    line_ranges = mask_to_ranges(hp_mask)

    line_rects = []
    for ys, ye in line_ranges:
        line_rect = (x1, y1 + ys, x2, y1 + ye)

        line_crop = crop_binary(binary, line_rect)
        lx1, ly1, lx2, ly2 = bounding_box_of_black(line_crop)
        refined_rect = (
            line_rect[0] + lx1,
            line_rect[1] + ly1,
            line_rect[0] + lx2,
            line_rect[1] + ly2,
        )
        line_rects.append(refined_rect)

    return line_rects


def segment_characters_in_line(
    binary: np.ndarray,
    line_rect: tuple[int, int, int, int],
    median_symbol_width: float
) -> list[tuple[int, int, int, int]]:
    """
    Улучшенная сегментация символов:
    - вертикальный профиль строится по центральной части строки;
    - короткие пробелы не склеиваются;
    - слишком широкие сегменты режутся по локальным минимумам профиля.
    """
    x1, y1, x2, y2 = line_rect
    line_crop = binary[y1:y2, x1:x2]

    body_crop, body_y1, body_y2 = get_char_body(line_crop)
    vp = vertical_profile(body_crop)

    vp_mask = thin_profile(
        vp,
        threshold=CHAR_PROFILE_THRESHOLD,
        min_gap=CHAR_MIN_GAP,
        min_run=CHAR_MIN_RUN,
        bridge_gaps=False,
    )

    coarse_ranges = mask_to_ranges(vp_mask)

    final_ranges = []
    for xs, xe in coarse_ranges:
        split_ranges = split_wide_segment_by_valleys(vp, xs, xe, median_symbol_width)
        final_ranges.extend(split_ranges)

    char_rects = []
    for xs, xe in final_ranges:
        char_rect = (x1 + xs, y1, x1 + xe, y2)

        char_crop = crop_binary(binary, char_rect)
        cx1, cy1, cx2, cy2 = bounding_box_of_black(char_crop)
        refined_rect = (
            char_rect[0] + cx1,
            char_rect[1] + cy1,
            char_rect[0] + cx2,
            char_rect[1] + cy2,
        )
        char_rects.append(refined_rect)

    return char_rects


def segment_text(
    binary: np.ndarray,
    median_symbol_width: float
) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    text_rect = bounding_box_of_black(binary)
    line_rects = segment_lines(binary, text_rect)

    char_rects = []
    for line_rect in line_rects:
        chars_in_line = segment_characters_in_line(binary, line_rect, median_symbol_width)
        chars_in_line = merge_adjacent_char_rects(chars_in_line, median_symbol_width)
        char_rects.extend(chars_in_line)

    return text_rect, line_rects, char_rects


# -----------------------------
# СОХРАНЕНИЕ ПРЯМОУГОЛЬНИКОВ И ФРАГМЕНТОВ
# -----------------------------
def save_rectangles_csv(
    text_rect: tuple[int, int, int, int],
    line_rects: list[tuple[int, int, int, int]],
    char_rects: list[tuple[int, int, int, int]],
    path: Path
) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["type", "index", "x1", "y1", "x2", "y2"])

        writer.writerow(["text", 1, *text_rect])

        for i, rect in enumerate(line_rects, start=1):
            writer.writerow(["line", i, *rect])

        for i, rect in enumerate(char_rects, start=1):
            writer.writerow(["char", i, *rect])


def save_segmented_characters(binary: np.ndarray, char_rects: list[tuple[int, int, int, int]]) -> None:
    for idx, rect in enumerate(char_rects, start=1):
        crop = crop_binary(binary, rect)
        save_binary_image(crop, SEGMENTS_DIR / f"char_{idx:03d}.png")


def save_rectangles_visualization(gray: np.ndarray, rects: list[tuple[int, int, int, int]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

    for rect in rects:
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        ax.add_patch(Rectangle((x1, y1), w, h, fill=False, linewidth=1.5, edgecolor="red"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_mixed_visualization(
    gray: np.ndarray,
    text_rect: tuple[int, int, int, int],
    line_rects: list[tuple[int, int, int, int]],
    char_rects: list[tuple[int, int, int, int]],
    out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Общий прямоугольник текста, строки и символы")
    ax.axis("off")

    x1, y1, x2, y2 = text_rect
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2.0, edgecolor="blue"))

    for rect in line_rects:
        rx1, ry1, rx2, ry2 = rect
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, linewidth=1.5, edgecolor="green"))

    for rect in char_rects:
        rx1, ry1, rx2, ry2 = rect
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, fill=False, linewidth=0.8, edgecolor="red"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# ПРОФИЛИ АЛФАВИТА
# -----------------------------
def render_symbol(symbol: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    image = Image.new("L", CANVAS_SIZE, 255)
    draw = ImageDraw.Draw(image)

    bbox = draw.textbbox((0, 0), symbol, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    x = (CANVAS_SIZE[0] - text_w) // 2 - bbox[0]
    y = (CANVAS_SIZE[1] - text_h) // 2 - bbox[1]

    draw.text((x, y), symbol, font=font, fill=0)
    return np.array(image, dtype=np.uint8)


def crop_binary_image(binary: np.ndarray, padding: int = 0) -> np.ndarray:
    ys, xs = np.where(binary == 1)

    if len(xs) == 0 or len(ys) == 0:
        return binary.copy()

    x_min = max(0, xs.min() - padding)
    x_max = min(binary.shape[1], xs.max() + 1 + padding)
    y_min = max(0, ys.min() - padding)
    y_max = min(binary.shape[0], ys.max() + 1 + padding)

    return binary[y_min:y_max, x_min:x_max]


def make_safe_name(symbol: str) -> str:
    return f"U+{ord(symbol):04X}_{symbol}"


def save_profiles_png(symbol: str, profile_x: np.ndarray, profile_y: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(np.arange(len(profile_x)), profile_x)
    axes[0].set_title(f"Профиль X: {symbol}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Сумма чёрных пикселей")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].yaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1].barh(np.arange(len(profile_y)), profile_y)
    axes[1].set_title(f"Профиль Y: {symbol}")
    axes[1].set_xlabel("Сумма чёрных пикселей")
    axes[1].set_ylabel("Y")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_alphabet_profiles(font_path: str) -> None:
    font = ImageFont.truetype(font_path, FONT_SIZE)

    with open(ALPHABET_PATH, "w", encoding="utf-8") as f:
        f.write(" ".join(ALPHABET))

    for symbol in ALPHABET:
        gray = render_symbol(symbol, font)
        binary = np.where(gray < BINARIZE_THRESHOLD, 1, 0).astype(np.uint8)
        binary = crop_binary_image(binary, padding=CROP_PADDING)

        safe_name = make_safe_name(symbol)

        symbol_path = ALPHABET_SYMBOLS_DIR / f"{safe_name}.png"
        save_binary_image(binary, symbol_path)

        px = vertical_profile(binary)
        py = horizontal_profile(binary)

        profile_path = ALPHABET_PROFILES_DIR / f"{safe_name}_profiles.png"
        save_profiles_png(symbol, px, py, profile_path)


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=== Лабораторная работа №6 ===")
    print("Сегментация текста")
    print()

    input_path = find_input_image()
    font_path = find_ponomar_font()
    median_symbol_width = estimate_symbol_widths_from_alphabet(font_path)

    print(f"Входное изображение: {input_path}")
    print(f"Используемый шрифт:  {font_path}")
    print(f"Оценка средней ширины символа: {median_symbol_width:.2f}")
    print(f"Алфавит: {' '.join(ALPHABET)}")
    print()

    gray, binary = load_image_as_binary(input_path, threshold=BINARIZE_THRESHOLD)

    input_original_path = INPUT_DIR / "input_phrase_original.png"
    input_gray_path = INPUT_DIR / "input_gray.bmp"
    input_binary_path = INPUT_DIR / "input_binary.bmp"

    save_original_image_with_frame(input_path, input_original_path)
    save_gray_image_with_frame(gray, input_gray_path)
    save_binary_image(binary, input_binary_path)

    total_hp = horizontal_profile(binary)
    total_vp = vertical_profile(binary)

    save_profile_plot(
        total_hp,
        PROFILES_DIR / "horizontal_profile_full.png",
        "Горизонтальный профиль всего изображения",
        "Y",
        "Сумма чёрных пикселей",
        horizontal=False,
    )

    save_profile_plot(
        total_vp,
        PROFILES_DIR / "vertical_profile_full.png",
        "Вертикальный профиль всего изображения",
        "Сумма чёрных пикселей",
        "X",
        horizontal=True,
    )

    text_rect, line_rects, char_rects = segment_text(binary, median_symbol_width)

    text_crop = crop_binary(binary, text_rect)
    hp_text = horizontal_profile(text_crop)
    vp_text = vertical_profile(text_crop)

    save_profile_plot(
        hp_text,
        PROFILES_DIR / "horizontal_profile_text.png",
        "Горизонтальный профиль текстовой области",
        "Y",
        "Сумма чёрных пикселей",
        horizontal=False,
    )

    save_profile_plot(
        vp_text,
        PROFILES_DIR / "vertical_profile_text.png",
        "Вертикальный профиль текстовой области",
        "Сумма чёрных пикселей",
        "X",
        horizontal=True,
    )

    for i, line_rect in enumerate(line_rects, start=1):
        line_crop = crop_binary(binary, line_rect)
        vp_line = vertical_profile(line_crop)
        hp_line = horizontal_profile(line_crop)

        save_profile_plot(
            hp_line,
            PROFILES_DIR / f"line_{i:02d}_horizontal_profile.png",
            f"Горизонтальный профиль строки {i}",
            "Y",
            "Сумма чёрных пикселей",
            horizontal=False,
        )

        save_profile_plot(
            vp_line,
            PROFILES_DIR / f"line_{i:02d}_vertical_profile.png",
            f"Вертикальный профиль строки {i}",
            "Сумма чёрных пикселей",
            "X",
            horizontal=True,
        )

    save_rectangles_csv(text_rect, line_rects, char_rects, RECTS_CSV_PATH)
    save_segmented_characters(binary, char_rects)

    save_rectangles_visualization(gray, [text_rect], VIS_DIR / "text_box.png", "Обрамляющий прямоугольник текста")
    save_rectangles_visualization(gray, line_rects, VIS_DIR / "lines_boxes.png", "Сегментация строк")
    save_rectangles_visualization(gray, char_rects, VIS_DIR / "chars_boxes.png", "Сегментация символов")
    save_mixed_visualization(gray, text_rect, line_rects, char_rects, VIS_DIR / "all_boxes.png")

    build_alphabet_profiles(font_path)

    print(f"Текстовый прямоугольник: {text_rect}")
    print(f"Количество строк:        {len(line_rects)}")
    print(f"Количество символов:     {len(char_rects)}")
    print()

    print("Результаты сохранены:")
    print(f"- Вход и фото фразы:  {INPUT_DIR.resolve()}")
    print(f"- Визуализации:       {VIS_DIR.resolve()}")
    print(f"- Профили текста:     {PROFILES_DIR.resolve()}")
    print(f"- Символы строки:     {SEGMENTS_DIR.resolve()}")
    print(f"- Профили алфавита:   {ALPHABET_PROFILES_DIR.resolve()}")
    print(f"- Символы алфавита:   {ALPHABET_SYMBOLS_DIR.resolve()}")
    print(f"- Прямоугольники CSV: {RECTS_CSV_PATH.resolve()}")
    print(f"- Алфавит:            {ALPHABET_PATH.resolve()}")


if __name__ == "__main__":
    main()