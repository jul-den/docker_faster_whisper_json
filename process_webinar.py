import os
import subprocess
import json
import glob
from pathlib import Path
import requests
from faster_whisper import WhisperModel

# ------------------------------------------------------------
# 1. Конфигурация (из переменных окружения)
# ------------------------------------------------------------
VIDEO_DIR = os.environ.get("VIDEO_DIR", "/source")           # папка с исходными файлами и JSON
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")     # папка для результатов
LLM_URL = os.environ.get("LLM_URL", "http://host.docker.internal:1234/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3-8b")
LLM_API_TOKEN = os.environ.get("LLM_API_TOKEN", None)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DELETE_WAV_AFTER_TRANSCRIBE = os.environ.get("DELETE_WAV_AFTER_TRANSCRIBE", "True").lower() == "true"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Глобальный кэш для модели Whisper
_whisper_model = None

# ------------------------------------------------------------
# 2. Вспомогательные функции (низкоуровневые операции)
# ------------------------------------------------------------
def get_whisper_model():
    """Ленивая загрузка модели Whisper с автоматическим переключением офлайн/онлайн."""
    global _whisper_model
    if _whisper_model is None:
        print(f"=== Загружаем Whisper ({WHISPER_MODEL}) на CPU ===", flush=True)
        # Сначала пробуем загрузить локально (офлайн)
        try:
            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                local_files_only=True
            )
            print("=== Модель успешно загружена из кэша ===", flush=True)
        except Exception as e:
            error_msg = str(e).lower()
            # Проверяем, связана ли ошибка с отсутствием локальной копии
            if any(key in error_msg for key in ["local_files_only", "no such file", "cannot find", "offline"]):
                print("=== Модель не найдена в кэше. Пробуем загрузить из интернета... ===", flush=True)
                try:
                    _whisper_model = WhisperModel(
                        WHISPER_MODEL,
                        device="cpu",
                        compute_type="int8",
                        local_files_only=False
                    )
                    print("=== Модель успешно загружена из интернета и сохранена в кэш ===", flush=True)
                except Exception as e2:
                    print("=== Не удалось загрузить модель даже из интернета ===", flush=True)
                    print(e2, flush=True)
                    raise RuntimeError("Не удалось загрузить модель Whisper. Проверьте подключение к интернету или наличие модели в кэше.") from e2
            else:
                # Другая ошибка
                print("=== Ошибка при загрузке модели (не связанная с отсутствием локальной копии) ===", flush=True)
                print(e, flush=True)
                raise
        print(f"=== Завершена загрузка Whisper ({WHISPER_MODEL}) ===", flush=True)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return _whisper_model

def resolve_file_list(input_spec, base_dir):
    """
    Преобразует входную спецификацию в список файлов.
    input_spec может быть строкой (одна маска) или списком строк (маски или точные имена).
    Возвращает отсортированный список Path объектов.
    """
    if isinstance(input_spec, str):
        patterns = [input_spec]
    elif isinstance(input_spec, list):
        patterns = input_spec
    else:
        raise TypeError("input должен быть строкой или списком строк")

    files = []
    for pat in patterns:
        # Если pat содержит glob-символы (*?), раскрываем, иначе считаем точным именем
        if any(c in pat for c in '*?[]'):
            found = sorted(base_dir.glob(pat))
        else:
            # Точное имя файла
            p = base_dir / pat
            if p.exists():
                found = [p]
            else:
                found = []
        files.extend(found)

    if not files:
        raise FileNotFoundError(f"Не найдено файлов по спецификации {input_spec} в {base_dir}")
    # Удаляем дубликаты (если вдруг одна маска перекрывает другую) и сортируем
    unique = sorted(set(files), key=lambda p: p.name)
    return unique

def get_audio_params(file_path):
    """Возвращает (sample_rate, channels) аудиофайла через ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().split()
    # Ожидаем две строки: sample_rate и channels
    if len(lines) >= 2:
        ar = int(lines[0])
        ac = int(lines[1])
    else:
        # Если не удалось, пробуем другие методы или ставим значения по умолчанию
        # Например, используем ffprobe с выводом в JSON
        cmd_json = [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json",
            str(file_path)
        ]
        res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
        import json
        data = json.loads(res.stdout)
        streams = data.get("streams", [])
        if streams:
            ar = int(streams[0].get("sample_rate", 16000))
            ac = int(streams[0].get("channels", 1))
        else:
            ar, ac = 16000, 1
    return ar, ac

def create_full_video(ts_files, output_video):
    """Склеивает список .ts файлов в один видеофайл (через concat demuxer)."""
    if output_video.exists():
        print(f"Полное видео {output_video} уже существует, пропускаем.", flush=True)
        return output_video
    list_file = output_video.parent / "concat_list.txt"
    with open(list_file, "w") as f:
        for ts in ts_files:
            f.write(f"file '{ts}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file), "-c", "copy", str(output_video)
    ]
    print("Склеиваем .ts файлы...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)
    return output_video

def cut_audio_segment(full_video, start_sec, end_sec, output_audio):
    """Вырезает аудиосегмент из видео и сохраняет как WAV (16 кГц, моно)."""
    if output_audio.exists():
        print(f"  Аудио {output_audio.name} уже существует, пропускаем.", flush=True)
        return
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-i", full_video,
        "-ss", str(start_sec),
        "-t", str(duration),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio
    ]
    subprocess.run(cmd, check=True, capture_output=True)

def transcribe_audio(audio_path, language="ru", beam_size=5):
    """Транскрибирует один аудиофайл, возвращает текст."""
    model = get_whisper_model()
    segments, _ = model.transcribe(str(audio_path), language=language, beam_size=beam_size)
    return " ".join(seg.text for seg in segments)

def call_lm_studio(prompt, output_file, system=None, max_tokens=8000, settings=None):
    """
    Отправляет запрос в LM Studio и сохраняет ответ.
    settings может содержать переопределённые lm_model и lm_url.
    """
    if output_file.exists():
        print(f"  Файл {output_file.name} уже существует, пропускаем запрос к LM Studio.", flush=True)
        return
    # Берём модель и URL из settings или глобальных переменных
    model = (settings.get("lm_model") if settings else None) or LLM_MODEL
    url = (settings.get("lm_url") if settings else None) or LLM_URL
    token = (settings.get("lm_token") if settings else None) or LLM_API_TOKEN

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens
    }

    headers = {
        "Content-Type": "application/json"
    }
    if token:  # добавляем токен, если он задан
        headers["Authorization"] = f"Bearer {token}"

    try:
        print(f"  Отправляем запрос к LM Studio ({model})...", flush=True)
        resp = requests.post(url, json=payload, headers=headers, timeout=30000)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0]["message"]["content"]
        elif "response" in data:
            answer = data["response"]
        else:
            answer = f"Неожиданный формат ответа: {data}"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(answer)
        print(f"  Ответ сохранён в {output_file}", flush=True)
    except Exception as e:
        print(f"  Ошибка при обращении к LM Studio: {e}", flush=True)
        with open(output_file.with_suffix(".error"), "w", encoding="utf-8") as f:
            f.write(f"Ошибка: {e}\n")

def find_input_file(filename, task_name):
    """
    Ищет файл сначала в подпапке задачи /output/<task_name>, затем в /source.
    Возвращает Path или None.
    """
    candidates = [
        Path(OUTPUT_DIR) / task_name / filename,
        Path(VIDEO_DIR) / filename
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

# ------------------------------------------------------------
# 3. Функции для выполнения шагов из JSON
# ------------------------------------------------------------
def step_concat(step, task_name, settings):
    """Шаг склейки файлов (поддерживает видео и аудио)."""
    input_spec = step["input"]
    output_file = step["output"]
    media_type = step.get("type", "video")  # по умолчанию video для обратной совместимости

    # Раскрываем входные файлы (всегда в VIDEO_DIR)
    source_dir = Path(VIDEO_DIR)
    input_files = resolve_file_list(input_spec, source_dir)

    output_path = Path(OUTPUT_DIR) / task_name / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Файл {output_path} уже существует, пропускаем concat.", flush=True)
        return

    if media_type == "video":
        # Используем существующую функцию для видео (копирование без перекодирования)
        create_full_video(input_files, output_path)
        return

    elif media_type == "audio":
        # Определяем целевые параметры аудио
        params = step.get("params", {})
        ar_target = params.get("ar")
        ac_target = params.get("ac")

        # Если параметры не заданы или "auto" – определяем по минимальным значениям среди файлов
        if ar_target == "auto" or ac_target == "auto" or (ar_target is None and ac_target is None):
            min_ar = float('inf')
            min_ac = float('inf')
            for f in input_files:
                ar, ac = get_audio_params(f)
                min_ar = min(min_ar, ar)
                min_ac = min(min_ac, ac)
            ar_target = min_ar if ar_target is None or ar_target == "auto" else ar_target
            ac_target = min_ac if ac_target is None or ac_target == "auto" else ac_target
        else:
            # Если заданы конкретные числа, используем их
            ar_target = ar_target if ar_target is not None else 16000
            ac_target = ac_target if ac_target is not None else 1

        # Подготавливаем временные файлы, если требуется перекодирование
        temp_files = []
        try:
            for i, f in enumerate(input_files):
                # Проверяем параметры исходного файла
                ar_src, ac_src = get_audio_params(f)
                if ar_src == ar_target and ac_src == ac_target:
                    # Совпадает – используем оригинал
                    temp_files.append(f)
                else:
                    # Нужно перекодировать
                    temp = output_path.parent / f"temp_concat_{i:04d}.wav"
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(f),
                        "-ar", str(ar_target),
                        "-ac", str(ac_target),
                        "-acodec", "pcm_s16le",
                        str(temp)
                    ]
                    print(f"  Перекодируем {f.name} -> {temp.name}", flush=True)
                    subprocess.run(cmd, check=True, capture_output=True)
                    temp_files.append(temp)

            # Теперь все файлы в едином формате, склеиваем через concat demuxer
            list_file = output_path.parent / "concat_audio_list.txt"
            with open(list_file, "w") as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")

            cmd_concat = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-c", "copy", str(output_path)
            ]
            print(f"  Склеиваем аудио в {output_path.name}", flush=True)
            subprocess.run(cmd_concat, check=True, capture_output=True)

        finally:
            # Удаляем временные файлы (кроме оригиналов)
            for tf in temp_files:
                if tf not in input_files:
                    tf.unlink(missing_ok=True)
            # Удаляем временный список
            if 'list_file' in locals():
                list_file.unlink(missing_ok=True)

        print(f"  Аудио склеено в {output_path}", flush=True)
    else:
        raise ValueError(f"Неизвестный тип media_type: {media_type}")

def step_extract_audio(step, task_name, settings):
    """Извлекает аудио из видеофайла."""
    input_file = step["input"]
    output_file = step["output"]
    input_path = find_input_file(input_file, task_name)
    if not input_path:
        raise FileNotFoundError(f"Входной файл {input_file} не найден")

    output_path = Path(OUTPUT_DIR) / task_name / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  Файл {output_path} уже существует, пропускаем extract_audio", flush=True)
        return

    params = step.get("params", {})
    ar = params.get("ar", 16000)
    ac = params.get("ac", 1)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(ar),
        "-ac", str(ac),
        str(output_path)
    ]
    print(f"  Извлекаем аудио из {input_path.name} -> {output_path.name}", flush=True)
    subprocess.run(cmd, check=True, capture_output=True)

def step_segment(step, task_name, settings):
    """Нарезает аудиофайл на сегменты по заданным интервалам."""
    input_file = step["input"]
    output_prefix = step["output_prefix"]
    segments = step["segments"]  # список объектов с "start" и "end"

    input_path = find_input_file(input_file, task_name)
    if not input_path:
        raise FileNotFoundError(f"Входной файл {input_file} не найден")

    out_fmt = step.get("format", "wav")
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        output_filename = f"{output_prefix}{i}.{out_fmt}"
        output_path = Path(OUTPUT_DIR) / task_name / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cut_audio_segment(input_path, start, end, output_path)

def step_transcribe(step, task_name, settings):
    """
    Транскрибирует один или несколько аудиофайлов (поддерживается маска).
    Если combine=True, все тексты объединяются через новую строку.
    """
    input_pattern = step["input"]
    output_file = step["output"]
    language = step.get("language", "ru")
    beam_size = step.get("beam_size", 5)
    combine = step.get("combine", True)

    # Ищем файлы сначала в output задачи, потом в source
    search_paths = [Path(OUTPUT_DIR) / task_name, Path(VIDEO_DIR)]
    input_files = []
    for base in search_paths:
        input_files.extend(base.glob(input_pattern))
    if not input_files:
        raise FileNotFoundError(f"Нет файлов по маске {input_pattern}")
    input_files = sorted(input_files)

    output_path = Path(OUTPUT_DIR) / task_name / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  Файл {output_path} уже существует, пропускаем transcribe", flush=True)
        return

    all_texts = []
    for fpath in input_files:
        print(f"  Транскрибируем {fpath.name}...", flush=True)
        text = transcribe_audio(fpath, language=language, beam_size=beam_size)
        # Сохраняем промежуточный результат (по желанию)
        seg_txt = fpath.with_suffix(".txt")
        with open(seg_txt, "w", encoding="utf-8") as f:
            f.write(text)
        all_texts.append(text)

    if combine:
        full_text = "\n".join(all_texts)
    else:
        full_text = all_texts[0] if all_texts else ""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"  Транскрибация сохранена в {output_path}", flush=True)

def step_llm(step, task_name, settings):
    """Отправляет текст в LM Studio и сохраняет результат."""
    input_file = step["input"]
    output_file = step["output"]
    prompt_template = step["prompt"]
    system = step.get("system", None)
    max_tokens = step.get("max_tokens", 8000)

    input_path = find_input_file(input_file, task_name)
    if not input_path:
        raise FileNotFoundError(f"Входной файл {input_file} не найден")

    output_path = Path(OUTPUT_DIR) / task_name / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"  Файл {output_path} уже существует, пропускаем llm", flush=True)
        return

    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read()

    prompt = prompt_template.replace("{input_text}", input_text)
    call_lm_studio(prompt, output_path, system=system, max_tokens=max_tokens, settings=settings)

# ------------------------------------------------------------
# 4. Основной цикл обработки JSON-задач
# ------------------------------------------------------------
def main():
    os.environ["HF_HOME"] = "/app/cache"
    os.makedirs("/app/cache", exist_ok=True)

    # Ищем все JSON-файлы в VIDEO_DIR
    json_files = list(Path(VIDEO_DIR).glob("*.json"))
    if not json_files:
        print("Нет JSON-файлов в /source. Завершаем.", flush=True)
        return

    for json_path in json_files:
        print(f"\n=== Обрабатываем задачу: {json_path.name} ===", flush=True)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                task = json.load(f)
        except Exception as e:
            print(f"Ошибка чтения {json_path.name}: {e}", flush=True)
            continue

        # Проверяем тип задачи
        if task.get("type") != "video_transcription":
            print(f"Пропускаем {json_path.name}: type != video_transcription", flush=True)
            continue

        task_name = task.get("name", json_path.stem)
        task_settings = task.get("settings", {})

        # Создаём выходную подпапку для задачи
        task_output_dir = Path(OUTPUT_DIR) / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        # Выполняем шаги последовательно
        steps = task.get("steps", [])
        for step_idx, step in enumerate(steps):
            action = step.get("action")
            out_desc = step.get("output") or step.get("output_prefix", "")
            print(f"  Шаг {step_idx+1}: {action} -> {out_desc}", flush=True)
            try:
                if action == "concat":
                    step_concat(step, task_name, task_settings)
                elif action == "extract_audio":
                    step_extract_audio(step, task_name, task_settings)
                elif action == "segment":
                    step_segment(step, task_name, task_settings)
                elif action == "transcribe":
                    step_transcribe(step, task_name, task_settings)
                elif action == "llm":
                    step_llm(step, task_name, task_settings)
                else:
                    print(f"    Неизвестное действие: {action}", flush=True)
            except Exception as e:
                print(f"    Ошибка в шаге {step_idx+1}: {e}", flush=True)
                # Прерываем обработку текущей задачи
                break
        else:
            print(f"Задача {task_name} выполнена успешно.", flush=True)
            # Можно переместить JSON в архив (опционально)
            # json_path.rename(json_path.with_suffix(".done.json"))

    print("\n=== Все задачи обработаны ===", flush=True)

if __name__ == "__main__":
    main()
