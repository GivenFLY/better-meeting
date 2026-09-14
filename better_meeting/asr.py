"""Етап 2: транскрипція з таймкодами.

Бекенди: mlx (mlx-whisper, macOS), faster (faster-whisper), parakeet
(parakeet-mlx, macOS; NVIDIA Parakeet TDT — один прогін на всі мови).

Whisper:
--lang uk        -> один прогін цією мовою.
--lang auto      -> повний прогін КОЖНОЮ з мов --langs (дефолт uk,ru,en),
                    потім злиття: таймлайн покривається сегментами тієї мови,
                    якій whisper дав кращу впевненість (avg_logprob) на цій
                    ділянці. Працює для будь-якого чергування мов у записі —
                    аж до перемикання мови між сусідніми репліками.
Кожен прогін кешується окремо (pass_<lang>.json у робочій теці) — падіння на
другій мові не змушує переганяти першу.

Parakeet:
один прогін незалежно від --lang/--langs (модель багатомовна, мову не приймає),
кеш pass_parakeet.json. Мова сегмента — з --lang, якщо закріплена, інакше
евристика по алфавіту (uk/ru/en). -O не підтримується.

Кожен сегмент має поле "lang".
"""

import hashlib
import json
import platform
import re
from pathlib import Path

from .utils import die, load, log, save

WHISPER_MODEL = "large-v3-turbo"                      # дефолт --asr-model
PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"  # дефолт для --asr-backend parakeet


def transcribe(wav: Path, lang: str, langs: list, model: str, backend: str,
               work: Path, opts: dict | None = None) -> list:
    """-> [{start, end, text, lang}], відсортовані по часу.
    opts — довільні whisper-параметри, летять у кожен прогін."""
    opts = opts or {}
    if backend == "auto":
        backend = "mlx" if platform.system() == "Darwin" else "faster"
    if backend == "parakeet":
        return [_public(s) for s in _transcribe_parakeet(wav, lang, langs, model, work, opts)]

    if lang != "auto":
        return [_public(s) for s in _run(wav, lang, model, backend, opts)]

    passes = []
    for l in langs:
        cache = _pass_path(work, backend, l)
        cached = load(cache) if cache.exists() else None
        if isinstance(cached, dict) and cached.get("model") == model \
                and cached.get("backend") == backend and cached.get("opts", {}) == opts:
            segs = cached["segments"]
            log(f"прогін [{l}]: з кешу, {len(segs)} сегментів")
        else:
            log(f"прогін [{l}]")
            segs = _run(wav, l, model, backend, opts)
            save(cache, {"model": model, "backend": backend, "opts": opts,
                         "segments": segs})
        passes.append(segs)

    merged = _merge(passes)
    shares = {}
    for s in merged:
        shares[s["lang"]] = shares.get(s["lang"], 0) + 1
    log("злито: " + ", ".join(f"{l}: {n}" for l, n in
                              sorted(shares.items(), key=lambda kv: -kv[1])))
    return [_public(s) for s in merged]


def _transcribe_parakeet(wav: Path, lang: str, langs: list, model: str,
                         work: Path, opts: dict) -> list:
    """Один прогін на весь запис. Кеш не залежить від --lang: закріплена мова
    лише підписує сегменти, auto — евристика по алфавіту."""
    _reject_whisper_opts(opts)
    if lang == "auto" and langs:
        log("parakeet: один прогін на всі мови, --langs не впливає")
    cache = _pass_path(work, "parakeet", None)
    cached = load(cache) if cache.exists() else None
    if isinstance(cached, dict) and cached.get("model") == model \
            and cached.get("backend") == "parakeet" and cached.get("opts", {}) == opts:
        segs = cached["segments"]
        log(f"прогін [parakeet]: з кешу, {len(segs)} сегментів")
    else:
        log("прогін [parakeet]")
        segs = _run_parakeet(wav, None, model, opts)
        save(cache, {"model": model, "backend": "parakeet", "opts": opts,
                     "segments": segs})
    if lang != "auto":
        segs = [{**s, "lang": lang} for s in segs]
    shares = {}
    for s in segs:
        shares[s["lang"]] = shares.get(s["lang"], 0) + 1
    log("мови: " + ", ".join(f"{l}: {n}" for l, n in
                              sorted(shares.items(), key=lambda kv: -kv[1])))
    return segs


def _pass_path(work: Path, backend: str, lang) -> Path:
    """Файл кешу повного прогону: pass_<lang>.json для whisper, один
    pass_parakeet.json для parakeet (мова на прогін не впливає)."""
    return work / ("pass_parakeet.json" if backend == "parakeet" else f"pass_{lang}.json")


def _public(s: dict) -> dict:
    return {"start": s["start"], "end": s["end"], "text": s["text"], "lang": s["lang"]}


def transcribe_range(video: Path, start: float, end, lang, model: str, backend: str,
                     work: Path, opts: dict, force: bool) -> list:
    """Точковий запит: транскрипт проміжку [start, end) з довільними whisper-опціями.

    Кешується за повним ключем параметрів (проміжок + мова + модель + бекенд +
    опції) у work/transcripts_at/ — той самий запит повертається з кешу миттєво,
    force перераховує."""
    if backend == "auto":
        backend = "mlx" if platform.system() == "Darwin" else "faster"
    if backend == "parakeet":
        _reject_whisper_opts(opts)     # до витягування аудіо, щоб не лишати кліп
    key = {"from": round(start, 3), "to": round(end, 3) if end is not None else None,
           "lang": lang, "model": model, "backend": backend, "opts": opts}
    digest = hashlib.sha1(
        json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
    cdir = work / "transcripts_at"
    cache = cdir / f"{int(start):05d}-{'end' if end is None else f'{int(end):05d}'}_{lang or 'auto'}_{digest}.json"

    if cache.exists() and not force:
        log(f"з кешу: {cache.name}")
        return load(cache)["segments"]

    if not force:
        reused = _slice_pipeline_cache(work, start, end, lang, model, backend, opts)
        if reused is not None:
            log("з кешу повного прогону (extract)")
            return reused

    from .audio import extract_audio
    cdir.mkdir(parents=True, exist_ok=True)
    clip = cdir / f"{digest}.wav"
    extract_audio(video, clip, start=start,
                  duration=None if end is None else end - start)
    segments = [_public(s) for s in _run(clip, lang, model, backend, opts)]
    clip.unlink()
    for s in segments:
        s["start"] += start
        s["end"] += start
    save(cache, {"params": key, "segments": segments})
    return segments


def _slice_pipeline_cache(work: Path, start: float, end, lang, model: str,
                          backend: str, opts: dict):
    """Якщо відео вже пройшло повний extract з тими самими моделлю/бекендом/опціями —
    діапазон нарізається з готових результатів, без нового прогону ASR.

    Конкретна мова -> зріз її pass_<lang>.json.
    Без мови       -> зріз злитого transcript.json (перевіривши по будь-якому
                      pass-файлу, що прогін був з тими самими параметрами).
    Повертає None, якщо придатного кешу немає."""
    def in_range(s):
        return s["end"] > start and (end is None or s["start"] < end)

    def match(data) -> bool:
        return isinstance(data, dict) and data.get("model") == model \
            and data.get("backend") == backend and data.get("opts", {}) == opts

    if lang is not None:
        f = _pass_path(work, backend, lang)
        if f.exists():
            data = load(f)
            if match(data):
                segs = [_public(s) for s in data["segments"] if in_range(s)]
                if backend == "parakeet":       # у кеші — евристичні мови, --lang їх перекриває
                    for s in segs:
                        s["lang"] = lang
                return segs
        return None

    f_transcript = work / "transcript.json"
    if f_transcript.exists():
        for f in work.glob("pass_*.json"):
            if match(load(f)):
                return [dict(s) for s in load(f_transcript) if in_range(s)]
    return None


def _merge(passes: list) -> list:
    """Жадібне покриття таймлайну: у кожній точці беремо сегмент з найкращим
    avg_logprob серед тих, що її накривають. Сегменти-галюцинації (стандартна
    whisper-евристика: високий no_speech і низький logprob) відкидаються."""
    segs = [s for p in passes for s in p
            if not (s["nospeech"] > 0.6 and s["score"] < -1.0)]
    if not segs:
        return []
    segs.sort(key=lambda s: s["start"])

    out = []
    cursor = segs[0]["start"]
    while True:
        cands = [s for s in segs
                 if s["end"] > cursor + 0.2 and s["start"] <= cursor + 2.0]
        if not cands:
            rest = [s["start"] for s in segs if s["start"] > cursor]
            if not rest:
                break
            cursor = min(rest)
            continue
        best = max(cands, key=lambda s: s["score"])
        out.append(best)
        cursor = best["end"]
    return out


def _run(wav: Path, lang, model: str, backend: str, opts: dict | None = None) -> list:
    if backend == "mlx":
        return _run_mlx(wav, lang, model, opts or {})
    if backend == "parakeet":
        return _run_parakeet(wav, lang, model, opts or {})
    return _run_faster(wav, lang, model, opts or {})


def _run_mlx(wav: Path, lang, model: str, opts: dict) -> list:
    try:
        import mlx_whisper  # type: ignore
    except ImportError:
        die("немає mlx-whisper. `pip install mlx-whisper` або --asr-backend faster")
    repo = model if "/" in model else f"mlx-community/whisper-{model}"
    kwargs = {"language": lang, "condition_on_previous_text": False, **opts}
    res = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, **kwargs)
    detected = lang or res.get("language")
    return [
        {"start": float(s["start"]), "end": float(s["end"]),
         "text": s["text"].strip(), "lang": detected,
         "score": float(s.get("avg_logprob", 0.0)),
         "nospeech": float(s.get("no_speech_prob", 0.0))}
        for s in res["segments"] if s["text"].strip()
    ]


_FASTER_CACHE = {}


def _run_faster(wav: Path, lang, model: str, opts: dict) -> list:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        die("немає faster-whisper. `pip install faster-whisper`")
    if model not in _FASTER_CACHE:
        _FASTER_CACHE[model] = WhisperModel(model, device="auto", compute_type="int8")
    kwargs = {"language": lang, "vad_filter": True,
              "condition_on_previous_text": False, **opts}
    segs, info = _FASTER_CACHE[model].transcribe(str(wav), **kwargs)
    detected = lang or info.language
    return [
        {"start": float(s.start), "end": float(s.end),
         "text": s.text.strip(), "lang": detected,
         "score": float(s.avg_logprob), "nospeech": float(s.no_speech_prob)}
        for s in segs if s.text.strip()
    ]


_PARAKEET_CACHE = {}


def _run_parakeet(wav: Path, lang, model: str, opts: dict) -> list:
    """Один прогін parakeet-mlx. lang закріплює мову сегментів; None — евристика.
    Довге аудіо ріжеться на 2-хвилинні шматки з перекриттям (як у CLI parakeet-mlx)."""
    _reject_whisper_opts(opts)
    if platform.system() != "Darwin":
        die("parakeet поки що лише на macOS (parakeet-mlx); тут — --asr-backend faster")
    try:
        from parakeet_mlx import from_pretrained  # type: ignore
    except ImportError:
        die("немає parakeet-mlx. `pip install \"better-meeting[parakeet]\"`")
    repo = _parakeet_repo(model)
    if repo not in _PARAKEET_CACHE:
        _PARAKEET_CACHE[repo] = from_pretrained(repo)
    res = _PARAKEET_CACHE[repo].transcribe(
        str(wav), chunk_duration=120.0, overlap_duration=15.0)
    segs = [
        {"start": float(s.start), "end": float(s.end), "text": s.text.strip(),
         "lang": lang, "score": float(s.confidence), "nospeech": 0.0}
        for s in res.sentences if s.text.strip()
    ]
    if lang is None:
        prev = None
        for s in segs:
            s["lang"] = prev = _guess_lang(s["text"], prev)
    return segs


def _reject_whisper_opts(opts: dict) -> None:
    if opts:
        die("-O стосується лише whisper; parakeet не має параметрів декодування")


def _parakeet_repo(model: str) -> str:
    """--asr-model під parakeet: HF repo id; дефолтна whisper-модель -> дефолт parakeet."""
    if "/" in model:
        return model
    if model == WHISPER_MODEL:
        return PARAKEET_MODEL
    die(f"--asr-model {model!r} не схоже на parakeet-модель; "
        f"вкажіть HF repo id, напр. {PARAKEET_MODEL}")


_UK_LETTERS = set("іїєґ")
_RU_LETTERS = set("ыэъё")
# службові слова, що існують лише в одній з мов (без спільних «не», «на», «так»)
_UK_WORDS = {"і", "й", "що", "це", "як", "був", "була", "було", "ще", "вже", "або",
             "коли", "де", "хто", "вони", "ми", "ви", "він", "вона", "воно", "з", "із",
             "зі", "від", "під", "але", "цей", "ця", "його", "її", "їх", "дуже", "зараз",
             "потім", "треба", "потрібно", "добре", "дякую", "ласка", "гаразд", "згоден",
             "питання", "вчора"}
_RU_WORDS = {"и", "что", "это", "как", "был", "была", "было", "есть", "нет", "ещё",
             "еще", "уже", "только", "если", "чтобы", "или", "когда", "где", "кто",
             "они", "мы", "вы", "он", "она", "оно", "из", "с", "к", "от", "под", "его",
             "её", "ее", "их", "очень", "сейчас", "потом", "надо", "нужно", "хорошо",
             "спасибо", "пожалуйста", "привет", "ладно", "согласен", "понятно", "вопрос",
             "сегодня", "вчера", "неделя", "встреча"}


def _guess_lang(text: str, prev):
    """Мова за письмом: латиниця -> en; кирилиця — за літерами-маркерами
    (і/ї/є/ґ проти ы/э/ъ/ё) і службовими словами однієї з мов; без маркерів
    (коротка репліка) -> мова попереднього сегмента, інакше uk."""
    low = text.lower()
    if not any("\u0400" <= c <= "\u04ff" for c in low):
        return "en" if any(c.isalpha() for c in low) else prev
    words = re.findall(r"[\u0400-\u04ff']+", low)
    uk = sum(c in _UK_LETTERS for c in low) + sum(w in _UK_WORDS for w in words)
    ru = sum(c in _RU_LETTERS for c in low) + sum(w in _RU_WORDS for w in words)
    if uk != ru:
        return "uk" if uk > ru else "ru"
    return prev if prev in ("uk", "ru") else "uk"
