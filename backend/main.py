from fastapi import FastAPI, HTTPException, Query
import os, httpx, datetime as dt, re, json
from typing import Optional, List
from collections import Counter
from openai import OpenAI

app = FastAPI()

# --- конфиг ---
LOKI = os.getenv("LOKI_URL")
PROM = os.getenv("PROM_URL")
VERIFY = False if os.getenv("ALLOW_INSECURE_TLS") == "1" else True
MAX_WINDOW_HOURS = int(os.getenv("MAX_WINDOW_HOURS", "168"))  # 7 days

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# OpenAI клиент инициализируем внутри обработчика, чтобы не падать при импорте без ключа

# --- утилиты времени ---
def parse_window(w: str) -> dt.timedelta:
    # нормализация: пробелы, регистр
    w = (w or "").strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d+)([smhd])", w)
    if not m:
        raise ValueError("bad window")
    n, u = int(m.group(1)), m.group(2)
    unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[u]
    td = dt.timedelta(**{unit: n})
    # ограничение максимального окна
    if td > dt.timedelta(hours=MAX_WINDOW_HOURS):
        raise ValueError("window too large")
    if td <= dt.timedelta(0):
        raise ValueError("bad window")
    return td

def ts_range(window: str):
    now = dt.datetime.utcnow()
    try:
        td = parse_window(window)
    except ValueError as e:
        raise HTTPException(400, str(e))
    start = (now - td).isoformat(timespec="seconds") + "Z"
    end = now.isoformat(timespec="seconds") + "Z"
    return start, end

# безопасная сборка селектора Loki
_DEF_LABEL_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Сопоставление "severity" -> фильтр по уровню (регекс для LogQL |~)
_VALID_SEVERITIES = {"errors", "warnings", "err+warn", "all"}

def severity_regex(severity: str | None) -> Optional[str]:
    s = (severity or "errors").lower()
    if s not in _VALID_SEVERITIES:
        raise HTTPException(400, f"bad severity: {severity}")
    if s == "errors":
        return "(?i)error|failed|exception|timeout|panic"
    if s == "warnings":
        return "(?i)warn|warning"
    if s == "err+warn":
        return "(?i)error|failed|exception|timeout|panic|warn|warning"
    # all
    return None

def build_loki_selector(labels: dict[str, str]) -> str:
    parts = []
    for k, v in labels.items():
        if v is None or v == "":
            continue
        if not _DEF_LABEL_RE.match(k):
            raise HTTPException(400, f"bad label name: {k}")
        # экранируем \ и " в значении
        sv = str(v).replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{k}="{sv}"')
    return "{" + ",".join(parts) + "}"

# --- helpers for level normalization ---
def _normalize_level_name(raw: str | None) -> str:
    """Map various level label spellings to canonical buckets.
    Returns one of: error, warning, info, unknown, debug (debug is later folded to unknown).
    """
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    if s in {"error", "err", "fatal", "crit", "critical"}:
        return "error"
    if s in {"warn", "warning"}:
        return "warning"
    if s in {"info", "notice"}:
        return "info"
    if s in {"debug", "trace"}:
        return "debug"
    return "unknown"

@app.get("/healthz")
def healthz(): return {"ok": True}

# ---------------- Loki/Prom (как было) ----------------

# internal helper for query_range
async def _loki_query_range(query: str, start: str, end: str, limit: int = 200, direction: str = "BACKWARD"):
    async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
        r = await s.get(f"{LOKI}/loki/api/v1/query_range",
                        params={"query": query, "start": start, "end": end, "limit": limit, "direction": direction})
    r.raise_for_status()
    return r.json()
@app.get("/api/loki/levels")
async def loki_levels(hostname: str, unit: str = "", window: str = "1h", job: str = "systemd-journal"):
    if not LOKI:
        raise HTTPException(500, "LOKI_URL is not set")

    selector = build_loki_selector({"job": job, "hostname": hostname, "unit": unit or None})
    _, end = ts_range(window)

    # 1) Основной путь — по ярлыку detected_level
    q = f"sum by (detected_level) (count_over_time({selector}[{window}]))"
    async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
        r = await s.get(f"{LOKI}/loki/api/v1/query", params={"query": q, "time": end})
        r.raise_for_status()
        data = r.json().get("data", {}).get("result", [])

    levels = {"error": 0, "warning": 0, "info": 0, "unknown": 0}
    detected_total = 0
    for row in data:
        raw_lvl = (row.get("metric", {}) or {}).get("detected_level") or "unknown"
        lvl = _normalize_level_name(raw_lvl)
        try:
            val = int(float(row.get("value", [0, "0"])[1]))
        except Exception:
            val = 0
        # fold anything outside our trio into unknown
        if lvl not in levels:
            lvl = "unknown"
        levels[lvl] += val
        detected_total += val

    # 2) Если ничего не получили по detected_level, используем эвристику по regex
    if detected_total == 0:
        rx_err = "(?i)error|failed|exception|timeout|panic"
        rx_warn = "(?i)warn|warning"
        rx_info = "(?i)info|started|listening|ready|connected|up|running"
        async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
            err_r = await s.get(f"{LOKI}/loki/api/v1/query", params={"query": f'sum(count_over_time(({selector} |~ "{rx_err}")[{window}]))', "time": end})
            warn_r = await s.get(f"{LOKI}/loki/api/v1/query", params={"query": f'sum(count_over_time(({selector} |~ "{rx_warn}")[{window}]))', "time": end})
            info_r = await s.get(f"{LOKI}/loki/api/v1/query", params={"query": f'sum(count_over_time(({selector} |~ "{rx_info}")[{window}]))', "time": end})
            for name, resp in (("error", err_r), ("warning", warn_r), ("info", info_r)):
                resp.raise_for_status()
                items = resp.json().get("data", {}).get("result", [])
                val = 0
                if items:
                    try:
                        val = int(float(items[0]["value"][1]))
                    except Exception:
                        val = 0
                levels[name] = val
        detected_total = levels["error"] + levels["warning"] + levels["info"]
        levels["unknown"] = 0

        # 2b) Если по regex всё ещё пусто — пробуем приорити из journald
        if detected_total == 0:
            pr_map = {
                "error": "0|1|2|3",
                "warning": "4",
                "info": "5|6",
            }
            levels = {"error": 0, "warning": 0, "info": 0, "unknown": 0}
            for name, patt in pr_map.items():
                # добавляем к селектору regex по метке priority
                sel_pr = selector[:-1] + f',priority=~"{patt}"' + "}"
                q_pr = f"sum(count_over_time({sel_pr}[{window}]))"
                async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
                    resp = await s.get(f"{LOKI}/loki/api/v1/query", params={"query": q_pr, "time": end})
                resp.raise_for_status()
                items = resp.json().get("data", {}).get("result", [])
                val = 0
                if items:
                    try:
                        val = int(float(items[0]["value"][1]))
                    except Exception:
                        val = 0
                levels[name] = val
            detected_total = levels["error"] + levels["warning"] + levels["info"]

    # 3) Отдельно считаем общий объём записей и корректно наполняем unknown
    total_q = f"sum(count_over_time({selector}[{window}]))"
    async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
        total_r = await s.get(f"{LOKI}/loki/api/v1/query", params={"query": total_q, "time": end})
        total_r.raise_for_status()
        total_items = total_r.json().get("data", {}).get("result", [])
    total = 0
    if total_items:
        try:
            total = int(float(total_items[0]["value"][1]))
        except Exception:
            total = 0

    # unknown = total - (error+warning+info)
    known = levels.get("error", 0) + levels.get("warning", 0) + levels.get("info", 0)
    unk = max(0, total - known)
    levels["unknown"] = unk

    return {"status": "success", "data": levels, "total": total}


@app.get("/api/loki/samples")
async def loki_samples(hostname: str, unit: str = "", window: str = "1h",
                       severity: str = "errors", level: Optional[str] = None, limit: int = 10, job: str = "systemd-journal"):
    # alias: если передан level, он имеет приоритет над severity
    lines = await loki_pull_lines(hostname=hostname, unit=unit, window=window,
                                  limit=limit, severity=severity, level=level)
    return {"status": "success", "count": len(lines), "items": lines}

@app.get("/api/loki")
async def loki(q: str = Query('{job=~".+"}'), window: str = "15m", limit: int = 200, severity: str = "all"):
    if not LOKI: raise HTTPException(500, "LOKI_URL is not set")
    start, end = ts_range(window)
    # если пришёл известный селектор, просто дополним фильтром по уровню
    lvl = severity_regex(severity)
    if lvl:
        q = f"{q} |~ \"{lvl}\""
    try:
        async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
            r = await s.get(f"{LOKI}/loki/api/v1/query_range",
                            params={"query": q, "start": start, "end": end,
                                    "limit": limit, "direction": "BACKWARD"})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Loki error: {e}")

@app.get("/api/prom")
async def prom(query: str = "up", window: str = "15m", step: str = "30s"):
    if not PROM: raise HTTPException(500, "PROM_URL is not set")
    start, end = ts_range(window)
    try:
        async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
            r = await s.get(f"{PROM}/api/v1/query_range",
                            params={"query": query, "start": start, "end": end, "step": step})
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Prometheus error: {e}")

@app.get("/api/loki/labels")
async def loki_labels(window: str = "24h"):
    start, end = ts_range(window)
    async with httpx.AsyncClient(timeout=20, verify=VERIFY) as s:
        r = await s.get(f"{LOKI}/loki/api/v1/labels", params={"start": start, "end": end})
    r.raise_for_status(); return r.json()

@app.get("/api/loki/label/{name}/values")
async def loki_label_values(name: str, window: str = "24h", hostname: Optional[str] = None):
    start, end = ts_range(window)
    params = [("start", start), ("end", end)]
    if hostname:
        sel = build_loki_selector({"job": "systemd-journal", "hostname": hostname})
        params = [("match[]", sel), ("start", start), ("end", end)]
    async with httpx.AsyncClient(timeout=20, verify=VERIFY) as s:
        r = await s.get(f"{LOKI}/loki/api/v1/label/{name}/values", params=params)
    r.raise_for_status(); return r.json()

@app.get("/api/loki/units")
async def loki_units(hostname: str, window: str = "24h", job: str = "systemd-journal"):
    """Вернёт *только те* unit, у которых реально были логи за окно.
    Используем агрегирующий запрос вместо /series, чтобы избежать усечения по лимитам
    и показывать ровно то, что видит Grafana.
    """
    # Проверим окно
    _, end = ts_range(window)
    # Собираем селектор и запрос: суммируем количество записей по unit за окно
    selector = build_loki_selector({"job": job, "hostname": hostname})
    query = f'sum by (unit) (count_over_time({selector}[{window}]))'

    async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
        # instant query на текущий момент времени (end)
        r = await s.get(f"{LOKI}/loki/api/v1/query", params={"query": query, "time": end})
    r.raise_for_status()

    data = r.json().get("data", {}).get("result", []) or []
    # Извлекаем уникальные unit-метки, игнорируя пустые
    units = sorted({item.get("metric", {}).get("unit") for item in data if item.get("metric", {}).get("unit")})

    return {"status": "success", "data": units}

@app.get("/api/loki/errors")
async def loki_errors(hostname: str, unit: str = "", window: str = "1h", limit: int = 200, severity: str = "errors"):
    if not LOKI: raise HTTPException(500, "LOKI_URL is not set")
    selector = build_loki_selector({
        "job": "systemd-journal",
        "hostname": hostname,
        "unit": unit or None,
    })
    q = selector
    lvl = severity_regex(severity)
    if lvl:
        q += f' |~ "{lvl}"'
    start, end = ts_range(window)
    async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
        r = await s.get(f"{LOKI}/loki/api/v1/query_range",
                        params={"query": q, "start": start, "end": end,
                                "limit": limit, "direction":"BACKWARD"})
    r.raise_for_status()
    data = r.json().get("data", {}).get("result", [])
    c = Counter()
    for stream in data:
        for _, line in stream.get("values", []):
            try:
                obj = json.loads(line)
                # пробуем вытащить уровень из json, чтобы сгруппировать точнее
                lvl_field = (obj.get("level") or obj.get("lvl") or obj.get("severity") or "").lower()
                keybase = obj.get("msg") or obj.get("message") or str(obj)
                key = f"[{lvl_field or 'n/a'}] {keybase}"[:120]
            except Exception:
                key = line[:120]
            c[key] += 1
    top = [{"msg": k, "count": v} for k, v in c.most_common(10)]
    return {"window": window, "hostname": hostname, "unit": unit or None,
            "severity": severity, "total": sum(c.values()), "top": top}

# ---------------- вспомогательные для LLM ----------------

async def loki_pull_lines(hostname: str, unit: str, window: str,
                          limit: int = 400, severity: str = "errors", level: Optional[str] = None) -> List[str]:
    if not LOKI:
        raise HTTPException(500, "LOKI_URL is not set")

    # Селектор по базовым меткам
    extra = {"job": "systemd-journal", "hostname": hostname, "unit": unit or None}
    lvl_label = (level or "").strip().lower()
    # Map user-visible names to actual label values present in Loki
    lvl_to_detected = {
        "error": "error",
        "err": "error",
        "fatal": "error",
        "crit": "error",
        "critical": "error",
        "warning": "warn",   # important: Loki often uses 'warn'
        "warn": "warn",
        "info": "info",
        "notice": "info",
        # do not filter by detected_level for 'unknown' — let regex/priority fallbacks work
    }
    if lvl_label in lvl_to_detected:
        extra["detected_level"] = lvl_to_detected[lvl_label]
    selector = build_loki_selector(extra)

    q = selector
    # Если level не задан, применим regex по severity
    if "detected_level" not in extra:
        rx = severity_regex(severity)
        if rx:
            q += f' |~ "{rx}"'

    start, end = ts_range(window)
    async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
        r = await s.get(
            f"{LOKI}/loki/api/v1/query_range",
            params={"query": q, "start": start, "end": end, "limit": limit, "direction": "BACKWARD"},
        )
    r.raise_for_status()

    def _collect(resp_json):
        lines_out = []
        for stream in resp_json.get("data", {}).get("result", []):
            for ts, line in stream.get("values", []):
                try:
                    t = dt.datetime.utcfromtimestamp(int(ts)//1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    t = str(ts)
                try:
                    obj = json.loads(line)
                    msg = obj.get("msg") or obj.get("message") or line
                except Exception:
                    msg = line
                lines_out.append(f"{t} {msg}")
                if len(lines_out) >= limit:
                    break
            if len(lines_out) >= limit:
                break
        return lines_out

    lines = _collect(r.json())

    # Fallback: если фильтрация по detected_level дала 0 строк, пробуем regex-уровень
    if not lines and "detected_level" in extra:
        # повторим запрос без ярлыка detected_level
        extra2 = {k: v for k, v in extra.items() if k != "detected_level"}
        selector2 = build_loki_selector(extra2)
        rx_map = {
            "error": "(?i)error|failed|exception|timeout|panic",
            "warning": "(?i)warn|warning",
            "info": r"(?i)\binfo\b|\bstarted\b|\blistening\b|\bready\b",
        }
        q2 = selector2
        if lvl_label in rx_map:
            q2 += f' |~ "{rx_map[lvl_label]}"'
        async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
            r2 = await s.get(
                f"{LOKI}/loki/api/v1/query_range",
                params={"query": q2, "start": start, "end": end, "limit": limit, "direction": "BACKWARD"},
            )
        r2.raise_for_status()
        lines = _collect(r2.json())

    # Второй fallback: используем priority из journald, если он присутствует
    if not lines and "detected_level" in extra:
        pr_map = {
            "error": "0|1|2|3",
            "warning": "4",
            "info": "5|6",
        }
        if lvl_label in pr_map:
            # строим селектор без detected_level и добавляем regex по priority
            extra3 = {k: v for k, v in extra.items() if k != "detected_level"}
            selector3 = build_loki_selector(extra3)
            sel_pr = selector3[:-1] + f',priority=~"{pr_map[lvl_label]}"' + "}"
            q3 = sel_pr
            async with httpx.AsyncClient(timeout=30, verify=VERIFY) as s:
                r3 = await s.get(
                    f"{LOKI}/loki/api/v1/query_range",
                    params={"query": q3, "start": start, "end": end, "limit": limit, "direction": "BACKWARD"},
                )
            r3.raise_for_status()
            lines = _collect(r3.json())

    return lines[:limit]

def scrub(lines: List[str]) -> str:
    out = []
    for x in lines:
        x = re.sub(r"\b[A-Fa-f0-9]{32,}\b", "[hex]", x)
        x = re.sub(r"\b(\d{1,3}\.){3}\d{1,3}\b", "[ip]", x)
        x = re.sub(r"(?i)(api|token|secret|key)=\S+", r"\1=[redacted]", x)
        out.append(x)
    text = "\n".join(out)
    # ограничим объём для токенов
    max_chars = int(os.getenv("AI_MAX_CHARS", "40000"))
    return text[:max_chars]

# --- простой raw для бота (по желанию) ---
@app.get("/api/loki/raw")
async def loki_raw(hostname: str, unit: str = "", window: str = "1h",
                   limit: int = 50, severity: str = "errors"):
    lines = await loki_pull_lines(hostname, unit, window, limit=limit, severity=severity)
    return {"count": len(lines), "items": lines}

# ---------------- LLM анализ ----------------

@app.post("/api/ai/analyze")
async def ai_analyze(payload: dict):
    """
    body:
      { "hostname": "...", "unit": "", "window": "1h", "limit": 300,
        "task": "summary|rca|actions|anomalies|qa", "question": "...", "severity": "errors" }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(500, "OPENAI_API_KEY not configured")

    host   = payload.get("hostname")
    if not host:
        raise HTTPException(400, "hostname is required")
    unit   = payload.get("unit", "")
    window = payload.get("window", "1h")
    limit  = int(payload.get("limit", 300))
    task   = (payload.get("task") or "summary").lower()
    question = (payload.get("question") or "").strip()
    severity = (payload.get("severity") or "errors").lower()

    lines = await loki_pull_lines(host, unit, window, limit=limit, severity=severity)
    text  = scrub(lines)

    system = (
        "Ты опытный SRE/DevOps. Твоя задача: {task}. Анализируй логи systemd journal Linux. "
        "Отвечай КРАТКО и по делу, только на русском. НЕ переводить и не изменять текст самих логов. "
        "Если доказательств недостаточно — явно укажи это. "
        "Верни JSON со схемой: title, summary, probable_causes[], evidence[], recommendations[], severity (low|medium|high|critical)."
    ).format(task=task)

    user = {
        "context": {"hostname": host, "unit": unit or None, "window": window, "lines": len(lines), "severity": severity},
        "task": task,
        "question": question or None,
        "logs_sample": text
    }

    client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
        )
        out = resp.choices[0].message.content
        data = json.loads(out)
        return {"ok": True, "model": OPENAI_MODEL, "result": data}
    except Exception as e:
        raise HTTPException(502, f"OpenAI error: {e}")
