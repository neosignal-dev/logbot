import asyncio, os, logging, json, html
from typing import Optional, List, Iterable, Dict
import httpx
from functools import lru_cache
from contextlib import suppress
from urllib.parse import urlparse, quote

from aiogram import Bot, Dispatcher, Router, types
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.exceptions import TelegramBadRequest

logging.basicConfig(level=logging.INFO)

# ---------------- small utils ----------------

def _normalize_ws(s: str) -> str:
    return " ".join((s or "").replace("\n", " ").replace("\t", " ").split())

def esc(s: str) -> str:
    return html.escape(str(s), quote=False)

# ---------------- env & globals ----------------

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    logging.error("TELEGRAM_BOT_TOKEN is not set"); raise SystemExit(1)

ALLOWED = {x.strip() for x in os.getenv("ALLOWED_TELEGRAM_IDS", "").split(",") if x.strip()}
BACKEND = os.getenv("BACKEND_URL", "http://backend:8000")
GRAFANA = os.getenv("GRAFANA_URL")

bot = Bot(TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp  = Dispatcher()
rt  = Router()
dp.include_router(rt)

# simple in-memory prefs
RECENT: Dict[int, dict] = {}
FAVS: Dict[int, dict] = {}

# ---------------- HTTP helper ----------------

async def fetch_json(url: str, params: Optional[dict] = None, method: str = "GET", json_body: Optional[dict] = None):
    timeout = 60 if "/api/ai/" in url else 20
    async with httpx.AsyncClient(timeout=timeout) as s:
        r = await (s.get(url, params=params) if method == "GET" else s.post(url, json=json_body))
    r.raise_for_status()
    return r.json()

# ---------------- auth ----------------

def ensure_allowed(m: types.Message | CallbackQuery) -> bool:
    uid = str(m.from_user.id)
    return (not ALLOWED) or (uid in ALLOWED)

async def cb_ack(q: CallbackQuery, text: Optional[str] = None):
    with suppress(TelegramBadRequest):
        await q.answer(text)

# ---------------- URL/Grafana ----------------

def _is_valid_http_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http","https") and bool(p.netloc)
    except Exception:
        return False

def grafana_explore_url(host: str, unit: str, window: str) -> Optional[str]:
    if not GRAFANA:
        return None
    base = GRAFANA.strip()
    if not base.startswith("http"):
        base = "https://" + base
    payload = {"datasource":"loki","queries":[{"query":"{" + f'"hostname":"{host}"' + (f',"unit":"{unit}"' if unit else '') + "}"}],"range":{"from":f"now-{window}","to":"now"}}
    try:
        left = quote(json.dumps(payload, separators=(",", ":")), safe="")
        url = f"{base}/explore?left={left}"
        if _is_valid_http_url(url) and len(url) <= 1000:
            return url
    except Exception:
        return None
    return None

# ---------------- formatting ----------------

def format_top(top: Iterable[dict], total: int, header: bool = True, limit: int = 3) -> str:
    top = list(top or [])[:limit]
    if not top and total == 0:
        return ""
    lines = []
    if header:
        lines.append(f"<b>Всего событий:</b> <b>{total}</b>")
    for item in top:
        msg = esc(item.get("msg", ""))[:180]
        cnt = item.get("count", 0)
        lines.append(f"• <code>{msg}</code> — <b>{cnt}×</b>")
    return "\n".join(lines)

async def fetch_levels(host: str, unit: str, window: str) -> dict:
    params = {"hostname": host, "unit": unit or "", "window": window}
    for ep in ("/api/loki/levels", "/api/loki/events_breakdown", "/api/loki/errors_levels"):
        try:
            js = await fetch_json(BACKEND + ep, params)
            if isinstance(js, dict):
                if "levels" in js and isinstance(js["levels"], dict):
                    return js["levels"]
                if "data" in js and isinstance(js["data"], dict):
                    return js["data"]
                return js
        except Exception:
            continue
    return {}

def format_levels(levels: dict) -> str:
    if not isinstance(levels, dict) or not levels:
        return ""
    order = ["error","warning","info","unknown"]
    icons = {"error":"🟥","warning":"🟧","info":"🟦","unknown":"⬛️"}
    parts = []
    for k in order:
        v = levels.get(k) or levels.get(k.upper()) or levels.get(k.capitalize())
        try:
            v = int(v)
        except Exception:
            continue
        if v>0:
            parts.append(f"{icons.get(k,'')} {k} <b>{v}</b>")
    return "📊 Уровни: " + " · ".join(parts) if parts else ""

# ---------------- keyboards ----------------

WINDOWS = [("15m","15m"),("1h","1h"),("6h","6h"),("24h","24h")]
LEVEL_FILTERS = [("Error","error"),("Warning","warning"),("Info","info")]
ANALYSES = [("🧠 Сводка","summary"),("🧩 Причины (RCA)","rca"),("🛠 Рекомендации","actions"),("📈 Аномалии","anomalies"),("❓ Q&A → /analyze","qa")]

# Canonicalization and titles for levels used by backend
LEVEL_CANON = {
    "error": "error", "err": "error", "fatal": "error", "critical": "error", "crit": "error",
    "warning": "warning", "warn": "warning",
    "info": "info"
}
LEVEL_TITLE = {"error": "Error", "warning": "Warning", "info": "Info"}


def kb_main() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="🖥 Хосты", callback_data="H")],
        [InlineKeyboardButton(text="⭐️ Избранное", callback_data="FAV")],
        [InlineKeyboardButton(text="❓ /analyze — свободный вопрос", callback_data="HELP_ANALYZE")],
    ])


def chunk_buttons(items: List[str], prefix: str, host: str, page: int, per_page: int = 8) -> InlineKeyboardMarkup:
    start = page * per_page
    slice_ = items[start:start+per_page]
    rows = [[InlineKeyboardButton(text=i, callback_data=f"{prefix}:{host}:{i}")] for i in slice_]
    nav = []
    if page>0: nav.append(InlineKeyboardButton(text="⬅️", callback_data=f"{prefix}_PG:{host}:{page-1}"))
    if start+per_page < len(items): nav.append(InlineKeyboardButton(text="➡️", callback_data=f"{prefix}_PG:{host}:{page+1}"))
    if nav: rows.append(nav)
    rows.append([InlineKeyboardButton(text="🏠 Меню", callback_data="M")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_windows(host: str, unit: str) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=f"🕒 {t}", callback_data=f"WIN:{host}:{unit or '-'}:{val}")] for t,val in WINDOWS]
    rows.append([InlineKeyboardButton(text="⬅️ Хосты", callback_data="H"), InlineKeyboardButton(text="🏠 Меню", callback_data="M")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def kb_analysis(host: str, unit: str, window: str) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(text=txt, callback_data=f"T:{task}:{host}:{unit or '-'}:{window}")] for (txt,task) in ANALYSES]
    # quick level filters
    rows.insert(0, [InlineKeyboardButton(text=txt, callback_data=f"LF:{lv}:{host}:{unit or '-'}:{window}") for (txt,lv) in LEVEL_FILTERS])
    url = grafana_explore_url(host, unit, window)
    if url and _is_valid_http_url(url):
        rows.insert(0, [InlineKeyboardButton(text="🔎 Открыть в Grafana", url=url)])
    rows.append([InlineKeyboardButton(text="⬅️ Окно", callback_data=f"W:{host}:{unit or '-'}"), InlineKeyboardButton(text="🏠 Меню", callback_data="M")])
    return InlineKeyboardMarkup(inline_keyboard=rows)

# ---------------- commands ----------------

@rt.message(Command("start","menu"))
async def cmd_start(m: types.Message):
    if not ensure_allowed(m):
        return await m.answer("⛔️ Доступ запрещён")
    await m.answer("Привет! Выбери хост для анализа логов с помощью OpenAI:", reply_markup=kb_main())

@rt.message(Command("analyze"))
async def cmd_analyze(m: types.Message, command: CommandObject):
    if not ensure_allowed(m):
        return await m.answer("⛔️")

    args = _normalize_ws(command.args or "")
    parts = args.split(" ", 2) if args else []
    if len(parts) < 3:
        return await m.answer("Формат: /analyze <host> <window> <вопрос>\nНапр.: /analyze lxc-caddy 6h почему таймауты?")

    host, window, question = parts[0], parts[1], parts[2].strip()
    import re
    if not re.fullmatch(r"\d+[smhd]", window):
        return await m.answer("Окно должно быть в формате 15m/1h/6h/24h и т.п.")

    body = {"hostname": host, "unit": "", "window": window, "limit": 300, "task": "qa", "question": question}
    try:
        js = await fetch_json(f"{BACKEND}/api/ai/analyze", method="POST", json_body=body)
        res = js.get("result", {})
    except Exception as e:
        return await m.answer(f"Ошибка AI: {e}")

    title = res.get("title") or "Q&A"
    summary = res.get("summary") or "Нет ответа"

    top_block = ""
    try:
        errs = await fetch_json(f"{BACKEND}/api/loki/errors", {"hostname": host, "unit": "", "window": window, "limit": 200})
        top_block = format_top(errs.get("top", []), errs.get("total", 0), header=True, limit=3)
    except Exception:
        pass

    parts_out = [f"🕒 <b>{esc(window)}</b> | <b>{esc(host)}</b>", f"🔖 <b>{esc(title)}</b>"]
    try:
        lv = await fetch_levels(host, "", window)
        lv_line = format_levels(lv)
        if lv_line:
            parts_out += [lv_line]
    except Exception:
        pass
    if top_block:
        parts_out += ["", top_block]
    if summary:
        short = summary[:300]
        rest = summary[300:900]
        if rest:
            parts_out += ["", esc(short) + f"<span class=\"tg-spoiler\">{esc(rest)}</span>"]
        else:
            parts_out += ["", esc(summary)]

    await m.answer("\n".join(parts_out))

# ---------------- callbacks ----------------

@rt.callback_query()
async def all_callbacks(q: CallbackQuery):
    if not ensure_allowed(q):
        return await q.answer("⛔️", show_alert=True)
    data = q.data or ""
    logging.info("CB DATA: %s", data)

    # Главное меню
    if data == "M":
        await cb_ack(q)
        await q.message.edit_text("Главное меню", reply_markup=kb_main())
        return

    if data == "HELP_ANALYZE":
        await q.answer("/analyze <host> <window> <вопрос>", show_alert=True)
        return

    # Список хостов
    if data == "H":
        await cb_ack(q)
        try:
            js = await fetch_json(f"{BACKEND}/api/loki/label/hostname/values")
            hosts = sorted(js.get("data", []))
        except Exception as e:
            return await q.answer(f"Ошибка: {e}", show_alert=True)
        kb = chunk_buttons(hosts, prefix="HOST", host="_", page=0)
        await q.message.edit_text("Выбери хост:", reply_markup=kb)
        return

    if data.startswith("HOST_PG:_:"):
        await cb_ack(q)
        _, _, _, page = data.split(":", 3)
        try:
            js = await fetch_json(f"{BACKEND}/api/loki/label/hostname/values")
            hosts = sorted(js.get("data", []))
        except Exception as e:
            return await q.answer(f"Ошибка: {e}", show_alert=True)
        kb = chunk_buttons(hosts, prefix="HOST", host="_", page=int(page))
        await q.message.edit_text("Выбери хост:", reply_markup=kb)
        return

    # Выбор хоста → юниты
    if data.startswith("HOST:_:"):
        await cb_ack(q)
        _, _, host = data.split(":", 2)
        try:
            js = await fetch_json(f"{BACKEND}/api/loki/units", {"hostname": host})
            units = sorted(set(js.get("data", []) or []))
        except Exception as e:
            return await q.answer(f"Ошибка: {e}", show_alert=True)
        RECENT[q.from_user.id] = {"host": host}
        units_display = ["(все юниты)"] + units
        kb = chunk_buttons(units_display, prefix="USEL", host=host, page=0)
        await q.message.edit_text(f"Хост {host}. Выберите unit:", reply_markup=kb)
        return

    if data.startswith("USEL_PG:"):
        await cb_ack(q)
        _, host, page = data.split(":", 2)
        try:
            js = await fetch_json(f"{BACKEND}/api/loki/units", {"hostname": host})
            units = sorted(set(js.get("data", []) or []))
        except Exception as e:
            return await q.answer(f"Ошибка: {e}", show_alert=True)
        units_display = ["(все юниты)"] + units
        kb = chunk_buttons(units_display, prefix="USEL", host=host, page=int(page))
        await q.message.edit_text(f"Хост {host}. Выберите unit:", reply_markup=kb)
        return

    # Выбор unit → окно
    if data.startswith("USEL:"):
        await cb_ack(q)
        _, host, unit = data.split(":", 2)
        unit = "" if unit == "(все юниты)" else unit
        RECENT[q.from_user.id] = {"host": host, "unit": unit}
        await q.message.edit_text(
            f"Хост {host}" + (f", unit {unit}" if unit else "") + ". Выберите окно:",
            reply_markup=kb_windows(host, unit)
        )
        return

    if data.startswith("W:"):
        await cb_ack(q)
        _, host, unit = data.split(":", 2)
        unit = "" if unit == "-" else unit
        await q.message.edit_text(
            f"Хост {host}" + (f", unit {unit}" if unit else "") + ". Выберите окно:",
            reply_markup=kb_windows(host, unit)
        )
        return

    # Выбор окна → тип анализа
    if data.startswith("WIN:"):
        await cb_ack(q)
        _, host, unit, window = data.split(":", 3)
        unit = "" if unit == "-" else unit
        RECENT[q.from_user.id] = {"host": host, "unit": unit, "window": window}
        preview = ""
        try:
            errs = await fetch_json(f"{BACKEND}/api/loki/errors", {"hostname": host, "unit": unit, "window": window, "limit": 200})
            preview = format_top(errs.get("top", []), errs.get("total", 0), header=True, limit=2)
        except Exception:
            preview = ""
        levels_line = ""
        try:
            lv = await fetch_levels(host, unit, window)
            levels_line = format_levels(lv)
        except Exception:
            pass
        text = f"🕒 <b>{esc(window)}</b> | <b>{esc(host)}</b>" + (f" (<i>{esc(unit)}</i>)" if unit else "")
        if levels_line: text += "\n" + levels_line
        if preview: text += "\n" + preview
        await q.message.edit_text(text, reply_markup=kb_analysis(host, unit, window))
        return

    # Быстрый просмотр примеров по уровню
    if data.startswith("LF:"):
        await cb_ack(q)
        _, level, host, unit, window = data.split(":", 4)
        unit = "" if unit == "-" else unit

        raw_level = level
        level = LEVEL_CANON.get(level.lower(), level.lower())
        title = LEVEL_TITLE.get(level, (raw_level or "").capitalize())

        # забираем примеры с бэка и терпимо парсим разные форматы
        try:
            js = await fetch_json(
                f"{BACKEND}/api/loki/samples",
                {"hostname": host, "unit": unit, "window": window, "level": level, "limit": 10},
            )
        except Exception as e:
            return await q.answer(f"Ошибка: {e}", show_alert=True)

        # извлечение списка примеров из разных возможных структур
        samples = []
        if isinstance(js, dict):
            if isinstance(js.get("samples"), list):
                samples = js["samples"]
            elif isinstance(js.get("data"), dict) and isinstance(js["data"].get("samples"), list):
                samples = js["data"]["samples"]
            elif isinstance(js.get("data"), list):
                samples = js["data"]
            elif isinstance(js.get("result"), list):
                # формат ответа из Loki: result -> streams -> values [[ts, line], ...]
                tmp = []
                for stream in js["result"]:
                    for item in stream.get("values", [])[:10]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            tmp.append({"line": item[1]})
                samples = tmp
            elif isinstance(js.get("data"), dict) and isinstance(js["data"].get("result"), list):
                tmp = []
                for stream in js["data"]["result"]:
                    for item in stream.get("values", [])[:10]:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            tmp.append({"line": item[1]})
                samples = tmp

        # формирование текста
        count = len(samples)
        lines = [f"🔎 Примеры ({title}) — {count} шт:"]
        for it in samples[:10]:
            line = (it.get("line") if isinstance(it, dict) else str(it)) or ""
            lines.append(f"• <code>{esc(line)[:180]}</code>")

        text = "\n".join(lines) if count else f"🔎 Примеры ({title}) — 0 шт.\nНет примеров за выбранное окно."

        kb = kb_analysis(host, unit, window)
        try:
            await q.message.edit_text(text, reply_markup=kb)
        except TelegramBadRequest as e:
            es = str(e).lower()
            # если текст не изменился — просто обновим клавиатуру
            if "message is not modified" in es:
                with suppress(TelegramBadRequest):
                    await q.message.edit_reply_markup(reply_markup=kb)
            else:
                # как fallback отправим новое сообщение
                await q.message.answer(text, reply_markup=kb)
        return

    # Тип анализа → AI
    if data.startswith("T:"):
        _, task, host, unit, window = data.split(":", 4)
        unit = "" if unit == "-" else unit
        if task == "qa":
            return await q.answer("Используй: /analyze <host> <window> <вопрос>", show_alert=True)
        await q.answer("Запускаю анализ…")
        body = {"hostname": host, "unit": unit, "window": window, "limit": 300, "task": task}
        try:
            js = await fetch_json(f"{BACKEND}/api/ai/analyze", method="POST", json_body=body)
            res = js.get("result", {})
        except Exception as e:
            return await q.message.edit_text(f"Ошибка AI: {e}")
        top_block = ""
        try:
            errs = await fetch_json(f"{BACKEND}/api/loki/errors", {"hostname": host, "unit": unit, "window": window, "limit": 200})
            top_block = format_top(errs.get("top", []), errs.get("total", 0), header=True, limit=3)
        except Exception:
            pass
        title = res.get("title") or f"{task} {host}"
        summary = res.get("summary") or ""
        sev = res.get("severity") or "unknown"
        causes = "\n".join(f"• {esc(c)}" for c in (res.get("probable_causes") or [])[:5])
        recs   = "\n".join(f"• {esc(r)}" for r in (res.get("recommendations") or [])[:5])
        lines = [f"🕒 <b>{esc(window)}</b> | <b>{esc(host)}</b> " + (f"(<i>{esc(unit)}</i>)" if unit else ""), f"🔖 <b>{esc(title)}</b>  ·  🔥 <b>{esc(sev)}</b>"]
        try:
            lv = await fetch_levels(host, unit, window)
            lv_line = format_levels(lv)
            if lv_line: lines += [lv_line]
        except Exception:
            pass
        if top_block: lines += ["", top_block]
        if summary:
            short = summary[:300]; rest = summary[300:900]
            if rest:
                lines += ["", esc(short) + f"<span class=\"tg-spoiler\">{esc(rest)}</span>"]
            else:
                lines += ["", esc(summary)]
        if causes: lines += ["", "Причины:", causes]
        if recs:   lines += ["", "Рекомендации:", recs]
        url = grafana_explore_url(host, unit, window)
        kb = InlineKeyboardMarkup(inline_keyboard=[
            *([[InlineKeyboardButton(text="🔎 Открыть в Grafana", url=url)]] if url and _is_valid_http_url(url) else []),
            [InlineKeyboardButton(text="⬅️ Тип анализа", callback_data=f"WIN:{host}:{unit or '-'}:{window}")],
            [InlineKeyboardButton(text="🏠 Меню", callback_data="M")],
        ])
        try:
            await q.message.edit_text("\n".join(lines), reply_markup=kb)
        except TelegramBadRequest:
            await q.message.answer("\n".join(lines), reply_markup=kb)
        return

    if data == "NOP":
        await cb_ack(q)
        return

@rt.message()
async def fallback(m: types.Message):
    if not ensure_allowed(m): return
    await m.answer("Команда не распознана. Попробуй /menu или /analyze.")

async def main():
    logging.info("Starting aiogram polling…")
    await dp.start_polling(bot, allowed_updates=["message","callback_query"])

if __name__ == "__main__":
    asyncio.run(main())
