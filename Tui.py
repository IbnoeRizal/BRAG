"""
tui.py - tampilan curses untuk prototype pencarian dokumen (ChromaDB).

Cara pakai:
    1. Taruh file ini di folder yang sama dengan main.py
    2. Windows: pip install windows-curses
    3. python tui.py

Semua pemanggilan ke database memakai fungsi yang sudah ada di main.py
(querytodb, add_to_vecdb, db), jadi main.py tidak perlu diubah.
"""
from __future__ import annotations

import inspect
import locale
import os
import sys
import textwrap
import threading
import traceback
from dataclasses import dataclass

try:
    import curses
except ImportError:
    sys.exit("Modul curses tidak ada. Di Windows jalankan: pip install windows-curses")

SPINNER = "|/-\\"
KEY_F2 = curses.KEY_F0 + 2
ENTER_KEYS = ("\n", "\r")
FETCH_N = 10  # berapa kandidat diminta ke database (kalau main.py mendukung)
MAX_SHOW = 5   # maksimal hasil unik yang ditampilkan

# atribut tampilan, diisi oleh init_colors()
A_HEAD = A_SEL = A_DIM = A_TITLE = A_OK = A_ERR = A_BUSY = 0


# --------------------------------------------------------------------------
# Helper tampilan
# --------------------------------------------------------------------------
def init_colors() -> None:
    global A_HEAD, A_SEL, A_DIM, A_TITLE, A_OK, A_ERR, A_BUSY
    bold = curses.A_BOLD
    A_SEL = curses.A_REVERSE
    A_DIM = curses.A_DIM
    A_TITLE = bold
    A_HEAD = curses.A_REVERSE | bold
    A_OK = A_ERR = A_BUSY = bold
    if curses.has_colors():
        curses.start_color()
        try:
            curses.use_default_colors()
            bg = -1
        except curses.error:
            bg = curses.COLOR_BLACK
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(2, curses.COLOR_GREEN, bg)
        curses.init_pair(3, curses.COLOR_RED, bg)
        curses.init_pair(4, curses.COLOR_YELLOW, bg)
        A_HEAD = curses.color_pair(1) | bold
        A_OK = curses.color_pair(2) | bold
        A_ERR = curses.color_pair(3) | bold
        A_BUSY = curses.color_pair(4) | bold


def put(win, y: int, x: int, text: str, width: int | None = None, attr: int = 0) -> None:
    """Tulis teks dengan aman: dipotong sesuai lebar, tidak error di tepi layar."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    room = (w - x) if width is None else min(width, w - x)
    if room <= 0:
        return
    try:
        win.addnstr(y, x, text, room, attr)
    except curses.error:
        pass


def fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..." if width > 3 else text[:width]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class Hit:
    id: str
    text: str
    meta: dict

    def doc(self) -> str:
        return str(self.meta.get("doc_name", "?"))

    def pages(self) -> str:
        a = self.meta.get("doc_metadata.page_start")
        b = self.meta.get("doc_metadata.page_end")
        if a is None:
            return "hal ?"
        return f"hal {a}" if b is None or a == b else f"hal {a}-{b}"


def parse_hits(res, limit: int = MAX_SHOW) -> tuple[list[Hit], int]:
    """Hasil query Chroma berbentuk list bersarang (satu list dalam per query).
    Hasil yang dokumen + isinya sama dengan hasil sebelumnya disembunyikan."""
    hits: list[Hit] = []
    seen: set[tuple] = set()
    hidden = 0
    for ids, docs, metas in zip(res["ids"], res["documents"], res["metadatas"]):
        for i, d, m in zip(ids, docs, metas):
            d, m = d or "", m or {}
            key = (m.get("doc_name"), " ".join(d.split()))
            if key in seen:
                hidden += 1
                continue
            seen.add(key)
            if len(hits) < limit:
                hits.append(Hit(str(i), d, m))
    return hits, hidden


def search(backend, query: str):
    """Pakai n_results lebih besar kalau querytodb() di main.py punya parameternya,
    supaya setelah duplikat dibuang masih tersisa beberapa hasil."""
    params = inspect.signature(backend.querytodb).parameters
    if "n_results" in params:
        return backend.querytodb(query, n_results=FETCH_N)
    return backend.querytodb(query)


def collection_names(backend) -> list[str]:
    names = []
    for c in backend.db.getcollection():
        names.append(c if isinstance(c, str) else getattr(c, "name", str(c)))
    return names


# --------------------------------------------------------------------------
# Input satu baris
# --------------------------------------------------------------------------
class LineEdit:
    def __init__(self) -> None:
        self.text = ""
        self.pos = 0

    def clear(self) -> None:
        self.text, self.pos = "", 0

    def feed(self, ch) -> None:
        t, p = self.text, self.pos
        if ch in ("\x7f", "\b") or ch == curses.KEY_BACKSPACE:
            if p:
                self.text, self.pos = t[: p - 1] + t[p:], p - 1
        elif ch == curses.KEY_DC:
            self.text = t[:p] + t[p + 1:]
        elif ch == "\x15":  # Ctrl+U: hapus sampai awal baris
            self.text, self.pos = t[p:], 0
        elif ch == curses.KEY_LEFT:
            self.pos = max(0, p - 1)
        elif ch == curses.KEY_RIGHT:
            self.pos = min(len(t), p + 1)
        elif ch == curses.KEY_HOME:
            self.pos = 0
        elif ch == curses.KEY_END:
            self.pos = len(t)
        elif isinstance(ch, str) and ch.isprintable():
            self.text, self.pos = t[:p] + ch + t[p:], p + 1


# --------------------------------------------------------------------------
# Pekerjaan di thread (supaya layar tidak membeku saat model bekerja)
# --------------------------------------------------------------------------
class Job:
    def __init__(self, kind: str, label: str, fn, *args) -> None:
        self.kind = kind
        self.label = label
        self.info = ""
        self.result = None
        self.error: Exception | None = None
        self.done = False
        threading.Thread(target=self._run, args=(fn, args), daemon=True).start()

    def _run(self, fn, args) -> None:
        try:
            self.result = fn(*args)
        except Exception as e:  # noqa: BLE001
            self.error = e
            traceback.print_exc()  # masuk ke tui.log
        finally:
            self.done = True


# --------------------------------------------------------------------------
# Layar pilih koleksi
# --------------------------------------------------------------------------
def pick_collection(stdscr, backend) -> str | None:
    NEW = "+ buat koleksi baru"
    items = collection_names(backend) + [NEW]
    sel = 0
    naming = False
    edit = LineEdit()
    msg = ""
    stdscr.timeout(-1)

    while True:
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        put(stdscr, 0, 0, " " * w, attr=A_HEAD)
        put(stdscr, 0, 1, "Pilih koleksi", attr=A_HEAD)
        for i, name in enumerate(items):
            attr = A_SEL if i == sel and not naming else 0
            put(stdscr, 2 + i, 2, fit(name, w - 4).ljust(min(40, w - 4)), attr=attr)
        if naming:
            label = "Nama baru > "
            put(stdscr, h - 3, 0, label, attr=A_TITLE)
            put(stdscr, h - 3, len(label), edit.text)
        put(stdscr, h - 2, 0, fit(msg, w - 1), attr=A_ERR)
        helptxt = " Enter pilih/simpan | Esc batal" if naming else " Up/Down pilih | Enter buka | Esc keluar"
        put(stdscr, h - 1, 0, fit(helptxt, w - 1), attr=A_DIM)
        if naming:
            try:
                stdscr.move(h - 3, len("Nama baru > ") + edit.pos)
            except curses.error:
                pass
        stdscr.refresh()

        ch = stdscr.get_wch()
        if ch == curses.KEY_RESIZE:
            continue
        if ch == "\x1b":
            if naming:
                naming, msg = False, ""
                edit.clear()
                continue
            return None
        if naming:
            if ch in ENTER_KEYS or ch == curses.KEY_ENTER:
                name = edit.text.strip()
                if not name:
                    continue
                try:
                    backend.db.createCollection(name)
                    return name
                except Exception as e:  # noqa: BLE001
                    msg = f"Gagal: {e}"
            else:
                edit.feed(ch)
            continue
        if ch == curses.KEY_UP:
            sel = max(0, sel - 1)
        elif ch == curses.KEY_DOWN:
            sel = min(len(items) - 1, sel + 1)
        elif ch in ENTER_KEYS or ch == curses.KEY_ENTER:
            if items[sel] == NEW:
                naming, msg = True, ""
                edit.clear()
            else:
                try:
                    backend.db.createCollection(items[sel])  # di main.py ini yang "memilih" koleksi
                    return items[sel]
                except Exception as e:  # noqa: BLE001
                    msg = f"Gagal: {e}"


# --------------------------------------------------------------------------
# Layar utama
# --------------------------------------------------------------------------
class App:
    def __init__(self, stdscr, backend, collection: str) -> None:
        self.s = stdscr
        self.backend = backend
        self.collection = collection

        self.mode = "search"  # "search" atau "path"
        self.edit_search = LineEdit()
        self.edit_path = LineEdit()

        self.hits: list[Hit] = []
        self.sel = 0
        self.list_off = 0
        self.scroll = 0
        self.text_h = 10
        self.wrap_cache: dict[tuple[str, int], list[str]] = {}

        self.job: Job | None = None
        self.tick = 0
        self.status = "Ketik pertanyaan lalu tekan Enter."
        self.status_kind = "info"

    # ---- status & job ----------------------------------------------------
    def set_status(self, text: str, kind: str = "info") -> None:
        self.status, self.status_kind = text, kind

    def start_job(self, kind: str, label: str, fn, *args) -> Job | None:
        if self.job:
            self.set_status("Tunggu, proses sebelumnya belum selesai.", "err")
            return None
        self.job = Job(kind, label, fn, *args)
        return self.job

    def poll_job(self) -> None:
        j = self.job
        if not j or not j.done:
            return
        self.job = None
        if j.error:
            self.set_status(f"Gagal: {j.error}  (detail: tui.log)", "err")
            return
        if j.kind == "search":
            self.hits, hidden = parse_hits(j.result)
            self.sel = self.list_off = self.scroll = 0
            self.wrap_cache.clear()
            n = len(self.hits)
            msg = f"{n} hasil" if n else "Tidak ada hasil."
            if hidden:
                msg += f" ({hidden} duplikat disembunyikan)"
            self.set_status(msg, "ok" if n else "info")
        elif j.kind == "add":
            self.set_status(f"Dokumen ditambahkan: {j.info}", "ok")

    # ---- aksi ------------------------------------------------------------
    def submit_search(self) -> None:
        q = self.edit_search.text.strip()
        if not q:
            return
        self.start_job("search", f"Mencari '{fit(q, 30)}'", search, self.backend, q)

    def submit_path(self) -> None:
        p = self.edit_path.text.strip().strip('"').strip("'")
        if not p:
            return
        p = os.path.normpath(p)
        if not os.path.isfile(p):
            self.set_status(f"File tidak ditemukan: {p}", "err")
            return
        job = self.start_job("add", "Memproses PDF (chunking + embedding)", self.backend.add_to_vecdb, p)
        if job:
            job.info = os.path.basename(p)
            self.mode = "search"
            self.edit_path.clear()

    def move(self, d: int) -> None:
        if self.hits:
            self.sel = max(0, min(len(self.hits) - 1, self.sel + d))
            self.scroll = 0

    def on_resize(self) -> None:
        try:
            h, w = self.s.getmaxyx()
            if curses.is_term_resized(h, w):
                curses.resize_term(0, 0)
        except (curses.error, AttributeError):
            pass
        self.s.clear()
        self.wrap_cache.clear()

    # ---- input -----------------------------------------------------------
    def handle(self, ch) -> bool:
        """Return False kalau program harus berhenti."""
        if ch == curses.KEY_RESIZE:
            self.on_resize()
            return True

        in_path = self.mode == "path"
        edit = self.edit_path if in_path else self.edit_search

        if ch == "\x1b":
            if in_path:
                self.mode = "search"
                self.set_status("Dibatalkan.")
                return True
            if edit.text:  # Esc pertama: kosongkan kotak cari, Esc kedua: keluar
                edit.clear()
                return True
            return False

        if ch in ENTER_KEYS or ch == curses.KEY_ENTER:
            if in_path:
                self.submit_path()
            else:
                self.submit_search()
            return True

        if not in_path:
            if ch == curses.KEY_UP:
                self.move(-1)
            elif ch == curses.KEY_DOWN:
                self.move(1)
            elif ch == curses.KEY_PPAGE:
                self.scroll = max(0, self.scroll - self.text_h)
            elif ch == curses.KEY_NPAGE:
                self.scroll += self.text_h  # dibatasi saat draw
            elif ch == KEY_F2 or ch == "\x01":  # F2 atau Ctrl+A
                self.mode = "path"
                self.set_status("Paste lokasi file PDF, lalu Enter.")
            else:
                edit.feed(ch)
        else:
            edit.feed(ch)
        return True

    # ---- gambar ----------------------------------------------------------
    def wrapped(self, hit: Hit, width: int) -> list[str]:
        key = (hit.id, width)
        if key not in self.wrap_cache:
            lines: list[str] = []
            for para in hit.text.split("\n"):
                lines.extend(textwrap.wrap(para, max(1, width)) or [""])
            self.wrap_cache[key] = lines
        return self.wrap_cache[key]

    def draw(self) -> None:
        s = self.s
        s.erase()
        h, w = s.getmaxyx()
        if h < 12 or w < 50:
            put(s, 0, 0, "Terminal terlalu kecil (min 50x12).")
            s.refresh()
            return

        # baris 0: judul
        put(s, 0, 0, " " * w, attr=A_HEAD)
        put(s, 0, 1, "Pencarian Dokumen (ChromaDB)", attr=A_HEAD)
        right = fit(f"koleksi: {self.collection} ", w - 32)
        put(s, 0, w - len(right), right, attr=A_HEAD)

        # baris 1: kotak input
        in_path = self.mode == "path"
        label = "PDF  > " if in_path else "Cari > "
        edit = self.edit_path if in_path else self.edit_search
        avail = w - len(label) - 1
        off = max(0, edit.pos - (avail - 1))
        put(s, 1, 0, label, attr=A_TITLE)
        put(s, 1, len(label), edit.text[off: off + avail])
        cursor_x = len(label) + edit.pos - off
        s.hline(2, 0, curses.ACS_HLINE, w)

        # area panel
        top = 3
        ph = h - 2 - top
        lw = min(max(26, w // 3), 40)
        s.vline(top, lw, curses.ACS_VLINE, ph)

        self.draw_list(top, ph, lw)
        self.draw_detail(top, ph, lw + 2, w)

        # status & bantuan
        if self.job:
            sp = SPINNER[self.tick % len(SPINNER)]
            put(s, h - 2, 0, fit(f" {sp} {self.job.label}...", w - 1), attr=A_BUSY)
        else:
            attr = {"ok": A_OK, "err": A_ERR}.get(self.status_kind, 0)
            put(s, h - 2, 0, fit(" " + self.status, w - 1), attr=attr)
        if in_path:
            helptxt = " Enter tambah | Esc batal | tip: paste path file (tanda kutip dibuang otomatis)"
        else:
            helptxt = " Enter cari | Up/Down pilih | PgUp/PgDn gulir | F2 tambah PDF | Esc hapus/keluar"
        put(s, h - 1, 0, fit(helptxt, w - 1), attr=A_DIM)

        try:
            s.move(1, min(cursor_x, w - 1))
        except curses.error:
            pass
        s.refresh()

    def draw_list(self, top: int, ph: int, lw: int) -> None:
        s = self.s
        if not self.hits:
            for k, line in enumerate(["Belum ada hasil.", "", "Ketik pertanyaan di atas,", "lalu tekan Enter."]):
                put(s, top + k, 1, fit(line, lw - 2), attr=A_DIM)
            return
        slots = max(1, ph // 2)
        if self.sel < self.list_off:
            self.list_off = self.sel
        if self.sel >= self.list_off + slots:
            self.list_off = self.sel - slots + 1
        for k in range(slots):
            i = self.list_off + k
            if i >= len(self.hits):
                break
            hit = self.hits[i]
            line1 = f"{i + 1}. {hit.pages()} {hit.id[:8]}"
            line2 = "   " + " ".join(hit.text.split())
            on = i == self.sel
            y = top + k * 2
            put(s, y, 0, fit(line1, lw).ljust(lw), lw, A_SEL if on else A_TITLE)
            put(s, y + 1, 0, fit(line2, lw).ljust(lw), lw, A_SEL if on else A_DIM)

    def draw_detail(self, top: int, ph: int, rx: int, w: int) -> None:
        s = self.s
        rw = w - rx - 1
        if not self.hits or rw < 10:
            return
        hit = self.hits[self.sel]
        lines = self.wrapped(hit, rw)
        th = max(1, ph - 3)
        self.text_h = th
        self.scroll = max(0, min(self.scroll, max(0, len(lines) - th)))

        ind = f"{self.scroll + 1}-{min(len(lines), self.scroll + th)}/{len(lines)}" if len(lines) > th else ""
        put(s, top, rx, fit(hit.doc(), rw), attr=A_TITLE)
        put(s, top + 1, rx, fit(f"{hit.pages()} | id {hit.id}", rw - len(ind) - 1), attr=A_DIM)
        if ind:
            put(s, top + 1, w - 1 - len(ind), ind, attr=A_DIM)
        for k in range(th):
            idx = self.scroll + k
            if idx >= len(lines):
                put(s, top + 3 + k, rx, " " * rw, rw)
                continue
            put(s, top + 3 + k, rx, lines[idx].ljust(rw), rw)  # padding: timpa sisa teks lama

    # ---- loop ------------------------------------------------------------
    def run(self) -> None:
        self.s.timeout(100)  # bangun tiap 0,1 detik untuk animasi spinner
        while True:
            self.tick += 1
            self.poll_job()
            self.draw()
            try:
                ch = self.s.get_wch()
            except curses.error:
                continue  # tidak ada tombol ditekan
            if not self.handle(ch):
                break


def run_app(stdscr, backend) -> None:
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    init_colors()
    stdscr.keypad(True)
    name = pick_collection(stdscr, backend)
    if name:
        App(stdscr, backend, name).run()


def start() -> None:
    locale.setlocale(locale.LC_ALL, "")
    os.environ.setdefault("ESCDELAY", "25")

    print("Memuat model dan database (sebentar)...", flush=True)
    import src.backend as backend  # main.py milikmu; fungsinya dipakai apa adanya

    # Selama curses aktif, print()/warning dari library lain akan merusak layar,
    # jadi dialihkan ke tui.log.
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tui.log")
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = log
    error = None
    try:
        curses.wrapper(run_app, backend)
    except KeyboardInterrupt:
        pass
    except Exception:  # noqa: BLE001
        error = traceback.format_exc()
        log.write(error)
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        log.close()
    if error:
        print(error)


if __name__ == "__main__":
    start()