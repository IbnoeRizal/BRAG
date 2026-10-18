"""
tui.py - tampilan curses untuk prototype RAG (ChromaDB + reranker + LLM).

Cara pakai:
    1. Taruh file ini di folder yang sama dengan main.py (root proyek)
    2. Windows: pip install windows-curses
    3. python tui.py

Fungsi yang dipakai dari main.py:
    querytodb(query, n_results)   reranking(pairs)   generation(query, context)
    add_to_vecdb(path)            db  (VecDB)        chromadb (untuk DOCMetadata)

Tombol:
    Enter        cari (lalu rerank kalau aktif)
    F4 / Ctrl+G  tanya LLM (cari -> rerank -> generation)
    Tab          ganti panel kanan: Isi <-> Jawaban
    Up/Down      pilih hasil          PgUp/PgDn  gulir panel kanan
    F3           rerank on/off
    F2 / Ctrl+A  tambah PDF
    F5           kelola dokumen (hapus dokumen)
    F6           ganti / hapus koleksi
    Esc          kosongkan kotak cari; kalau sudah kosong, keluar
"""
from __future__ import annotations

import locale
import os
import re
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
KEY_F2, KEY_F3, KEY_F4, KEY_F5, KEY_F6 = (curses.KEY_F0 + n for n in (2, 3, 4, 5, 6))
ENTER_KEYS = ("\n", "\r")

FETCH_N = 10   # kandidat yang diminta ke database
MAX_SHOW = 5   # hasil unik yang ditampilkan
K_CTX = 2      # potongan teratas yang dikirim ke LLM sebagai konteks
MIN_H, MIN_W = 14, 60

# aturan nama koleksi ChromaDB: 3-512 karakter [a-zA-Z0-9._-], awal & akhir huruf/angka
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]$")

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


def kind_attr(kind: str) -> int:
    return {"ok": A_OK, "err": A_ERR}.get(kind, 0)


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


def is_enter(ch) -> bool:
    return ch in ENTER_KEYS or ch == curses.KEY_ENTER


def fix_resize(stdscr) -> None:
    try:
        h, w = stdscr.getmaxyx()
        if curses.is_term_resized(h, w):
            curses.resize_term(0, 0)
    except (curses.error, AttributeError):
        pass
    stdscr.clear()


def scroll_window(sel: int, off: int, slots: int) -> int:
    if sel < off:
        off = sel
    if sel >= off + slots:
        off = sel - slots + 1
    return max(0, off)


# --------------------------------------------------------------------------
# Data & pipeline
# --------------------------------------------------------------------------
@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float | None = None  # skor reranker (None kalau rerank mati)

    def doc(self) -> str:
        return str(self.meta.get("doc_name", "?"))

    def pages(self) -> str:
        a = self.meta.get("doc_metadata.page_start")
        b = self.meta.get("doc_metadata.page_end")
        if a is None:
            return "hal ?"
        return f"hal {a}" if b is None or a == b else f"hal {a}-{b}"


def parse_hits(res) -> tuple[list[Hit], int]:
    """Hasil query Chroma berbentuk list bersarang (satu list dalam per query).
    Hasil yang dokumen + isinya sama dengan hasil sebelumnya dibuang."""
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
            hits.append(Hit(str(i), d, m))
    return hits, hidden


def run_search(backend, q: str, use_rerank: bool) -> tuple[list[Hit], int]:
    res = backend.querytodb(q, FETCH_N)
    hits, hidden = parse_hits(res)
    if use_rerank and hits:
        # asumsi: reranking() mengembalikan satu skor per pasangan, urutannya sama dengan input
        scores = backend.reranking([[q, h.text] for h in hits])
        for h, s in zip(hits, scores):
            h.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:MAX_SHOW], hidden


def run_ask(backend, q: str, use_rerank: bool) -> tuple[list[Hit], int, str]:
    hits, hidden = run_search(backend, q, use_rerank)
    if not hits:
        return hits, hidden, ""
    context = "\n\n".join(h.text for h in hits[:K_CTX])
    answer = backend.generation(query=q, context=context)
    return hits, hidden, str(answer)


def collection_names(backend) -> list[str]:
    return [str(n) for n in backend.db.getcollection(limit=None)]  # default-nya cuma 10


def doc_stats(backend) -> list[tuple[str, int]]:
    """Daftar (nama dokumen, jumlah chunk) di koleksi aktif."""
    res = backend.db.collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for m in res.get("metadatas") or []:
        name = str((m or {}).get("doc_name", "(tanpa nama)"))
        counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items())


def delete_document(backend, name: str) -> int:
    """Hapus semua chunk milik satu dokumen. Return jumlah chunk yang terhapus."""
    col = backend.db.collection
    before = col.count()
    # delete_data hanya bisa dengan doc_name saja; kalau doc_metadata ikut diisi,
    # ChromaDB menolak filternya (butuh operator $and).
    backend.db.delete_data(backend.chromadb.DOCMetadata(doc_name=name, doc_metadata=None))
    return before - col.count()


def delete_collection(backend, name: str, keep: str | None) -> bool:
    """Hapus koleksi `name`. `keep` = koleksi yang harus tetap terpilih sesudahnya."""
    backend.db.createCollection(name)  # delete_collection() menolak kalau belum ada koleksi terpilih
    ok = backend.db.delete_collection(name)
    if keep:
        backend.db.createCollection(keep)
    return ok


# --------------------------------------------------------------------------
# Input satu baris & job di thread
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


class Job:
    """Menjalankan fungsi lambat (model/DB) di thread supaya layar tidak membeku."""

    def __init__(self, kind: str, label: str, fn, *args, info: str = "") -> None:
        self.kind = kind
        self.label = label
        self.info = info
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
# Layar-layar modal (berjalan sendiri, memblokir sampai selesai)
# --------------------------------------------------------------------------
def modal_frame(stdscr, title: str):
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    put(stdscr, 0, 0, " " * w, attr=A_HEAD)
    put(stdscr, 0, 1, fit(title, w - 2), attr=A_HEAD)
    return h, w


def confirm(stdscr, prompt: str, expect: str | None = None) -> bool:
    """Konfirmasi di dasar layar. expect=None -> tekan y; selain itu harus mengetik teks tsb."""
    edit = LineEdit()
    while True:
        h, w = stdscr.getmaxyx()
        for r in (h - 3, h - 2, h - 1):
            put(stdscr, r, 0, " " * w)
        put(stdscr, h - 3, 0, fit(" " + prompt, w - 1), attr=A_ERR)
        if expect is None:
            put(stdscr, h - 2, 0, " Tekan y untuk melanjutkan, tombol lain untuk batal.", attr=A_TITLE)
        else:
            label = f" Ketik '{expect}' lalu Enter (Esc batal) > "
            put(stdscr, h - 2, 0, fit(label + edit.text, w - 1), attr=A_TITLE)
        stdscr.refresh()
        ch = stdscr.get_wch()
        if ch == curses.KEY_RESIZE:
            continue
        if expect is None:
            return ch in ("y", "Y")
        if ch == "\x1b":
            return False
        if is_enter(ch):
            return edit.text == expect
        edit.feed(ch)


def pick_collection(stdscr, backend, current: str | None = None) -> str | None:
    """Pilih / buat / hapus koleksi. Return nama koleksi terpilih, atau None kalau dibatalkan."""
    NEW = "+ buat koleksi baru"
    names = collection_names(backend)
    alive = current          # koleksi aktif yang masih ada (None kalau sudah dihapus)
    must_pick = False        # True kalau koleksi aktif terhapus -> wajib pilih yang lain
    sel = names.index(current) if current in names else 0
    off = 0
    naming = False
    edit = LineEdit()
    msg, kind = "", "info"

    while True:
        items = names + [NEW]
        sel = max(0, min(sel, len(items) - 1))
        h, w = modal_frame(stdscr, "Pilih koleksi")
        slots = max(1, h - 5)
        off = scroll_window(sel, off, slots)
        for k in range(slots):
            i = off + k
            if i >= len(items):
                break
            mark = "* " if items[i] == alive else "  "
            attr = A_SEL if (i == sel and not naming) else 0
            put(stdscr, 2 + k, 1, fit(mark + items[i], w - 3).ljust(min(44, w - 3)), attr=attr)
        if naming:
            put(stdscr, h - 3, 0, " Nama baru > " + edit.text, attr=A_TITLE)
        put(stdscr, h - 2, 0, fit(" " + msg, w - 1), attr=kind_attr(kind))
        if naming:
            helptxt = " Enter simpan | Esc batal | min. 3 karakter: huruf, angka, . _ -"
        elif must_pick:
            helptxt = " Up/Down pilih | Enter buka | Del/d hapus koleksi | pilih koleksi aktif dulu"
        else:
            helptxt = " Up/Down pilih | Enter buka | Del/d hapus koleksi | Esc " + ("kembali" if alive else "keluar")
        put(stdscr, h - 1, 0, fit(helptxt, w - 1), attr=A_DIM)
        stdscr.refresh()

        ch = stdscr.get_wch()
        if ch == curses.KEY_RESIZE:
            fix_resize(stdscr)
            continue

        if ch == "\x1b":
            if naming:
                naming, msg = False, ""
                edit.clear()
            elif must_pick:
                msg, kind = "Koleksi aktif sudah dihapus. Pilih atau buat koleksi lain.", "err"
            else:
                return None
            continue

        if naming:
            if is_enter(ch):
                name = edit.text.strip()
                if not NAME_RE.match(name):
                    msg, kind = "Nama tidak valid (min. 3 karakter, huruf/angka/./_/-).", "err"
                    continue
                try:
                    backend.db.createCollection(name)
                    return name
                except Exception as e:  # noqa: BLE001
                    msg, kind = f"Gagal: {e}", "err"
            else:
                edit.feed(ch)
            continue

        if ch == curses.KEY_UP:
            sel -= 1
        elif ch == curses.KEY_DOWN:
            sel += 1
        elif is_enter(ch):
            if items[sel] == NEW:
                naming, msg = True, ""
                edit.clear()
            else:
                try:
                    backend.db.createCollection(items[sel])  # di backend, ini yang "memilih" koleksi
                    return items[sel]
                except Exception as e:  # noqa: BLE001
                    msg, kind = f"Gagal: {e}", "err"
        elif ch in (curses.KEY_DC, "d", "D") and items[sel] != NEW:
            name = items[sel]
            if confirm(stdscr, f"Hapus koleksi '{name}' beserta SEMUA isinya? Tidak bisa dibatalkan.", expect=name):
                keep = alive if (alive and alive != name) else None
                try:
                    ok = delete_collection(backend, name, keep)
                except Exception as e:  # noqa: BLE001
                    ok, msg, kind = False, f"Gagal: {e}", "err"
                if ok:
                    names = collection_names(backend)
                    msg, kind = f"Koleksi '{name}' dihapus.", "ok"
                    if name == alive:
                        alive, must_pick = None, True
                elif kind != "err":
                    msg, kind = f"Gagal menghapus '{name}'.", "err"
            else:
                msg, kind = "Dibatalkan.", "info"


def manage_documents(stdscr, backend, collection: str) -> bool:
    """Daftar dokumen di koleksi aktif + hapus dokumen. Return True kalau ada yang dihapus."""
    docs: list[tuple[str, int]] = []
    msg, kind = "", "info"

    def load() -> None:
        nonlocal docs, msg, kind
        try:
            docs = doc_stats(backend)
        except Exception as e:  # noqa: BLE001
            docs, msg, kind = [], f"Gagal membaca dokumen: {e}", "err"

    load()
    sel = off = 0
    changed = False
    while True:
        h, w = modal_frame(stdscr, f"Dokumen di koleksi '{collection}' ({len(docs)} dokumen)")
        slots = max(1, h - 5)
        sel = max(0, min(sel, len(docs) - 1)) if docs else 0
        off = scroll_window(sel, off, slots)
        if not docs:
            put(stdscr, 2, 2, "Koleksi ini belum berisi dokumen.", attr=A_DIM)
        for k in range(slots):
            i = off + k
            if i >= len(docs):
                break
            name, n = docs[i]
            line = (" " + fit(name, w - 16)).ljust(max(0, w - 12)) + f"{n:>4} chunk"
            put(stdscr, 2 + k, 0, line, attr=A_SEL if i == sel else 0)
        put(stdscr, h - 2, 0, fit(" " + msg, w - 1), attr=kind_attr(kind))
        put(stdscr, h - 1, 0, fit(" Up/Down pilih | Del/d hapus dokumen | Esc kembali", w - 1), attr=A_DIM)
        stdscr.refresh()

        ch = stdscr.get_wch()
        if ch == curses.KEY_RESIZE:
            fix_resize(stdscr)
        elif ch == "\x1b":
            return changed
        elif ch == curses.KEY_UP:
            sel -= 1
        elif ch == curses.KEY_DOWN:
            sel += 1
        elif ch in (curses.KEY_DC, "d", "D") and docs:
            name, n = docs[sel]
            if confirm(stdscr, f"Hapus dokumen '{name}' ({n} chunk) dari '{collection}'? Tidak bisa dibatalkan."):
                try:
                    removed = delete_document(backend, name)
                    changed = True
                    msg, kind = f"'{name}' dihapus ({removed} chunk).", "ok"
                except Exception as e:  # noqa: BLE001
                    msg, kind = f"Gagal: {e}", "err"
                load()
            else:
                msg, kind = "Dibatalkan.", "info"


# --------------------------------------------------------------------------
# Layar utama
# --------------------------------------------------------------------------
class App:
    def __init__(self, stdscr, backend, collection: str) -> None:
        self.s = stdscr
        self.backend = backend
        self.collection = collection

        self.mode = "search"  # "search" atau "path" (kotak input dipakai untuk lokasi PDF)
        self.edit_search = LineEdit()
        self.edit_path = LineEdit()
        self.use_rerank = True

        self.hits: list[Hit] = []
        self.sel = 0
        self.list_off = 0
        self.answer = ""
        self.answer_q = ""
        self.answer_src: list[str] = []
        self.view = "isi"  # panel kanan: "isi" atau "jawaban"
        self.scroll = {"isi": 0, "jawaban": 0}
        self.text_h = 10
        self.wrap_cache: dict[tuple, list[str]] = {}

        self.job: Job | None = None
        self.tick = 0
        self.status = "Ketik pertanyaan lalu Enter (cari) atau F4 (tanya LLM)."
        self.status_kind = "info"

    # ---- status & job ----------------------------------------------------
    def set_status(self, text: str, kind: str = "info") -> None:
        self.status, self.status_kind = text, kind

    def start_job(self, kind: str, label: str, fn, *args, info: str = "") -> Job | None:
        if self.job:
            self.set_status("Tunggu, proses sebelumnya belum selesai.", "err")
            return None
        self.job = Job(kind, label, fn, *args, info=info)
        return self.job

    def reset_results(self) -> None:
        self.hits = []
        self.sel = self.list_off = 0
        self.answer, self.answer_q, self.answer_src = "", "", []
        self.scroll = {"isi": 0, "jawaban": 0}
        self.view = "isi"
        self.wrap_cache.clear()

    def poll_job(self) -> None:
        j = self.job
        if not j or not j.done:
            return
        self.job = None
        if j.error:
            self.set_status(f"Gagal: {type(j.error).__name__}: {j.error}  (detail: tui.log)", "err")
            return

        if j.kind in ("search", "ask"):
            answer = ""
            if j.kind == "search":
                hits, hidden = j.result
            else:
                hits, hidden, answer = j.result
            self.hits = hits
            self.sel = self.list_off = 0
            self.scroll["isi"] = 0
            self.view = "isi"
            self.wrap_cache.clear()
            n = len(hits)
            msg = f"{n} hasil" if n else "Tidak ada hasil."
            if n and hits[0].score is not None:
                msg += ", diurutkan ulang (rerank)"
            if hidden:
                msg += f", {hidden} duplikat disembunyikan"
            if j.kind == "ask":
                if answer:
                    self.answer, self.answer_q = answer, j.info
                    self.answer_src = [f"[{i + 1}] {h.doc()} {h.pages()}" for i, h in enumerate(hits[:K_CTX])]
                    self.scroll["jawaban"] = 0
                    self.view = "jawaban"
                    msg = "Jawaban siap. " + msg
                else:
                    msg = "Tidak ada konteks untuk dijawab. " + msg
            self.set_status(msg, "ok" if n else "info")
        elif j.kind == "add":
            self.set_status(f"Dokumen ditambahkan: {j.info}", "ok")

    # ---- aksi ------------------------------------------------------------
    def submit_search(self, ask: bool = False) -> None:
        q = self.edit_search.text.strip()
        if not q:
            self.set_status("Ketik pertanyaan dulu.", "err")
            return
        if ask:
            self.start_job("ask", f"Mencari dan bertanya ke LLM '{fit(q, 24)}'",
                           run_ask, self.backend, q, self.use_rerank, info=q)
        else:
            self.start_job("search", f"Mencari '{fit(q, 30)}'",
                           run_search, self.backend, q, self.use_rerank, info=q)

    def submit_path(self) -> None:
        p = self.edit_path.text.strip().strip('"').strip("'")
        if not p:
            return
        p = os.path.normpath(p)
        if not os.path.isfile(p):
            self.set_status(f"File tidak ditemukan: {p}", "err")
            return
        job = self.start_job("add", "Memproses PDF (chunking + embedding)",
                             self.backend.add_to_vecdb, p, info=os.path.basename(p))
        if job:
            self.mode = "search"
            self.edit_path.clear()

    def modal(self, fn, *args):
        """Jalankan layar modal (blocking), lalu kembalikan pengaturan layar utama."""
        self.s.timeout(-1)
        try:
            return fn(self.s, self.backend, *args)
        finally:
            self.s.timeout(100)
            self.s.clear()

    def open_documents(self) -> None:
        if self.job:
            self.set_status("Tunggu, proses sebelumnya belum selesai.", "err")
            return
        if self.modal(manage_documents, self.collection):
            self.reset_results()
            self.set_status("Dokumen dihapus. Hasil pencarian lama dibersihkan.", "ok")

    def open_collections(self) -> None:
        if self.job:
            self.set_status("Tunggu, proses sebelumnya belum selesai.", "err")
            return
        name = self.modal(pick_collection, self.collection)
        if name:
            self.collection = name
            self.reset_results()
            self.set_status(f"Koleksi aktif: {name}", "ok")

    def move(self, d: int) -> None:
        if self.hits:
            self.sel = max(0, min(len(self.hits) - 1, self.sel + d))
            self.scroll["isi"] = 0

    # ---- input -----------------------------------------------------------
    def handle(self, ch) -> bool:
        """Return False kalau program harus berhenti."""
        if ch == curses.KEY_RESIZE:
            fix_resize(self.s)
            self.wrap_cache.clear()
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

        if is_enter(ch):
            if in_path:
                self.submit_path()
            else:
                self.submit_search(ask=False)
            return True

        if in_path:
            edit.feed(ch)
            return True

        if ch == curses.KEY_UP:
            self.move(-1)
        elif ch == curses.KEY_DOWN:
            self.move(1)
        elif ch == curses.KEY_PPAGE:
            self.scroll[self.view] = max(0, self.scroll[self.view] - self.text_h)
        elif ch == curses.KEY_NPAGE:
            self.scroll[self.view] += self.text_h  # dibatasi saat draw
        elif ch == "\t":
            self.view = "jawaban" if self.view == "isi" else "isi"
        elif ch == KEY_F3:
            self.use_rerank = not self.use_rerank
            self.set_status(f"Rerank {'aktif' if self.use_rerank else 'mati'} (berlaku di pencarian berikutnya).")
        elif ch in (KEY_F4, "\x07"):  # F4 atau Ctrl+G
            self.submit_search(ask=True)
        elif ch in (KEY_F2, "\x01"):  # F2 atau Ctrl+A
            if self.job:
                self.set_status("Tunggu, proses sebelumnya belum selesai.", "err")
            else:
                self.mode = "path"
                self.set_status("Paste lokasi file PDF, lalu Enter.")
        elif ch == KEY_F5:
            self.open_documents()
        elif ch == KEY_F6:
            self.open_collections()
        else:
            edit.feed(ch)
        return True

    # ---- gambar ----------------------------------------------------------
    def wrapped(self, key: str, text: str, width: int) -> list[str]:
        ck = (key, width)
        if ck not in self.wrap_cache:
            lines: list[str] = []
            for para in text.split("\n"):
                lines.extend(textwrap.wrap(para, max(1, width)) or [""])
            self.wrap_cache[ck] = lines
        return self.wrap_cache[ck]

    def draw(self) -> None:
        s = self.s
        s.erase()
        h, w = s.getmaxyx()
        if h < MIN_H or w < MIN_W:
            put(s, 0, 0, f"Terminal terlalu kecil (min {MIN_W}x{MIN_H}).")
            s.refresh()
            return

        # baris 0: judul
        put(s, 0, 0, " " * w, attr=A_HEAD)
        put(s, 0, 1, "Pencarian Dokumen (RAG)", attr=A_HEAD)
        right = fit(f"koleksi: {self.collection} | rerank: {'on' if self.use_rerank else 'off'} ", w - 26)
        put(s, 0, w - len(right), right, attr=A_HEAD)

        # baris 1: kotak input (dipakai bersama untuk lokasi PDF)
        in_path = self.mode == "path"
        label = "PDF  > " if in_path else "Cari > "
        edit = self.edit_path if in_path else self.edit_search
        avail = w - len(label) - 1
        off = max(0, edit.pos - (avail - 1))
        put(s, 1, 0, label, attr=A_TITLE)
        put(s, 1, len(label), edit.text[off: off + avail])
        cursor_x = len(label) + edit.pos - off
        s.hline(2, 0, curses.ACS_HLINE, w)

        # panel: baris 3 .. h-4  (3 baris terbawah: status + 2 baris bantuan)
        top = 3
        ph = h - 3 - top
        lw = min(max(26, w // 3), 40)
        s.vline(top, lw, curses.ACS_VLINE, ph)
        self.draw_list(top, ph, lw)
        self.draw_right(top, ph, lw + 2, w)

        # status
        if self.job:
            sp = SPINNER[self.tick % len(SPINNER)]
            put(s, h - 3, 0, fit(f" {sp} {self.job.label}...", w - 1), attr=A_BUSY)
        else:
            put(s, h - 3, 0, fit(" " + self.status, w - 1), attr=kind_attr(self.status_kind))

        # bantuan
        if in_path:
            put(s, h - 2, 0, fit(" Enter tambah | Esc batal", w - 1), attr=A_DIM)
            put(s, h - 1, 0, fit(" Tip: paste lokasi file PDF (tanda kutip dibuang otomatis)", w - 1), attr=A_DIM)
        else:
            put(s, h - 2, 0, fit(" Enter cari  F4 tanya LLM  Tab isi/jawaban  Up/Down pilih  PgUp/PgDn gulir", w - 1), attr=A_DIM)
            put(s, h - 1, 0, fit(" F2 +PDF  F3 rerank  F5 dokumen  F6 koleksi  Esc hapus/keluar", w - 1), attr=A_DIM)

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
        self.list_off = scroll_window(self.sel, self.list_off, slots)
        for k in range(slots):
            i = self.list_off + k
            if i >= len(self.hits):
                break
            hit = self.hits[i]
            tail = f"skor {hit.score:.2f}" if hit.score is not None else hit.id[:6]
            line1 = f"{i + 1}. {hit.pages()}  {tail}"
            line2 = "   " + " ".join(hit.text.split())
            on = i == self.sel
            y = top + k * 2
            put(s, y, 0, fit(line1, lw).ljust(lw), lw, A_SEL if on else A_TITLE)
            put(s, y + 1, 0, fit(line2, lw).ljust(lw), lw, A_SEL if on else A_DIM)

    def draw_right(self, top: int, ph: int, rx: int, w: int) -> None:
        s = self.s
        rw = w - rx - 1
        if rw < 10:
            return

        # tab: Isi | Jawaban
        x = rx
        for name, key in (("Isi", "isi"), ("Jawaban", "jawaban")):
            text = f" {name} "
            put(s, top, x, text, attr=A_SEL if self.view == key else A_DIM)
            x += len(text) + 1

        th = max(1, ph - 3)
        self.text_h = th
        info, lines = "", []
        if self.view == "isi":
            if not self.hits:
                return
            hit = self.hits[self.sel]
            lines = self.wrapped("hit:" + hit.id, hit.text, rw)
            info = f"{hit.doc()} | {hit.pages()}"
            if hit.score is not None:
                info += f" | skor {hit.score:.2f}"
        else:
            if not self.answer:
                for k, line in enumerate(["Belum ada jawaban.", "Ketik pertanyaan, lalu tekan F4 (atau Ctrl+G)."]):
                    put(s, top + 2 + k, rx, fit(line, rw), attr=A_DIM)
                return
            full = self.answer
            if self.answer_src:
                full += "\n\nSumber:\n" + "\n".join(self.answer_src)
            lines = self.wrapped("ans", full, rw)
            info = "T: " + self.answer_q

        sc = max(0, min(self.scroll[self.view], max(0, len(lines) - th)))
        self.scroll[self.view] = sc
        ind = f"{sc + 1}-{min(len(lines), sc + th)}/{len(lines)}" if len(lines) > th else ""
        put(s, top + 1, rx, fit(info, rw - len(ind) - 1), attr=A_DIM)
        if ind:
            put(s, top + 1, w - 1 - len(ind), ind, attr=A_DIM)
        for k in range(th):
            idx = sc + k
            text = lines[idx] if idx < len(lines) else ""
            put(s, top + 3 + k, rx, text.ljust(rw), rw)

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
    stdscr.timeout(-1)
    name = pick_collection(stdscr, backend, current=None)
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