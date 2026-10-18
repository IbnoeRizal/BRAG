import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dataclasses import dataclass

@dataclass
class Metadata:
    page_start:int
    line_start:int
    page_end:int
    line_end:int

@dataclass
class DOCMetadata:
    doc_name:str|None
    doc_metadata:Metadata|None

class ChunkIng:
    model: SentenceTransformer | None = None

    def __init__(self,doc:list[list[str]],model:SentenceTransformer) -> None:
        self.pages = doc
        self.l_md: list[Metadata] = []
        self.model = model


    def semantic_chunking(self, cut_percent: float = 20):
        if self.model is None:
            raise RuntimeError(f"model status {self.model}")

        self.l_md = []

        # 1. embed semua baris dan catat posisinya
        vecs, positions = [], []
        for idx, page in enumerate(self.pages):
            batch = [f"passage: {t}" for t in page]
            emb = self.model.encode(batch, normalize_embeddings=True)
            for i in range(len(emb)):
                vecs.append(emb[i])
                positions.append((idx, i))

        vecs = np.array(vecs)
        if len(vecs) == 0:
            return

        # 2. similarity tiap baris dengan baris sebelumnya
        sims = np.sum(vecs[1:] * vecs[:-1], axis=1)

        # 3. threshold = nilai terendah cut_percent%
        threshold = np.percentile(sims, cut_percent)
        print(f"threshold={threshold:.3f}, min={sims.min():.3f}, max={sims.max():.3f}")

        # 4. potong di tempat similarity < threshold
        start = 0
        for k, s in enumerate(sims, start=1):
            if s < threshold:
                self._close(start, k - 1, positions)
                start = k
        self._close(start, len(vecs) - 1, positions)

        print("jumlah chunk:", len(self.l_md))

    def _close(self, a, b, positions):
        page_start, line_start = positions[a]
        page_end, line_end = positions[b]
        
        self.l_md.append(Metadata(
            page_start,
            line_start,
            page_end,
            line_end
        ))

    def _chunk_text(self, m:Metadata):
        """Ambil teks sebuah chunk dari self.pages berdasarkan metadata-nya."""
        parts = []
        for p in range(m.page_start, m.page_end + 1):
            page = self.pages[p]
            a = m.line_start if p == m.page_start else 0
            b = m.line_end if p == m.page_end else len(page) - 1
            parts.extend(page[a:b + 1])
        return " ".join(parts).strip()

    def merge_short_chunks(self):
        """Gabungkan chunk satu kata:
        - diawali huruf kecil -> ke chunk sebelumnya
        - diawali huruf besar -> ke chunk sesudahnya
        """
        i = 0
        while i < len(self.l_md) and len(self.l_md) > 1:
            m = self.l_md[i]
            text = self._chunk_text(m)

            if len(text.split()) != 1:
                i += 1
                continue

            to_prev = text[0].islower()

            # kasus pinggir: tidak ada tetangga di arah yang diinginkan
            if i == 0:
                to_prev = False
            elif i == len(self.l_md) - 1:
                to_prev = True

            if to_prev:
                prev = self.l_md[i - 1]
                prev.page_end = m.page_end
                prev.line_end = m.line_end
                del self.l_md[i]          # i sekarang menunjuk chunk berikutnya
            else:
                nxt = self.l_md[i + 1]
                nxt.page_start = m.page_start
                nxt.line_start = m.line_start
                del self.l_md[i]          # i sekarang menunjuk chunk hasil gabungan