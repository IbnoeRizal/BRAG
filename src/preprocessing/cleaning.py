import pymupdf
import re
from collections import Counter


class TextCleaning:
        
    @staticmethod
    def clean_reference(doc:list[list[str]]):
        for i in reversed(range(len(doc))):
            halaman = doc[i]
            for j in range(len(halaman)):
                baris = halaman[j]
                if re.search(r"references?|daftar\s+pustaka",baris,flags=re.IGNORECASE):
                    del halaman[j:]
                    del doc[i+1:]
                    return
        
    @staticmethod
    def clean_before_abstract(doc:list[list[str]]):
        for i in range(len(doc)):
            halaman = doc[i]
            for j in range(len(halaman)):
                baris = halaman[j]
                if re.search(r"abstract|abstrak",baris,flags=re.IGNORECASE):
                    del halaman[:j]
                    return
    
    @staticmethod
    def clean_block(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
    
    @staticmethod
    def header_footercleaning(doc:pymupdf.Document):
        # 1. ambil baris-baris tiap halaman
        pages = [[TextCleaning.clean_block(b[4]) for b in page.get_text("blocks")
         if b[6] == 0 and TextCleaning.clean_block(b[4])]
         for page in doc]

        # 2. hitung di berapa halaman tiap baris muncul
        counter = Counter()
        for lines in pages:
            counter.update(set(lines))   # set() = hitung maksimal 1x per halaman

        # 3. buang baris yang muncul di semua halaman (header/footer)
        doc_len = len(doc)
        return [[b for b in blocks if counter[b] < doc_len * 0.7 and not b.isdigit()] for blocks in pages]