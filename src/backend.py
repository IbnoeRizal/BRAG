from src.preprocessing.chunking import ChunkIng, DOCMetadata
from dataclasses import asdict
from src.preprocessing.cleaning import TextCleaning
import pymupdf
from src.storage import chromadb
import uuid
import os
from config import ENV
from src.model.EmbeddingModel import Model_EMB
from src.model.RerankerModel import Model_Reranker
from src.model.GenerationModel import Model_Generation


embd = Model_EMB(ENV.model_embd.model,ENV.model_embd.path)
rerank = Model_Reranker(ENV.model_rerank.model,ENV.model_rerank.path)
llm = Model_Generation(ENV.model_generation.model,ENV.model_generation.path)
db = chromadb.VecDB()


def add_to_vecdb(path:str,collection:str|None = None)->None:
    if not os.path.isfile(path=path):
        print("File tidak ada")
        return

    doc = pymupdf.open(path)
    with doc:
        _,tail = os.path.split(path)

        newdoc = TextCleaning.header_footercleaning(doc)
        TextCleaning.clean_before_abstract(newdoc)
        TextCleaning.clean_reference(newdoc)

        chunk = ChunkIng(newdoc,model=embd.model)
        chunk.semantic_chunking()
        chunk.merge_short_chunks()

        data = []
        for md in chunk.l_md:
            txtchunk = chunk._chunk_text(md)
            embedding = embd.model.encode(f"passage: {txtchunk}")
            data.append(chromadb.DataToDB(
                id=str(uuid.uuid4()),
                embedding=embedding,
                metadata=chromadb.EmbeddingMetadata(
                    **asdict(DOCMetadata(tail,md))
                ),
                document=txtchunk
            ))

        if(collection):
            db.createCollection(collection.strip())
        db.add_data(data)


def querytodb(query:str,n_results:int):
    return db.query_data(chromadb.QueryToDB(
        query_embeddings=embd.model.encode(f"query: {query.strip()}"),
        include=["metadatas","documents"],
        n_results=n_results
    ))

reranking = rerank.rank
generation = llm.generate