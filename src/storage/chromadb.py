from __future__ import annotations
import chromadb
from config import ENV
from dataclasses import dataclass,asdict
from numpy import ndarray
from src.preprocessing.chunking import Metadata, DOCMetadata
from typing import Literal

@dataclass
class EmbeddingMetadata:
    doc_name:str
    doc_metadata:Metadata
    
@dataclass
class DataToDB:
    id:str
    embedding:ndarray
    metadata:EmbeddingMetadata
    document:str

    @staticmethod
    def to_chroma(items: "DataToDB | list[DataToDB]") -> dict:
        if not isinstance(items, list):
            items = [items]
        return {
            "ids": [d.id for d in items],
            "embeddings": [d.embedding for d in items],
            "metadatas": [flatten(asdict(d.metadata)) for d in items],
            "documents": [d.document for d in items],
        }
    
def flatten(d: dict, parent: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out
    
@dataclass
class QueryToDB:
    query_embeddings: ndarray | None
    include:list[Literal["metadatas","documents","distances"]]
    n_results:int
    ids: None = None
    where: None = None
    where_document: None = None

class VecDB:
    def __init__(self) -> None:
        self.collection = None    
        self.client = chromadb.PersistentClient(
            path=ENV.db.path
        )

    def __repr__(self) -> str:
        return f"client is currently {self.client}"
    
    def createCollection(self,name):
        self.collection = self.client.get_or_create_collection(name)
        
    def add_data(self,data:DataToDB | list[DataToDB]):
        if self.collection is None:
            raise ValueError(f"collection is {self.collection}","you need to call createcollection method first")      
        self.collection.add(**DataToDB.to_chroma(data))
        
    def getcollection(self,limit:int | None =10,skip:int | None=0):
        return [c.name for c in self.client.list_collections(limit=limit,offset=skip)]
    
    def query_data(self,query:QueryToDB):
        if self.collection is None:
           raise ValueError(f"collection is {self.collection}","you need to call createcollection method first")
        
        return self.collection.query(**asdict(query))
    
    def delete_collection(self,name:str)->bool:
        if self.collection is None:
            return False
             
        collection_names = set(self.getcollection(limit=None))
        if name not in collection_names:
            return False
        self.client.delete_collection(name)
        return True
    
    def delete_data(self,where:DOCMetadata):
        if self.collection is None:
            return
        res = {k: v for k, v in asdict(where).items() if v is not None}
        self.collection.delete(where=res)