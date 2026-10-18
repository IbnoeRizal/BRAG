import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True, slots=True)
class DB:
   path:str

@dataclass(frozen=True, slots=True)
class Model:
   model:str
   path:str

@dataclass(frozen=True, slots=True)
class ENVAR:
   db:DB
   model_embd:Model
   model_rerank:Model
   model_generation:Model


ENV = ENVAR(
   db=DB(
      path = os.environ.get("CHROMA_DB_PATH","db")
   ),
   model_embd = Model(
      model=os.environ.get("ENCODER_MODEL","intfloat/multilingual-e5-small"),
      path=os.environ.get("ENCODER_MODEL_PATH","savepoint/small")
   ),
   model_rerank=Model(
      model=os.environ.get("RERANKER_MODEL","Alibaba-NLP/gte-multilingual-reranker-base"),
      path=os.environ.get("RERANKER_MODEL_PATH","savepoint/reranker")
   ),
   model_generation=Model(
      model=os.environ.get("GENERATION_MODEL","bigscience/bloomz-560m"),
      path=os.environ.get("GENERATION_MODEL_PATH","savepoint/generation"),
   )
)