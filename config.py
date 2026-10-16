import os
from dataclasses import dataclass

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


ENV = ENVAR(
   db=DB(
      path = os.environ.get("CHROMA_DB_PATH","db")
   ),
   model_embd = Model(
      model=os.environ.get("ENCODER_MODEL","intfloat/multilingual-e5-small"),
      path=os.environ.get("ENCODER_MODEL_PATH","savepoint")
   )
)