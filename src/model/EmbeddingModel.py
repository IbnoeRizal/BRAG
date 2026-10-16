from sentence_transformers import SentenceTransformer 
import os

class Model_EMB:
    
    def __init__(self,name:str,path:str) -> None:
        self.name = name.strip()
        self.path = os.path.normpath((path.strip()))
        self.model = self._createmodel()

    def _createmodel(self) -> SentenceTransformer:
        fd_exist = os.path.exists(self.path)
                
        if not fd_exist:
            os.mkdir(self.path)

        return SentenceTransformer(
            model_name_or_path=self.name,
            cache_folder=self.path,
            local_files_only=fd_exist
        )