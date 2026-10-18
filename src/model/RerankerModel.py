from config import ENV
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os

class Model_Reranker:
    
    def __init__(self,name:str,path:str) -> None:
        self.name = name.strip()
        self.path = os.path.normpath((path.strip()))
        self._createmodel()


    def _createmodel(self):
        fd_exist = os.path.exists(self.path)
        name_or_path = self.name
                
        if not fd_exist:
            os.mkdir(self.path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            cache_dir=self.path
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=name_or_path,
            trust_remote_code=True,
            dtype=torch.float16,
            cache_dir=self.path
        )

    def rank(self,pairs:list[list[str]] | list[str])->torch.Tensor:
        npairs = [pairs] if isinstance(pairs[0], str) else pairs

        with torch.no_grad():
            inputs = self.tokenizer(npairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            return self.model(**inputs, return_dict=True).logits.view(-1).float()

if __name__ == "__main__":

    model = Model_Reranker(ENV.model_rerank.model,ENV.model_rerank.path)

    # Define the query and candidate documents
    query = "How to build a deep learning model?"
    documents = [
        "A guide on training neural networks from scratch using PyTorch and TensorFlow.",
        "The best recipes for baking chocolate chip cookies in a traditional oven.",
        "An introductory overview of deep learning techniques and model optimization."
    ]

    # Construct query-document pairs
    pairs = [[query, doc] for doc in documents]

    # Run the reranker model
    scores = model.rank(pairs)

    # Print the results ranked by their relevance scores
    ranked_results = sorted(zip(documents, scores.tolist()), key=lambda x: x[1], reverse=True)
    for res in ranked_results:
        print(res[0])
        print(res[1],end="\n\n")
