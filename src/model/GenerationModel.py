from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class Model_Generation:
    def __init__(self,name:str,path:str) -> None:
        self.path = os.path.normpath(path.strip())
        self.name = name.strip()
        self._buildmodel()
    
    def _buildmodel(self):
        fd_exist = os.path.exists(self.path)
                
        if not fd_exist:
            os.mkdir(self.path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.name,
            cache_dir=self.path
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.name,
            cache_dir=self.path
        )

    def generate(self, query: str, context: str) -> str:
        prompt = (
            f"Teks: {context.strip()}\n\n"
            "Berdasarkan teks di atas, jawab pertanyaan berikut dengan singkat.\n"
            f"Pertanyaan: {query.strip()}\n"
            "Jawaban:"
        )
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inp,
            max_new_tokens=128,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        new_tokens = outputs[0][inp["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    