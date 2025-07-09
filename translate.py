import json

class Translator:
    def __init__(self, lang="en", path="labels.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.lang = lang
    
    def set_lang(self, lang):
        self.lang = lang
    
    def translate(self, key, **kwargs):
        value = self.labels.get(key, {}).get(self.lang, key)
        return value.format(**kwargs)

    def option(self, key):
        return self.labels.get("options", {}).get(key, {}).get(self.lang, key)

    def var(self, key, dat) -> str:
        return self.labels.get("variables", {}).get(key, {}).get(dat, {}).get(self.lang, key)