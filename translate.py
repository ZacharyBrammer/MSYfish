import json


class Translator:
    def __init__(self, lang="en", path="new_labels.json"):
        with open(path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)
        self.lang = lang

    def set_lang(self, lang):
        self.lang = lang

    def translate(self, *keys, default=None, **kwargs):
        node = self.labels

        for key in keys:
            if not isinstance(node, dict):
                return default or keys[-1]
            node = node.get(key)
            if node is None:
                return default or keys[-1]
        
        # If ended at a language dictionary
        if isinstance(node, dict):
            value = node.get(self.lang, default or keys[-1])
        else:
            value = node

        try:
            return value.format(**kwargs)
        except (AttributeError, KeyError):
            return value
