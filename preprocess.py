import re

# Bangla Unicode block (U+0980-U+09FF) theke shudhu letter/matra rakhi.
# Bangla digit (০-৯) ar punctuation remove kori.
_URL = re.compile(r"http\S+|www\.\S+")
_NON_BANGLA = re.compile(r"[^\u0980-\u09E5\u09F0-\u09FF\s]")  # \u09E6-\u09EF = Bangla digits bad
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = str(text)
    text = _URL.sub(" ", text)
    text = _NON_BANGLA.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
