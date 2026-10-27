"""
Internationalization (i18n) and localization service.
Handles translations between English and Hindi with graceful English fallbacks.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json

LOCALES_DIR = Path(__file__).resolve().parent / "locales"


def load_locale(lang: str = "en") -> Dict[str, Any]:
    """Loads JSON locale file for specified language code."""
    file_path = LOCALES_DIR / f"{lang}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """
    Translates a translation key into the target language.
    Falls back to English if the key is missing in the target locale.
    Never returns mixed bilingual text.
    """
    locales = load_locale(lang)
    val = locales.get(key)
    if not val and lang != "en":
        # Fallback to English
        en_locales = load_locale("en")
        val = en_locales.get(key)

    if not val:
        val = key

    if kwargs and isinstance(val, str):
        try:
            return val.format(**kwargs)
        except Exception:
            return val
    return val
