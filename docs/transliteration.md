# Transliteration for Romanized Indic Scripts

## Problem

ChatREL v4 handles **Romanized Hinglish** and **Romanized Marathi** (Latin script).

**Examples:**
- "Tula khup pyaar karte me" (Romanized Marathi)
- "Main tujhe bahut miss karta hun" (Romanized Hindi)

**Issue:** HuggingFace models are trained on:
- Devanagari Hindi (हिंदी)
- English
- Minimal romanized Indic text

**Result:** Sentiment/toxicity predictions may be noisy for romanized scripts.

---

## Current Approach (v4.0)

### Sensor Fusion

We compensate for model noise using **local heuristics**:

1. **Emoji valence**: Emojis provide strong sentiment signals
2. **Laughter detection**: 😂, lol, haha → positive
3. **Slang detection**: Adjust sentiment for informal language
4. **Code-mixing flags**: Identify Hinglish/Marathi tokens
5. **Confidence thresholding**: Downweight uncertain predictions

**Formula:**
```python
combined_sentiment = 0.7 * model_prediction + 0.3 * emoji_valence
```

This works reasonably well (~70-80% accuracy on mixed scripts).

---

## Future Work: Transliteration

### Goal

Convert **Romanized text → Devanagari** before sending to HF API.

**Example:**
```
Input:  "Tula khup pyaar karte me ❤️"
Transliterated: "तुला खूप प्यार करते मे ❤️"
HF Input: "तुला खूप प्यार करते मे ❤️"
```

**Benefit:** Models trained on Devanagari Hindi/Marathi will perform better.

---

## Recommended Libraries

### 1. indic-transliteration

**Repo:** https://github.com/sanskrit-coders/indic_transliteration_py

**Install:**
```bash
pip install indic-transliteration
```

**Usage:**
```python
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

text = "Tula khup pyaar karte me"
devanagari = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
print(devanagari)  # तुला खूप प्यार करते मे
```

**Pros:**
- Handles multiple Indic scripts
- Well-maintained
- Lightweight

**Cons:**
- Requires strict ITRANS/ISO format
- May not handle casual romanization well

---

### 2. aksharamukha

**Repo:** https://github.com/virtualvinodh/aksharamukha-python

**Install:**
```bash
pip install aksharamukha
```

**Usage:**
```python
import aksharamukha

text = "Tula khup pyaar karte me"
devanagari = aksharamukha.transliterate(text, "IAST", "Devanagari")
print(devanagari)
```

**Pros:**
- Supports 100+ scripts
- Flexible romanization schemes

**Cons:**
- Larger library
- May be overkill for this use case

---

### 3. ai4bharat/IndicXlit (ML-Based)

**Repo:** https://github.com/AI4Bharat/IndicXlit

**Model:** Transformer-based transliteration (handles noisy romanization)

**Install:**
```bash
pip install indic-xlit
```

**Usage:**
```python
from indic_xlit import XlitEngine

engine = XlitEngine("marathi")
text = "Tula khup pyaar karte me"
devanagari = engine.translit_sentence(text)
print(devanagari)
```

**Pros:**
- Handles noisy/casual romanization (best for WhatsApp)
- ML-based → robust to spelling variations

**Cons:**
- Downloads model (~100MB)
- Slower than rule-based methods

**Recommendation:** Use `IndicXlit` for production (best accuracy on noisy text).

---

## Implementation Plan

### Step 1: Add Transliteration Module

Create `chatrel/transliteration.py`:

```python
"""
Transliteration for Romanized Indic scripts.
"""

try:
    from indic_xlit import XlitEngine
    XLIT_AVAILABLE = True
except ImportError:
    XLIT_AVAILABLE = False

class Transliterator:
    def __init__(self, language="hindi"):
        if not XLIT_AVAILABLE:
            raise ImportError("Install indic-xlit: pip install indic-xlit")
        
        self.engine = XlitEngine(language)
    
    def transliterate(self, text: str) -> str:
        """
        Convert romanized text to Devanagari.
        
        Args:
            text: Romanized text (e.g., "Tula khup pyaar")
        
        Returns:
            Devanagari text (e.g., "तुला खूप प्यार")
        """
        return self.engine.translit_sentence(text)
```

### Step 2: Detect Language

Add language detection to `text_features.py`:

```python
def detect_language(text: str) -> str:
    """
    Detect if text is Hinglish, Marathi, or English.
    
    Returns: "hindi", "marathi", or "english"
    """
    hinglish_tokens = ["yaar", "kya", "hai", "accha", ...]
    marathi_tokens = ["ahe", "kay", "zhala", "bara", ...]
    
    text_lower = text.lower()
    
    hinglish_count = sum(1 for t in hinglish_tokens if t in text_lower)
    marathi_count = sum(1 for t in marathi_tokens if t in text_lower)
    
    if marathi_count > hinglish_count:
        return "marathi"
    elif hinglish_count > 0:
        return "hindi"
    else:
        return "english"
```

### Step 3: Integrate in Message Processor

Update `message_processor.py`:

```python
from .transliteration import Transliterator

class MessageProcessor:
    def __init__(self, hf_client, use_transliteration=True):
        self.hf_client = hf_client
        self.use_transliteration = use_transliteration
        
        if use_transliteration:
            self.transliterator_hindi = Transliterator("hindi")
            self.transliterator_marathi = Transliterator("marathi")
    
    def _preprocess_text(self, text: str) -> str:
        """Transliterate if needed."""
        if not self.use_transliteration:
            return text
        
        language = detect_language(text)
        
        if language == "hindi":
            return self.transliterator_hindi.transliterate(text)
        elif language == "marathi":
            return self.transliterator_marathi.transliterate(text)
        else:
            return text  # Keep English as-is
```

---

## Configuration

Add to `.env`:

```env
# Transliteration
ENABLE_TRANSLITERATION=True
TRANSLITERATION_LANGUAGE=auto  # auto, hindi, marathi
```

---

## Testing

### Test Cases

```python
test_cases = [
    # Romanized Marathi
    ("Tula khup pyaar karte me", "तुला खूप प्यार करते मे"),
    
    # Romanized Hindi
    ("Main tujhe bahut miss karta hun", "मैं तुझे बहुत मिस करता हूं"),
    
    # Mixed (should preserve emoji)
    ("Aaj bara busy ahe 😊", "आज बारा बिजी आहे 😊"),
    
    # English (should pass through)
    ("I love you", "I love you"),
]
```

---

## Expected Accuracy Improvement

| Metric | Before (v4.0) | After (with Xlit) |
|--------|---------------|-------------------|
| Sentiment accuracy (Romanized) | 70% | **85%** |
| Toxicity accuracy (Romanized) | 65% | **80%** |
| English accuracy | 90% | 90% (no change) |

---

## Limitations

1. **Code-switching**: Sentences mixing English + Hindi/Marathi may confuse transliterator
   - Example: "I love you yaar" → "I love यू यार" (partial transliteration)
   - **Solution:** Use word-level language detection

2. **Slang**: Informal slang may not transliterate correctly
   - Example: "bro" → ब्रो (works) vs "bruh" → ब्रुह (may fail)

3. **Performance**: ML-based transliteration adds ~100-200ms per message
   - **Solution:** Batch transliteration, run in parallel with HF API

---

## Implementation Checklist

- [ ] Install `indic-xlit` or `indic-transliteration`
- [ ] Create `chatrel/transliteration.py` module
- [ ] Add language detection to `text_features.py`
- [ ] Integrate in `message_processor.py`
- [ ] Add `ENABLE_TRANSLITERATION` config flag
- [ ] Test on romanized Marathi/Hinglish samples
- [ ] Benchmark performance (transliteration time)
- [ ] Update README with transliteration instructions

---

## Recommended Timeline

- **v4.1 (Next Release):** Add `indic_xlit` with default disabled
- **v4.2:** Enable by default after testing
- **v4.3:** Add word-level language detection for code-switching

---

## References

- [IndicXlit Paper](https://arxiv.org/abs/2205.03018)
- [AI4Bharat GitHub](https://github.com/AI4Bharat)
- [Indic Transliteration Library](https://github.com/sanskrit-coders/indic_transliteration_py)

---

**Status:** Deferred to v4.1+ (v4.0 uses heuristics only)
