# Heuristics and Scoring Formulas

## Overview

ChatREL v4 uses a hybrid approach combining:
1. **ML Models** (HuggingFace): Sentiment & Toxicity detection
2. **Local Heuristics**: Emoji, laughter, code-mixing, slang
3. **Sensor Fusion**: Weighted combination of model outputs + heuristics

---

## Emoji Mapping

### Romantic (+1.0)
❤️, 😍, 💖, 💕, 💘, 💓, 💝, 💞, 😘, 🥰, 💗, 💌, 💋

### Affectionate (+0.6-0.8)
😊, 😇, 💛, 💙, 💚, 💜, 🫶, 🌹, 🌸, 💐, 🫂

### Playful/Laughter (+0.3-0.4)
😂, 🤣, 😹, 😝, 😜, 😛, 🙃, 😏, 🤪

### Positive (+0.4-0.6)
👍, 👏, 💯, ✅, 🙏, 🎉, ✨, 💫, 🏆, 🥳, 💪

### Neutral (0.0)
😐, 😑, 😶, 🤔, 😌, 😴, 💀

### Negative (-0.6 to -1.0)
😡, 🤬, 😠, 👿, 💢, 😤, 😾, 👎, ☹️, 😖

### Sad (-0.4 to -0.9)
😢, 😭, 😞, 🥺, 😿, 💔, 😪, 😥, 😓, 😰, 😟, 😔

---

## Message-Level Fusion Rules

### Sentiment Combination

```python
# 1. Get HF sentiment (categorical)
if label == "positive":
    sentiment_numeric = score  # 0 to 1
elif label == "negative":
    sentiment_numeric = -score  # -1 to 0
else:
    sentiment_numeric = 0

# 2. Extract emoji valence
emoji_valence = mean([EMOJI_LEXICON[e] for e in emojis])

# 3. Combine (70% model, 30% emoji)
combined = 0.7 * sentiment_numeric + 0.3 * emoji_valence

# 4. Slang adjustment
if has_slang and sentiment_numeric < 0:
    combined -= 0.1  # Increase negative weight

# 5. Uncertainty downweighting
if sentiment_confidence < 0.45:
    combined *= 0.5  # Reduce impact of uncertain predictions

# Result: combined_sentiment in [-1, 1]
```

### Conflict Detection

```python
conflict = (toxicity_score > 0.7)
```

### Teasing Detection (Sarcasm)

```python
# Negative sentiment + laughter + low toxicity = teasing (not conflict)
teasing = (
    sentiment_label == "negative"
    and laughter_flag  # Contains 😂, lol, haha, etc.
    and toxicity_score < 0.2
)

if teasing:
    conflict = False  # Override conflict flag
```

### Code-Mixing Detection

```python
# Simple keyword-based detection
hinglish_tokens = ["yaar", "kya", "bhai", "hai", "accha", ...]
marathi_tokens = ["ahe", "kay", "zhala", "bara", "mhanun", ...]

is_code_mixed = any(token in text.lower() for token in hinglish_tokens + marathi_tokens)
```

---

## Aggregate Metrics

### Sentiment Metrics

```python
mean_sentiment = mean(combined_sentiment)
median_sentiment = median(combined_sentiment)
sd_sentiment = std(combined_sentiment)
percent_positive = % messages with combined_sentiment > 0.2
percent_negative = % messages with combined_sentiment < -0.2
```

### Toxicity Metrics

```python
avg_toxicity = mean(toxicity_score)
max_toxicity = max(toxicity_score)
toxicity_spike_count = count(toxicity_score > 0.7)
```

### Emoji Metrics

```python
emoji_density = total_emojis / total_messages
positive_emojis = count(emoji_valence > 0.3)
emoji_affinity = positive_emojis / total_emojis
```

### Reciprocity

```python
# For top 2 senders
counts = [sender1_messages, sender2_messages]
imbalance = abs(counts[0] - counts[1]) / sum(counts)
reciprocity = 1 - imbalance  # Range: [0, 1]
```

### Reply Times

```python
# Compute time diff between consecutive messages from different senders
reply_times = [time_diff where sender_changed]
median_reply_time_sec = median(reply_times)
percent_replies_under_1h = % reply_times < 3600 seconds
```

### Engagement

```python
date_range = max(timestamp) - min(timestamp)
msgs_per_day = total_messages / date_range.days
avg_words = mean(word_count)
```

---

## Sub-Scores (0-1)

### Warmth Score

```python
warmth = (
    0.5 * normalize(percent_positive, 0, 80)  # High % positive
    + 0.3 * emoji_affinity                     # Positive emojis
    + 0.2 * normalize(mean_sentiment, -0.5, 0.5)  # Average sentiment
)
```

### Conflict Score

```python
conflict = (
    0.5 * avg_toxicity                        # Toxicity level
    + 0.3 * normalize(percent_negative, 0, 50)  # % negative messages
    + 0.2 * min(1, toxicity_spike_count / 5)  # Toxicity spikes
)
```

### Engagement Score

```python
engagement = (
    0.5 * normalize(msgs_per_day, 0, 20)      # Messages per day
    + 0.3 * reciprocity                        # Balance
    + 0.2 * normalize(percent_replies_under_1h, 0, 80)  # Fast replies
)
```

### Stability Score

```python
stability = 1 - min(1, sd_sentiment / 0.5)   # Low variance = high stability
```

---

## Overall Health (0-100)

```python
# Configurable weights (default shown)
WARMTH_WEIGHT = 0.45
ENGAGEMENT_WEIGHT = 0.30
CONFLICT_WEIGHT = 0.20
STABILITY_WEIGHT = 0.05

raw_score = (
    WARMTH_WEIGHT * warmth
    + ENGAGEMENT_WEIGHT * engagement
    - CONFLICT_WEIGHT * conflict  # Negative contribution
    + STABILITY_WEIGHT * stability
)

overall_health = 100 * clip(raw_score, 0, 1)
```

---

## Relationship Type Classification

### Heuristic Rules

#### Couple
```python
if (
    warmth >= 0.7
    and emoji_affinity >= 0.6
    and reciprocity >= 0.5
):
    type = "Couple"
    confidence = 0.8 + 0.2 * warmth
```

#### Crush
```python
if (
    warmth >= 0.65
    and emoji_affinity >= 0.5
    and reciprocity < 0.7  # Asymmetric
):
    type = "Crush"
    confidence = 0.7 + 0.3 * (1 - reciprocity)
```

#### Friend
```python
if (
    0.4 <= warmth < 0.65
    and engagement >= 0.5
):
    type = "Friend"
    confidence = 0.6 + 0.4 * engagement
```

#### Family
```python
if (
    0.3 <= warmth < 0.6
    and avg_words >= 10
    and emoji_density < 0.5
):
    type = "Family"
    confidence = 0.5 + 0.5 * (avg_words / 20)
```

**Priority**: Couple > Crush > Friend > Family (default)

---

## Normalize Function

```python
def normalize(value, min_val, max_val, clip=True):
    """Map [min_val, max_val] to [0, 1]"""
    normalized = (value - min_val) / (max_val - min_val)
    if clip:
        return max(0, min(1, normalized))
    return normalized
```

---

## Notes

- **Hinglish/Marathi**: Models may produce noisy outputs for romanized scripts. Local heuristics (emoji, laughter, slang) help compensate.
- **Confidence Thresholds**: Sentiment predictions with `score < 0.45` are marked as uncertain and downweighted.
- **Toxicity Threshold**: `toxicity > 0.7` triggers conflict flag.
- **Teasing Override**: Negative + laughter + low toxicity prevents false conflict detection.

---

## Future Improvements

1. **Learned Classifier**: Replace heuristic relationship classifier with ML model
2. **Fine-Tuned Models**: Domain-adapt sentiment model on 500 labeled Hinglish messages
3. **Emotion Detection**: Add joy/anger/sadness dimensions
4. **Transliteration**: Convert Roman Marathi → Devanagari before API calls
