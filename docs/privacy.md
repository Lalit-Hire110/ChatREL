# Privacy and Data Handling

## Overview

ChatREL v4 processes personal chat data. This document explains:
- What data is collected
- How it's processed
- Where it's sent
- How to enhance privacy

---

## Data Flow

```
WhatsApp Export → Parser → HF API (Sentiment/Toxicity) → Aggregator → Results
                             ↓
                          Cache (SQLite)
```

---

## What Data is Sent to HuggingFace

### Text Messages

By default, **message text is sent to HuggingFace Inference API** for:
- Sentiment analysis (`cardiffnlp/twitter-xlm-roberta-base-sentiment`)
- Toxicity detection (`textdetox/xlmr-large-toxicity-classifier`)

**What's sent:**
- Message text (verbatim)
- No sender names
- No timestamps
- No metadata

**Example:**
```
Input: "I love you yaar ❤️"
Sent to HF: "I love you yaar ❤️"
```

### Batching

Messages are sent in batches (default: 16 messages per request) to reduce API calls.

---

## What's Stored Locally

### Cache Database

- **Location**: `.cache/responses.db` (SQLite)
- **Contents**: 
  - Model name
  - Message text (hashed SHA256 as key)
  - API response JSON
  - Timestamp
- **TTL**: 7 days (configurable)
- **Purpose**: Avoid redundant API calls

**Example cache entry:**
```sql
key: "a3f5e8... (SHA256 hash)"
model_name: "cardiffnlp/twitter-xlm-roberta-base-sentiment"
text: "I love you yaar ❤️"
response: {"label": "positive", "score": 0.92}
timestamp: "2025-11-28T10:30:00"
```

### Uploads Directory

- **Location**: `uploads/`
- **Contents**: Uploaded WhatsApp .txt files (temporary)
- **Retention**: Manual cleanup required
- **Recommendation**: Delete after analysis

---

## Privacy Controls

### 1. Pseudonymization (Optional)

Enable in `.env`:
```env
PSEUDONYMIZE_NAMES=True
```

**What it does:**
- Hashes sender names before API calls
- Masks phone numbers (replaces with `[PHONE]`)
- Removes email addresses

**Example:**
```
Original:  "Hey Rahul, call me on 9876543210"
Pseudonymized: "Hey [SENDER_1], call me on [PHONE]"
```

**Tradeoff:** May reduce sentiment accuracy for name-based emotions ("I miss Rahul" → "I miss [SENDER_1]")

### 2. Disable Raw Text Sending

Enable in `.env`:
```env
SEND_RAW_TEXT_TO_HF=False
```

**What it does:**
- Applies aggressive pseudonymization before API calls
- Hashes all names, numbers, emails
- Preserves emoji and general structure

**Tradeoff:** Significantly reduced accuracy. Use only for highly sensitive data.

### 3. Mock Mode (No API Calls)

Run CLI/Flask with mock mode:
```bash
python -m chatrel.cli analyze chat.txt --mock
```

**What it does:**
- No data sent to HuggingFace
- Uses simulated responses from `mock_responses.json`
- Useful for testing or offline analysis

---

## HuggingFace Privacy Policy

HuggingFace Inference API:
- **Does NOT store** inference inputs/outputs permanently
- **Logs** may include request metadata (timestamp, model name) but not content
- **Privacy Policy**: https://huggingface.co/privacy

**Recommendation:** Review HF privacy policy before using with sensitive data.

---

## User Consent Requirements

### Before Analyzing Someone's Chat

✅ **DO:**
- Obtain explicit consent from both parties
- Explain data is sent to HuggingFace API
- Offer pseudonymization options
- Allow opt-out

❌ **DON'T:**
- Analyze private chats without permission
- Share analysis results without consent
- Use for surveillance or monitoring

### Recommended Consent Flow

1. Inform: "This tool uses HuggingFace API for sentiment analysis"
2. Explain: "Your messages will be sent to external servers (anonymized)"
3. Offer: "Enable pseudonymization? (reduces accuracy)"
4. Confirm: "Do you consent to proceed?"

---

## Data Minimization

### Best Practices

1. **Delete uploads after analysis:**
   ```bash
   rm -rf uploads/*
   ```

2. **Clear cache periodically:**
   ```python
   from chatrel.cache import ResponseCache
   cache = ResponseCache()
   cache.clear_all()
   ```

3. **Use small windows:**
   - Analyze last 100 messages instead of full history
   - Reduces data exposure

4. **Avoid logging sensitive messages:**
   - Set `logging.level = WARNING` to reduce log verbosity

---

## GDPR Compliance Considerations

If deploying ChatREL v4 in EU/EEA:

### User Rights

- **Right to Access**: Users can request their cached data
- **Right to Deletion**: Implement cache/upload purge
- **Right to Portability**: Export analysis results as JSON

### Data Processing Agreement

- HuggingFace is "Data Processor"
- You are "Data Controller"
- Ensure GDPR-compliant Data Processing Agreement with HF

### Recommendations

1. Add consent banner to web UI
2. Implement user data export/deletion
3. Log all API calls for audit trail
4. Privacy notice on homepage

---

## Security

### API Token Protection

- **NEVER commit `.env` to Git**
- Use `.env.example` as template
- Rotate HF tokens periodically
- Use read-only tokens if possible

### Web UI Security

- **No authentication** in default Flask app
- **Recommendation**: Add login for production
- Use HTTPS in production (not HTTP)
- Set `app.secret_key` for sessions

### Cache Security

- Cache database is **unencrypted**
- Store `.cache/` directory with restricted permissions:
  ```bash
  chmod 700 .cache
  ```
- Consider encrypting cache for sensitive deployments

---

## Compliance Checklist

Before deploying ChatREL v4:

- [ ] Review HuggingFace privacy policy
- [ ] Add consent mechanism to UI
- [ ] Enable pseudonymization (if applicable)
- [ ] Set cache TTL to minimum required
- [ ] Implement upload cleanup automation
- [ ] Add privacy notice to web UI
- [ ] Secure API token storage
- [ ] Use HTTPS in production
- [ ] Audit log for API calls (optional)
- [ ] GDPR DPA with HuggingFace (if in EU)

---

## Questions?

For privacy concerns:
- Email: privacy@yourorganization.com
- Issue tracker: github.com/yourrepo/ChatREL_v4/issues

**When in doubt, prioritize privacy over accuracy.**
