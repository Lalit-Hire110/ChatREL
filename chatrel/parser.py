"""
WhatsApp Chat Parser for ChatREL v4
Adapted from v3_1 - Robust parser handling multiline messages, media, system messages
"""

import re
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Unicode quirks
NBSP = "\u00A0"
NNBSP = "\u202F"
ZWSP = "\u200B"
LRM = "\u200E"
RLM = "\u200F"
BOM = "\ufeff"
DASHES = "-\u2013\u2014"

# Message regex pattern (12h or 24h format)
MESSAGE_RE = re.compile(
    rf"""^
    (?P<date>\d{{1,2}}/\d{{1,2}}/\d{{2,4}})
    ,\s*
    (?P<time>\d{{1,2}}:\d{{2}})
    (?:[ \t{NBSP}{NNBSP}]?(?P<ampm>[AaPp][Mm]))?
    \s*[{DASHES}]\s*
    (?P<sender>[^:]+):
    \s*
    (?P<message>.*)
    $""",
    re.VERBOSE,
)

# System message patterns
SYSTEM_SNIPPETS = [
    "Messages and calls are end-to-end encrypted",
    "message was deleted",
    "You deleted this message",
    "changed the subject",
    "added you",
    "left",
    "joined using this group",
    "created group",
    "changed this group",
    "security code changed",
    "Missed voice call",
    "Missed video call",
]

# Media patterns
MEDIA_SNIPPETS = [
    "<Media omitted>", "image omitted", "video omitted",
    "audio omitted", "document omitted", "GIF omitted",
    "sticker omitted", "contact card omitted"
]


def _strip_weird_unicode(s: str) -> str:
    """Normalize WhatsApp quirks: remove invisible chars, unify spaces/dashes."""
    if not s:
        return s
    s = s.replace(BOM, "")
    s = s.replace(LRM, "").replace(RLM, "")
    s = s.replace(ZWSP, " ")  # Replace with space, not remove
    s = s.replace(NBSP, " ").replace(NNBSP, " ")
    s = re.sub(r"[\u2013\u2014]", "-", s)
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def _looks_like_system(line: str) -> bool:
    """Check if line is a system message."""
    low = line.lower()
    return any(snippet.lower() in low for snippet in SYSTEM_SNIPPETS)


def _is_media(msg: str) -> bool:
    """Check if message is a media placeholder."""
    low = msg.lower()
    return any(snip.lower() in low for snip in MEDIA_SNIPPETS)


class WhatsAppParser:
    """Parse WhatsApp chat exports into structured DataFrame."""
    
    def __init__(self):
        pass
    
    def parse_file(self, file_path: str) -> pd.DataFrame:
        """Parse WhatsApp export file from disk."""
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err: Optional[Exception] = None
        
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc, errors="replace") as f:
                    text = f.read()
                return self.parse_text(text)
            except Exception as e:
                last_err = e
                continue
        
        raise ValueError(f"Failed to read/parse file {file_path}: {last_err}")
    
    def parse_text(self, text: str) -> pd.DataFrame:
        """Parse WhatsApp export text already loaded in memory."""
        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        messages: List[Dict] = []
        current: Optional[Dict] = None
        
        for i, raw in enumerate(lines, start=1):
            line = _strip_weird_unicode(raw)
            if not line:
                continue
            if _looks_like_system(line):
                continue
            
            m = MESSAGE_RE.match(line)
            if m:
                # Save previous message
                if current:
                    messages.append(self._finalize(current, len(messages)))
                
                date_str = m.group("date")
                time_str = m.group("time")
                ampm = m.group("ampm")
                sender = m.group("sender").strip()
                message = m.group("message").strip()
                
                try:
                    ts = self._parse_timestamp(date_str, time_str, ampm)
                except ValueError as e:
                    logger.warning(f"Line {i}: {e}; skipping")
                    continue
                
                current = {
                    "timestamp": ts,
                    "sender": sender,
                    "text": message,
                    "line_number": i,
                }
            else:
                # Multiline continuation
                if current is not None:
                    current["text"] += "\n" + line
                else:
                    logger.debug(f"Orphaned line {i}: {line[:80]}")
        
        # Save last message
        if current:
            messages.append(self._finalize(current, len(messages)))
        
        if not messages:
            raise ValueError("No valid messages found - check export format")
        
        df = pd.DataFrame(messages).sort_values("timestamp").reset_index(drop=True)
        logger.info(f"Parsed {len(df)} messages from {df['sender'].nunique()} senders")
        return df
    
    def _parse_timestamp(self, date_str: str, time_str: str, ampm: Optional[str]) -> datetime:
        """Parse timestamp with multiple format attempts."""
        norm_date = _strip_weird_unicode(date_str)
        norm_time = _strip_weird_unicode(time_str)
        norm_ampm = _strip_weird_unicode(ampm) if ampm else None
        
        if norm_ampm:
            fmts = ["%d/%m/%y, %I:%M %p", "%d/%m/%Y, %I:%M %p", 
                    "%m/%d/%y, %I:%M %p", "%m/%d/%Y, %I:%M %p"]
            candidates = [f"{norm_date}, {norm_time} {norm_ampm.upper()}"]
        else:
            fmts = ["%d/%m/%y, %H:%M", "%d/%m/%Y, %H:%M",
                    "%m/%d/%y, %H:%M", "%m/%d/%Y, %H:%M"]
            candidates = [f"{norm_date}, {norm_time}"]
        
        for s in candidates:
            for fmt in fmts:
                try:
                    return datetime.strptime(s, fmt)
                except ValueError:
                    continue
        
        raise ValueError(f"Unrecognized timestamp: '{date_str}, {time_str}{' ' + (ampm or '')}'")
    
    def _finalize(self, msg: Dict, msg_id: int) -> Dict:
        """Add computed fields to message."""
        text = msg["text"]
        msg["msg_id"] = msg_id
        msg["is_media"] = _is_media(text)
        msg["is_system"] = _looks_like_system(text)
        msg["word_count"] = len(text.split()) if text else 0
        msg["char_count"] = len(text) if text else 0
        return msg


def validate_format(file_path: str, min_hits: int = 3) -> tuple[bool, str]:
    """
    Validate WhatsApp export format.
    Returns (is_valid, reason).
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        return False, f"Could not read file: {e}"
    
    hits = 0
    for raw in text.splitlines():
        line = _strip_weird_unicode(raw)
        if not line or _looks_like_system(line):
            continue
        if MESSAGE_RE.match(line):
            hits += 1
            if hits >= min_hits:
                return True, "Format appears valid"
    
    return False, "Not enough lines match expected WhatsApp format"


if __name__ == "__main__":
    # Test parser
    parser = WhatsAppParser()
    print("WhatsApp Parser Test")
    print("Expected format: dd/mm/yy, hh:mm - Name: Message")
    print("\nExample:")
    print("25/12/23, 14:30 - Alice: Hey! How are you? 😊")
    print("25/12/23, 14:32 - Bob: I'm great! ❤️")
