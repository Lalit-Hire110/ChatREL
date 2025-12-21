import argparse
import sqlite3
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from chatrel import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("clear_demo_cache")

def clear_cache(profile: str, chat_hash: str = None, all_profiles: bool = False):
    """Clear demo cache entries."""
    conn = sqlite3.connect(config.CSM_DB_PATH)
    cursor = conn.cursor()
    
    try:
        if all_profiles:
            cursor.execute("DELETE FROM demo_chat_cache")
            logger.info(f"Cleared ALL demo cache entries ({cursor.rowcount} rows)")
        elif chat_hash:
            cursor.execute(
                "DELETE FROM demo_chat_cache WHERE chat_hash = ? AND demo_profile = ?",
                (chat_hash, profile)
            )
            logger.info(f"Cleared cache for hash={chat_hash} profile={profile} ({cursor.rowcount} rows)")
        else:
            cursor.execute(
                "DELETE FROM demo_chat_cache WHERE demo_profile = ?",
                (profile,)
            )
            logger.info(f"Cleared cache for profile={profile} ({cursor.rowcount} rows)")
            
        conn.commit()
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear demo chat cache")
    parser.add_argument("--profile", default="demo_v1", help="Demo profile name")
    parser.add_argument("--chat_hash", help="Specific chat hash to clear")
    parser.add_argument("--all", action="store_true", help="Clear ALL entries regardless of profile")
    
    args = parser.parse_args()
    
    clear_cache(args.profile, args.chat_hash, args.all)
