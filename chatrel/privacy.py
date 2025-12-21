"""
Privacy and PII pseudonymization utilities for ChatREL v4
"""

import re
import hashlib
from typing import Dict, Tuple


class Pseudonymizer:
    """Handle PII pseudonymization for privacy protection."""
    
    def __init__(self):
        """Initialize pseudonymizer with pattern matchers and counters."""
        self.phone_counter = 0
        self.email_counter = 0
        self.number_counter = 0
        self.sender_map: Dict[str, str] = {}
        
        # Regex patterns
        self.phone_pattern = re.compile(r'\b\d{10,15}\b')  # 10-15 digit sequences
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.number_pattern = re.compile(r'\b\d{6,9}\b')  # 6-9 digit sequences (IDs, etc.)
    
    def pseudonymize_text(self, text: str) -> str:
        """
        Pseudonymize text by replacing PII with tokens.
        
        Args:
            text: Original text
        
        Returns:
            Pseudonymized text
        """
        if not text:
            return text
        
        # Replace phones
        text = self.phone_pattern.sub(lambda m: self._get_phone_token(), text)
        
        # Replace emails
        text = self.email_pattern.sub(lambda m: self._get_email_token(), text)
        
        # Replace other long numbers (be conservative to avoid dates/times)
        text = self.number_pattern.sub(lambda m: self._get_number_token(), text)
        
        return text
    
    def pseudonymize_sender(self, sender: str) -> str:
        """
        Map sender name to pseudonymous ID.
        
        Args:
            sender: Original sender name
        
        Returns:
            Pseudonymous ID (e.g., "User_1", "User_2")
        """
        if sender not in self.sender_map:
            # Create deterministic but anonymized ID
            user_num = len(self.sender_map) + 1
            self.sender_map[sender] = f"User_{user_num}"
        
        return self.sender_map[sender]
    
    def get_sender_mapping(self) -> Dict[str, str]:
        """Get the sender name to pseudo-ID mapping."""
        return self.sender_map.copy()
    
    def _get_phone_token(self) -> str:
        """Generate phone number token."""
        self.phone_counter += 1
        return f"<PHONE_{self.phone_counter}>"
    
    def _get_email_token(self) -> str:
        """Generate email token."""
        self.email_counter += 1
        return f"<EMAIL_{self.email_counter}>"
    
    def _get_number_token(self) -> str:
        """Generate number token."""
        self.number_counter += 1
        return f"<NUM_{self.number_counter}>"


def pseudonymize_dataframe(df, columns_to_mask=None):
    """
    Pseudonymize a pandas DataFrame.
    
    Args:
        df: DataFrame with chat messages
        columns_to_mask: List of column names to pseudonymize (default: ['text'])
    
    Returns:
        Tuple of (pseudonymized_df, pseudonymizer instance with mappings)
    """
    import pandas as pd
    
    if columns_to_mask is None:
        columns_to_mask = ['text']
    
    df_pseudo = df.copy()
    pseudo = Pseudonymizer()
    
    # Pseudonymize sender names
    if 'sender' in df_pseudo.columns:
        df_pseudo['sender'] = df_pseudo['sender'].apply(pseudo.pseudonymize_sender)
    
    # Pseudonymize text columns
    for col in columns_to_mask:
        if col in df_pseudo.columns:
            df_pseudo[col] = df_pseudo[col].apply(
                lambda x: pseudo.pseudonymize_text(str(x)) if pd.notna(x) else x
            )
    
    return df_pseudo, pseudo


if __name__ == "__main__":
    # Test pseudonymization
    pseudo = Pseudonymizer()
    
    test_cases = [
        "Call me at 9876543210 or email john@example.com",
        "My ID is 12345678 and phone is +91-9876543210",
        "Meeting at 3:45 PM tomorrow",  # Should not mask times
    ]
    
    print("Pseudonymization Test:")
    for text in test_cases:
        masked = pseudo.pseudonymize_text(text)
        print(f"Original: {text}")
        print(f"Masked:   {masked}\n")
    
    # Test sender mapping
    senders = ["Alice", "Bob", "Alice", "Charlie"]
    print("Sender Mapping:")
    for sender in senders:
        pseudo_id = pseudo.pseudonymize_sender(sender)
        print(f"{sender} → {pseudo_id}")
    
    print(f"\nMapping: {pseudo.get_sender_mapping()}")
