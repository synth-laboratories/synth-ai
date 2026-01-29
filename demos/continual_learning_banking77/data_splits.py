#!/usr/bin/env python3
"""
Define progressive Banking77 data splits for continual learning experiments.

Each split is a superset of the previous:
- Split 1: 2 intents (simple card queries)
- Split 2: 7 intents (expanded card operations)
- Split 3: 27 intents (common banking operations)
- Split 4: 77 intents (full Banking77 dataset)
"""

from __future__ import annotations

from typing import Dict, List, Set

from datasets import load_dataset

# Split 1: 2 intents - Simple card queries
SPLIT_1_INTENTS = [
    "card_arrival",
    "lost_or_stolen_card",
]

# Split 2: 7 intents - Expanded card operations (superset of Split 1)
SPLIT_2_INTENTS = SPLIT_1_INTENTS + [
    "activate_my_card",
    "card_not_working",
    "card_delivery_estimate",
    "pin_blocked",
    "change_pin",
]

# Split 3: 27 intents - Common banking operations (superset of Split 2)
SPLIT_3_INTENTS = SPLIT_2_INTENTS + [
    # Card-related
    "compromised_card",
    "contactless_not_working",
    "declined_card_payment",
    "card_payment_not_recognised",
    "card_swallowed",
    "get_physical_card",
    "order_physical_card",
    "getting_virtual_card",
    "get_disposable_virtual_card",
    # Account & transfers
    "transfer_not_received_by_recipient",
    "failed_transfer",
    "pending_transfer",
    "cancel_transfer",
    "receiving_money",
    # Balance & top-up
    "balance_not_updated_after_bank_transfer",
    "top_up_failed",
    "pending_top_up",
    # Identity & account
    "verify_my_identity",
    "passcode_forgotten",
    "edit_personal_details",
]

# Split 4: All 77 intents (full Banking77 dataset)
ALL_77_INTENTS = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "get_disposable_virtual_card",
    "get_physical_card",
    "getting_spare_card",
    "getting_virtual_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

SPLIT_4_INTENTS = ALL_77_INTENTS

# Mapping of split number to intent list
SPLITS = {
    1: SPLIT_1_INTENTS,
    2: SPLIT_2_INTENTS,
    3: SPLIT_3_INTENTS,
    4: SPLIT_4_INTENTS,
}


def get_split_intents(split_num: int) -> List[str]:
    """Get the list of intents for a given split number."""
    if split_num not in SPLITS:
        raise ValueError(f"Invalid split number: {split_num}. Must be 1-4.")
    return SPLITS[split_num]


def get_split_size(split_num: int) -> int:
    """Get the number of intents in a given split."""
    return len(get_split_intents(split_num))


class Banking77SplitDataset:
    """Banking77 dataset filtered by progressive intent splits.
    
    Loads the Banking77 dataset and provides methods to sample from
    specific splits (subsets of intents).
    """
    
    # Load directly from GitHub CSV
    _DATA_URLS = {
        "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    }
    
    def __init__(self):
        self._cache: Dict[str, list] = {}
        self._split_cache: Dict[tuple, list] = {}
        self._all_labels: List[str] = []
    
    def _load_split(self, data_split: str) -> list:
        """Load a data split (train/test) from the raw CSV."""
        if data_split not in self._cache:
            url = self._DATA_URLS.get(data_split)
            if not url:
                raise ValueError(f"Unknown data split: {data_split}")
            ds = load_dataset("csv", data_files=url, split="train")
            # Convert to list of dicts
            self._cache[data_split] = [
                {"text": row["text"], "category": row["category"]}
                for row in ds
            ]
            # Build label names from unique categories
            if not self._all_labels:
                self._all_labels = sorted(set(row["category"] for row in self._cache[data_split]))
        return self._cache[data_split]
    
    def _get_filtered_data(self, data_split: str, intent_split: int) -> list:
        """Get data filtered to only include intents from the specified split."""
        cache_key = (data_split, intent_split)
        if cache_key not in self._split_cache:
            all_data = self._load_split(data_split)
            allowed_intents = set(get_split_intents(intent_split))
            self._split_cache[cache_key] = [
                row for row in all_data
                if row["category"] in allowed_intents
            ]
        return self._split_cache[cache_key]
    
    def ensure_ready(self, data_splits: List[str] = None):
        """Pre-load data splits."""
        if data_splits is None:
            data_splits = ["train", "test"]
        for split in data_splits:
            self._load_split(split)
    
    def size(self, data_split: str, intent_split: int) -> int:
        """Get the number of samples for a given data/intent split combination."""
        return len(self._get_filtered_data(data_split, intent_split))
    
    def sample(self, *, data_split: str, intent_split: int, index: int) -> dict:
        """Sample a single example from the filtered dataset.
        
        Args:
            data_split: "train" or "test"
            intent_split: 1, 2, 3, or 4 (which intent subset)
            index: Index into the filtered dataset
            
        Returns:
            Dict with 'index', 'split', 'intent_split', 'text', 'label'
        """
        filtered_data = self._get_filtered_data(data_split, intent_split)
        if not filtered_data:
            raise ValueError(f"No data for split {data_split} with intent_split {intent_split}")
        idx = index % len(filtered_data)
        row = filtered_data[idx]
        return {
            "index": idx,
            "split": data_split,
            "intent_split": intent_split,
            "text": row["text"],
            "label": row["category"],
        }
    
    def get_split_labels(self, intent_split: int) -> List[str]:
        """Get the list of intent labels for a given split."""
        return get_split_intents(intent_split)
    
    @property
    def all_labels(self) -> List[str]:
        """Get all 77 intent labels."""
        if not self._all_labels:
            self._load_split("train")
        return self._all_labels


def format_available_intents(label_names: List[str]) -> str:
    """Format intent labels for display in prompts."""
    return "\n".join(f"{i + 1}. {label}" for i, label in enumerate(label_names))


def print_split_info():
    """Print information about the data splits."""
    print("Banking77 Progressive Data Splits")
    print("=" * 60)
    
    dataset = Banking77SplitDataset()
    dataset.ensure_ready()
    
    for split_num in [1, 2, 3, 4]:
        intents = get_split_intents(split_num)
        train_size = dataset.size("train", split_num)
        test_size = dataset.size("test", split_num)
        
        print(f"\nSplit {split_num}: {len(intents)} intents")
        print(f"  Train samples: {train_size}")
        print(f"  Test samples: {test_size}")
        print(f"  Intents: {', '.join(intents[:5])}{'...' if len(intents) > 5 else ''}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_split_info()
