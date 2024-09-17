from typing import Dict, Optional, Type, Union
from dataclasses import dataclass
from pydantic import BaseModel
import json
import sqlite3
import os

@dataclass
class PersistentCache:
    def __init__(self, db_path: str = ".cache/persistent_cache.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS cache
                              (key TEXT PRIMARY KEY, response TEXT)""")
        self.conn.commit()

    def hit_cache(self, key: str, response_model: Type[BaseModel]) -> Optional[Dict]:
        self.cursor.execute("SELECT response FROM cache WHERE key = ?", (key,))
        result = self.cursor.fetchone()
        if not result:
            return None
        result = json.loads(result[0])
        if result and response_model:
            return response_model(**result)
        elif result and not isinstance(result, dict):
            return result
        elif result and isinstance(result, dict) and "response" in result:
            return result["response"]
        return None

    def add_to_cache(self, key: str, response: Union[BaseModel, str]) -> None:
        if isinstance(response, BaseModel):
            response_dict = response.model_dump()
            response_class = response.__class__.__name__
        elif isinstance(response, str):
            response_dict = response
            response_class = None
        else:
            raise ValueError(f"Invalid response type: {type(response)}")

        cache_data = {
            "response": response_dict,
            "response_class": response_class,
        }
        self.cursor.execute(
            "INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
            (key, json.dumps(cache_data)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
