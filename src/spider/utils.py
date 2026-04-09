import os
import re
import time
import json
import pickle
import requests
import dirtyjson
from pathlib import Path
from typing import Any, List, Dict, Optional


class spider:
    def __init__(
        self,
        api_url: str,
        headers: dict,
        target_tables: str,
        target_fields: list[str],
        limit: int = 0,
        per_fetch: int = 500,
        offset: int = 0,
        cache_file: Optional[str] = None,
    ):
        self.api_url = api_url
        self.headers = headers
        self.target_tables = target_tables
        self.target_fields = target_fields
        self.limit = limit

        self.per_fetch = (
            200
            if any(
                heavy in target_tables.lower()
                for heavy in ["items", "recipes", "drops"]
            )
            else min(per_fetch, 500)
        )
        self.offset = offset
        self.cache_file = cache_file

        if self.cache_file and self._load_cache():
            print(
                f"[{self.target_tables}] Restoring progress from cache file {self.cache_file}..."
            )

    def _get_cache_filename(self) -> str:
        """Get cache file name

        Returns:
            Cache filename based on target tables and fields
        """
        if self.cache_file:
            return self.cache_file
        fields_part = "_".join(self.target_fields[:3])
        return f"./cache/{self.target_tables}_{fields_part}_cache.pkl"

    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        cache_path = Path(self._get_cache_filename())
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_cache(self, data: List[Dict], current_offset: int):
        """Save cache data to file

        Args:
            data: Current data list
            current_offset: Current offset position
        """
        if not self.cache_file and not self._get_cache_filename():
            return

        self._ensure_cache_dir()
        cache_data = {
            "data": data,
            "offset": current_offset,
            "target_tables": self.target_tables,
            "target_fields": self.target_fields,
            "timestamp": time.time(),
        }
        cache_filename = self._get_cache_filename()
        try:
            with open(cache_filename, "wb") as f:
                pickle.dump(cache_data, f)
            print(f"[{self.target_tables}] Cache saved to {cache_filename}")
        except Exception as e:
            print(f"[{self.target_tables}] Failed to save cache: {e}")

    def _load_cache(self) -> bool:
        """Load cache data from file

        Returns:
            True if cache loaded successfully, False otherwise
        """
        cache_filename = self._get_cache_filename()
        if not os.path.exists(cache_filename):
            return False

        try:
            with open(cache_filename, "rb") as f:
                cache_data = pickle.load(f)

            if (
                cache_data.get("target_tables") == self.target_tables
                and cache_data.get("target_fields") == self.target_fields
            ):
                cached_data = cache_data.get("data", [])
                cached_offset = cache_data.get("offset", 0)

                if cached_data:
                    print(
                        f"[{self.target_tables}] Restored {len(cached_data)} records from cache"
                    )
                    self.cached_raw_data = cached_data
                    self.offset = cached_offset
                    return True
                return False
            else:
                print(
                    f"[{self.target_tables}] Cache file does not match current task, ignoring"
                )
                return False

        except Exception as e:
            print(f"[{self.target_tables}] Failed to load cache: {e}")
            return False

    def _clear_cache(self):
        """Clear cache file"""
        cache_filename = self._get_cache_filename()
        if os.path.exists(cache_filename):
            try:
                os.remove(cache_filename)
                print(f"[{self.target_tables}] Cache file deleted")
            except Exception as e:
                print(f"[{self.target_tables}] Failed to delete cache file: {e}")

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text by finding matching brackets

        Args:
            text: Input text containing JSON

        Returns:
            Extracted JSON string or original text if no matching brackets found
        """
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def fetch(self):
        if hasattr(self, "cached_raw_data"):
            raw = self.cached_raw_data.copy()
            print(f"[{self.target_tables}] Using {len(raw)} cached records")
        else:
            raw = []

        current_per_fetch = self.per_fetch

        while True:
            if self.limit > 0:
                remaining = self.limit - len(raw)
                if remaining <= 0:
                    print(
                        f"[{self.target_tables}] Reached the set limit: {self.limit} records, stopping fetch."
                    )
                    self._clear_cache()
                    break
                fetch_size = min(current_per_fetch, remaining)
            else:
                fetch_size = current_per_fetch

            params_cargo = {
                "action": "cargoquery",
                "format": "json",
                "tables": self.target_tables,
                "fields": ", ".join(self.target_fields),
                "limit": fetch_size,
                "offset": self.offset,
            }

            max_retries = 5
            retry_delay = 2
            request_success = False
            response = None

            for attempt in range(max_retries):
                try:
                    response = requests.get(
                        self.api_url,
                        params=params_cargo,
                        headers=self.headers,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        request_success = True
                        if current_per_fetch < self.per_fetch:
                            current_per_fetch = min(
                                self.per_fetch, current_per_fetch + 50
                            )
                        break
                    elif response.status_code in [500, 502, 503, 504]:
                        current_per_fetch = max(20, current_per_fetch // 2)
                        print(
                            f"[{self.target_tables}] Server pressure ({response.status_code}). Reducing speed to {current_per_fetch} records and retrying..."
                        )
                        time.sleep(retry_delay * (attempt + 2))
                    else:
                        print(
                            f"[{self.target_tables}] Fatal error HTTP {response.status_code}, skipping current batch."
                        )
                        break

                except Exception as e:
                    print(
                        f"[{self.target_tables}] Network exception: {e}, retrying ({attempt + 1}/{max_retries})..."
                    )
                    time.sleep(retry_delay * 2)

            if not request_success or response is None:
                print(
                    f"[{self.target_tables}] Failed {max_retries} times consecutively, forcing pagination, skipping Offset {self.offset}"
                )
                self.offset += fetch_size
                self._save_cache(raw, self.offset)
                continue

            response_text = response.text
            try:
                res = json.loads(response_text, strict=False)
            except json.JSONDecodeError:
                print(f"[{self.target_tables}] JSON corrupted, attempting repair...")
                try:
                    cleaned_text = self._extract_json(response_text)
                    res = dirtyjson.loads(cleaned_text)
                except Exception as e:
                    print(
                        f"[{self.target_tables}] Repair failed: {e}, forcing pagination."
                    )
                    self.offset += fetch_size
                    self._save_cache(raw, self.offset)
                    continue

            if isinstance(res, dict) and "error" in res:
                info = res["error"].get("info", res["error"])
                print(f"❌ Wiki API returned error: {info}")
                self._save_cache(raw, self.offset)
                break

            batch = res.get("cargoquery", []) if isinstance(res, dict) else []

            if not batch:
                print(f"[{self.target_tables}] All data exhausted, no new records.")
                self._clear_cache()
                break

            raw.extend(batch)
            self.offset += len(batch)

            self._save_cache(raw, self.offset)

            if self.limit > 0:
                print(
                    f"[{self.target_tables}] Progress: {len(raw)} / {self.limit} records..."
                )
            else:
                print(f"[{self.target_tables}] Currently fetched {len(raw)} records...")

            time.sleep(0.5)

        return self._process_results(raw)

    def _process_results(self, raw_data):
        """Process raw data into final results

        Args:
            raw_data: Raw data from API response

        Returns:
            List of processed results
        """
        result = [item.get("title", {}) for item in raw_data if isinstance(item, dict)]
        return [item for item in result if item]


class DropsCleaner:
    @staticmethod
    def _clean_html(text: str) -> str:
        """Advanced HTML cleaning to prevent text sticking together

        Args:
            text: Input text containing HTML

        Returns:
            Cleaned text with HTML tags removed
        """
        if not isinstance(text, str) or not text:
            return ""

        text = re.sub(r"<(div|p|br|/div|/p)[^>]*>", " ", text)

        text = re.sub(r"<[^>]+>", "", text)

        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return text

    @staticmethod
    def _clean_wikitext(text: str) -> str:
        """Clean wiki-specific syntax

        Args:
            text: Input text containing wiki syntax

        Returns:
            Cleaned text with wiki syntax removed
        """
        if not isinstance(text, str) or not text:
            return ""

        text = re.sub(r"\[\[[^\]]+\|([^\]]+)\]\]", r"\1", text)

        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _to_bool(val: Any) -> bool:
        """Convert Cargo boolean value, usually '1' represents True

        Args:
            val: Input value to convert

        Returns:
            Boolean representation of the input
        """
        return str(val).strip() in ["1", "true", "yes"]

    @classmethod
    def clean_drop_dict(cls, raw_dict: dict) -> dict:
        """Main function to clean Drops dictionary

        Args:
            raw_dict: Raw dictionary containing drop data

        Returns:
            Cleaned dictionary with processed drop data
        """

        def deep_clean(text: str) -> str:
            if text is None:
                return ""
            no_html = cls._clean_html(text)
            return cls._clean_wikitext(no_html)

        cleaned = {
            "name": deep_clean(raw_dict.get("name", "")),
            "item": deep_clean(raw_dict.get("item", "")),
            "rate": deep_clean(raw_dict.get("rate", "")),
            "isfromnpc": cls._to_bool(raw_dict.get("isfromnpc")),
            "normal": deep_clean(raw_dict.get("normal", "")),
            "expert": deep_clean(raw_dict.get("expert", "")),
            "master": deep_clean(raw_dict.get("master", "")),
        }

        return cleaned

    @classmethod
    def clean_drops_list(cls, raw_list: List[dict]) -> List[dict]:
        """Main function to clean Drops list

        Args:
            raw_list: Raw list of drop dictionaries

        Returns:
            List of cleaned drop dictionaries
        """
        return [
            cls.clean_drop_dict(item) for item in raw_list if isinstance(item, dict)
        ]


class ItemsCleaner:
    @staticmethod
    def _clean_html(text: str, keep_newlines: bool = False) -> str:
        """HTML cleaning, optionally keep <br> as newline (useful for tooltips)

        Args:
            text: Input text containing HTML
            keep_newlines: Whether to keep newlines or convert to space

        Returns:
            Cleaned text with HTML removed
        """
        if not isinstance(text, str) or not text:
            return ""

        if keep_newlines:
            text = re.sub(r"<(br|/p|/div)[^>]*>", "\n", text)
        else:
            text = re.sub(r"<(div|p|br|/div|/p)[^>]*>", " ", text)

        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&#32;", " ")
        return text.strip()

    @staticmethod
    def _clean_wikitext(text: str) -> str:
        """Clean wiki-specific syntax

        Args:
            text: Input text containing wiki syntax

        Returns:
            Cleaned text with wiki syntax removed
        """
        if not isinstance(text, str) or not text:
            return ""
        text = re.sub(r"\[\[[^\]]+\|([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _to_bool(val: Any) -> bool:
        """Process boolean values

        Args:
            val: Input value to convert

        Returns:
            Boolean representation of the input
        """
        return str(val).strip().lower() in ["1", "true", "yes", "y"]

    @staticmethod
    def _to_int(val: Any) -> int:
        """Extract integer from string (e.g. rare, axe, pick fields)

        Args:
            val: Input value to convert

        Returns:
            Integer representation of the input
        """
        if not val:
            return 0
        if isinstance(val, int):
            return val
        match = re.search(r"(-?\d+)", str(val))
        return int(match.group(1)) if match else 0

    @staticmethod
    def _parse_coins(html_text: str) -> int:
        """Parse wiki coin system, prioritize data-sort-value (returns pure copper value), extract pure number if not

        Args:
            html_text: Input text containing coin information

        Returns:
            Parsed coin value as integer
        """
        if not html_text:
            return 0

        sort_value_match = re.search(r'data-sort-value="(\d+)"', html_text)
        if sort_value_match:
            return int(sort_value_match.group(1))

        clean_text = ItemsCleaner._clean_html(html_text)
        match = re.search(r"(\d+)", clean_text)
        return int(match.group(1)) if match else 0

    @classmethod
    def clean_item_dict(cls, raw_dict: dict) -> dict:
        """Main function to clean Items dictionary

        Args:
            raw_dict: Raw dictionary containing item data

        Returns:
            Cleaned dictionary with processed item data
        """

        def deep_clean(text: str, keep_newlines: bool = False) -> str:
            if text is None:
                return ""
            no_html = cls._clean_html(text, keep_newlines)
            return cls._clean_wikitext(no_html)

        cleaned = {
            "itemid": cls._to_int(raw_dict.get("itemid")),
            "name": deep_clean(raw_dict.get("name", "")),
            "hardmode": cls._to_bool(raw_dict.get("hardmode")),
            "type": deep_clean(raw_dict.get("type", "")),
            "rare": cls._to_int(raw_dict.get("rare")),
            "buy": cls._parse_coins(raw_dict.get("buy", "")),
            "sell": cls._parse_coins(raw_dict.get("sell", "")),
            "axe": cls._to_int(raw_dict.get("axe")),
            "pick": cls._to_int(raw_dict.get("pick")),
            "hammer": cls._to_int(raw_dict.get("hammer")),
            "tooltip": deep_clean(raw_dict.get("tooltip", ""), keep_newlines=True),
        }

        return cleaned

    @classmethod
    def clean_items_list(cls, raw_list: List[dict]) -> List[dict]:
        """Main function to clean Items list

        Args:
            raw_list: Raw list of item dictionaries

        Returns:
            List of cleaned item dictionaries
        """
        return [
            cls.clean_item_dict(item) for item in raw_list if isinstance(item, dict)
        ]


class NpcsCleaner:
    @staticmethod
    def _clean_html(text: str) -> str:
        """HTML cleaning, with special removal of Terraria-specific version tags

        Args:
            text: Input text containing HTML

        Returns:
            Cleaned text with HTML tags removed
        """
        if not isinstance(text, str) or not text:
            return ""

        text = re.sub(
            r"\((Desktop|Console|Mobile|Old-gen console|3DS|tModLoader)[^)]*versions?\)",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(r"<(div|p|br|/div|/p)[^>]*>", " ", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return text.strip()

    @staticmethod
    def _clean_wikitext(text: str) -> str:
        """Clean wiki-specific syntax

        Args:
            text: Input text containing wiki syntax

        Returns:
            Cleaned text with wiki syntax removed
        """
        if not isinstance(text, str) or not text:
            return ""
        text = re.sub(r"\[\[[^\]]+\|([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _to_int(val: Any) -> int:
        """Extract integer from string (for npcid and banner fields)

        Args:
            val: Input value to convert

        Returns:
            Integer representation of the input
        """
        if not val:
            return 0
        if isinstance(val, int):
            return val
        match = re.search(r"(-?\d+)", str(val))
        return int(match.group(1)) if match else 0

    @classmethod
    def clean_npc_dict(cls, raw_dict: dict) -> dict:
        """Main function to clean NPCs dictionary

        Args:
            raw_dict: Raw dictionary containing NPC data

        Returns:
            Cleaned dictionary with processed NPC data
        """

        def deep_clean(text: Optional[str]) -> str:
            if not text:
                return ""
            no_html = cls._clean_html(text)
            return cls._clean_wikitext(no_html)

        cleaned = {
            "npcid": cls._to_int(raw_dict.get("npcid")),
            "name": deep_clean(raw_dict.get("name")),
            "nameraw": deep_clean(raw_dict.get("nameraw")),
            "type": deep_clean(raw_dict.get("type")),
            "environment": deep_clean(raw_dict.get("environment")),
            "ai": deep_clean(raw_dict.get("ai")),
            "banner": cls._to_int(raw_dict.get("banner")),
            "bannername": deep_clean(raw_dict.get("bannername")),
        }

        return cleaned

    @classmethod
    def clean_npcs_list(cls, raw_list: List[dict]) -> List[dict]:
        """Main function to clean NPCs list

        Args:
            raw_list: Raw list of NPC dictionaries

        Returns:
            List of cleaned NPC dictionaries
        """
        return [cls.clean_npc_dict(item) for item in raw_list if isinstance(item, dict)]


class RecipesCleaner:
    @staticmethod
    def _clean_html(text: str) -> str:
        """HTML cleaning

        Args:
            text: Input text containing HTML

        Returns:
            Cleaned text with HTML removed
        """
        if not isinstance(text, str) or not text:
            return ""
        text = re.sub(r"<(div|p|br|/div|/p)[^>]*>", " ", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return text.strip()

    @staticmethod
    def _clean_wikitext(text: str) -> str:
        """Clean wiki-specific syntax

        Args:
            text: Input text containing wiki syntax

        Returns:
            Cleaned text with wiki syntax removed
        """
        if not isinstance(text, str) or not text:
            return ""
        text = re.sub(r"\[\[[^\]]+\|([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _to_int(val: Any) -> int:
        """Safely extract integer

        Args:
            val: Input value to convert

        Returns:
            Integer representation of the input
        """
        if not val:
            return 0
        if isinstance(val, int):
            return val
        match = re.search(r"(-?\d+)", str(val))
        return int(match.group(1)) if match else 0

    @classmethod
    def _parse_ingredients(cls, raw_str: str) -> List[Dict[str, Any]]:
        """Core method: Parse Cargo list text into structured ingredient list
        e.g. "Wood^10¦Iron Bar^5" -> [{"item": "Wood", "amount": 10}, {"item": "Iron Bar", "amount": 5}]

        Args:
            raw_str: Raw string containing ingredient list

        Returns:
            List of ingredient dictionaries with name and amount
        """
        if not raw_str or not isinstance(raw_str, str):
            return []

        cleaned_str = raw_str.strip("¦, ")
        if not cleaned_str:
            return []

        parts = re.split(r"[¦,]+", cleaned_str)
        ingredients = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if "^" in part:
                item_name, amount_str = part.split("^", 1)
                amount = cls._to_int(amount_str)
            else:
                item_name = part
                amount = 1
            item_name = cls._clean_wikitext(cls._clean_html(item_name))

            ingredients.append(
                {"name": item_name, "amount": amount if amount > 0 else 1}
            )

        return ingredients

    @classmethod
    def clean_recipe_dict(cls, raw_dict: dict) -> dict:
        """Main function to clean Recipes dictionary

        Args:
            raw_dict: Raw dictionary containing recipe data

        Returns:
            Cleaned dictionary with processed recipe data
        """

        def deep_clean(text: Optional[str]) -> str:
            if not text:
                return ""
            no_html = cls._clean_html(text)
            return cls._clean_wikitext(no_html)

        station = deep_clean(raw_dict.get("station", ""))
        station = station.replace("¦", ", ").strip(", ")

        result_item = deep_clean(raw_dict.get("result", ""))
        result_amount = 1
        if "^" in result_item:
            result_item, amount_str = result_item.split("^", 1)
            result_amount = cls._to_int(amount_str) or 1

        cleaned = {
            "result": result_item,
            "amount": result_amount,
            "station": station if station else "By Hand",
            "ingredients": cls._parse_ingredients(raw_dict.get("ingredients", "")),
        }

        return cleaned

    @classmethod
    def clean_recipes_list(cls, raw_list: List[dict]) -> List[dict]:
        """Main function to clean Recipes list

        Args:
            raw_list: Raw list of recipe dictionaries

        Returns:
            List of cleaned recipe dictionaries
        """
        return [
            cls.clean_recipe_dict(item) for item in raw_list if isinstance(item, dict)
        ]
