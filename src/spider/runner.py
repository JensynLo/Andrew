import json
import argparse
import os
from ..utils import load_config
from .utils import spider, DropsCleaner, ItemsCleaner, NpcsCleaner, RecipesCleaner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="path of the config file"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="path to save the cleaned data"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    cfgs = load_config(args.cfg)
    api_url = cfgs["configs"]["api_url"]
    headers = cfgs["configs"]["headers"]
    per_fetch = cfgs["configs"]["per_fetch"]
    offset = cfgs["configs"]["offset"]
    limit = cfgs["configs"]["limit"]
    table_fields_list = [
        (val["table"], val["fields"]) for _, val in cfgs["datas"].items()
    ]

    cleaner_map = {
        "items": ItemsCleaner.clean_items_list,
        "drops": DropsCleaner.clean_drops_list,
        "npcs": NpcsCleaner.clean_npcs_list,
        "recipes": RecipesCleaner.clean_recipes_list,
    }

    for table, fields in table_fields_list:
        print(f"\n[{table}] fetching from {api_url} ...")
        print(f"[{table}] target fields: {fields}")

        spider_instance = spider(
            api_url=api_url,
            headers=headers,
            target_tables=table,
            target_fields=fields,
            limit=limit,
            offset=offset,
            per_fetch=per_fetch,
        )

        raw_data = spider_instance.fetch()

        if not raw_data:
            print(
                f"[{table}] Warning: No data fetched for {table}. Skipping cleaning and saving."
            )
            continue

        print(
            f"[{table}] fetched {len(raw_data)} records. Starting cleaning process..."
        )

        clean_func = cleaner_map.get(table.lower())

        if clean_func:
            final_data = clean_func(raw_data)
            print(
                f"[{table}] cleaning completed. {len(final_data)} records after cleaning."
            )
        else:
            print(f"[{table}] No cleaning function found. Saving raw data.")
            final_data = raw_data

        file_path = os.path.join(args.output_path, f"{table}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)

        print(f"[{table}] Data successfully saved to -> {file_path}")

    print("\n All tasks completed!")


if __name__ == "__main__":
    main()
