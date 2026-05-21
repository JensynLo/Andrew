import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def drop2text(drops: dict) -> str:
    npc_name = drops["name"]
    item_name = drops["item"]
    rate = drops["rate"]
    text = f"{npc_name} drops {item_name} with a rate of {rate}%."
    return text


def item2text(items: dict) -> str:
    item_name = items["name"]
    item_type = items["type"]
    hardmode = "hard_model" if items["hardmode"] else "pre_hard_model"
    rare = items["rare"]
    buy_price = items["buy"]
    sell_price = items["sell"]
    axe = items["axe"]
    pick = items["pick"]
    hammer = items["hammer"]
    tooltip = items["tooltip"]
    # 组装文本
    axe_text = f"you need at least {axe} axe power to get this item." if axe else ""
    hammer_text = (
        f"you need at least {hammer} hammer power to get this item." if hammer else ""
    )
    pick_text = (
        f"you need at least {pick} pickaxe power to get this item." if pick else ""
    )
    price_text = (
        f"you can buy it for {buy_price} coins and sell it for {sell_price} coins."
        if buy_price and sell_price
        else ""
    )
    text = f"{item_name} is a {item_type} item. you can get it in {hardmode} mode. its rarity is {rare}. {price_text} {axe_text} {hammer_text} {pick_text} With tooltip: {tooltip}"
    return text


def npc2text(npcs: dict) -> str:
    npc_name = npcs["name"]
    npc_type = npcs["type"]
    environment = npcs["environment"]
    ai = npcs["ai"]
    banner = npcs["banner"]
    banner_name = npcs["bannername"]
    # 组装文本
    banner_text = (
        f"you can get its banner called {banner_name} by killing it." if banner else ""
    )
    text = f"{npc_name} is a {npc_type} npc that can be found in {environment} environment. with {ai}. {banner_text}"
    return text


def recipe2text(recipes: dict) -> str:
    result = recipes["result"]
    amount = recipes["amount"]
    ingredients = recipes["ingredients"]
    station = recipes["station"]
    # 组装文本
    ingredients_text = ", ".join(
        [f"{ingredient['amount']} {ingredient['name']}" for ingredient in ingredients]
    )
    text = f"To craft {amount} {result}, you need {ingredients_text} at a {station}."
    return text


class VectorIndexer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.st_model = SentenceTransformer(model_name)
        self.embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store: FAISS | None = None
        self.documents = []

    def load_and_process_data(self, data_dir: Path = Path("output")):
        """Load JSON data and transform to text using indexer functions"""
        data_dir = Path(data_dir)

        # Process Items.json
        items_file = data_dir / "Items.json"
        if items_file.exists():
            with open(str(items_file), "r", encoding="utf-8") as f:
                items = json.load(f)

            for item in items:
                try:
                    text = item2text(item)
                    self.documents.append(
                        Document(
                            page_content=text,
                            metadata={"type": "item", "name": item.get("name")},
                        )
                    )
                except Exception as e:
                    print(f"Error processing item {item.get('name', 'Unknown')}: {e}")

        # Process NPCs.json
        npcs_file = data_dir / "NPCs.json"
        if npcs_file.exists():
            with open(npcs_file, "r", encoding="utf-8") as f:
                npcs = json.load(f)

            for npc in npcs:
                try:
                    text = npc2text(npc)
                    self.documents.append(
                        Document(
                            page_content=text,
                            metadata={"type": "npc", "name": npc.get("name")},
                        )
                    )
                except Exception as e:
                    print(f"Error processing NPC {npc.get('name', 'Unknown')}: {e}")

        # Process Recipes.json
        recipes_file = data_dir / "Recipes.json"
        if recipes_file.exists():
            with open(recipes_file, "r", encoding="utf-8") as f:
                recipes = json.load(f)

            for recipe in recipes:
                try:
                    text = recipe2text(recipe)
                    self.documents.append(
                        Document(
                            page_content=text,
                            metadata={"type": "recipe", "result": recipe.get("result")},
                        )
                    )
                except Exception as e:
                    print(
                        f"Error processing recipe for {recipe.get('result', 'Unknown')}: {e}"
                    )

        # Process Drops.json
        drops_file = data_dir / "Drops.json"
        if drops_file.exists():
            with open(drops_file, "r", encoding="utf-8") as f:
                drops = json.load(f)

            for drop in drops:
                try:
                    text = drop2text(drop)
                    self.documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "type": "drop",
                                "npc": drop.get("name"),
                                "item": drop.get("item"),
                            },
                        )
                    )
                except Exception as e:
                    print(
                        f"Error processing drop {drop.get('name', 'Unknown')} -> {drop.get('item', 'Unknown')}: {e}"
                    )

        print(f"Loaded {len(self.documents)} documents")

    def create_index(self):
        """Generate embeddings and create FAISS index using LangChain"""
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings_model)

    def save_index(self, folder_path: str = "output/ragindex"):
        """Save the index to disk using LangChain's format"""
        if self.vector_store is None:
            raise ValueError("Index not created yet.")
        self.vector_store.save_local(folder_path)

    def load_index(self, folder_path: str):
        """Load the index from disk"""
        self.vector_store = FAISS.load_local(
            folder_path, self.embeddings_model, allow_dangerous_deserialization=True
        )
        print(f"Loaded index from {folder_path}")

    def search(self, query: str, k: int = 5):
        """Search for the most similar documents to the query"""
        if self.vector_store is None:
            raise ValueError("Index not created or loaded.")

        results = self.vector_store.similarity_search_with_score(query, k=k)

        # Format results to match previous interface
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "score": float(score),
                    "document": doc.metadata,
                    "text": doc.page_content,
                }
            )

        return formatted_results


def build_index(data_dir: str = "output", folder_path: str = "output/ragindex"):
    """Convenience function to build the entire index"""
    indexer = VectorIndexer()
    indexer.load_and_process_data(Path(data_dir))
    indexer.create_index()
    indexer.save_index(folder_path)
    return indexer


if __name__ == "__main__":
    # Example usage
    print("Building Terraria vector index...")
    indexer = build_index()
    # 一直查询
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        results = indexer.search(query, k=3)
        for i, result in enumerate(results):
            print(f"\nResult {i + 1} (Score: {result['score']:.4f}):")
            print(f"Type: {result['document']['type']}")
            print(f"Text: {result['text']}")
