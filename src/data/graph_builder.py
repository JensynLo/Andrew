"""Terraria Knowledge Graph Builder Module"""

import networkx as nx


class GraphBuilder:
    """Terraria Knowledge Graph Builder"""

    PROGRESS_TIERS = {
        0: "Just built first house (Pre-Boss)",
        1: "Defeated Eye of Cthulhu / King Slime",
        2: "Defeated Eater of Worlds / Brain of Cthulhu",
        3: "Defeated Skeletron",
        4: "Defeated Wall of Flesh (Entered Hardmode)",
        5: "Defeated the Mechanical Bosses",
        6: "Defeated Plantera",
        7: "Defeated Golem",
        8: "Defeated Lunatic Cultist",
    }

    ENTITY_TIER_MAP = {
        "Eye of Cthulhu": 1,
        "King Slime": 1,
        "Eater of Worlds": 2,
        "Brain of Cthulhu": 2,
        "Skeletron": 3,
        "Wall of Flesh": 4,
        "The Twins": 5,
        "The Destroyer": 5,
        "Skeletron Prime": 5,
        "Plantera": 6,
        "Golem": 7,
        "Lunatic Cultist": 8,
        "Moon Lord": 8,
    }

    def __init__(self):
        pass

    def build_graph(self, data_dict: dict) -> nx.DiGraph:
        """Build Terraria knowledge graph from input data dictionary

        Args:
            data_dict: Dictionary containing npcs, items, drops, recipes data

        Returns:
            nx.DiGraph: Built knowledge graph
        """
        G = nx.DiGraph()

        for npc in data_dict.get("npcs", []):
            G.add_node(
                npc["name"],
                node_type="Entity",
                entity_type=npc.get("type", "Unknown"),
                environment=npc.get("environment", "Unknown"),
            )

        for item in data_dict.get("items", []):
            G.add_node(
                item["name"],
                node_type="Item",
                hardmode=item.get("hardmode", False),
                pick_power=item.get("pick", 0),
                axe_power=item.get("axe", 0),
            )

        for drop in data_dict.get("drops", []):
            npc_name = drop["name"]
            item_name = drop["item"]

            if "Banner" in item_name:
                continue

            if not G.has_node(npc_name):
                G.add_node(npc_name, node_type="Entity")
            if not G.has_node(item_name):
                G.add_node(item_name, node_type="Item")

            G.add_edge(
                npc_name, item_name, edge_type="DROPS_TO", rate=drop.get("rate", "100%")
            )

        for recipe in data_dict.get("recipes", []):
            result_item = recipe["result"]
            station = recipe.get("station")

            if not G.has_node(result_item):
                G.add_node(result_item, node_type="Item")

            if station and station != "By Hand":  # Exclude crafting by hand
                if not G.has_node(station):
                    G.add_node(station, node_type="Station")
                G.add_edge(station, result_item, edge_type="REQUIRED_FOR")

            ingredients = recipe.get("ingredients", [])
            for ing in ingredients:
                if isinstance(ing, dict):
                    if "name" in ing:
                        ing_name = ing["name"]
                    elif "item_name" in ing:
                        ing_name = ing["item_name"]
                    else:
                        print(
                            f"Warning: Ingredient dict missing 'name' or 'item_name': {ing}"
                        )
                        continue  # Skip this invalid ingredient
                else:
                    ing_name = ing

                if not G.has_node(ing_name):
                    G.add_node(ing_name, node_type="Item")

                G.add_edge(ing_name, result_item, edge_type="CRAFTS_INTO")

        return G


def search_node(G, keyword):
    """Fuzzy search for nodes containing the specified keyword in the graph

    Args:
        G: The graph to search in
        keyword: Keyword to search for
    """
    print(f"\n🔎 Searching for nodes containing '{keyword}'...")
    matches = [
        n for n in G.nodes() if isinstance(n, str) and keyword.lower() in n.lower()
    ]

    if matches:
        for m in matches:
            node_type = G.nodes[m].get("node_type", "Unknown")
            print(f"  ✅ Found precise node name: '{m}' (Type: {node_type})")
    else:
        print(f"  ❌ No nodes containing '{keyword}' found")


def inspect_item(G, item_name):
    """View properties of a single item and its upstream/downstream associations

    Args:
        G: The graph to inspect in
        item_name: Name of the item to inspect
    """
    if not G.has_node(item_name):
        print(
            f"❌ Node does not exist in graph: '{item_name}'. Please check spelling or Chinese/English."
        )
        return

    print(f"\n{'=' * 40}")
    print(f" 🔍 Item Analysis Report: {item_name}")
    print(f"{'=' * 40}")

    print("\n[Node Properties]")
    attrs = G.nodes[item_name]
    for k, v in attrs.items():
        print(f"  - {k}: {v}")

    print("\n[Source / Crafting Recipes (Direct Predecessors)]")
    in_edges = G.in_edges(item_name, data=True)
    if not in_edges:
        print("  (No predecessor nodes, possibly basic material or not crawled)")
    else:
        for source, _, data in in_edges:
            edge_type = data.get("edge_type", "UNKNOWN")
            if edge_type == "CRAFTS_INTO":
                print(f"  🛠️  Requires Material: {source}")
            elif edge_type == "REQUIRED_FOR":
                print(f"  ⚙️  Crafting Station: {source}")
            elif edge_type == "DROPS_TO":
                print(f"  ⚔️  Drops From: {source} (Rate: {data.get('rate', 'N/A')})")

    print("\n[Can be used to craft (Direct Successors)]")
    out_edges = G.out_edges(item_name, data=True)
    if not out_edges:
        print("  (Reached end, no further crafting)")
    else:
        for _, target, data in out_edges:
            edge_type = data.get("edge_type", "UNKNOWN")
            if edge_type == "CRAFTS_INTO":
                print(f"  ➡️  Can craft: {target}")


def print_crafting_tree(G, target_item, depth=0, visited=None):
    """Recursively print the complete crafting/acquisition tree for an item

    Args:
        G: The graph to print the tree from
        target_item: The target item to start the tree from
        depth: Current depth in the recursion
        visited: Set of already visited nodes to prevent cycles
    """
    if visited is None:
        visited = set()

    if target_item in visited:
        print("  " * depth + f"└─ 🔄 {target_item} (already expanded)")
        return
    visited.add(target_item)

    if depth == 0:
        print(f"\n🌳 Complete dependency tree for [{target_item}]:")
    else:
        print("  " * depth + f"└─ {target_item}")

    for source, _, data in G.in_edges(target_item, data=True):
        edge_type = data.get("edge_type")
        if edge_type == "CRAFTS_INTO":
            print_crafting_tree(G, source, depth + 1, visited.copy())
        elif edge_type == "REQUIRED_FOR":
            print("  " * (depth + 1) + f"└─ ⚙️ [Crafting Station] {source}")
        elif edge_type == "DROPS_TO":
            print("  " * (depth + 1) + f"└─ ⚔️ [Drops From] {source}")
