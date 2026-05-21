"""
RAG (Retrieval Augmented Generation) Integration Module
This module integrates the vector indexing system with an LLM for question answering.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


class RAGIntegration:
    """
    RAG (Retrieval Augmented Generation) Integration Class
    Combines the indexing capabilities of VectorIndexer with an LLM for inference.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
    ):
        self.model_name = model_name
        self.llm_model_name = llm_model_name

        # Initialize embedding model
        self.st_model = SentenceTransformer(model_name)
        self.embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

        # Initialize LLM components
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name, device_map="auto", torch_dtype="auto"
        )

        # Initialize text generation pipeline
        self.text_generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            temperature=temperature,
            do_sample=True,
        )

        self.llm = HuggingFacePipeline(pipeline=self.text_generation_pipeline)

        # Initialize vector store and documents
        self.vector_store: Optional[FAISS] = None
        self.documents = []

        # Define RAG prompt template
        self.rag_template = """
        You are a knowledgeable assistant familiar with Terraria game mechanics.
        Use the provided context to answer the user's question accurately.

        If the answer cannot be found in the context, please respond with "I don't know based on the provided information."

        Context:
        {context}

        Question: {question}

        Answer:"""

        self.prompt = PromptTemplate.from_template(self.rag_template)

    def load_and_process_data(self, data_dir: Path = Path("output")):
        """Load JSON data and transform to text using indexer functions (from indexer.py)"""
        data_dir = Path(data_dir)

        # Import functions from indexer.py to avoid duplication
        from .indexer import item2text, npc2text, recipe2text, drop2text

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
        if not self.documents:
            raise ValueError("No documents loaded. Call load_and_process_data first.")

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

    def format_docs(self, docs):
        """Format retrieved documents into a string"""
        return "\n\n".join(doc.page_content for doc in docs)

    def build_rag_chain(self, k: int = 3):
        """Build the RAG chain for question answering"""
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Load or create an index first."
            )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # Create the RAG chain
        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

    def query(self, question: str, k: int = 3) -> str:
        """Query the RAG system with a question"""
        rag_chain = self.build_rag_chain(k)
        return rag_chain.invoke(question)

    def query_with_context(self, question: str, k: int = 3) -> Dict[str, Any]:
        """Query the RAG system and return both answer and context"""
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. Load or create an index first."
            )

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        # Get retrieved documents
        docs = retriever.invoke(question)
        context = self.format_docs(docs)

        # Format prompt with context and question
        formatted_prompt = self.prompt.format(context=context, question=question)

        # Generate answer
        answer = self.llm.invoke(formatted_prompt)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "retrieved_docs": [
                {"content": doc.page_content, "metadata": doc.metadata} for doc in docs
            ],
        }


def initialize_rag_system(
    data_dir: str = "output",
    index_folder: str = "output/ragindex",
    load_existing: bool = True,
    llm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
) -> RAGIntegration:
    """
    Convenience function to initialize the complete RAG system
    """
    rag_system = RAGIntegration(llm_model_name=llm_model_name)

    if load_existing:
        try:
            rag_system.load_index(index_folder)
            print("Successfully loaded existing RAG index.")
        except Exception as e:
            print(f"Could not load existing index: {e}. Building new index...")
            rag_system.load_and_process_data(Path(data_dir))
            rag_system.create_index()
            rag_system.save_index(index_folder)
    else:
        rag_system.load_and_process_data(Path(data_dir))
        rag_system.create_index()
        rag_system.save_index(index_folder)

    return rag_system


if __name__ == "__main__":
    # Example usage of the RAG integration
    print("Initializing RAG system...")

    # Initialize the RAG system
    rag_system = initialize_rag_system(
        data_dir="output",
        index_folder="output/ragindex",
        load_existing=True,
        llm_model_name="meta-llama/Llama-3.2-3B-Instruct",
    )

    print("\nRAG system ready! You can now ask questions.")
    print("Type 'exit' to quit or 'context' to see detailed results with context.")

    while True:
        query_input = input("\nEnter your query: ").strip()

        if query_input.lower() == "exit":
            break
        elif query_input.lower() == "context":
            question = input("Enter your question: ").strip()
            result = rag_system.query_with_context(question)

            print(f"\n❓ Question: {result['question']}")
            print(
                f"📄 Retrieved Context:\n{result['context'][:500]}..."
            )  # Truncate for display
            print(f"🤖 Answer: {result['answer']}")

            print(f"\n📚 Retrieved {len(result['retrieved_docs'])} documents:")
            for i, doc in enumerate(result["retrieved_docs"]):
                print(f"  [{i + 1}] Type: {doc['metadata'].get('type', 'unknown')}")
                print(f"      Content preview: {doc['content'][:100]}...")
        else:
            answer = rag_system.query(query_input)
            print(f"\n🤖 Answer: {answer}")
