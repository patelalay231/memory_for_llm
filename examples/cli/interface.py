from core.api.config import MemoryAPIConfig, EmbeddingConfig, OpenAIEmbeddingConfig, LLMProviderConfig, GeminiConfig, HuggingFaceConfig
from core.api.memory_api import MemoryAPI
from core.api.retrieval_api import RetrievalAPI
from logger import Logger
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def format_memories_for_context(memories):
    """
    Format retrieved memories for inclusion in LLM context.
    
    Args:
        memories: List of Memory objects
        
    Returns:
        Formatted string with memories
    """
    if not memories:
        return ""
    
    memory_lines = []
    for idx, memory in enumerate(memories, 1):
        memory_lines.append(f"{idx}. {memory.content}")
    
    return "\n".join(memory_lines)

def run_cli():
    """Run the CLI chat interface."""
    base_system_instruction = """You are a friendly, helpful, and supportive friend. 
                    Your goal is to have natural, warm conversations with the user. 
                    Be conversational, empathetic, and genuine in your responses. 
                    Use a friendly tone, show interest in what they're saying, and respond 
                    as if you're chatting with a close friend. Keep responses concise but 
                    engaging, and feel free to ask follow-up questions to keep the 
                    conversation flowing naturally."""
    
    # Configuration: use LLM_PROVIDER=gemini or LLM_PROVIDER=huggingface (default: gemini)
    llm_provider_name = os.getenv("LLM_PROVIDER", "gemini").lower()
    if llm_provider_name == "huggingface":
        hf_cfg = {
            "api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
            "model": os.getenv("HUGGINGFACE_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        }
        if os.getenv("HUGGINGFACE_PROVIDER"):
            hf_cfg["provider"] = os.getenv("HUGGINGFACE_PROVIDER")
        llm_config = {"huggingface": hf_cfg}
    else:
        llm_config = {
            "gemini": {
                "api_key": os.getenv("GEMINI_API_KEY", ""),
                "model": os.getenv("GEMINI_MODEL_NAME", "gemini-pro"),
            }
        }

    config_dict = {
        "llm": llm_config,
        "storage": {
            "mongodb": {
                "uri": "mongodb://localhost:27017",
                "database": "memory_db",
                "collection": "memories"
            }
        },
        "embedding": {
            "huggingface": {
                "api_key": os.getenv("HUGGINGFACE_API_KEY", ""),
                "model": os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"),
                "provider": os.getenv("HUGGINGFACE_EMBEDDING_PROVIDER", "auto"),
                "normalize": True,
                "output_dimensionality": 1536,  # Qwen3-Embedding-8B default; use same in vector.dimension
            }
        },
        "vector": {
            "faiss": {
                "dimension": 1536,  # Must match embedding model output (e.g. 1536 for Qwen3-Embedding-8B)
                "index_path": "./faiss_index",
                "index_type": "COSINE"  # Options: "L2", "IP", or "COSINE"
            }
        },
        # "debug": True
    }
    
    
    # Initialize memory API
    try:
        memory_api = MemoryAPI(config_dict)
    except Exception as e:
        print(f"Error initializing memory API: {e}")
        sys.exit(1)
    
    # Initialize retrieval API using components from memory_api
    try:
        retrieval_api = RetrievalAPI(
            storage=memory_api.memory_store.storage,
            vector_store=memory_api.memory_store.vector_store,
            embedding_generator=memory_api.memory_store.embedding_generator
        )
    except Exception as e:
        print(f"Error initializing retrieval API: {e}")
        sys.exit(1)
    
    # Get LLM provider from memory API for chat interface
    llm_provider = memory_api.extractor.provider

    history_messages = []

    while True:
        user_message = input("👤 User: ")
        if user_message.lower() in ["exit", "quit", "q", "bye", "goodbye"]:
            print("👋 Goodbye!")
            break

        # Retrieve top 5 relevant memories before generating response
        retrieved_memories = []
        try:
            Logger.debug(f"Retrieving relevant memories for query: '{user_message}'", "[CLI]")
            retrieved_memories = retrieval_api.retrieve(query=user_message, top_k=5)
            if retrieved_memories:
                Logger.debug(f"Retrieved {len(retrieved_memories)} relevant memories", "[CLI]")
        except Exception as e:
            Logger.debug(f"Memory retrieval failed: {e}", "[CLI]")
            # Continue even if retrieval fails

        # Build enhanced system instruction with retrieved memories
        system_instruction = base_system_instruction
        if retrieved_memories:
            memories_context = format_memories_for_context(retrieved_memories)
            system_instruction += f"\n\nRelevant context from past conversations (user and assistant):\n{memories_context}\n\nUse this context to provide consistent, personalized, and relevant responses."

        print(f"🤖 AI: ", end="", flush=True)
        assistant_response = llm_provider.send_message(user_message, system_instruction)
        print(assistant_response)
    
        # Extract and store memories from this conversation turn
        try:
            memories = memory_api.add_memory(
                recent_messages=history_messages[-10:], # Only use last 10 messages
                user_message=user_message,
                assistant_message=assistant_response
            )
            if memories:
                Logger.info(f"\n💾 Summary: Successfully stored {len(memories)} memory/memories")
            else:       
                Logger.info("\n💾 Summary: No memories extracted from this conversation")
        except Exception as e:
            print(f"\n⚠️ Memory extraction/storage failed: {e}")

        history_messages.append({"user": user_message, "assistant": assistant_response})


if __name__ == "__main__":
    run_cli()
