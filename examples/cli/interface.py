from core.providers.gemini import GeminiProvider
from core.api.memory_api import MemoryAPI
from logger import Logger

def run_cli():
    """Run the CLI chat interface."""
    system_instruction = """You are a friendly, helpful, and supportive friend. 
                    Your goal is to have natural, warm conversations with the user. 
                    Be conversational, empathetic, and genuine in your responses. 
                    Use a friendly tone, show interest in what they're saying, and respond 
                    as if you're chatting with a close friend. Keep responses concise but 
                    engaging, and feel free to ask follow-up questions to keep the 
                    conversation flowing naturally."""
    
    llm_provider = GeminiProvider()
    
    # Initialize memory API
    memory_api = MemoryAPI({
        "llm_provider": llm_provider,
        "storage": {
            "type": "mongodb",
            "uri": "mongodb://localhost:27017",
            "database": "memory_db",
            "collection": "memories"
        },
        "max_retries": 3,
        # "debug": True  # Can also set debug here
    })

    history_messages = []

    while True:
        user_message = input("👤 User: ")
        if user_message.lower() in ["exit", "quit", "q", "bye", "goodbye"]:
            print("👋 Goodbye!")
            break

        print(f"🤖 AI: ", end="", flush=True)
        assistant_response = llm_provider.send_message(user_message, system_instruction)
        print(assistant_response)

        # Extract and store memories
        try:
            memories = memory_api.add_memory(
                recent_messages=history_messages,
                user_message=user_message,
                assistant_message=assistant_response
            )
            if memories:
                print(f"\n💾 Summary: Successfully stored {len(memories)} memory/memories")
            else:
                print("\n💾 Summary: No memories extracted from this conversation")
        except Exception as e:
            print(f"\n⚠️ Memory extraction/storage failed: {e}")

        history_messages.append({"user": user_message, "assistant": assistant_response})


if __name__ == "__main__":
    run_cli()
