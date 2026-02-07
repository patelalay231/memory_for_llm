from core.providers.gemini import GeminiProvider
from core.extraction.memory_extract import MemoryExtract
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
    memory_extract = MemoryExtract()

    conversation_summary = ""
    history_messages = []

    while True:
        user_message = input("👤 User: ")
        if user_message.lower() in ["exit", "quit", "q", "bye", "goodbye"]:
            print("👋 Goodbye!")
            break

        print(f"🤖 AI: ", end="", flush=True)
        assistant_response = llm_provider.send_message(user_message, system_instruction)

        extracted_memory = memory_extract.extract_memory(conversation_summary, history_messages, user_message, assistant_response)
        print(assistant_response)

        print("Extracted Memory: ", extracted_memory)

        history_messages.append({"user": user_message, "assistant": assistant_response})


if __name__ == "__main__":
    run_cli()