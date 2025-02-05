from llama_cpp import Llama

# Load the model with custom context length, maximum response length, and GPU layers
model = Llama(
    model_path="llama.gguf",
    context_length=2048,  # Set context length to 2048 tokens
    max_length=4096,      # Set max response length to 4096 tokens
    gpu_layers=32         # Load 32 model layers into VRAM
)

# Define a prompt that encourages the assistant to give a direct, non-roleplaying response
prompt_injection = "You are an assistant. Respond directly to the user's input without continuing or roleplaying."

# Define the prompt template
def generate_prompt(user_input):
    return f"{prompt_injection} User asked: {user_input}\nAssistant answer:"

def chat_with_llama():
    print("Chat with LlamaBot! Type 'exit' to quit.")
    
    while True:
        # Get user input from the console
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Generate the prompt using the template
        full_prompt = generate_prompt(user_input)
        
        # Generate a response from the model with exact settings
        response = model(
            full_prompt,
            max_tokens=4096,
            temperature=0.7,       # Set temperature to 0.7 as specified
            top_p=0.4,             # Set nucleus sampling factor to 0.4
            min_p=0,               # Set minimum token probability to 0
            top_k=40,              # Use a token selection pool size of 40
            repeat_penalty=1.18    # Set repetition penalty factor to 1.18
        )
        
        # Display the model's response
        print("LlamaBot: " + response['choices'][0]['text'].strip())

if __name__ == "__main__":
    chat_with_llama()
