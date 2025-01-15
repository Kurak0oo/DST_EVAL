import openai

# Set your OpenAI API key
openai.api_key = "sk-zk226cfbfd412c556205a3a5d139989e93b677b729db741d"  # Replace with your actual API key

# Test by making a request to the text completion endpoint
response = openai.Completion.create(
    engine="text-davinci-003",  # Use a specific model, for example: text-davinci-003
    prompt="Hello, OpenAI! How are you?",
    max_tokens=50
)

# Print the generated text
print(response.choices[0].text.strip())
