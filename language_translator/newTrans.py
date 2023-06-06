import transformers

# Load the Transformer model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr")

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr")

# Define a function to translate text


def translate(text):
    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate the translation
    outputs = model.generate(input_ids, max_length=512)

    # Decode the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translation


# Start the translation app
while True:
    # Get the input text from the user
    text = input("Enter text to translate: ")

    # Translate the text
    translation = translate(text)

    # Print the translation
    print(f"Translation: {translation}")
