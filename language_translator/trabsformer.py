import tkinter as tk
import transformers

# Load the Transformer model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr")

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr")

# Define a function to translate text


def translate(text):
    # Get the input text from the input field
    # text = input_text.get()
    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate the translation
    outputs = model.generate(input_ids, max_length=512)

    # Decode the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return translation


# Start the translation app
# Get the input text from the user
text = input("Enter text to translate: ")

# Translate the text
translation = translate(text)

# Print the translation
print(f"Translation: {translation}")

# --------- Result with GUI --------- #


# Load the Transformer model
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr")

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-en-fr")

# Define a function to translate text


def translate():
    # Get the input text from the input field
    text = input_field.get("1.0", "end-1c")

    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors="pt")

    # Generate the translation
    outputs = model.generate(input_ids, max_length=512)

    # Decode the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Update the output field with the translation
    output_field.delete("1.0", "end")
    output_field.insert("1.0", translation)


# Create the GUI
root = tk.Tk()
root.geometry("600x150")

# Create the input field
input_field = tk.Text(root, width=20, height=5, bd=2,
                      relief="groove", font=("Arial", 12), wrap="word")
input_field.grid(row=0, column=0, padx=20, pady=20)

# Create the translate button
translate_button = tk.Button(root, text="Translate", width=8, bg="cyan", font=(
    "Arial", 12), command=translate)
translate_button.grid(row=0, column=1, padx=20, pady=20)

# Create the output field
output_field = tk.Text(root, width=20, height=5, bd=2,
                       relief="groove", font=("Arial", 12))
output_field.grid(row=0, column=2, padx=20, pady=20)

# Configure the grid layout to be responsive
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=0)
root.columnconfigure(2, weight=1)
root.rowconfigure(0, weight=1)

# Start the main event loop
root.mainloop()
