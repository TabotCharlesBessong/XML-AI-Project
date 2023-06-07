import tkinter as tk
import transformers


def translate_text():
    # Get the input text from the input field
    text = input_field.get()

    # Translate the text
    input_ids = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=512)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Update the GUI with the translation
    result_label.config(text=f"Translation: {translation}")


# Load the Transformer model and tokenizer
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")

# Create the GUI
window = tk.Tk()
window.title("Translation App")

input_label = tk.Label(window, text="Enter text to translate:")
input_label.pack()

input_field = tk.Entry(window, width=50)
input_field.pack()

translate_button = tk.Button(window, text="Translate", command=translate_text)
translate_button.pack()

result_label = tk.Label(window, text="")
result_label.pack()

# Start the app
window.mainloop()