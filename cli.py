import torch
from tkinter import Tk, Label, Text, Button, Scrollbar, END, RIGHT, Y, Frame
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# Step 1: Load Models and Tokenizers
def load_models():
    # Load Bug Detection Model
    detection_model_name = "microsoft/codebert-base"
    detection_tokenizer = RobertaTokenizer.from_pretrained(detection_model_name)
    detection_model = RobertaForSequenceClassification.from_pretrained(detection_model_name, num_labels=2)
    detection_model.load_state_dict(torch.load("bug_detection_model.pt", map_location=torch.device('cpu')))
    detection_model.eval()

    # Load Fix Recommendation Model
    fix_model_name = "microsoft/CodeGPT-small-py"
    fix_tokenizer = AutoTokenizer.from_pretrained(fix_model_name)
    fix_model = AutoModelForCausalLM.from_pretrained(fix_model_name)

    return detection_model, detection_tokenizer, fix_model, fix_tokenizer

# Initialize models and tokenizers
detection_model, detection_tokenizer, fix_model, fix_tokenizer = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detection_model.to(device)
fix_model.to(device)

# Step 2: Define Functionality for Analysis
def analyze_code(code_snippet):
    # Detect if the code is buggy
    inputs = detection_tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = detection_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    status = "buggy" if prediction == 1 else "bug-free"

    # Generate a suggested fix if buggy
    suggested_fix = None
    if status == "buggy":
        inputs_fix = fix_tokenizer.encode(code_snippet, return_tensors="pt", max_length=128, truncation=True).to(device)
        outputs_fix = fix_model.generate(
            inputs_fix,
            max_length=256,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            top_k=50
        )
        suggested_fix = fix_tokenizer.decode(outputs_fix[0], skip_special_tokens=True)

    return status, suggested_fix

# Step 3: Build the GUI
def run_gui():
    # Initialize the main window
    root = Tk()
    root.title("Bug Detection and Fix Suggestion Tool")
    root.geometry("800x600")

    # Input Frame
    input_frame = Frame(root)
    input_frame.pack(pady=10)

    Label(input_frame, text="Enter Your Code:").pack()
    input_text = Text(input_frame, height=15, width=90, wrap="word")
    input_text.pack(side="left", pady=10)

    # Output Frame
    output_frame = Frame(root)
    output_frame.pack(pady=10)

    Label(output_frame, text="Output:").pack()
    output_text = Text(output_frame, height=15, width=90, wrap="word", state="disabled")
    output_text.pack(side="left", pady=10)

    # Scrollbars
    input_scrollbar = Scrollbar(input_frame, orient="vertical", command=input_text.yview)
    input_scrollbar.pack(side=RIGHT, fill=Y)
    input_text.configure(yscrollcommand=input_scrollbar.set)

    output_scrollbar = Scrollbar(output_frame, orient="vertical", command=output_text.yview)
    output_scrollbar.pack(side=RIGHT, fill=Y)
    output_text.configure(yscrollcommand=output_scrollbar.set)

    # Function to Analyze Code on Button Click
    def analyze_button_click():
        code = input_text.get("1.0", END).strip()  # Get the code from the input field
        if not code:
            update_output("Error: Please enter some code!")
            return
        try:
            status, suggested_fix = analyze_code(code)
            if status == "buggy":
                output = f"Status: {status}\nSuggested Fix:\n{suggested_fix}"
            else:
                output = "Status: bug-free\nThe code appears to be correct!"
        except Exception as e:
            output = f"Error: {str(e)}"
        update_output(output)

    # Update the output box
    def update_output(text):
        output_text.configure(state="normal")
        output_text.delete("1.0", END)
        output_text.insert("1.0", text)
        output_text.configure(state="disabled")

    # Analyze Code Button
    analyze_button = Button(root, text="Analyze Code", command=analyze_button_click, bg="blue", fg="white")
    analyze_button.pack(pady=10)

    # Run the application
    root.mainloop()

# Step 4: Launch the GUI
if __name__ == "__main__":
    run_gui()
    
