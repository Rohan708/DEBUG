import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# Load models and tokenizers
def load_models():
    # Bug Detection Model
    detection_model_name = "microsoft/codebert-base"
    detection_tokenizer = RobertaTokenizer.from_pretrained(detection_model_name)
    detection_model = RobertaForSequenceClassification.from_pretrained(detection_model_name, num_labels=2)
    detection_model.load_state_dict(torch.load("bug_detection_model.pt", map_location=torch.device('cpu')))
    detection_model.eval()

    # Fix Recommendation Model
    fix_model_name = "microsoft/CodeGPT-small-py"
    fix_tokenizer = AutoTokenizer.from_pretrained(fix_model_name)
    fix_model = AutoModelForCausalLM.from_pretrained(fix_model_name)

    return detection_model, detection_tokenizer, fix_model, fix_tokenizer

# Initialize models
detection_model, detection_tokenizer, fix_model, fix_tokenizer = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detection_model.to(device)
fix_model.to(device)

def analyze_code(code_snippet):
    # Detect if the code is buggy
    inputs = detection_tokenizer(code_snippet, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = detection_model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    status = "buggy" if prediction == 1 else "bug-free"

    # Generate fix if buggy
    suggested_fix = None
    if status == "buggy":
        inputs_fix = fix_tokenizer.encode(code_snippet, return_tensors="pt", max_length=128, truncation=True).to(device)
        outputs_fix = fix_model.generate(inputs_fix, max_length=256, num_return_sequences=1, temperature=0.7)
        suggested_fix = fix_tokenizer.decode(outputs_fix[0], skip_special_tokens=True)

    return status, suggested_fix

if __name__ == "__main__":
    print("Welcome to the Bug Detection and Fix Suggestion Tool!")
    while True:
        # Accept code input from the user
        code_snippet = input("\nEnter your code snippet (or type 'exit' to quit):\n")
        if code_snippet.lower() == "exit":
            print("Goodbye!")
            break

        # Analyze the code
        status, suggested_fix = analyze_code(code_snippet)

        # Print the results
        print(f"\nStatus: {status}")
        if status == "buggy" and suggested_fix:
            print(f"Suggested Fix:\n{suggested_fix}")
        elif status == "bug-free":
            print("The code appears to be bug-free!")