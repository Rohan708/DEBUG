from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Step 1: Load Pre-trained Generative Model
def load_model():
    model_name = "microsoft/CodeGPT-small-py"  # Replace with your chosen generative model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# Step 2: Preprocess Buggy Code Snippet
def preprocess_input(buggy_code, tokenizer, max_length=128):
    inputs = tokenizer.encode(buggy_code, return_tensors="pt", max_length=max_length, truncation=True)
    return inputs

# Step 3: Generate Fix Recommendation
def generate_fix(inputs, model, tokenizer, device, max_length=256):
    inputs = inputs.to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, temperature=0.7)
    suggested_fix = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return suggested_fix

# Step 4: Main Function to Run the Pipeline
def main():
    print("Loading model...")
    tokenizer, model, device = load_model()

    print("Enter the buggy code snippet:")
    buggy_code = input("> ")

    print("Preprocessing input...")
    inputs = preprocess_input(buggy_code, tokenizer)

    print("Generating fix recommendation...")
    suggested_fix = generate_fix(inputs, model, tokenizer, device)
    
    print("\nSuggested Fix:")
    print(suggested_fix)

if __name__ == "__main__":
    main()
    