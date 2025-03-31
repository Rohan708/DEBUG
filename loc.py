from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForCausalLM

# Step 1: Ensure the app object is correctly defined for FastAPI
app = FastAPI()

# Step 2: Load Models and Tokenizers
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

# Step 3: Define API schema
class CodeSnippet(BaseModel):
    code: str

# Step 4: Unified Endpoint for Analyze Code (Bug Detection + Fix Recommendation)
@app.post("/analyze-code")
async def analyze_code(snippet: CodeSnippet):
    try:
        # Validate input
        if not snippet.code.strip():
            return {"error": "Code snippet is empty or invalid"}

        # Detect if the code is buggy
        inputs = detection_tokenizer(snippet.code, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = detection_model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        status = "buggy" if prediction == 1 else "bug-free"

        # Generate a suggested fix if the code is buggy
        suggested_fix = None
        if status == "buggy":
            inputs_fix = fix_tokenizer.encode(snippet.code, return_tensors="pt", max_length=128, truncation=True).to(device)
            outputs_fix = fix_model.generate(inputs_fix, max_length=256, num_return_sequences=1, temperature=0.7)
            suggested_fix = fix_tokenizer.decode(outputs_fix[0], skip_special_tokens=True)

        # Return structured response
        return {
            "status": status,
            "suggested_fix": suggested_fix if suggested_fix else "No fixes needed"
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Step 5: Serve index.html as the Frontend
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Step 6: Serve Static Files (Optional for CSS/JS/Images)
app.mount("/static", StaticFiles(directory="static"), name="static")