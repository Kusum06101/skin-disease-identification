import torch
import io
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.templating import Jinja2Templates # For serving HTML
from fastapi.staticfiles import StaticFiles # For serving static assets
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any 
import uvicorn
import json # ADDED: Import the json library for data loading

# --- 0. DATA LOADING CONFIGURATION ---
DISEASE_DATA_FILE = 'disease_data.json' # Define the JSON file name

def load_disease_details(file_path: str) -> Dict[str, Dict[str, str]]:
    """Loads detailed disease information from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The data file '{file_path}' was not found. Please create it.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return {}

# --- GLOBAL DATA VARIABLE (LOADED HERE) ---
DISEASE_DETAILS: Dict[str, Dict[str, str]] = load_disease_details(DISEASE_DATA_FILE)

# --- UTILITY FUNCTION ---
def get_disease_details(disease_name: str) -> Dict[str, str]:
    """Retrieves the detailed data from the source, including Ayurvedic diagnosis."""
    # Use .get() for safe retrieval with a fallback for all fields
    default_details = {
        'common_name': 'Details Unavailable',
        'cause': 'Cause details are not yet available for this disease in the data source.',
        'remedy': 'Remedy details are not yet available for this disease in the data source.',
        'usage': 'Usage instructions are not yet available for this disease in the data source.',
        'note': 'Consult a dermatologist for professional diagnosis and treatment.',
        'ayurvedic_diagnosis': 'Ayurvedic diagnosis is not yet available for this disease in the data source.', # ADDED FALLBACK
    }
    # Uses the globally loaded DISEASE_DETAILS
    return DISEASE_DETAILS.get(disease_name, default_details)


# --- 1. CONFIGURATION AND MODEL LOADING ---
repo_name = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
# Define device (use CUDA if available, otherwise use CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables for model and processor
image_processor = None
model = None

# Define the prediction result structure using Pydantic 
# ADDED 'ayurvedic_diagnosis'
class PredictionResult(BaseModel):
    predicted_disease: str
    confidence: float
    common_name: str
    cause: str
    remedy: str
    usage: str
    note: str
    ayurvedic_diagnosis: str # NEW FIELD ADDED

# Define the class names (must match the model's output)
CLASS_NAMES = [
    'Basal Cell Carcinoma', 'Darier_s Disease', 'Epidermolysis Bullosa Pruriginosa', 
    'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans', 
    'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid', 
    'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma', 
    'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis', 
    'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis', 
    'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis', 'Tinea Corporis', 
    'Tinea Nigra', 'Tungiasis', 'actinic keratosis', 'dermatofibroma', 'nevus', 
    'pigmented benign keratosis', 'seborrheic keratosis', 'squamous cell carcinoma', 
    'vascular lesion'
]

app = FastAPI()

# Configure template directory for HTML
templates = Jinja2Templates(directory="templates")
# Optional: Mount a directory for static files (e.g., CSS/JS) if you create a 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def load_resources():
    """Load the model and processor when the application starts."""
    global image_processor, model
    
    # Check if data was loaded successfully. If not, raise an error to stop startup.
    if not DISEASE_DETAILS:
        print("Data initialization failed. Cannot proceed without disease details.")
        raise RuntimeError("Failed to load critical disease data from JSON file.")

    try:
        print(f"Loading image processor and model from {repo_name} to {DEVICE}...")
        image_processor = AutoImageProcessor.from_pretrained(repo_name)
        model = AutoModelForImageClassification.from_pretrained(repo_name).to(DEVICE)
        model.eval() # Set model to evaluation mode
        print("Model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading model components: {e}")
        # Re-raise the exception to prevent the server from starting with a broken model
        raise RuntimeError("Failed to load machine learning model components.") from e
        
# --- 2. API ENDPOINT FOR PREDICTION (UPDATED TO RETURN AYURVEDIC DIAGNOSIS) ---

@app.post("/api/predict", response_model=PredictionResult)
async def predict_skin_disease(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the top predicted skin disease and the 
    detailed structured data, now including Ayurvedic diagnosis.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # 1. Read the image file contents into a BytesIO object
        image_bytes = await file.read()
        image_stream = io.BytesIO(image_bytes)
        
        # 2. Open the image using PIL and convert to RGB
        image = Image.open(image_stream).convert("RGB")

        # 3. Preprocess the image and move it to the correct device
        encoding = image_processor(images=image, return_tensors="pt").to(DEVICE)

        # 4. Make a prediction
        with torch.no_grad():
            outputs = model(**encoding)
            # Logits are the raw outputs from the model
            logits = outputs.logits.cpu() # Move logits back to CPU for standard ops

        # 5. Calculate probabilities and confidence
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get the single top result
        top_k = 1
        top_k_values, top_k_indices = torch.topk(probabilities, top_k)
        
        # Get the single predicted class index and confidence
        predicted_class_idx = top_k_indices[0].item()
        predicted_confidence = top_k_values[0].item()
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        
        # 6. Fetch the detailed structured data
        details = get_disease_details(predicted_class_name)

        # 7. Return the structured result (NEW FIELD INCLUDED HERE)
        # Note: confidence is multiplied by 100 for percentage display on the frontend
        return PredictionResult(
            predicted_disease=predicted_class_name,
            confidence=round(predicted_confidence * 100, 2),
            common_name=details['common_name'],
            cause=details['cause'],
            remedy=details['remedy'],
            usage=details['usage'],
            note=details['note'],
            ayurvedic_diagnosis=details['ayurvedic_diagnosis'] # Mapped from the dictionary
        )

    except Exception as e:
        # Catch any unexpected errors during prediction
        print(f"\n--- INTERNAL PREDICTION ERROR ---\n{type(e).__name__}: {e}\n---------------------------------")
        
        # Return a 500 Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed due to an internal server error: {type(e).__name__}. Check server logs for details."
        )

# --- 3. HTML Frontend Route ---

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the main index.html file located in the 'templates' directory."""
    return templates.TemplateResponse("index.html", {"request": request})

# --- 4. RUNNING THE APPLICATION ---
if __name__ == "__main__":
    # Recommended host for local testing is 127.0.0.1 (localhost)
    uvicorn.run(app, host="127.0.0.1", port=8000)