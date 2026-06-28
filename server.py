import os
import json
import threading
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from dataset import MultimodalNIHDataset, FINDINGS, CLINICAL_COLS, create_mock_nih_dataset
from model import MultimodalAttentionFusionModel
from train import run_training_pipeline

app = FastAPI(title="Multimodal Medical Diagnosis System API")

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project paths
DATA_DIR = './data_mock'
STATIC_DIR = './static'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Training status global variables
training_status = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "message": "System Idle. Ready to train.",
    "completed": False
}

# Loaded scaler statistics
scaler_stats = {}
multimodal_model = None
model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_scaler_stats_and_model():
    """
    Computes mean/std on the current dataset to scale incoming single predictions.
    Also loads the trained multimodal model weights if they exist.
    """
    global scaler_stats, multimodal_model
    csv_path = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Check numericals
        num_cols = ['Patient Age', 'Body Temperature', 'Oxygen Saturation', 'WBC Count', 'Heart Rate', 'Systolic BP']
        
        # Simple stats calculator
        stats = {}
        for col in num_cols:
            vals = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median()).values
            stats[col] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()) if vals.std() > 0 else 1.0
            }
        scaler_stats = stats
        print("Scaler statistics loaded successfully.")
    
    # Load model weights
    weight_path = os.path.join(DATA_DIR, 'multimodal_weights.pth')
    if os.path.exists(weight_path):
        # We need clinical dimension (which is 10)
        multimodal_model = MultimodalAttentionFusionModel(clinical_dim=10, num_classes=len(FINDINGS))
        try:
            multimodal_model.load_state_dict(torch.load(weight_path, map_location=model_device))
            multimodal_model.to(model_device)
            multimodal_model.eval()
            print("Trained multimodal model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            multimodal_model = None


def background_training_task(epochs: int):
    global training_status
    training_status["is_training"] = True
    training_status["total_epochs"] = epochs
    training_status["completed"] = False
    
    try:
        # Check if dataset exists, if not generate it
        csv_path = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')
        if not os.path.exists(csv_path):
            training_status["message"] = "Generating synthetic NIH chest X-ray images..."
            create_mock_nih_dataset(DATA_DIR, num_patients=50, num_images=100)
            
        training_status["message"] = "Initializing training pipeline..."
        
        # Override the print/status reporting within training
        # We run it synchronously in this thread
        run_training_pipeline(DATA_DIR, epochs=epochs, batch_size=16)
        
        training_status["message"] = "Training completed successfully. Loading model weights..."
        load_scaler_stats_and_model()
        training_status["is_training"] = False
        training_status["completed"] = True
    except Exception as e:
        training_status["message"] = f"Training failed with error: {str(e)}"
        training_status["is_training"] = False
        training_status["completed"] = False


@app.on_event("startup")
def startup_event():
    # Make sure we have a mock dataset and trained weights so the app is instantly usable
    csv_path = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')
    if not os.path.exists(csv_path):
        print("No dataset found. Generating default mock NIH dataset...")
        create_mock_nih_dataset(DATA_DIR, num_patients=50, num_images=100)
        
    load_scaler_stats_and_model()
    
    # If weights do not exist, run a fast training (1 epoch) in the background so weights are available
    weight_path = os.path.join(DATA_DIR, 'multimodal_weights.pth')
    if not os.path.exists(weight_path):
        print("No pre-trained weights found. Starting a quick 1-epoch background training...")
        threading.Thread(target=background_training_task, args=(1,)).start()


@app.post("/api/train")
def train_model(epochs: int = Form(5)):
    if training_status["is_training"]:
        return JSONResponse(status_code=400, content={"message": "Training is already in progress."})
        
    threading.Thread(target=background_training_task, args=(epochs,)).start()
    return {"message": f"Training started in the background for {epochs} epochs."}


@app.get("/api/status")
def get_status():
    return training_status


@app.get("/api/evaluate")
def get_evaluation_metrics():
    metrics_path = os.path.join(DATA_DIR, 'metrics.json')
    if not os.path.exists(metrics_path):
        return JSONResponse(status_code=404, content={"message": "Evaluation metrics not found. Please train models first."})
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


@app.get("/api/samples")
def get_samples():
    """
    Returns a set of 4 curated clinical samples representing 
    different findings (Normal, Pneumonia, Cardiomegaly, Pneumothorax)
    for easy frontend testing.
    """
    csv_path = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Dataset not initialized.")
        
    df = pd.read_csv(csv_path)
    
    samples = []
    # Find normal sample
    normal_df = df[df['Finding Labels'] == 'No Finding']
    if len(normal_df) > 0:
        samples.append(normal_df.iloc[0].to_dict())
        
    # Find Cardiomegaly sample
    cardio_df = df[df['Finding Labels'].str.contains('Cardiomegaly', na=False)]
    if len(cardio_df) > 0:
        samples.append(cardio_df.iloc[0].to_dict())
        
    # Find Pneumonia sample
    pneum_df = df[df['Finding Labels'].str.contains('Pneumonia', na=False)]
    if len(pneum_df) > 0:
        samples.append(pneum_df.iloc[0].to_dict())
        
    # Find Pneumothorax sample
    pneumothorax_df = df[df['Finding Labels'].str.contains('Pneumothorax', na=False)]
    if len(pneumothorax_df) > 0:
        samples.append(pneumothorax_df.iloc[0].to_dict())
        
    # Fallbacks if none found
    while len(samples) < 4 and len(df) > len(samples):
        samples.append(df.iloc[len(samples)].to_dict())
        
    # Remove path extensions for safety, return local URL or indices
    return samples


@app.post("/api/predict")
async def predict_diagnosis(
    file: UploadFile = File(...),
    age: float = Form(...),
    gender: str = Form(...),
    view_pos: str = Form(...),
    temp: float = Form(...),
    spo2: float = Form(...),
    wbc: float = Form(...),
    hr: float = Form(...),
    sbp: float = Form(...),
    cough: str = Form(...),
    pain: str = Form(...),
    explain_class: str = Form("Cardiomegaly") # Default class to compute Grad-CAM for
):
    global multimodal_model, scaler_stats
    
    if multimodal_model is None:
        # Attempt reloading
        load_scaler_stats_and_model()
        if multimodal_model is None:
            return JSONResponse(status_code=400, content={"message": "Model weights are not trained/loaded yet. Please train the model."})
            
    # 1. Preprocess uploaded image
    try:
        image = Image.open(file.file).convert('L')
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Invalid image file: {str(e)}"})
        
    # Image resize and conversion to PyTorch float tensor (1, 1, 128, 128)
    image_resized = image.resize((128, 128))
    img_np = np.array(image_resized).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_np).unsqueeze(0).unsqueeze(0).to(model_device)
    
    # 2. Preprocess Clinical Record features
    # Ensure stats are loaded
    if not scaler_stats:
        load_scaler_stats_and_model()
        
    # Scale numericals using training metrics
    try:
        s_age = (age - scaler_stats.get('Patient Age', {'mean': 45.0, 'std': 15.0})['mean']) / scaler_stats.get('Patient Age', {'mean': 45.0, 'std': 15.0})['std']
        s_temp = (temp - scaler_stats.get('Body Temperature', {'mean': 36.8, 'std': 0.5})['mean']) / scaler_stats.get('Body Temperature', {'mean': 36.8, 'std': 0.5})['std']
        s_spo2 = (spo2 - scaler_stats.get('Oxygen Saturation', {'mean': 98.0, 'std': 1.5})['mean']) / scaler_stats.get('Oxygen Saturation', {'mean': 98.0, 'std': 1.5})['std']
        s_wbc = (wbc - scaler_stats.get('WBC Count', {'mean': 7000.0, 'std': 1500.0})['mean']) / scaler_stats.get('WBC Count', {'mean': 7000.0, 'std': 1500.0})['std']
        s_hr = (hr - scaler_stats.get('Heart Rate', {'mean': 75.0, 'std': 10.0})['mean']) / scaler_stats.get('Heart Rate', {'mean': 75.0, 'std': 10.0})['std']
        s_sbp = (sbp - scaler_stats.get('Systolic BP', {'mean': 120.0, 'std': 8.0})['mean']) / scaler_stats.get('Systolic BP', {'mean': 120.0, 'std': 8.0})['std']
    except Exception:
        # Fallbacks
        s_age, s_temp, s_spo2, s_wbc, s_hr, s_sbp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Categorical encodings (same as in dataset.py preprocessing)
    gender_val = 0.0 if gender == 'M' else 1.0
    view_val = 0.0 if view_pos == 'PA' else 1.0
    
    cough_map = {'None': 0.0, 'Mild': 1.0, 'Moderate': 2.0, 'Severe': 3.0}
    cough_val = cough_map.get(cough, 0.0)
    
    pain_map = {'None': 0.0, 'Dull': 1.0, 'Sharp': 2.0}
    pain_val = pain_map.get(pain, 0.0)
    
    # Pack into feature array (10 columns matching CLINICAL_COLS)
    clin_array = np.array([
        s_age, gender_val, view_val,
        s_temp, s_spo2, s_wbc,
        s_hr, s_sbp, cough_val, pain_val
    ], dtype=np.float32)
    
    clinical_tensor = torch.tensor(clin_array).unsqueeze(0).to(model_device)
    
    # 3. Model Inference & Grad-CAM Backprop
    # Set grads enabled to compute backprop for Grad-CAM
    multimodal_model.train() # Set to train or keep eval with grad enabled
    for param in multimodal_model.parameters():
        param.requires_grad = True
        
    logits = multimodal_model(img_tensor, clinical_tensor)
    probs = torch.sigmoid(logits).squeeze(0)
    prob_dict = {FINDINGS[i]: float(probs[i]) for i in range(len(FINDINGS))}
    
    # Grad-CAM for selected class
    if explain_class not in FINDINGS:
        explain_class = FINDINGS[0]
        
    class_idx = FINDINGS.index(explain_class)
    score = logits[0, class_idx]
    
    multimodal_model.zero_grad()
    score.backward(retain_graph=True)
    
    # Extract ViT activations and gradients
    grads = multimodal_model.vit.gradients
    acts = multimodal_model.vit.activations
    
    cam_grid = []
    if grads is not None and acts is not None:
        grads = grads[0] # (65, 64)
        acts = acts[0]   # (65, 64)
        
        # Compute mean gradient per embedding channel over patches (1 to 64)
        weights = torch.mean(grads[1:], dim=0) # (64,)
        
        # Weighted combination of patch activations
        cam = torch.zeros(grads.shape[0] - 1, dtype=torch.float32, device=grads.device)
        for i in range(len(weights)):
            cam += weights[i] * acts[1:, i]
            
        # ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalize
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max
            
        cam_np = cam.cpu().detach().numpy()
        cam_grid = cam_np.reshape(8, 8).tolist() # Return as 8x8 list
    else:
        # Fallback uniform attention grid
        cam_grid = (np.ones((8, 8), dtype=np.float32) * 0.1).tolist()
        
    # 4. Extract Clinical Attention weights
    # multimodal_model.attention_weights contains 11 values:
    # Index 0: visual token. Index 1-10: clinical features
    attn_weights = multimodal_model.attention_weights
    clin_attn = {}
    if attn_weights is not None:
        # Convert to numpy
        attn_np = attn_weights[0].cpu().detach().numpy()
        # Normalise clinical features relative to each other for display
        raw_clin_attn = attn_np[1:]
        sum_attn = np.sum(raw_clin_attn) if np.sum(raw_clin_attn) > 0 else 1.0
        norm_clin_attn = (raw_clin_attn / sum_attn) * 100 # percentage
        
        for idx, col in enumerate(CLINICAL_COLS):
            clin_attn[col] = float(norm_clin_attn[idx])
    else:
        # Equal distribution fallback
        for col in CLINICAL_COLS:
            clin_attn[col] = 10.0
            
    # Reset model to eval state
    multimodal_model.eval()
    for param in multimodal_model.parameters():
        param.requires_grad = False
        
    return {
        "predictions": prob_dict,
        "grad_cam_grid": cam_grid,
        "clinical_attention": clin_attn
    }


# Endpoint to serve mock images if requested by index
@app.get("/api/images/{filename}")
def get_dataset_image(filename: str):
    img_path = os.path.join(DATA_DIR, 'images', filename)
    if os.path.exists(img_path):
        return FileResponse(img_path)
    return JSONResponse(status_code=404, content={"message": "Image not found."})


# Root serves dashboard
@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Server online. Frontend folder ./static is empty or index.html is missing."}

# Mount static files (app.js, style.css, etc.)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
