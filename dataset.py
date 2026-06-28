import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

# 14 NIH ChestX-ray14 findings
FINDINGS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
    'Emphysema', 'Fibrosis', 'Pleural Thickening', 'Hernia'
]

CLINICAL_COLS = [
    'Patient Age', 'Patient Gender', 'View Position',
    'Body Temperature', 'Oxygen Saturation', 'WBC Count',
    'Heart Rate', 'Systolic BP', 'Cough Severity', 'Chest Pain'
]

def generate_mock_image(findings, size=(128, 128)):
    """
    Generates a synthetic chest X-ray image with visual features correlated 
    to specific finding labels.
    """
    # Create dark gray chest cavity background
    img = Image.new('L', size, color=40)
    draw = ImageDraw.Draw(img)
    w, h = size
    
    # Draw spine column (vertical light gray line in the middle)
    draw.rectangle([w//2 - 4, 10, w//2 + 4, h - 10], fill=120)
    
    # Draw rib cages (curved light gray lines)
    for i in range(20, h - 15, 12):
        # Left rib
        draw.arc([10, i - 5, w//2 - 5, i + 10], start=180, end=270, fill=100, width=2)
        # Right rib
        draw.arc([w//2 + 5, i - 5, w - 10, i + 10], start=270, end=360, fill=100, width=2)
        
    # Draw lungs (two dark lobes)
    # Left Lung
    draw.ellipse([15, 20, w//2 - 8, h - 20], fill=15)
    # Right Lung
    draw.ellipse([w//2 + 8, 20, w - 15, h - 20], fill=15)
    
    # Draw diaphragm (curved bottom)
    draw.chord([10, h - 30, w - 10, h + 20], start=180, end=360, fill=50)

    # Draw clinical findings
    # Cardiomegaly: enlarged heart
    has_cardiomegaly = 'Cardiomegaly' in findings
    heart_width = 32 if has_cardiomegaly else 18
    heart_brightness = 160 if has_cardiomegaly else 130
    draw.ellipse([w//2 - heart_width, h//2 - 10, w//2 + heart_width, h - 25], fill=heart_brightness)
    
    # Effusion: fluid levels (accumulation of fluid at the bottom of lung fields)
    if 'Effusion' in findings:
        draw.rectangle([15, h - 35, w//2 - 8, h - 20], fill=140)
        draw.rectangle([w//2 + 8, h - 35, w - 15, h - 20], fill=140)
        
    # Pneumonia / Infiltration / Consolidation: cloudy light gray opacities in lungs
    if any(f in findings for f in ['Pneumonia', 'Infiltration', 'Consolidation', 'Edema']):
        # Random location in lungs
        # Left lung opacity
        draw.ellipse([20, 35, w//2 - 15, h//2 + 10], fill=90) # semi-opaque
        # Right lung opacity
        draw.ellipse([w//2 + 15, 45, w - 25, h//2 + 20], fill=80)
        
    # Pneumothorax: collapsed lung (dark area, with lung edge)
    if 'Pneumothorax' in findings:
        # Erase part of right lung (make it pitch black/air-filled) and draw thin white border
        draw.ellipse([w//2 + 18, 20, w - 18, h - 25], fill=5)
        draw.arc([w//2 + 12, 20, w - 22, h - 25], start=270, end=90, fill=180, width=1)
        
    # Mass / Nodule: small circular nodules
    if 'Mass' in findings or 'Nodule' in findings:
        r = 10 if 'Mass' in findings else 4
        # Left upper zone
        draw.ellipse([25 - r, 30 - r, 25 + r, 30 + r], fill=210)
        if 'Mass' in findings:
            # Right lower zone
            draw.ellipse([w - 35 - r, h - 50 - r, w - 35 + r, h - 50 + r], fill=200)

    # Apply small gaussian blur (represented simply as averaging locally)
    # Convert image back to numpy array to add slight noise, then back to PIL
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_clinical_vitals(findings, age, gender):
    """
    Generates realistic clinical vitals and lab results based on 
    assigned finding labels to match expected pathophysiology.
    """
    # Healthy baselines
    temp = np.random.normal(36.8, 0.2)
    spo2 = np.random.normal(98.0, 1.0)
    wbc = np.random.normal(7000, 1500)
    hr = np.random.normal(75, 10)
    sbp = np.random.normal(120, 8)
    cough = 0 # 0: None, 1: Mild, 2: Moderate, 3: Severe
    pain = 0  # 0: None, 1: Dull, 2: Sharp
    
    # Pathophysiology modifications
    if 'Cardiomegaly' in findings or 'Edema' in findings:
        sbp += np.random.normal(35, 12)  # hypertension
        hr += np.random.normal(20, 10)   # tachycardia
        
    if 'Pneumonia' in findings or 'Infiltration' in findings or 'Consolidation' in findings:
        temp += np.random.normal(1.8, 0.4) # fever
        wbc += np.random.normal(9000, 3000) # leukocytosis
        cough = np.random.choice([2, 3], p=[0.4, 0.6])
        
    if 'Edema' in findings or 'Pneumothorax' in findings or 'Effusion' in findings or 'Atelectasis' in findings:
        spo2 -= np.random.normal(7.0, 2.5) # hypoxia
        hr += np.random.normal(15, 8)
        
    if 'Pneumothorax' in findings:
        pain = 2 # sharp chest pain
        
    if 'Hernia' in findings or 'Pleural Thickening' in findings:
        pain = max(pain, 1) # dull chest pain
        
    if any(f in findings for f in ['Emphysema', 'Fibrosis', 'Atelectasis']):
        cough = max(cough, np.random.choice([1, 2], p=[0.5, 0.5]))
        spo2 -= np.random.normal(2.0, 1.0)
        
    # Boundary constraints
    temp = np.clip(temp, 35.0, 41.5)
    spo2 = np.clip(spo2, 60.0, 100.0)
    wbc = np.clip(wbc, 2000, 35000)
    hr = np.clip(hr, 45, 180)
    sbp = np.clip(sbp, 80, 220)
    
    # Map back to structured types
    cough_map = {0: 'None', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    pain_map = {0: 'None', 1: 'Dull', 2: 'Sharp'}
    
    return {
        'Body Temperature': round(temp, 1),
        'Oxygen Saturation': round(spo2, 1),
        'WBC Count': int(wbc),
        'Heart Rate': int(hr),
        'Systolic BP': int(sbp),
        'Cough Severity': cough_map[cough],
        'Chest Pain': pain_map[pain]
    }


def create_mock_nih_dataset(data_dir, num_patients=200, num_images=300):
    """
    Creates a full mock directory matching the NIH ChestX-ray14 format,
    including CSV entry file and image folder.
    """
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)
    csv_path = os.path.join(data_dir, 'Data_Entry_2017.csv')
    
    # Generate patient IDs
    patient_ids = [f'{i:05d}' for i in range(1, num_patients + 1)]
    
    records = []
    
    print(f"Generating {num_images} mock NIH medical images and clinical metadata...")
    
    for i in range(num_images):
        img_name = f'images/000{i:05d}.png'
        patient_id = np.random.choice(patient_ids)
        age = int(np.random.randint(18, 88))
        gender = np.random.choice(['M', 'F'], p=[0.52, 0.48])
        view_pos = np.random.choice(['PA', 'AP'], p=[0.6, 0.4])
        
        # Decide finding labels (can be multiple labels, separated by '|')
        # Probability of having no findings (Normal) is high, representing real dataset
        has_finding = np.random.choice([True, False], p=[0.6, 0.4])
        if has_finding:
            # Choose 1 to 3 random findings
            num_findings = np.random.choice([1, 2, 3], p=[0.75, 0.2, 0.05])
            chosen = np.random.choice(FINDINGS, size=num_findings, replace=False)
            finding_str = '|'.join(chosen)
        else:
            finding_str = 'No Finding'
            
        # Draw image
        findings_list = finding_str.split('|') if finding_str != 'No Finding' else []
        img = generate_mock_image(findings_list)
        img.save(os.path.join(data_dir, img_name))
        
        # Vitals
        vitals = generate_clinical_vitals(findings_list, age, gender)
        
        record = {
            'Image Index': f'000{i:05d}.png',
            'Finding Labels': finding_str,
            'Follow-up #': np.random.randint(0, 10),
            'Patient ID': patient_id,
            'Patient Age': age,
            'Patient Gender': gender,
            'View Position': view_pos,
            **vitals
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated at {data_dir} successfully!")


class MultimodalNIHDataset(Dataset):
    """
    PyTorch Dataset loading images and structured clinical records 
    from NIH ChestX-ray14 dataset directory, with pre-processing logic.
    """
    def __init__(self, data_dir, csv_file='Data_Entry_2017.csv', transform=None, is_train=True, split_ratio=0.8):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load CSV
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}. Please check your path.")
            
        self.df = pd.read_csv(csv_path)
        
        # Ensure clinical columns exist (if missing, augment them on the fly)
        # This handles loading the real NIH csv file which lacks vitals
        for col in CLINICAL_COLS:
            if col not in self.df.columns:
                print(f"Clinical column '{col}' missing. Augmenting CSV with synthetic clinical parameters...")
                self._augment_missing_clinical_columns()
                # Save augmented file back
                self.df.to_csv(csv_path, index=False)
                break
                
        # Split into train/validation sets based on Patient ID to prevent leakage
        unique_patients = self.df['Patient ID'].unique()
        np.random.seed(42) # determinism
        np.random.shuffle(unique_patients)
        
        split_idx = int(len(unique_patients) * split_ratio)
        train_patients = unique_patients[:split_idx]
        val_patients = unique_patients[split_idx:]
        
        if is_train:
            self.df = self.df[self.df['Patient ID'].isin(train_patients)].reset_index(drop=True)
        else:
            self.df = self.df[self.df['Patient ID'].isin(val_patients)].reset_index(drop=True)
            
        # Extract visual targets (Multi-label one-hot vectors)
        self.labels = np.zeros((len(self.df), len(FINDINGS)), dtype=np.float32)
        for idx, row in self.df.iterrows():
            findings = str(row['Finding Labels']).split('|')
            for f in findings:
                if f in FINDINGS:
                    f_idx = FINDINGS.index(f)
                    self.labels[idx, f_idx] = 1.0

        # Preprocess Clinical Records
        self._preprocess_clinical_data()
        
    def _augment_missing_clinical_columns(self):
        """
        Helper to add clinically logical EHR variables to the existing NIH records.
        """
        vitals_list = []
        for _, row in self.df.iterrows():
            findings = str(row['Finding Labels']).split('|') if str(row['Finding Labels']) != 'No Finding' else []
            vitals = generate_clinical_vitals(findings, row['Patient Age'], row['Patient Gender'])
            vitals_list.append(vitals)
            
        vitals_df = pd.DataFrame(vitals_list)
        for col in vitals_df.columns:
            self.df[col] = vitals_df[col]

    def _preprocess_clinical_data(self):
        """
        Handles categorical variables, standardizes numerical data, and manages missing values.
        """
        # Copy to avoid settings with copy warnings
        df_cli = self.df[CLINICAL_COLS].copy()
        
        # 1. Fill missing values (Impute with medians/modes)
        # Check numericals
        num_cols = ['Patient Age', 'Body Temperature', 'Oxygen Saturation', 'WBC Count', 'Heart Rate', 'Systolic BP']
        for col in num_cols:
            df_cli[col] = pd.to_numeric(df_cli[col], errors='coerce')
            median_val = df_cli[col].median()
            df_cli[col] = df_cli[col].fillna(median_val)
            
        # Check categoricals
        cat_cols = ['Patient Gender', 'View Position', 'Cough Severity', 'Chest Pain']
        for col in cat_cols:
            df_cli[col] = df_cli[col].astype(str)
            mode_val = df_cli[col].mode()[0]
            df_cli[col] = df_cli[col].replace('nan', mode_val).replace('', mode_val)

        # 2. Standardize numerical attributes (Z-score Scaling)
        # Using fixed parameters so it scales consistently (or fit stats dynamically)
        # For simplicity, we scale based on the current split's stats
        self.numerical_means = df_cli[num_cols].mean().to_dict()
        self.numerical_stds = df_cli[num_cols].std().replace(0, 1.0).to_dict()
        
        for col in num_cols:
            mean = self.numerical_means[col]
            std = self.numerical_stds[col]
            df_cli[col] = (df_cli[col] - mean) / std

        # 3. Categorical encoding
        # Explicit mappings for safety
        gender_map = {'M': 0.0, 'F': 1.0}
        view_map = {'PA': 0.0, 'AP': 1.0}
        cough_map = {'None': 0.0, 'Mild': 1.0, 'Moderate': 2.0, 'Severe': 3.0}
        pain_map = {'None': 0.0, 'Dull': 1.0, 'Sharp': 2.0}
        
        # Map values
        df_cli['Patient Gender'] = df_cli['Patient Gender'].map(gender_map).fillna(0.0)
        df_cli['View Position'] = df_cli['View Position'].map(view_map).fillna(0.0)
        df_cli['Cough Severity'] = df_cli['Cough Severity'].map(cough_map).fillna(0.0)
        df_cli['Chest Pain'] = df_cli['Chest Pain'].map(pain_map).fillna(0.0)
        
        # Convert categoricals to one-hot or keep mapped values as dense inputs.
        # Keeping mapped values is cleaner and works perfectly with MLPs.
        # We will convert the dataframe to numpy
        self.clinical_features = df_cli.values.astype(np.float32)
        self.clinical_dim = self.clinical_features.shape[1]

    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        # 1. Load image
        img_name = self.df.iloc[idx]['Image Index']
        img_path = os.path.join(self.data_dir, 'images', os.path.basename(img_name))
        
        try:
            # Grayscale load
            img = Image.open(img_path).convert('L')
        except Exception as e:
            # Fallback if image doesn't exist (e.g. broken link in real dataset)
            findings_list = str(self.df.iloc[idx]['Finding Labels']).split('|') if str(self.df.iloc[idx]['Finding Labels']) != 'No Finding' else []
            img = generate_mock_image(findings_list)
            
        if self.transform:
            img = self.transform(img)
        else:
            # Standard tensor transformation: resize to 128x128, scale 0-1
            img = img.resize((128, 128))
            img_np = np.array(img).astype(np.float32) / 255.0
            # Add channel dimension: (1, 128, 128)
            img = torch.tensor(img_np).unsqueeze(0)
            
        # 2. Get clinical feature vector
        clinical_vec = torch.tensor(self.clinical_features[idx])
        
        # 3. Get labels
        label_vec = torch.tensor(self.labels[idx])
        
        return img, clinical_vec, label_vec
