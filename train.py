import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

from dataset import MultimodalNIHDataset, FINDINGS
from model import MultimodalAttentionFusionModel, ImageOnlyModel, ClinicalOnlyModel

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, clinical, labels in dataloader:
        imgs = imgs.to(device)
        clinical = clinical.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs, clinical)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, clinical, labels in dataloader:
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            labels = labels.to(device)
            
            logits = model(imgs, clinical)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return val_loss, all_preds, all_targets


def calculate_metrics(targets, preds, threshold=0.5):
    """
    Computes Accuracy, Precision, Recall, F1, and ROC-AUC per class,
    and returns a summary of average values.
    """
    # Binary predictions based on threshold
    preds_binary = (preds >= threshold).astype(float)
    
    # Calculate global metrics
    # Using 'macro' average for multi-label classification
    precision, recall, f1, _ = precision_recall_fscore_support(targets, preds_binary, average='macro', zero_division=0)
    
    # Calculate simple accuracy (ratio of exactly matched label sets or element-wise accuracy)
    # Element-wise binary accuracy
    accuracy = (preds_binary == targets).mean()
    
    # Class-wise ROC-AUC and general metrics
    auc_scores = {}
    class_roc_curves = {}
    
    for idx, name in enumerate(FINDINGS):
        target_cls = targets[:, idx]
        pred_cls = preds[:, idx]
        
        # Calculate AUC (safely handle cases with only 1 class present)
        if len(np.unique(target_cls)) > 1:
            auc = roc_auc_score(target_cls, pred_cls)
            # Downsample ROC curve data points to limit JSON size (max 20 points)
            fpr, tpr, _ = roc_curve(target_cls, pred_cls)
            indices = np.linspace(0, len(fpr) - 1, min(len(fpr), 20), dtype=int)
            class_roc_curves[name] = {
                'fpr': fpr[indices].tolist(),
                'tpr': tpr[indices].tolist()
            }
        else:
            auc = 0.5  # baseline default
            class_roc_curves[name] = {'fpr': [0.0, 1.0], 'tpr': [0.0, 1.0]}
            
        auc_scores[name] = float(auc)
        
    mean_auc = np.mean(list(auc_scores.values()))
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mean_auc': float(mean_auc),
        'class_auc': auc_scores,
        'roc_curves': class_roc_curves
    }


def run_training_pipeline(data_dir, epochs=5, batch_size=32, lr=0.001):
    """
    Trains all three models (Multimodal, Image-only, Clinical-only)
    and saves results and weights.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training pipeline running on device: {device}")
    
    # Load dataset splits
    train_dataset = MultimodalNIHDataset(data_dir, is_train=True)
    val_dataset = MultimodalNIHDataset(data_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    clinical_dim = train_dataset.clinical_dim
    num_classes = len(FINDINGS)
    
    # Initialize models
    models = {
        'Multimodal': MultimodalAttentionFusionModel(clinical_dim=clinical_dim, num_classes=num_classes),
        'Image-only': ImageOnlyModel(num_classes=num_classes),
        'Clinical-only': ClinicalOnlyModel(clinical_dim=clinical_dim, num_classes=num_classes)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n--- Training Model: {model_name} ---")
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {'train_loss': [], 'val_loss': []}
        
        best_val_loss = float('inf')
        best_metrics = {}
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_targets = evaluate_model(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save metrics if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = calculate_metrics(val_targets, val_preds)
                # Save model weight file
                weight_path = os.path.join(data_dir, f'{model_name.lower().replace("-", "_")}_weights.pth')
                torch.save(model.state_dict(), weight_path)
                
        results[model_name] = {
            'history': history,
            'metrics': {
                'accuracy': best_metrics['accuracy'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'f1': best_metrics['f1'],
                'mean_auc': best_metrics['mean_auc']
            },
            'class_auc': best_metrics['class_auc'],
            'roc_curves': best_metrics['roc_curves']
        }
        
    # Write metrics to json file
    metrics_path = os.path.join(data_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nTraining pipeline completed. Metrics saved to {metrics_path}")
    return results


if __name__ == '__main__':
    # Test script in mock mode
    from dataset import create_mock_nih_dataset
    data_dir = './data_mock'
    if not os.path.exists(data_dir):
        create_mock_nih_dataset(data_dir, num_patients=50, num_images=100)
    run_training_pipeline(data_dir, epochs=3)
