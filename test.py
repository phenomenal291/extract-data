import pandas as pd
import re
import rule
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def extract_tagged_phones(tagged_text):
    """
    Extract phone numbers from tagged text between <PHONE> tags.
    
    Args:
        tagged_text (str): Text with <PHONE>number</PHONE> tags
        
    Returns:
        list: Cleaned phone numbers starting with '0'
    """
    if pd.isna(tagged_text):
        return []
    
    # Extract content between <PHONE> and </PHONE> tags
    phone_pattern = r'<PHONE>(.*?)</PHONE>'
    matches = re.findall(phone_pattern, tagged_text, re.IGNORECASE)
    
    cleaned_phones = []
    for phone in matches:
        # Clean phone number same as rule.py
        clean_num = re.sub(r'[^\d]', '', phone)
        
        if clean_num.startswith('84'):
            clean_num = '0' + clean_num[2:]
        if clean_num.startswith('0084'):
            clean_num = '0' + clean_num[4:]
            
        if clean_num:
            cleaned_phones.append(clean_num)
    
    return cleaned_phones

def evaluate_phone_extraction(excel_file='data.xlsx'):
    """
    Evaluate phone number extraction performance.
    
    Args:
        excel_file (str): Path to Excel file with data
        
    Returns:
        dict: Evaluation metrics
    """
    # Read data
    df = pd.read_excel(excel_file)
    
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for idx, row in df.iterrows():
        content = row.get('content', '')
        tagged_content = row.get('human_tagged_content', '')
        
        # Extract ground truth phones
        true_phones = set(extract_tagged_phones(tagged_content))
        
        # Extract predicted phones using rule.py
        pred_phones = set(rule.phone(content))
        
        # Calculate metrics for this sample
        tp = len(true_phones.intersection(pred_phones))
        fp = len(pred_phones - true_phones)
        fn = len(true_phones - pred_phones)
        
        true_positives += tp
        false_positives += fp
        false_negatives += fn
        
        # For binary classification metrics
        has_true_phone = len(true_phones) > 0
        has_pred_phone = len(pred_phones) > 0
        
        y_true.append(1 if has_true_phone else 0)
        y_pred.append(1 if has_pred_phone else 0)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Binary classification metrics
    binary_precision = precision_score(y_true, y_pred, zero_division=0)
    binary_recall = recall_score(y_true, y_pred, zero_division=0)
    binary_f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp_binary, fn_binary, tp_binary = confusion_matrix(y_true, y_pred).ravel()
    
    results = {
        'exact_match_metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        },
        'binary_classification_metrics': {
            'precision': binary_precision,
            'recall': binary_recall,
            'f1_score': binary_f1,
            'true_positives': tp_binary,
            'true_negatives': tn,
            'false_positives': fp_binary,
            'false_negatives': fn_binary
        }
    }
    
    return results

def print_evaluation_report(results):
    """Print formatted evaluation report."""
    print("PHONE NUMBER EXTRACTION EVALUATION")
    print("=" * 50)
    
    print("\n1. EXACT MATCH METRICS:")
    exact = results['exact_match_metrics']
    print(f"   Precision: {exact['precision']:.4f}")
    print(f"   Recall:    {exact['recall']:.4f}")
    print(f"   F1-Score:  {exact['f1_score']:.4f}")
    print(f"   TP: {exact['true_positives']}, FP: {exact['false_positives']}, FN: {exact['false_negatives']}")
    
    print("\n2. BINARY CLASSIFICATION METRICS:")
    binary = results['binary_classification_metrics']
    print(f"   Precision: {binary['precision']:.4f}")
    print(f"   Recall:    {binary['recall']:.4f}")
    print(f"   F1-Score:  {binary['f1_score']:.4f}")
    print(f"   TP: {binary['true_positives']}, TN: {binary['true_negatives']}")
    print(f"   FP: {binary['false_positives']}, FN: {binary['false_negatives']}")

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_phone_extraction()
    print_evaluation_report(results)