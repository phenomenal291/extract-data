#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Vietnamese NER Prediction
Chỉ có function cơ bản để dự đoán NER từ model đã train
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List, Dict


class VietnameseNERPredictor:
    """Simple Vietnamese NER Predictor"""
    
    def __init__(self, model_path: str = "./mdeberta_ner_model/final"):
        """Initialize predictor with model"""
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")
    
    def predict_text(self, text: str) -> Dict:
        """
        Dự đoán entities trong văn bản
        
        Args:
            text: Văn bản cần dự đoán
            
        Returns:
            Dictionary chứa kết quả dự đoán
        """
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            return_offsets_mapping=True
        )
        
        offset_mapping = inputs.pop("offset_mapping")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=2)
            confidences = torch.softmax(outputs.logits, dim=2).max(dim=2)[0]
        
        # Convert to CPU
        predictions = predictions.cpu().numpy()[0]
        confidences = confidences.cpu().numpy()[0]
        offset_mapping = offset_mapping.cpu().numpy()[0]
        
        # Extract entities
        entities = []
        current_entity = None
        
        for i, (pred_id, conf, (start, end)) in enumerate(zip(predictions, confidences, offset_mapping)):
            if start == end:  # Skip special tokens
                continue
                
            label = self.model.config.id2label[int(pred_id)]
            
            if label == "O":
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
                continue
            
            entity_type = label[2:]  # Remove B- or I-
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                
                current_entity = {
                    "text": text[start:end],
                    "label": entity_type,
                    "start": int(start),
                    "end": int(end),
                    "confidence": float(conf)
                }
            elif label.startswith("I-") and current_entity and current_entity["label"] == entity_type:
                current_entity["text"] += text[start:end]
                current_entity["end"] = int(end)
                current_entity["confidence"] = max(current_entity["confidence"], float(conf))
        
        if current_entity:
            entities.append(current_entity)
        
        # Group by type
        entities_by_type = {}
        for entity in entities:
            label = entity["label"]
            if label not in entities_by_type:
                entities_by_type[label] = []
            entities_by_type[label].append(entity)
        
        return {
            "text": text,
            "entities": entities,
            "entities_by_type": entities_by_type,
            "total_entities": len(entities),
            "entity_types": list(entities_by_type.keys())
        }
    
    def format_results(self, results: Dict, show_confidence: bool = True) -> str:
        """Format kết quả để hiển thị"""
        output = []
        output.append(f"📝 Text: {results['text'][:100]}{'...' if len(results['text']) > 100 else ''}")
        output.append(f"📊 Found {results['total_entities']} entities")
        output.append("")
        
        if results['entities']:
            output.append("🏷️ Detected Entities:")
            for i, entity in enumerate(results['entities'], 1):
                confidence_str = f" (confidence: {entity['confidence']:.3f})" if show_confidence else ""
                output.append(f"   {i}. '{entity['text']}' → {entity['label']}{confidence_str}")
            
            output.append("")
            output.append("📋 By Category:")
            for entity_type in sorted(results['entities_by_type'].keys()):
                entities = results['entities_by_type'][entity_type]
                output.append(f"   {entity_type}: {len(entities)} entities")
                for entity in entities:
                    output.append(f"      • {entity['text']}")
        else:
            output.append("❌ No entities detected")
        
        return "\n".join(output)


def main():
    """Test function đơn giản"""
    # Simple test
    predictor = VietnameseNERPredictor()
    
    test_text ="""Bác sĩ Nguyễn Văn A làm việc tại Bệnh viện Bạch Mai ở Hà Nội."""
    
    results = predictor.predict_text(test_text)
    print(results)


if __name__ == "__main__":
    main()
