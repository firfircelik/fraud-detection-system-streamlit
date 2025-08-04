#!/usr/bin/env python3
"""
🔍 SHAP Explainability Module for Fraud Detection
Advanced model explanation using SHAP (SHapley Additive exPlanations)
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import json
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """SHAP-based model explainer for fraud detection"""
    
    def __init__(self):
        self.explainers = {}
        self.background_data = None
        self.feature_names = None
        self.explanation_cache = {}
        
    def initialize_explainers(self, models: Dict, background_data: np.ndarray, 
                            feature_names: List[str] = None):
        """Initialize SHAP explainers for all models"""
        
        self.background_data = background_data
        self.feature_names = feature_names or [f'feature_{i}' for i in range(background_data.shape[1])]
        
        for model_name, model_info in models.items():
            try:
                model = model_info['model']
                
                # Choose appropriate explainer based on model type
                if hasattr(model, 'predict_proba'):
                    # Tree-based models
                    if hasattr(model, 'estimators_'):
                        self.explainers[model_name] = shap.TreeExplainer(model)
                    else:
                        # Linear models
                        self.explainers[model_name] = shap.LinearExplainer(
                            model, background_data
                        )
                else:
                    # Kernel explainer for other models
                    self.explainers[model_name] = shap.KernelExplainer(
                        model.predict, background_data
                    )
                
                logger.info(f"SHAP explainer initialized for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize SHAP explainer for {model_name}: {e}")
    
    def explain_prediction(self, model_name: str, features: np.ndarray, 
                         transaction_id: str = None) -> Dict[str, Any]:
        """Generate SHAP explanation for a single prediction"""
        
        cache_key = f"{model_name}_{hash(str(features.tolist()))}"
        
        # Check cache first
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        try:
            if model_name not in self.explainers:
                raise ValueError(f"No explainer found for model {model_name}")
            
            explainer = self.explainers[model_name]
            
            # Generate SHAP values
            shap_values = explainer.shap_values(features)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Binary classification - take positive class
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            # Create explanation dictionary
            explanation = {
                'transaction_id': transaction_id,
                'model_name': model_name,
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                'feature_names': self.feature_names,
                'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0.0,
                'feature_contributions': self._calculate_feature_contributions(shap_values),
                'top_features': self._get_top_features(shap_values),
                'explanation_summary': self._generate_explanation_summary(shap_values),
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache the explanation
            self.explanation_cache[cache_key] = explanation
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP explanation failed for {model_name}: {e}")
            return {
                'transaction_id': transaction_id,
                'model_name': model_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_feature_contributions(self, shap_values: np.ndarray) -> Dict[str, float]:
        """Calculate feature contributions from SHAP values"""
        
        contributions = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(shap_values):
                contributions[feature_name] = float(shap_values[i])
        
        return contributions
    
    def _get_top_features(self, shap_values: np.ndarray, top_k: int = 10) -> List[Dict]:
        """Get top contributing features"""
        
        # Get absolute values for ranking
        abs_values = np.abs(shap_values)
        
        # Get top indices
        top_indices = np.argsort(abs_values)[-top_k:][::-1]
        
        top_features = []
        for idx in top_indices:
            if idx < len(self.feature_names):
                top_features.append({
                    'feature_name': self.feature_names[idx],
                    'shap_value': float(shap_values[idx]),
                    'abs_shap_value': float(abs_values[idx]),
                    'contribution_type': 'positive' if shap_values[idx] > 0 else 'negative'
                })
        
        return top_features
    
    def _generate_explanation_summary(self, shap_values: np.ndarray) -> Dict[str, Any]:
        """Generate human-readable explanation summary"""
        
        # Calculate total positive and negative contributions
        positive_contrib = np.sum(shap_values[shap_values > 0])
        negative_contrib = np.sum(shap_values[shap_values < 0])
        
        # Find most important features
        abs_values = np.abs(shap_values)
        most_important_idx = np.argmax(abs_values)
        
        summary = {
            'total_positive_contribution': float(positive_contrib),
            'total_negative_contribution': float(negative_contrib),
            'net_contribution': float(positive_contrib + negative_contrib),
            'most_important_feature': {
                'name': self.feature_names[most_important_idx] if most_important_idx < len(self.feature_names) else f'feature_{most_important_idx}',
                'value': float(shap_values[most_important_idx]),
                'impact': 'increases fraud risk' if shap_values[most_important_idx] > 0 else 'decreases fraud risk'
            },
            'explanation_text': self._generate_explanation_text(shap_values, positive_contrib, negative_contrib)
        }
        
        return summary
    
    def _generate_explanation_text(self, shap_values: np.ndarray, 
                                 positive_contrib: float, negative_contrib: float) -> str:
        """Generate human-readable explanation text"""
        
        # Get top positive and negative features
        positive_features = []
        negative_features = []
        
        for i, value in enumerate(shap_values):
            feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
            
            if value > 0.01:  # Threshold for significance
                positive_features.append((feature_name, value))
            elif value < -0.01:
                negative_features.append((feature_name, value))
        
        # Sort by absolute value
        positive_features.sort(key=lambda x: x[1], reverse=True)
        negative_features.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanation text
        explanation_parts = []
        
        if positive_features:
            top_positive = positive_features[:3]
            feature_names = [f[0] for f in top_positive]
            explanation_parts.append(f"Features increasing fraud risk: {', '.join(feature_names)}")
        
        if negative_features:
            top_negative = negative_features[:3]
            feature_names = [f[0] for f in top_negative]
            explanation_parts.append(f"Features decreasing fraud risk: {', '.join(feature_names)}")
        
        if positive_contrib > abs(negative_contrib):
            explanation_parts.append("Overall prediction leans towards fraud")
        else:
            explanation_parts.append("Overall prediction leans towards legitimate transaction")
        
        return ". ".join(explanation_parts) + "."

class EnsembleSHAPExplainer:
    """SHAP explainer for ensemble predictions"""
    
    def __init__(self):
        self.individual_explainers = {}
        self.ensemble_weights = {}
    
    def add_explainer(self, model_name: str, explainer: SHAPExplainer, weight: float = 1.0):
        """Add individual model explainer"""
        self.individual_explainers[model_name] = explainer
        self.ensemble_weights[model_name] = weight
    
    def explain_ensemble_prediction(self, features: np.ndarray, 
                                  transaction_id: str = None) -> Dict[str, Any]:
        """Generate ensemble SHAP explanation"""
        
        individual_explanations = {}
        aggregated_shap_values = None
        
        # Get explanations from all models
        for model_name, explainer in self.individual_explainers.items():
            explanation = explainer.explain_prediction(model_name, features, transaction_id)
            individual_explanations[model_name] = explanation
            
            # Aggregate SHAP values
            if 'shap_values' in explanation:
                shap_values = np.array(explanation['shap_values'])
                weight = self.ensemble_weights.get(model_name, 1.0)
                
                if aggregated_shap_values is None:
                    aggregated_shap_values = shap_values * weight
                else:
                    aggregated_shap_values += shap_values * weight
        
        # Normalize by total weight
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0 and aggregated_shap_values is not None:
            aggregated_shap_values /= total_weight
        
        # Create ensemble explanation
        ensemble_explanation = {
            'transaction_id': transaction_id,
            'ensemble_shap_values': aggregated_shap_values.tolist() if aggregated_shap_values is not None else [],
            'individual_explanations': individual_explanations,
            'model_weights': self.ensemble_weights.copy(),
            'aggregated_contributions': self._aggregate_feature_contributions(individual_explanations),
            'consensus_analysis': self._analyze_model_consensus(individual_explanations),
            'timestamp': datetime.now().isoformat()
        }
        
        return ensemble_explanation
    
    def _aggregate_feature_contributions(self, individual_explanations: Dict) -> Dict[str, float]:
        """Aggregate feature contributions across models"""
        
        aggregated = {}
        
        for model_name, explanation in individual_explanations.items():
            if 'feature_contributions' in explanation:
                weight = self.ensemble_weights.get(model_name, 1.0)
                
                for feature, contribution in explanation['feature_contributions'].items():
                    if feature not in aggregated:
                        aggregated[feature] = 0.0
                    aggregated[feature] += contribution * weight
        
        # Normalize by total weight
        total_weight = sum(self.ensemble_weights.values())
        if total_weight > 0:
            for feature in aggregated:
                aggregated[feature] /= total_weight
        
        return aggregated
    
    def _analyze_model_consensus(self, individual_explanations: Dict) -> Dict[str, Any]:
        """Analyze consensus between model explanations"""
        
        # Get top features from each model
        all_top_features = []
        
        for model_name, explanation in individual_explanations.items():
            if 'top_features' in explanation:
                top_features = explanation['top_features'][:5]  # Top 5
                all_top_features.extend([f['feature_name'] for f in top_features])
        
        # Count feature frequency
        feature_counts = {}
        for feature in all_top_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        # Find consensus features (appearing in multiple models)
        consensus_features = [f for f, count in feature_counts.items() if count > 1]
        
        # Calculate agreement score
        total_models = len(individual_explanations)
        agreement_score = len(consensus_features) / max(1, len(set(all_top_features)))
        
        consensus_analysis = {
            'consensus_features': consensus_features,
            'agreement_score': agreement_score,
            'total_models': total_models,
            'feature_frequency': feature_counts
        }
        
        return consensus_analysis

# Usage example
if __name__ == "__main__":
    print("🔍 Testing SHAP Explainer...")
    
    try:
        # Create sample data
        np.random.seed(42)
        background_data = np.random.random((100, 10))
        test_features = np.random.random((1, 10))
        feature_names = [f'feature_{i}' for i in range(10)]
        
        # Create a simple model for testing
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Generate sample labels
        y = np.random.choice([0, 1], 100)
        model.fit(background_data, y)
        
        # Initialize SHAP explainer
        explainer = SHAPExplainer()
        models = {'test_model': {'model': model}}
        explainer.initialize_explainers(models, background_data, feature_names)
        
        # Generate explanation
        explanation = explainer.explain_prediction('test_model', test_features, 'test_tx_001')
        
        print("✅ SHAP explanation generated successfully!")
        print(f"   Top features: {len(explanation.get('top_features', []))}")
        print(f"   Feature contributions: {len(explanation.get('feature_contributions', {}))}")
        print(f"   Explanation summary available: {'explanation_summary' in explanation}")
        
        print("\n🎉 SHAP Explainer test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()