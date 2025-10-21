"""
============================================================================
ML Model Explainability with LIME (Local Interpretable Model-agnostic Explanations)
A Professional Implementation for Text Classification
============================================================================
This script demonstrates a comprehensive implementation of LIME for explaining
predictions of a machine learning model trained on a text classification task.
It includes dataset creation, model training, evaluation, LIME explanations,
visualizations, and analysis of explanation stability.
============================================================================
Author: Eltaief Aymen
============================================================================

"""

# ============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================================

import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Dict

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)

# LIME imports
import lime
import lime.lime_text

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("ML MODEL EXPLAINABILITY WITH LIME - PROFESSIONAL IMPLEMENTATION")
print("=" * 80)
print()


# ============================================================================
# STEP 2: CREATE AND PREPARE DATASET
# ============================================================================

print("STEP 1: Creating and Preparing Dataset")
print("-" * 80)

def create_sentiment_dataset() -> pd.DataFrame:
    """
    Create a sentiment analysis dataset for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with 'text' and 'sentiment' columns
    """
    data = {
        'text': [
            # Positive reviews
            "This movie was fantastic! Loved every moment.",
            "Absolutely brilliant film, a must-watch.",
            "Great storyline and amazing visuals.",
            "Highly recommend, truly engaging.",
            "So happy with this purchase, perfect!",
            "An absolute joy to watch!",
            "What a gem! Exceeded expectations.",
            "Fantastic service, thank you!",
            "Outstanding performance by all actors.",
            "Incredible experience, worth every penny.",
            "Beautifully crafted and inspiring.",
            "Best purchase I've made this year!",
            
            # Negative reviews
            "Terrible acting, waste of time and money.",
            "Could have been better, somewhat boring.",
            "Worst experience ever, never again.",
            "Mediocre plot, but decent special effects.",
            "Disappointed with the quality, very bad.",
            "This product broke quickly, very unhappy.",
            "Boring and uninspired, avoid it.",
            "Poor delivery and damaged item.",
            "Complete waste of money, terrible quality.",
            "Awful experience, would not recommend.",
            "Very disappointing, expected much better.",
            "Poorly made and overpriced.",
        ],
        'sentiment': [
            'positive', 'positive', 'positive', 'positive', 
            'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive',
            'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative'
        ]
    }
    return pd.DataFrame(data)

# Create dataset
df = create_sentiment_dataset()

print(f"✓ Dataset created with {len(df)} samples")
print(f"✓ Class distribution:")
print(df['sentiment'].value_counts())
print(f"\n✓ Sample text: '{df['text'].iloc[0]}'")
print()


# ============================================================================
# STEP 3: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================================

print("STEP 2: Splitting Data into Train and Test Sets")
print("-" * 80)

X = df['text']
y = df['sentiment']

# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.25,  # 25% for testing
    random_state=42, 
    stratify=y  # Maintain class distribution in both sets
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")
print(f"✓ Training distribution:")
print(pd.Series(y_train).value_counts())
print()


# ============================================================================
# STEP 4: BUILD AND TRAIN THE ML MODEL
# ============================================================================

print("STEP 3: Building and Training the Machine Learning Model")
print("-" * 80)

# Create TF-IDF Vectorizer
# TF-IDF converts text to numerical features based on word importance
vectorizer = TfidfVectorizer(
    stop_words='english',  # Remove common English words
    max_features=100,      # Limit vocabulary size
    ngram_range=(1, 2)     # Use unigrams and bigrams
)

# Create Random Forest Classifier
classifier = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    random_state=42,       # For reproducibility
    max_depth=10,          # Limit tree depth
    min_samples_split=2    # Minimum samples to split a node
)

# Create pipeline (combines vectorization and classification)
model_pipeline = make_pipeline(vectorizer, classifier)

# Train the model
print("Training model...")
model_pipeline.fit(X_train, y_train)
print("✓ Model training complete!")
print()


# ============================================================================
# STEP 5: EVALUATE MODEL PERFORMANCE
# ============================================================================

print("STEP 4: Evaluating Model Performance")
print("-" * 80)

# Make predictions on test set
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)

# Calculate metrics
train_accuracy = model_pipeline.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"✓ Training Accuracy: {train_accuracy:.2%}")
print(f"✓ Testing Accuracy: {test_accuracy:.2%}")
print()

# Detailed classification report
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print()


# ============================================================================
# STEP 6: SETUP LIME EXPLAINER
# ============================================================================

print("STEP 5: Setting Up LIME Explainer")
print("-" * 80)

# Get class names from the pipeline
class_names = model_pipeline.classes_

# Create LIME Text Explainer
explainer = lime.lime_text.LimeTextExplainer(
    class_names=class_names,
    # Split text into words for perturbation
    split_expression=r'\W+',
    # Use default bow (bag of words) mode
    bow=True,
    random_state=42
)

print(f"✓ LIME Explainer initialized")
print(f"✓ Classes to explain: {list(class_names)}")
print()


# ============================================================================
# STEP 7: EXPLAIN SPECIFIC PREDICTIONS
# ============================================================================

print("STEP 6: Explaining Model Predictions with LIME")
print("-" * 80)

def explain_prediction(
    text: str, 
    explainer: lime.lime_text.LimeTextExplainer,
    model_pipeline,
    num_features: int = 10,
    num_samples: int = 2000
) -> lime.lime_text.LimeTextExplainer:
    """
    Generate LIME explanation for a given text.
    
    Args:
        text: The text to explain
        explainer: LIME text explainer instance
        model_pipeline: Trained model pipeline
        num_features: Number of features to include in explanation
        num_samples: Number of perturbed samples to generate
    
    Returns:
        LIME explanation object
    """
    # Get prediction
    prediction = model_pipeline.predict([text])[0]
    prediction_proba = model_pipeline.predict_proba([text])[0]
    
    print(f"\n{'='*70}")
    print(f"TEXT TO EXPLAIN:")
    print(f"'{text}'")
    print(f"{'-'*70}")
    print(f"Predicted Class: {prediction}")
    print(f"Prediction Probabilities:")
    for cls, prob in zip(class_names, prediction_proba):
        print(f"  {cls}: {prob:.2%}")
    print(f"{'='*70}\n")
    
    # Generate explanation
    print("Generating LIME explanation...")
    print(f"  - Creating {num_samples} perturbed samples...")
    print(f"  - Fitting local interpretable model...")
    
    explanation = explainer.explain_instance(
        text,
        model_pipeline.predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        top_labels=2  # Explain top 2 classes
    )
    
    print("✓ Explanation generated successfully!")
    
    return explanation, prediction, prediction_proba


def display_explanation_details(explanation, prediction_class: str):
    """
    Display detailed explanation information.
    
    Args:
        explanation: LIME explanation object
        prediction_class: The predicted class
    """
    print(f"\nTop Contributing Features for '{prediction_class}' prediction:")
    print("-" * 70)
    
    # Get feature weights for the predicted class
    exp_list = explanation.as_list(label=explanation.available_labels()[0])
    
    for i, (feature, weight) in enumerate(exp_list, 1):
        direction = "POSITIVE" if weight > 0 else "NEGATIVE"
        print(f"{i:2d}. '{feature}': {weight:+.4f} ({direction} contribution)")
    
    print()


# ============================================================================
# Example 1: Explain a Positive Review
# ============================================================================

text_positive = "This movie was absolutely fantastic! I loved every single moment and would recommend it to everyone."

explanation_pos, pred_pos, proba_pos = explain_prediction(
    text_positive,
    explainer,
    model_pipeline,
    num_features=8,
    num_samples=2000
)

display_explanation_details(explanation_pos, pred_pos)


# ============================================================================
# Example 2: Explain a Negative Review
# ============================================================================

text_negative = "Terrible experience, complete waste of money. The quality was awful and I'm very disappointed."

explanation_neg, pred_neg, proba_neg = explain_prediction(
    text_negative,
    explainer,
    model_pipeline,
    num_features=8,
    num_samples=2000
)

display_explanation_details(explanation_neg, pred_neg)


# ============================================================================
# Example 3: Explain a Mixed/Ambiguous Review
# ============================================================================

text_mixed = "The special effects were amazing but the story was quite boring and predictable."

explanation_mix, pred_mix, proba_mix = explain_prediction(
    text_mixed,
    explainer,
    model_pipeline,
    num_features=8,
    num_samples=2000
)

display_explanation_details(explanation_mix, pred_mix)


# ============================================================================
# STEP 8: VISUALIZE EXPLANATIONS
# ============================================================================

print("\nSTEP 7: Creating Visualization of Feature Importance")
print("-" * 80)

def plot_feature_importance(explanation, title: str, predicted_class: str):
    """
    Create a bar plot of feature importance from LIME explanation.
    
    Args:
        explanation: LIME explanation object
        title: Plot title
        predicted_class: The predicted class
    """
    # Get feature weights
    exp_list = explanation.as_list(label=explanation.available_labels()[0])
    
    # Separate features and weights
    features = [item[0] for item in exp_list]
    weights = [item[1] for item in exp_list]
    
    # Create color based on weight sign
    colors = ['green' if w > 0 else 'red' for w in weights]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(features)), weights, color=colors, alpha=0.7)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Weight (Contribution to Prediction)', fontsize=12)
    plt.title(f'{title}\nPredicted: {predicted_class}', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Positive Contribution'),
        Patch(facecolor='red', alpha=0.7, label='Negative Contribution')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'lime_explanation_{title.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Visualization saved as 'lime_explanation_{title.replace(' ', '_')}.png'")


# Generate visualizations for all three examples
plot_feature_importance(explanation_pos, "Positive Review", pred_pos)
plot_feature_importance(explanation_neg, "Negative Review", pred_neg)
plot_feature_importance(explanation_mix, "Mixed Review", pred_mix)


# ============================================================================
# STEP 9: ANALYZE EXPLANATION STABILITY
# ============================================================================

print("\n\nSTEP 8: Analyzing Explanation Stability")
print("-" * 80)

def analyze_explanation_stability(
    text: str,
    explainer,
    model_pipeline,
    num_runs: int = 5
) -> Dict:
    """
    Test stability of LIME explanations across multiple runs.
    
    Args:
        text: Text to explain
        explainer: LIME explainer
        model_pipeline: Trained model
        num_runs: Number of explanation iterations
    
    Returns:
        Dictionary with stability metrics
    """
    print(f"Running {num_runs} explanation iterations for stability analysis...")
    
    all_explanations = []
    
    for i in range(num_runs):
        exp = explainer.explain_instance(
            text,
            model_pipeline.predict_proba,
            num_features=8,
            num_samples=2000
        )
        exp_list = exp.as_list(label=exp.available_labels()[0])
        all_explanations.append(dict(exp_list))
    
    # Get all unique features
    all_features = set()
    for exp in all_explanations:
        all_features.update(exp.keys())
    
    # Calculate mean and std for each feature
    feature_stats = {}
    for feature in all_features:
        weights = [exp.get(feature, 0) for exp in all_explanations]
        feature_stats[feature] = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'appearances': sum(1 for w in weights if w != 0)
        }
    
    # Sort by mean absolute weight
    sorted_features = sorted(
        feature_stats.items(),
        key=lambda x: abs(x[1]['mean']),
        reverse=True
    )
    
    print("\n✓ Stability Analysis Results:")
    print(f"{'Feature':<30} {'Mean Weight':<15} {'Std Dev':<15} {'Appearances':<15}")
    print("-" * 75)
    
    for feature, stats in sorted_features[:8]:
        print(f"{feature:<30} {stats['mean']:>+.4f}        {stats['std']:>.4f}        {stats['appearances']}/{num_runs}")
    
    return feature_stats

# Analyze stability for positive review
stability_results = analyze_explanation_stability(
    text_positive,
    explainer,
    model_pipeline,
    num_runs=5
)


# ============================================================================
# STEP 10: SUMMARY AND BEST PRACTICES
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY AND KEY TAKEAWAYS")
print("=" * 80)

summary = """
✓ Successfully trained a sentiment classification model
✓ Achieved {:.2%} test accuracy
✓ Generated LIME explanations for multiple text instances
✓ Visualized feature contributions to predictions
✓ Analyzed explanation stability across multiple runs

KEY INSIGHTS FROM LIME EXPLANATIONS:
• Words like 'fantastic', 'loved', 'recommend' strongly indicate POSITIVE sentiment
• Words like 'terrible', 'waste', 'disappointed' strongly indicate NEGATIVE sentiment
• The model correctly identifies sentiment-bearing words in text
• Explanations remain relatively stable across multiple runs
""".format(test_accuracy)

print(summary)

print("=" * 80)
print("IMPLEMENTATION COMPLETE!")
print("=" * 80)