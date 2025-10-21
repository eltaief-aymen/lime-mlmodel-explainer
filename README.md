# 🔍 ML Model Explainability with LIME

A simple, professional implementation demonstrating how to explain machine learning predictions using LIME (Local Interpretable Model-agnostic Explanations).

**LIME** helps you understand **why** your machine learning model made a specific prediction. Instead of just getting "positive" or "negative" from your classifier, LIME shows you which words or features influenced that decision.

### How LIME Works:
1. **Takes your prediction**: Select an instance you want to explain
2. **Creates variations**: Generates thousands of similar examples by slightly modifying the original
3. **Tests all variations**: Runs them through your model to see what it predicts
4. **Fits a simple model**: Uses a simple, interpretable model (like linear regression) around your instance
5. **Shows you why**: Reveals which features pushed the prediction one way or another

Think of it like asking: *"Which words made my model think this review is positive?"*

## 📊 What This Code Does

1. **Creates a sentiment dataset** - Movie/product reviews labeled as positive or negative
2. **Trains a Random Forest classifier** - Learns to predict sentiment from text
3. **Explains predictions with LIME** - Shows which words influence each prediction
4. **Visualizes the results** - Creates bar charts showing feature importance
5. **Tests stability** - Checks if explanations are consistent across multiple runs

## 🔑 Key Features

✅ **Complete pipeline** from data to explanations  
✅ **Multiple examples** (positive, negative, mixed reviews)  
✅ **Professional visualizations** with color-coded importance  
✅ **Stability analysis** to test explanation reliability  
✅ **Well-documented code** with clear comments  
✅ **Production-ready** with proper train/test splits  

## 📖 Learn More

- [LIME Original Paper](https://arxiv.org/abs/1602.04938)
- [LIME GitHub Repository](https://github.com/marcotcr/lime)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

---

**Made for anyone who wants to understand their ML models better** 🚀

⭐ Star this repo if you find it helpful!
