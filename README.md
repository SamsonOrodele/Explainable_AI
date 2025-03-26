# 🔍 German Credit Risk Analysis with Explainable Deep Learning

This repository houses the implementation of my MSc dissertation project, titled **"Generating Counterfactual and Global Explanations for German Credit Data Using Deep Learning Models"**. The project tackles the critical challenge of transparency in credit risk assessment by leveraging **Explainable Artificial Intelligence (XAI)** techniques with deep learning models. Using the German Credit Dataset, I developed two neural networks, a **Multilayer Perceptron (MLP)** and a **Convolutional Neural Network (CNN)** enhanced with **SMOTE** to address data imbalance, and explained their decisions using **SHAP** for global feature importance and **counterfactual explanations** for individual decision insights.

---

## 💼 Background and Motivation

Credit scoring is a cornerstone of financial decision-making, determining whether individuals receive loans or credit cards based on their likelihood of repayment. However, traditional machine learning models, often dubbed "black boxes," obscure the reasoning behind their predictions, eroding trust among stakeholders, be it applicants seeking clarity or institutions ensuring fairness. This project bridges that gap by integrating **predictive power** with **interpretability**, a pressing need in the financial sector where transparency is not just a luxury but a regulatory and ethical necessity.

The German Credit Dataset, a widely recognized benchmark, presents a real-world challenge with its imbalanced classes (700 good credit cases vs. 300 bad credit cases) and mix of categorical and numerical features. My work aims to not only classify creditworthiness accurately but also explain *why* decisions are made and *how* they can be altered, empowering both financial institutions and their clients.

---

## 🎯 Project Objectives

This project pursues the following goals:

1. **Develop Robust Predictive Models**: Construct and optimize two deep learning models, MLP and CNN, for credit risk classification.
2. **Mitigate Data Imbalance**: Apply the **Synthetic Minority Oversampling Technique (SMOTE)** to balance the dataset and improve model fairness.
3. **Enhance Global Interpretability**: Use **SHAP (SHapley Additive exPlanations)** to uncover the most influential features driving credit decisions.
4. **Provide Individual Insights**: Generate **counterfactual explanations** to demonstrate how specific feature changes (e.g., increasing credit amount) can flip a credit decision.
5. **Visualize and Communicate Results**: Create intuitive visualizations to make findings accessible to technical and non-technical audiences alike.

---

## 📊 Dataset Overview

The **German Credit Dataset**, sourced from the UCI Machine Learning Repository, comprises 1,000 instances with 21 features, including the target variable `Creditability` (1 for good credit, 0 for bad credit). Key features include:

- **Numerical**: `Duration of Credit (month)`, `Credit Amount`, `Age (years)`
- **Categorical**: `Account Balance`, `Purpose`, `Sex & Marital Status`, `Occupation`, etc.

Exploratory data analysis revealed an imbalance (700 good vs. 300 bad credit cases) and a strong correlation between `Account Balance` and creditworthiness, while `Duration of Credit` showed a negative correlation (-0.21).

---

## 🧠 Methodology and Models

### Preprocessing
The dataset required careful preparation:
- **Categorical Features**: One-hot encoded 17 categorical variables (e.g., `Purpose`, `Guarantors`) to avoid ordinal misinterpretation, expanding the feature set to 71 columns.
- **Numerical Features**: Standardized `Credit Amount`, `Duration`, and `Age` using `StandardScaler` for equal contribution to gradient descent.
- **SMOTE**: Oversampled the minority class (bad credit) from 300 to 700 instances, balancing the dataset to 1,400 total samples.

### Models Developed

| Model            | Architecture Details                                                                 | Regularization Techniques         |
|------------------|-------------------------------------------------------------------------------------|-----------------------------------|
| **MLP**          | Input layer, two hidden layers (64 nodes each, ReLU), output layer (sigmoid)         | Dropout (0.5), L2 regularization, Early Stopping |
| **CNN**          | 1D Conv layers (filters tuned), MaxPooling, Dense layers, output layer (sigmoid)    | Dropout (0.35, 0.55), Early Stopping, Hyperband tuning |

- **MLP**: A feedforward neural network trained with backpropagation, optimized with Adam and binary cross-entropy loss. Initial overfitting (training accuracy 90.21% vs. validation 83.21%) was mitigated with regularization.
- **CNN**: Adapted for tabular data using 1D convolutions, tuned with Keras Tuner's Hyperband algorithm over 30 trials to optimize filters, dropout rates, and layer sizes.

### Explainability Techniques

1. **Global Explanation with SHAP**:
   - Employed the Kernel Explainer to compute SHAP values, revealing feature importance across the dataset.
   - Visualized with Beeswarm and bar plots to highlight top contributors like `Account Balance_4` and `Duration of Credit`.

2. **Counterfactual Explanations**:
   - Manually perturbed features (e.g., increasing `Credit Amount` by 3000 for MLP, adjusting `Credit Amount`, `Duration`, and `Age` for CNN) to flip predictions.
   - Aimed to answer "What if?" scenarios, such as how an applicant could become creditworthy.

---

## ⚖️ Addressing Data Imbalance

The dataset’s imbalance posed a risk of biased predictions favoring the majority class (good credit). **SMOTE** was integrated to synthesize minority class samples, ensuring both classes were equally represented (700 each). This step significantly improved the models' ability to identify bad credit cases, as evidenced by confusion matrix results.

---

## 📈 Results and Findings

### Performance Metrics

| Metric         | MLP Model | CNN Model |
|----------------|-----------|-----------|
| **Accuracy**   | 72.50%    | 78.21%    |
| **Precision**  | 83.20%    | 79.72%    |
| **Recall**     | 75.36%    | 78.08%    |
| **F1 Score**   | 79.09%    | 78.89%    |
| **ROC AUC**    | 0.76      | 0.782     |
| **AUPRC**      | 0.861     | 0.893     |

- **MLP**: Achieved 72.5% test accuracy after regularization addressed overfitting. Precision (83.2%) was strong, but recall (75.36%) indicated challenges with minority class detection (38 true negatives vs. 25 false negatives).
- **CNN**: Outperformed with 78.21% accuracy, benefiting from SMOTE and hyperparameter tuning. Balanced precision (79.72%) and recall (78.08%) reflected improved class differentiation (105 true negatives, 114 true positives).

### Explainability Insights

- **MLP Counterfactual**: Increasing `Credit Amount` by 3000 flipped a prediction from 0.176 (bad) to 1.0 (good), suggesting sensitivity to this feature, though the result was counterintuitive and warrants further study.
- **CNN Counterfactual**: Adjusting `Credit Amount` (6615 to 3271.2), `Duration` (24 to 20.9 months), and `Age` (75 to 239.78) flipped a prediction from "not creditworthy" to "creditworthy." The unrealistic age perturbation highlighted a need for better clamping.
- **SHAP Findings**:
  - **MLP**: Top features: `Account Balance_4`, `Value Savings/Stocks_1`, `Duration of Credit (month)`.
  - **CNN**: Top features: `Account Balance_4`, `Duration of Credit (month)`, `Purpose_3`.

---



---

## 📸 Visual Outputs

### SHAP Feature Importance
![SHAP Summary](Plots/shap_summary.png)

### Counterfactual Sample Example
![Counterfactual Example](Plots/counterfactual_example.png)

---


---

## ✅ Ideal For

-Financial Institutions: Seeking transparent AI tools for credit risk assessment.

-Data Scientists: Interested in XAI applications with deep learning.

-Regulators and Ethicists: Focused on fairness and interpretability in AI systems.

-Students/Researchers: Exploring SMOTE, SHAP, and counterfactuals in real-world scenarios.



---

## 👨‍💻 Author

**Samson Orodele**  
- **Email**: [samorodele@gmail.com](mailto:samorodele@gmail.com)  
- **GitHub**: [github.com/SamsonOrodele](https://github.com/SamsonOrodele)  

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and extend it.

---
