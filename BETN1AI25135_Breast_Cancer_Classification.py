# Breast Cancer Classification Project
# Student Name: Vansh Sikarwar
# Enrollment Number: BETN1AI25135
#
# Converted from the updated Jupyter Notebook.
# Run this file in an environment with the required Python packages installed.

#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification Using Machine Learning
# 
# **Student Name:** Vansh Sikarwar  
# **Enrollment Number:** BETN1AI25135
# 
# ## Project Overview
# This practical explores breast cancer diagnosis classification using Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), and Gaussian Naive Bayes. The workflow includes data inspection, preprocessing, feature scaling, model training, and performance evaluation.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv("data.csv")
print(df)


# In[3]:


df.head(10)


# In[4]:


df.tail(10)


# In[5]:


df.info()


# In[6]:


print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# In[7]:


df.describe()


# In[8]:


df.describe(include="all")


# In[9]:


print("Total Missing Values:", df.isnull().sum().sum())


# In[10]:


(df.isnull().mean() * 100).round(2)


# In[11]:


features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean'
]


# In[12]:


df[features].hist(
    figsize=(12, 8),
    bins=20,
    edgecolor='black'
)

plt.suptitle("Histograms of Important Features")
plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(12, 6))

sns.boxplot(data=df[features])

plt.title("Boxplot of Important Features")
plt.xticks(rotation=45)
plt.ylabel("Feature Values")
plt.show()


# In[14]:


sns.countplot(data=df, x='diagnosis')

plt.title("Distribution of Breast Cancer Diagnosis")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()


# task 2 data preposseing

# In[15]:


data = df.copy()

print("Dataset copied successfully")
print("Shape:", data.shape)


# In[16]:


print(data.columns)


# In[17]:


data.columns = data.columns.str.strip()
data = data.drop(columns=['Unnamed: 32'], errors='ignore')


# In[18]:


print("Updated Dataset Shape:", data.shape)
print(data.head())


# In[19]:


print(data.isnull().sum())


# In[20]:


print("Total Missing Values:", data.isnull().sum().sum())


# In[21]:


for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

print("Missing values handled successfully!")


# In[22]:


print(data.isnull().sum().sum())


# In[23]:


print(data['diagnosis'].value_counts())


# In[24]:


data['diagnosis'] = data['diagnosis'].map({
    'B': 0,
    'M': 1
})


# In[25]:


print(data['diagnosis'].value_counts())
print(data.head())


# In[26]:


X = data.drop(columns=['diagnosis', 'id'])


# In[27]:


y = data['diagnosis']


# In[28]:


print("Input Features Shape:", X.shape)
print("Target Variable Shape:", y.shape)

print("\nInput Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())


# In[29]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# In[30]:


print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)


# In[31]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt



# In[33]:


lr_model = LogisticRegression(random_state=42)

lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)


# In[34]:


print("Logistic Regression Performance")

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1-Score:", f1_score(y_test, y_pred_lr))


# In[35]:


cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[36]:


dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("Decision Tree Model Trained Successfully!")


# In[37]:


print("Decision Tree Performance")

print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Precision:", precision_score(y_test, y_pred_dt))
print("Recall:", recall_score(y_test, y_pred_dt))
print("F1-Score:", f1_score(y_test, y_pred_dt))


# In[38]:


cm_dt = confusion_matrix(y_test, y_pred_dt)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[39]:


comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_dt)
    ],
    'Precision': [
        precision_score(y_test, y_pred_lr),
        precision_score(y_test, y_pred_dt)
    ],
    'Recall': [
        recall_score(y_test, y_pred_lr),
        recall_score(y_test, y_pred_dt)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr),
        f1_score(y_test, y_pred_dt)
    ]
})

comparison


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# In[41]:


rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)


# In[42]:


print("Random Forest Performance")

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1-Score:", f1_score(y_test, y_pred_rf))


# In[43]:


cm_rf = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Oranges',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[44]:


svm_model = SVC(kernel='linear', random_state=42)

svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

print("SVM Model Trained Successfully!")


# In[45]:


print("SVM Performance")

print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Precision:", precision_score(y_test, y_pred_svm))
print("Recall:", recall_score(y_test, y_pred_svm))
print("F1-Score:", f1_score(y_test, y_pred_svm))


# In[46]:


cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[47]:


comparison_task4 = pd.DataFrame({
    'Model': ['Random Forest', 'SVM'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_svm)
    ],
    'Precision': [
        precision_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_svm)
    ],
    'Recall': [
        recall_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_svm)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_svm)
    ]
})

comparison_task4


# In[48]:


best_model = comparison_task4.loc[
    comparison_task4['F1-Score'].idxmax()
]

print("Better Performing Model:", best_model['Model'])
print("F1-Score:", best_model['F1-Score'])


# In[49]:


from sklearn.naive_bayes import GaussianNB


# In[50]:


nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)

y_pred_nb = nb_model.predict(X_test_scaled)


# In[51]:


print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Precision:", precision_score(y_test, y_pred_nb))
print("Recall:", recall_score(y_test, y_pred_nb))
print("F1-Score:", f1_score(y_test, y_pred_nb))


# In[52]:


cm_nb = confusion_matrix(y_test, y_pred_nb)

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm_nb,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[53]:


from sklearn.metrics import roc_auc_score, roc_curve


# In[54]:


y_scores = {
    'Logistic Regression': lr_model.predict_proba(X_test_scaled)[:, 1],
    'Decision Tree': dt_model.predict_proba(X_test)[:, 1],
    'Random Forest': rf_model.predict_proba(X_test)[:, 1],
    'SVM': svm_model.decision_function(X_test_scaled),
    'Naive Bayes': nb_model.predict_proba(X_test_scaled)[:, 1]
}

for model_name, scores in y_scores.items():
    auc_score = roc_auc_score(y_test, scores)
    print(name, "ROC-AUC:", auc_score)


# In[55]:


models = {
    'Logistic Regression': y_pred_lr,
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf,
    'SVM': y_pred_svm,
    'Naive Bayes': y_pred_nb
}

results = []

for model_name, predictions in models.items():

    results.append({
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1-Score': f1_score(y_test, predictions),
        'ROC-AUC': roc_auc_score(y_test, y_scores[model_name])
    })

final_comparison = pd.DataFrame(results)

final_comparison


# In[56]:


fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, (model_name, predictions) in zip(
    axes.flat, models.items()
):

    cm = confusion_matrix(y_test, predictions)

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant'],
        ax=ax
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

axes.flat[-1].axis('off')

plt.tight_layout()
plt.show()


# In[57]:


plt.figure(figsize=(9, 6))

for model_name, scores in y_scores.items():

    fpr, tpr, thresholds = roc_curve(y_test, scores)

    auc_value = roc_auc_score(y_test, scores)

    plt.plot(
        fpr,
        tpr,
        label=f'{model_name} (AUC = {auc_value:.3f})'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.title("ROC Curve Comparison of All Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.grid()
plt.show()


# In[58]:


best_model = final_comparison.loc[
    final_comparison['F1-Score'].idxmax()
]

print("Best Performing Model:", best_model['Model'])
print("Accuracy:", best_model['Accuracy'])
print("Precision:", best_model['Precision'])
print("Recall:", best_model['Recall'])
print("F1-Score:", best_model['F1-Score'])
print("ROC-AUC:", best_model['ROC-AUC'])


# In[59]:


data.to_csv("cleaned_dataset.csv", index=False)
print("Cleaned Dataset Saved Successfully!")


# In[60]:


cleaned_data = pd.read_csv("cleaned_dataset.csv")

print("Dataset Shape:", cleaned_data.shape)
print(cleaned_data.head())


# ### Conclusion
# 
# In this practical, we used five Machine Learning algorithms to predict breast cancer. These algorithms are Logistic Regression, Decision Tree, Random Forest, SVM and Naive Bayes.
# 
# First, we loaded the dataset and checked its information. Then we cleaned the data, handled missing values and converted text values into numbers. After that, we divided the data into 70% training and 30% testing.
# 
# We trained all five models and checked their performance using Accuracy, Precision, Recall, F1-Score, ROC-AUC and Confusion Matrix.
# 
# At the end, we compared all five models to find the best-performing model. Through this practical, we learned how to use Machine Learning algorithms and how to check their performance.

# In[ ]:




