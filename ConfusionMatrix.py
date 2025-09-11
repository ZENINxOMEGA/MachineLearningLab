import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

y_act = [1,1,1,0,1,0,1,1,0,0]
y_pred = [1,1,0,1,0,1,1,1,0,0]

cm = confusion_matrix(y_act, y_pred)
tn,fp,fn,tp = cm.ravel()
print("Confusion_matrix\n", cm)
print("True Negative :", tn , "\nFalse Positive :", fp, "\nFalse Negative :", fn, "\nTrue Positive :", tp)

acc = accuracy_score(y_act, y_pred)
prec = precision_score(y_act, y_pred)
rec = recall_score(y_act, y_pred)
spec = tn / (tn + fp)
f1 = f1_score(y_act, y_pred)
print("\nAccuracy :", acc, "\nPrecision :", prec, "\nRecall :", rec, "\nSpecificity :", spec, "\nF1 Score :", f1)

plt.figure(figsize=(5,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=['Actual Positive', 'Actual Negative'], 
            yticklabels=['Predicted Positive', 'Predicted Negative'])

plt.title('Actual vs Predicted')
plt.ylabel('Predicted')
plt.xlabel('Actual')    
plt.show()