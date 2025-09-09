#!/usr/bin/env python
# coding: utf-8

# In[22]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


iris = load_iris()
x = iris.data        
y = iris.target      

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

lda = LinearDiscriminantAnalysis()

lda.fit(x_train, y_train)

y_pred = lda.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {accuracy*100}%")


sample = [[5.0, 3.4, 1.5, 0.2]]
predicted_class = lda.predict(sample)
predicted_species = iris.target_names[predicted_class[0]]

print(f"The predicted species for the flower is: {predicted_species}")

