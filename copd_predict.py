import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#metrics and stuff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


data = pd.read_csv("mortalities.csv")


def fill_gen(row):
    if row == "M":
        return 1
    return 0
data["gendera"] = data["gendera"].apply(fill_gen)

def fill_yes(row):
    if row == "YES":
        return 1
    return 0

for column in data:
    data[column].fillna(-1, inplace=True)

data.pop("ID")

import seaborn
#GRAPHSSSS
#seaborn.pairplot(data, hue="COPD", palette="Blues")

#g = seaborn.jointplot(data=data, x="Respiratory rate", y="glucose", hue="COPD", kind="kde")
#seaborn.relplot(data=data, x="age", y="Respiratory rate", hue="COPD")
#plt.show()

#copd_data = data[data["COPD"]==1]
#copd_data.to_csv("copd_dataset.csv", index=False)


x = data.drop(columns=["COPD"])
y = data["COPD"]


train_data, test_data, train_labels, test_labels = train_test_split(x, y, test_size = 0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

# Create an instance of the Gaussian Naive Bayes classifier
classifier = GaussianNB()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix1: \n", confusion_matrix(test_labels, predictions))
print("Accuracy1: ", accuracy_score(test_labels, predictions)*100)
"""
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

# Display precision, recall, and F1-score
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Calculate and display classification report
report = classification_report(test_labels, predictions)
print("Classification Report:\n", report)"""

 
 
# Create an instance of the GaussianProcessClassifier
classifier = GaussianProcessClassifier()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix2: \n", confusion_matrix(test_labels, predictions))
print("Accuracy2: ", accuracy_score(test_labels, predictions)*100)



# Create an instance of the RandomForestClassifier
classifier = RandomForestClassifier()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix3: \n", confusion_matrix(test_labels, predictions))
print("Accuracy3: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the LogisticRegression
classifier = LogisticRegression(max_iter=1000)
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix4: \n", confusion_matrix(test_labels, predictions))
print("Accuracy4: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the SVC
classifier = SVC()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix5: \n", confusion_matrix(test_labels, predictions))
print("Accuracy5: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the KNeighborsClassifier
classifier = KNeighborsClassifier()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix6: \n", confusion_matrix(test_labels, predictions))
print("Accuracy6: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the MLPClassifier
classifier = MLPClassifier(max_iter=1000)
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix7: \n", confusion_matrix(test_labels, predictions))
print("Accuracy7: ", accuracy_score(test_labels, predictions)*100)


"""
##VISUALSSS
g = seaborn.jointplot(data=data, x="Respiratory rate", y="glucose", hue="outcome", kind="kde")

plt.show()
seaborn.relplot(data=data, x="age", y="Respiratory rate", hue="outcome")
plt.show()

###
SP O2: reduced
Respiratory rate: higher
Depression
Diabetes

import seaborn
data.replace(-1, pd.NA, inplace=True)
seaborn.pairplot(
    data.dropna(),
    x_vars=["age", "Respiratory rate", "glucose", "SP O2", "depression", "diabetes"],
    y_vars=["age", "Respiratory rate", "glucose", "SP O2", "depression", "diabetes"],
    hue="COPD",
    kind="scatter"

)
plt.show()
"""

"""
# Load an example dataset with long-form data


# Plot the responses for different events and regions
seaborn.lineplot(x="Respiratory rate", y="SP O2",
             hue="COPD", style="hypertensive",
             data=data)

plt.show()
"""
"""
filtered_patients = data[(data['glucose'] >= 200) & (data['glucose'] <= 350)]

# Print the filtered patient information
print(filtered_patients)
filtered_patients.to_csv('filtered_patients2.csv', index=False)
filtered = pd.read_csv("filtered_patients2.csv")

seaborn.pairplot(
    data=filtered,
    x_vars=["age", "Respiratory rate", "glucose", "BMI"],
    y_vars=["age", "Respiratory rate", "glucose", "BMI"],
    hue="COPD",
    kind="scatter" 

)
plt.show()

"""



"""
##CORRELATION MATRIXES for (linear) comparisons and trend findings
import scipy.stats as stats
correlation_matrix = data.corr()
print(correlation_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

for column in data.columns:
    if column != 'COPD':
        correlation, p_value = stats.pearsonr(data[column], data['age'])
        print(f"Correlation between {column} and outcome: {correlation:.2f}, p-value: {p_value:.4f}")
"""