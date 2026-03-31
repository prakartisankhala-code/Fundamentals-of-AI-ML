import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# loading dataset
data = pd.read_csv("data.csv")

# These are the features and targets
X = data[["ScreenTime", "SleepHours", "AppUsage"]]
y = data["Addicted"]

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Testing accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Sample prediction
sample = [[9, 5, 20]]  # ScreenTime, SleepHours, AppUsage
prediction = model.predict(sample)

print("Prediction (1 = Addicted, 0 = Not Addicted):", prediction[0])