import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pickle

# Step 1: Define the Dataset
data = pd.DataFrame({
    'age': [25, 40, 60, 30, 50, 20],
    'time_since_injury': [1.5, 5, 2, 8, 6, 3],
    'gcs': [15, 8, 12, 6, 10, 13],
    'gos': [5, 3, 4, 2, 3, 4],
    'mortality': [0, 1, 0, 1, 1, 0],  # 0 = unlikely, 1 = likely
    'morbidity': [0.9, 0.3, 0.8, 0.2, 0.4, 0.95]  # Scale of 0 to 1
})

# Step 2: Separate Features and Targets
X = data[['age', 'time_since_injury', 'gcs', 'gos']]  # Features
y_mortality = data['mortality']  # Target for mortality (classification)
y_morbidity = data['morbidity']  # Target for morbidity (regression)

# Step 3: Split Dataset into Training and Testing Sets
X_train, X_test, y_train_mortality, y_test_mortality = train_test_split(
    X, y_mortality, test_size=0.2, random_state=42
)

_, _, y_train_morbidity, y_test_morbidity = train_test_split(
    X, y_morbidity, test_size=0.2, random_state=42
)

# Step 4: Train the Mortality Model (Classification)
print("Training mortality model...")
mortality_model = RandomForestClassifier(random_state=42)
mortality_model.fit(X_train, y_train_mortality)

# Step 5: Train the Morbidity Model (Regression)
print("Training morbidity model...")
morbidity_model = RandomForestRegressor(random_state=42)
morbidity_model.fit(X_train, y_train_morbidity)

# Step 6: Save the Models
print("Saving models...")
with open('mortality_model.pkl', 'wb') as f:
    pickle.dump(mortality_model, f)

with open('morbidity_model.pkl', 'wb') as f:
    pickle.dump(morbidity_model, f)

print("Models trained and saved successfully!")
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import pickle

# # 1. Load or Create a Dataset
# # Replace this with the actual dataset path
# # You can use your own dataset in CSV format
# data = pd.DataFrame({
#     'age': [25, 40, 60, 30, 50, 20],
#     'time_since_injury': [1.5, 5, 2, 8, 6, 3],
#     'gcs': [15, 8, 12, 6, 10, 13],
#     'gos': [5, 3, 4, 2, 3, 4],
#     'outcome': [1, 0, 1, 0, 0, 1]  # 1 = Recovery, 0 = Critical
# })

# # 2. Separate Features (X) and Target (y)
# X = data[['age', 'time_since_injury', 'gcs', 'gos']]
# y = data['outcome']

# # 3. Split Dataset into Training and Testing Sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 4. Train the Logistic Regression Model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 5. Evaluate the Model
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # 6. Save the Trained Model to a File
# with open('prognosis/model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# print("Model trained and saved successfully!")
