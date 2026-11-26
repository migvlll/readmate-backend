import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

np.random.seed(42)

def generate_elementary_reading_dataset(samples_per_class=250):
    """
    Generates calibrated elementary reading assessment feature vectors:
    Features: [reading_accuracy (%), wpm, comprehension_score (%)]
    Labels: 0 = Beginner, 1 = Developing, 2 = Proficient, 3 = Advanced
    """
    X = []
    y = []

    # 0: Beginner (Struggling decoding, slow pace, low comprehension)
    for _ in range(samples_per_class):
        acc = np.random.uniform(40.0, 74.9)
        wpm = np.random.uniform(20.0, 59.9)
        comp = np.random.uniform(20.0, 59.9)
        X.append([acc, wpm, comp])
        y.append(0)

    # 1: Developing (Moderate accuracy, hesitant pace, partial comprehension)
    for _ in range(samples_per_class):
        acc = np.random.uniform(75.0, 84.9)
        wpm = np.random.uniform(60.0, 89.9)
        comp = np.random.uniform(60.0, 74.9)
        X.append([acc, wpm, comp])
        y.append(1)

    # 2: Proficient (Solid accuracy, grade-level WPM, good comprehension)
    for _ in range(samples_per_class):
        acc = np.random.uniform(85.0, 94.9)
        wpm = np.random.uniform(90.0, 124.9)
        comp = np.random.uniform(75.0, 89.9)
        X.append([acc, wpm, comp])
        y.append(2)

    # 3: Advanced (High accuracy, fluent pace, strong comprehension)
    for _ in range(samples_per_class):
        acc = np.random.uniform(95.0, 100.0)
        wpm = np.random.uniform(125.0, 175.0)
        comp = np.random.uniform(90.0, 100.0)
        X.append([acc, wpm, comp])
        y.append(3)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = generate_elementary_reading_dataset(samples_per_class=300)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    target_names = ["Beginner", "Developing", "Proficient", "Advanced"]

    print(f"Validation Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=target_names))

    model_path = os.path.join("models", "rf_reading_model.pkl")
    joblib.dump(rf_model, model_path)
    print(f"Saved Random Forest model to {model_path}")