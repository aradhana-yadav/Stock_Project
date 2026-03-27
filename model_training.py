import pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model(df, stock_name):
    filename = f"models/{stock_name}.pkl"
    if os.path.exists(filename):
        # Load model AND accuracy
        with open(filename, "rb") as f:
            data = pickle.load(f)
            model = data["model"]
            accuracy = data["accuracy"]
        return model, accuracy

    # Features & target
    X = df[['MA10', 'MA50', 'Return']]
    y = df['Signal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    os.makedirs("models", exist_ok=True)
    # Save both model AND accuracy
    with open(filename, "wb") as f:
        pickle.dump({"model": model, "accuracy": accuracy}, f)

    return model, accuracy