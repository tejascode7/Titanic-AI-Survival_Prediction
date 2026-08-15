from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("titanic_knn_model.pkl")

# Load the same scaler used during training
scaler = joblib.load("titanic_scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # Create the passenger data with the exact
    # same 18 features used during training
    input_data = pd.DataFrame([{
        "Pclass": data["Pclass"],
        "Sex": data["Sex"],
        "Age": data["Age"],
        "SibSp": data["SibSp"],
        "Parch": data["Parch"],
        "Fare": data["Fare"],

        "Embarked_C": data["Embarked_C"],
        "Embarked_Q": data["Embarked_Q"],
        "Embarked_S": data["Embarked_S"],

        "CabinDeck_A": data["CabinDeck_A"],
        "CabinDeck_B": data["CabinDeck_B"],
        "CabinDeck_C": data["CabinDeck_C"],
        "CabinDeck_D": data["CabinDeck_D"],
        "CabinDeck_E": data["CabinDeck_E"],
        "CabinDeck_F": data["CabinDeck_F"],
        "CabinDeck_G": data["CabinDeck_G"],
        "CabinDeck_T": data["CabinDeck_T"],
        "CabinDeck_Unknown": data["CabinDeck_Unknown"]
    }])


    # These are the same columns we scaled during training
    columns_to_scale = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]


    # Apply the SAME scaler used during training
    input_data[columns_to_scale] = scaler.transform(
        input_data[columns_to_scale]
    )


    # Make prediction
    prediction = model.predict(input_data)


    return jsonify({
        "prediction": int(prediction[0])
    })


if __name__ == "__main__":
    app.run(debug=True)