from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model
model_path = r"C:\Users\dhari\Documents\deploying-reg-model\bulldozer_price_prediction_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Dummy encoders (Replace with real mappings if needed)
product_size_encoder = LabelEncoder()
fi_model_desc_encoder = LabelEncoder()

# Sample categorical values (Update with real values)
product_size_encoder.fit(["Small", "Medium", "Large"])  # Adjust based on dataset
fi_model_desc_encoder.fit(["521d", "950fii", "226", "pc120-6e", "s175"])  # Adjust based on dataset

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user inputs
            year_made = int(request.form["YearMade"])
            product_size = request.form["ProductSize"]
            sale_year = int(request.form["SaleYear"])
            fi_model_desc = request.form["fiModelDesc"]
            model_id = int(request.form["ModelID"])

            # Convert categorical values using Label Encoding
            product_size = product_size_encoder.transform([product_size])[0]
            fi_model_desc = fi_model_desc_encoder.transform([fi_model_desc])[0]

            # Prepare data for prediction
            input_data = np.array([[year_made, product_size, sale_year, fi_model_desc, model_id]])

            # Make prediction
            predicted_price = model.predict(input_data)[0]

            return render_template("index.html", prediction=f"Predicted Sale Price: ${predicted_price:.2f}")

        except Exception as e:
            return render_template("index.html", error="Invalid input. Please try again.")

    return render_template("index.html", prediction=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
