from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# ----------------------------
# FIXED MODEL LOADING (RENDER SAFE)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "bulldozer_price_prediction_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ----------------------------
# ENCODERS
# ----------------------------
product_size_encoder = LabelEncoder()
fi_model_desc_encoder = LabelEncoder()

product_size_encoder.fit(["Small", "Medium", "Large"])
fi_model_desc_encoder.fit(["521d", "950fii", "226", "pc120-6e", "s175"])

# ----------------------------
# ROUTE
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            year_made = int(request.form["YearMade"])
            product_size = request.form["ProductSize"]
            sale_year = int(request.form["SaleYear"])
            fi_model_desc = request.form["fiModelDesc"]
            model_id = int(request.form["ModelID"])

            product_size = product_size_encoder.transform([product_size])[0]
            fi_model_desc = fi_model_desc_encoder.transform([fi_model_desc])[0]

            input_data = np.array([[year_made, product_size, sale_year, fi_model_desc, model_id]])

            predicted_price = model.predict(input_data)[0]

            return render_template(
                "index.html",
                prediction=f"Predicted Sale Price: ${predicted_price:.2f}",
                error=None
            )

        except Exception as e:
            return render_template("index.html", error=str(e), prediction=None)

    return render_template("index.html", prediction=None, error=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
