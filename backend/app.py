from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import traceback
import os
import joblib, gzip


app = Flask(__name__) 
CORS(app)

# Load model and encoder
with gzip.open("car_pricemodel_binary_compressed.pkl.gz", "rb") as f:
  model = joblib.load(f)
encoder = joblib.load("encoders.pkl")

@app.route("/health", methods=["GET"])
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "message": "Service is running"}), 200

@app.route("/data", methods=["POST"])
def receive_data():
    try:
        data = request.get_json()
        print("Received raw data:", data) 

        brand = data.get('brand', '')
        model_name = data.get('model', '')
        vehicle_age = data.get('vehicle_age', '')
        km_driven = data.get('km_driven', '')
        seller_type = data.get('seller_type', '')
        fuel_type = data.get('fuel_type', '')
        transmission_type = data.get('transmission_type', '')
        condition = data.get('condition', '')
        color = data.get('color', '')

        
        if not all([brand, model_name, vehicle_age, km_driven, seller_type, fuel_type, transmission_type, condition, color]):
            return jsonify({"error": "All fields are required."}), 400

        
        input_data = pd.DataFrame({
            "brand": [brand],
            "model": [model_name],
            "vehicle_age": [int(vehicle_age)],
            "km_driven": [int(km_driven)],
            "seller_type": [seller_type],
            "fuel_type": [fuel_type],
            "transmission_type": [transmission_type],
            "condition": [condition],
            "color": [color]
        })

        
        categorical_columns = ['brand', 'model', 'seller_type', 'fuel_type', 'transmission_type', 'condition', 'color']
        for col in categorical_columns:
            value = input_data[col].values[0]
            if value in encoder[col].classes_:
                input_data[col] = encoder[col].transform(input_data[col])
            else:
                return jsonify({"error": f"Unknown label '{value}' in column '{col}'"}), 400

        #  Predict
        predicted_price = model.predict(input_data)[0]
        # predicted_price = round(predicted_price/2, 2)

        return jsonify({"predicted_price": predicted_price}), 200

    except Exception as e:
        print("Error during prediction:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=port,debug=True)



