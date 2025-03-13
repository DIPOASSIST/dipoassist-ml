from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model dan scaler dari file pkl
model_data = joblib.load("model.pkl")
rf_model = model_data["model"]
scaler = model_data["scaler"]

# Mapping label ke teks deskripsi
label_mapping = {
    "A": "makan",
    "B": "sakit",
    "C": "tolong",
    "D": "toilet",
    "E": "haus"
}

# Inisialisasi Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Flask API untuk Prediksi dengan Random Forest!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data JSON dari request
        data = request.json
        if not data or "features" not in data:
            return jsonify({"error": "Data tidak valid"}), 400
        
        # Convert input ke array numpy
        features = np.array(data["features"]).reshape(1, -1)
        
        # Normalisasi data menggunakan scaler yang sudah disimpan
        scaled_features = scaler.transform(features)
        
        # Lakukan prediksi
        prediction = rf_model.predict(scaled_features)[0]  # Ambil nilai pertama

        # Ambil deskripsi dari mapping
        prediction_text = label_mapping.get(prediction, "Tidak diketahui")

        # Kirim hasil prediksi
        return jsonify({
            "prediction": prediction,
            "description": prediction_text
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Jalankan API
if __name__ == "__main__":
    app.run(debug=True)
