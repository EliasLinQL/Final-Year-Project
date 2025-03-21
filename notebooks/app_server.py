# backend/app_server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time
import sys
import subprocess

from Data_collection import fetch_crypto_data

app = Flask(__name__)
CORS(app)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(project_root, 'data')
os.makedirs(data_dir, exist_ok=True)

@app.route('/api/fetch_crypto_data', methods=['POST'])
def api_fetch_data():
    try:
        req = request.json
        start_date = req.get("start_date")
        end_date = req.get("end_date")
        symbols = req.get("symbols")

        if not (start_date and end_date and symbols):
            return jsonify({"status": "error", "message": "Missing parameters"}), 400

        start_time = int(time.mktime(time.strptime(start_date, '%Y-%m-%d')) * 1000)
        end_time = int(time.mktime(time.strptime(end_date, '%Y-%m-%d')) * 1000)

        print(f"📥 API Request Received: {start_date} → {end_date}, Symbols: {symbols}")

        file_paths = fetch_crypto_data(symbols, start_time, end_time)
        if not file_paths:
            return jsonify({"status": "error", "message": "No data fetched"}), 400

        result = execute_data_processing()
        return jsonify({
            "status": result.get("status"),
            "message": result.get("message")
        }), 200

    except Exception as e:
        print("❌ API Error (fetch_crypto_data):", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    try:
        print("🚀 Model training request received.")
        result = execute_model_training()
        return jsonify({
            "status": result.get("status"),
            "message": result.get("message")
        }), 200
    except Exception as e:
        print("❌ API Error (train_model):", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

def execute_data_processing():
    try:
        script = os.path.join(os.path.dirname(__file__), 'Data_processing.py')
        python_exec = sys.executable
        print(f"⚙️ Executing: {python_exec} {script}")
        result = subprocess.run(
            [python_exec, script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'  # 👈 防止 UnicodeDecodeError
        )
        print("📤 STDOUT:", result.stdout)
        print("📥 STDERR:", result.stderr)
        if result.returncode == 0:
            return {"status": "success", "message": "Data processing completed."}
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def execute_model_training():
    try:
        script = os.path.join(os.path.dirname(__file__), 'LSTM-GCN Model.py')
        python_exec = sys.executable
        print(f"⚙️ Executing: {python_exec} {script}")
        result = subprocess.run(
            [python_exec, script],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print("📤 STDOUT:", result.stdout)
        print("📥 STDERR:", result.stderr)
        if result.returncode == 0:
            return {"status": "success", "message": "Model training completed."}
        else:
            return {"status": "error", "message": result.stderr}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    print("🚀 Starting Flask server at http://localhost:5000")
    print(f"📁 Project Root: {project_root}")
    app.run(host='0.0.0.0', port=5000, debug=True)
