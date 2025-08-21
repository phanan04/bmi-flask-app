# Cell 6: Flask web app (fixed)
from flask import Flask, request, render_template_string
import joblib, pandas as pd

app = Flask(__name__)

# Load model và encoder
model = joblib.load("best_model.pkl")
le = joblib.load("label_encoder.pkl")

def health_advice(label, bmi):
    if "Under" in label:
        return "Bạn đang thiếu cân. Ăn đủ dinh dưỡng, tập sức mạnh, ngủ đủ."
    elif "Over" in label:
        return "Bạn đang thừa cân. Giảm calo nhẹ, tăng vận động, tập luyện đều đặn."
    return "BMI bình thường. Duy trì chế độ ăn cân bằng và vận động hợp lý."

TEMPLATE = """
<!doctype html>
<html lang="vi">
  <head>
    <meta charset="utf-8">
    <title>BMI Classifier</title>
    <style>
      body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
      }
      .container {
        background: #fff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        width: 400px;
        text-align: center;
      }
      h2 {
        margin-bottom: 20px;
        color: #0072ff;
      }
      label {
        display: block;
        margin: 10px 0 5px;
        font-weight: bold;
        text-align: left;
      }
      input {
        width: 100%;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 8px;
        outline: none;
        font-size: 14px;
      }
      input:focus {
        border-color: #0072ff;
      }
      button {
        margin-top: 20px;
        width: 100%;
        padding: 12px;
        border: none;
        border-radius: 8px;
        background: #0072ff;
        color: white;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.3s;
      }
      button:hover {
        background: #0056cc;
      }
      .result {
        margin-top: 25px;
        padding: 15px;
        border-radius: 10px;
        background: #f1f7ff;
        text-align: left;
      }
      .result p {
        margin: 8px 0;
      }
      .advice {
        color: #333;
        font-style: italic;
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h2>BMI Classifier</h2>
      <form method="post">
        <label>Weight (kg):</label>
        <input name="weight" type="number" step="any" required>

        <label>Height (m):</label>
        <input name="height" type="number" step="any" required>

        <button type="submit">Predict</button>
      </form>
      {% if result %}
        <div class="result">
          <p><b>BMI:</b> {{ result.bmi }}</p>
          <p><b>Category:</b> {{ result.label }}</p>
          <p class="advice"><b>Advice:</b> {{ result.advice }}</p>
        </div>
      {% endif %}
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    if request.method == "POST":
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        print("DEBUG - Weight:", weight, "Height:", height)  # Debug

        bmi = weight / (height ** 2)
        print("DEBUG - BMI:", bmi)  # Debug

        # Predict
        X_new = pd.DataFrame([[height, weight]], columns=["Height", "Weight"])
        pred = model.predict(X_new)[0]
        label = pred

        result = {
            "bmi": f"{bmi:.2f}",
            "label": label,
            "advice": health_advice(label, bmi)
        }

    return render_template_string(TEMPLATE, result=result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

