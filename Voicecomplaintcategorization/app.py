from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load("maintenance_classifier.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    complaint = request.form["complaint"]
    prediction = model.predict([complaint])[0]
    return render_template("index.html", prediction=prediction, complaint=complaint)

if __name__ == "__main__":
    app.run(debug=True)
