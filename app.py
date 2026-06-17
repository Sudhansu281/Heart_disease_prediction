from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("heart_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    age = request.form['age']
    sex = request.form['sex']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    features = [[
        float(age),
        float(sex),
        float(cp),
        float(trestbps),
        float(chol),
        float(fbs),
        float(restecg),
        float(thalach),
        float(exang),
        float(oldpeak),
        float(slope),
        float(ca),
        float(thal)
    ]]

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "⚠ Heart Disease Detected"
    else:
        result = "✅ No Heart Disease"

    return render_template(
        'index.html',
        prediction=result,
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal
    )


if __name__ == "__main__":
    app.run(debug=False)