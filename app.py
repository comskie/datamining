from flask import Flask, render_template, request
import predictor

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.post('/predict')
def predict():
    # Get the data from the POST request.
    data = request.form.to_dict()
    
    # make prediction
    result = predictor.predict_naive_bayes(data)

    return render_template('predict.html')
