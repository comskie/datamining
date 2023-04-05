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

    message = 'You\'re good!' if result[0] > result[1] else 'You suck!'
    return render_template('predict.html', yes="{:.2f}".format(result[0]), no="{:.2f}".format(result[1]), message=message)
