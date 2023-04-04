from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.post('/predict')
def predict():
    # Get the data from the POST request.
    data = request.form.to_dict()
    
    # make prediction

    return render_template('predict.html')
