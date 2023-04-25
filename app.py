from flask import Flask, render_template, request
import predictor

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['post'])
def predict():
    # Get the data from the POST request.
    data = request.form.to_dict()

    # make prediction
    result = predictor.predict_naive_bayes(data)

    shift = result[0] >= predictor.THRESHOLD
    message = 'You\'re good!' if not shift else 'You suck!'

    return render_template(
        'predict.html',
        shift=shift,
        yes="{:.2f}%".format(result[0] * 100),
        no="{:.2f}%".format(result[1] * 100),
        message=message,
    )
