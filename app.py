from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


def is_spam(text):
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]




@app.route('/', methods=['GET', 'POST'])
def index():
    message = ''
    if request.method == 'POST':
        user_input = request.form['text']
        if is_spam(user_input):
            message = 'This message is likely spam.'
        else:
            message = 'This message is not spam.'
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
