from flask import Flask, render_template, request
from editor import funforfrontend

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])

def analyze():
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        summary, original_text = funforfrontend(rawtext)
    return render_template('summary.html', summary=summary, original_text=original_text)

if __name__ == "__main__":
    app.run(debug=True)