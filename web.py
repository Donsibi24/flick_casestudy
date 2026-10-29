from flask import Flask, render_template, request
import pickle  

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            prediction_result = "Pass" 
            
            return render_template("index.html", prediction_text=prediction_result)
        
        except Exception as e:
            return str(e) 

if __name__ == "__main__":
    app.run(debug=True)