from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    income = float(request.form["income"])
    loan = float(request.form["loan"])
    credit = int(request.form["credit"])

    prediction = model.predict([[income, loan, credit]])

    result = "Loan Approved" if prediction[0] == 1 else "Loan Rejected"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__=="__main__":
    app.run(debug=True)