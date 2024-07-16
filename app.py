from flask import Flask , render_template , request 
import pickle 
import pandas as pd

ridge_model = pickle.load(open('reg.pkl','rb'))
standard_scaler = pickle.load(open('scl.pkl','rb'))
application = Flask(__name__)
app = application
@app.route('/')
def welcome():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
  if request.method == 'POST':
    Weight = float(request.form.get('Weight'))
    features = pd.DataFrame([[Weight]], columns=['Weight'])
    new_data_scaled = standard_scaler.transform(features)
    result = ridge_model.predict(new_data_scaled)
    return render_template("home.html",result=result[0][0])
  else:
    print("else")
    return render_template("home.html")