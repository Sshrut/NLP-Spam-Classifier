from flask import Flask,render_template,request
import pickle



cv=pickle.load(open('transform.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vect = cv.transform(data).toarray()
    my_prediction = model.predict(vect)
    
    return render_template('home.html', prediction_text=my_prediction)
        
if __name__=='__main__':
      app.run(debug=True)  
    
