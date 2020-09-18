from flask import Flask,render_template,request
import pickle

flname='model.pkl'
model=pickle.load(open(flname,'rb'))

cv=pickle.load(open('change.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message = str(request.form['message'])
        words=message.split()

        final_words=list(map(str.lower,words))
        data = [' '.join(final_words)]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
    
        return render_template('home.html', prediction_text=my_prediction)
        
if __name__=='__main__':
      app.run()  
    
