from flask import Flask,render_template,request
import pickle


model=pickle.load(open('model.pkl','rb'))

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
        final_words=[]
        from nltk.corpus import stopwords
        stop=stopwords.words('english')
        
        for i in words:
            if i in stop:
                continue
        
            import re
            # Replace email addresses with 'email'
            processed = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$',
                                 'emailaddress',i)

            # Replace URLs with 'webaddress'
            processed = re.sub(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$',
                                      'webaddress',processed)
    
            # Replace money symbols with 'moneysymb' 
            processed = re.sub(r'£|\$', 'moneysymb',processed)
            
            # Replace 10 digit phone numbers (formats include paranthesis, spaces, 
            #                                   no spaces, dashes) with 'phonenumber'
            processed = re.sub(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$',
                                  'phonenumbr',processed)
    
            # Replace numbers with 'numbr'
            processed = re.sub(r'\d+(\.\d+)?', 'numbr',processed)

            # Remove punctuation
            processed = re.sub(r'[^\w\d\s]', ' ',processed)

            # Replace whitespace between terms with a single space
            processed = re.sub(r'\s+', ' ',processed)
        
            # Remove leading and trailing whitespace
            processed = re.sub(r'^\s+|\s+?$', '',processed)

            processed = processed.lower()
            
            final_words.append(processed)
        
        data = [' '.join(final_words)]
        vect = cv.transform(data).toarray()
        my_prediction = model.predict(vect)
    
        return render_template('home.html', prediction_text=my_prediction)
        
if __name__=='__main__':
      app.run()  
    
