from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import warnings
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


app = Flask(__name__)


#loading pickle files
tfidf = pickle.load(open("Tfidf.pkl","rb"))
MNB = pickle.load(open("MNB.pkl","rb"))

ps = PorterStemmer()


#Input Cleaning
def test_preprocess(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()

    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()

    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    

    return " ".join(y)

@app.route('/',methods=["Get","POST"])
def index():
    result = None
    if request.method == "POST":
        input_text= request.form["email"]
        if input_text:
            transform_input= test_preprocess(input_text)  #preproceesing input test
            vectorized_input = tfidf.transform([transform_input])  #converting input to tf-idf matrix
            predicted_output = MNB.predict(vectorized_input) # predicting the output
            if predicted_output[0] == 1:
                result = "This email is most likely to be a Spam"
            else:
                result = "This email doesn't seem to be a spam"
    return render_template('index.html',result=result)


if __name__ == "__main__":
    app.run(debug=True)
