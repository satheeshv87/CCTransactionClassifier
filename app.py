from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
    
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # Read input Data
    data = pd.read_csv('data.csv')
    col = ['category_desc', 'description']
    df = data[col]
    df = df[pd.notnull(df['description'])]
    df.columns = ['category_desc', 'description']
    df['category_id'] = df['category_desc'].factorize()[0]

    # Create and fit model
    X_train, X_test, y_train, y_test = train_test_split(df['description'], df['category_desc'], random_state = 42)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    
    # Get the message that is typed and predict the category
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = count_vect.transform(data).toarray()
        my_prediction = clf.predict(vect)
        my_prediction = clf.predict(count_vect.transform(data))        
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)