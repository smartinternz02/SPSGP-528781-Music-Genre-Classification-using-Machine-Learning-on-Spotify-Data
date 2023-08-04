from flask import Flask, request, render_template,url_for,redirect
import numpy as np
import joblib
app=Flask (__name__)# initializing a flask app 

model=joblib.load('bagging.model')
sc=joblib.load('transform.save')
@app.route('/')# route to display the home page
def home():
    return render_template('home.html') #rendering the home page
@app.route('/prediction', methods=['POST', 'GET'])
def prediction(): # route which will take you to the prediction page
    return render_template('indexnew.html')
@app.route('/home', methods=['POST', 'GET'])
def my_home(): 
    return redirect(url_for(''))
@app.route('/predict', methods=["POST", "GET"])# route to show the predictions in a web ut
def predict():
#reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]
    x=[np.array(input_feature)]
    x= sc.transform(x)
    print(x)
    prediction=model.predict(x)
    labels=['acoustic','afrobeat','alt-rock','alternative','ambient','anime','black-metal','bluegrass','blues','brazil','breakbeat','british','cantopop','chicago-house','children','chill','classical','club','comedy','country','dance','dancehall','death-metal','deep-house','detroit-techno','disco','disney','drum-and-bass','dub','dubstep','edm','electro','electronic','emo','folk','forro','french','funk','garage','german','gospel','goth','grindcore','groove','grunge','guitar','happy','hard-rock','hardcore','hardstyle','heavy-metal','hip-hop','honky-tonk','house','idm','indian','indie','indie-pop','industrial','iranian','j-dance','j-idol','j-pop','j-rock','jazz','k-pop','kids','latin','latino','malay','mandopop','metal','metalcore','minimal-techno','mpb','new-age','opera','pagode','party','piano','pop','pop-film','power-pop','progressive-house','psych-rock','punk','punk-rock','r-n-b','reggae','reggaeton','rock','rock-n-roll','rockabilly','romance','sad','salsa','samba','sertanejo','show-tunes','singer-songwriter','ska','sleep','songwriter','soul','spanish','study','swedish','synth-pop','tango','techno','trance','trip-hop','turkish','world-music']
    return render_template('result.html',data=labels [prediction[0]])

if __name__ =="__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)