

from flask import (Flask,  render_template, request, send_from_directory, url_for,jsonify)
from chat import get_response,extract_fatwa,retrive_fatwa,save_fatwa


app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        
    )

@app.post("/predict")
def predict():
    text= request.get_json().get("message")
    global response
    response= get_response(text)
    if response['mode']==0 or response['mode']==1:
        message={"answer":response["res"]}
    elif response['mode']==2:
        message={"answer":response["res"]['answer']+"<br><br>"+"المصدر: كتاب الحج والعمرة 1060 سؤال وجواب "}
    elif response['mode']==3 and response["res"]['bestResult']!=None:
        message={"answer":response["res"]['bestResult']['answer']+"<br><br>"+"https://binbaz.org.sa :المصدر"}
    else:
        message={"answer":" لا اجد فتوى مناسبة ..."}

    return jsonify(message)

@app.post("/save")
def save():
    #text= request.get_json().get("message")
    #response= get_response(text)
    #message={"answer":response}
    save_fatwa(response)
    return jsonify({"answer":"شكرا للدعم"})


if __name__ == '__main__':
   app.run(debug=True)
