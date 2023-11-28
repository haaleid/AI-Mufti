import random
import json
import csv 
import torch

   
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

from arabert import ArabertPreprocessor
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
import traceback
import logging



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#requirements for intention classfication
with open('intentsAr.json', 'r',encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Mufti"

#requirements for responce retrival
from transformers import logging

logging.set_verbosity_warning()
with open("QAPool.csv") as csv_file:
    reader = csv.DictReader(csv_file)
    questions=[]
    answers=[]
    for row in reader:
        questions.append(row["Question"])
        answers.append(row["Answer"])
embedder = SentenceTransformer('Models/runSArabertDQ')
corpus = questions
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

#requirements for responce extraction
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.headless = True
driver = webdriver.Firefox(options )
driver1 = webdriver.Firefox(options)
prep = ArabertPreprocessor("aubmindlab/araelectra-base-discriminator")
qa_pipe =pipeline("question-answering",model="Models/runAraelectraQA")
# requirment for google search
from googlesearch import search
def search_fatwa(msg):
    search_result=search(msg,  lang='ar', num=10, start=0, stop=10, pause=2)
    #for i in search_result:
        #print(i)
    return search_result
# requirment to compute similarity
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
tokenizerSM = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
modelSM = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def compute_similarity(msg1,msg2):
    
    input_texts = ['query: '+msg1,
               "passage: "+ msg2]
    
    # Tokenize the input texts
    batch_dict = tokenizerSM(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

    outputs = modelSM(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    print(scores.tolist())
    print(scores[0])
    print(type(scores[0]))
    return scores[0].item()

def retrive_fatwa(msg):
    query = [msg]
    top_k = min(1, len(corpus))
    query_embedding = embedder.encode(query, convert_to_tensor=True)    
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)  
    return {'question1':msg,'question2':questions[top_results[1][0].item()],'answer':answers[top_results[1][0].item()],'score':top_results[0][0].item()}
    
#extract from binbaz sit directlty
def extract_fatwa(msg):
    bestResult=None
    
    driver.get("https://binbaz.org.sa")
    try:
        prequestion=prep.preprocess(msg)
    except BaseException as error:
        print('An exception in preprocessing question {} {}')
        return {'question':None,'bestFtitle':None, 'bestFquestion':None,'bestFanswer':None,'bestResult':None,'bestURL':None}
            
    search = None
    flag=0
    while search is None:
        try:
            search = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.NAME, "q")))
        except TimeoutException as ex:
            print ("Time Out to find the search key")
            flag=flag+1
            if (flag>4):
                
                return {'question':None,'bestFtitle':None, 'bestFquestion':None,'bestFanswer':None,'bestResult':None,'bestURL':None}
        
    search.send_keys(msg)
    search.send_keys(Keys.RETURN)
        
    results=None
    flag=0
    while results is None:
        try:
                
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'div.search-results-count')))    
            results = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'article.box__body__element')))
            #print("all elements is appear") 
            #print(len(results))
            #for result in results:
                #print (result.get_attribute("innerHTML"))
        except TimeoutException as ex:
            print ("Time Out to get search results")
            flag=flag+1
            if (flag>4):
                return {'question':None,'bestFtitle':None, 'bestFquestion':None,'bestFanswer':None,'bestResult':None,'bestURL':None}
        
    
        
        
    for result in results:
           
        try:
            #print ("try to get the url")      
            fatwaURL=result.find_element("xpath", ".//a").get_attribute("href")
        except NoSuchElementException:
            print ("No Fatwa link")
            continue  
        except StaleElementReferenceException as e:
            print ("StaleElement Fatwa link as {}",e)
            continue
            
        driver1.get(fatwaURL)
        #print (fatwaURL)
        flag=0
        fatwa = None
        while fatwa is None:
            try:
                fatwa = WebDriverWait(driver1, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'fatwa')))
            except TimeoutException as ex:
                print ("Time Out to get fatwa content")
                flag=flag+1
                if (flag>4):
                    break
        if (flag>4):
            continue
            
        try:
            Ftitle=fatwa.find_element("xpath", ".//h1").text
            if compute_similarity(msg,Ftitle)<80:
                print ("Unrelated Fatwa")
                continue
        except NoSuchElementException as ex:
            print ("No Fatwa Title")
            continue
            
            
        try:
            Fquestion=fatwa.find_element("xpath", ".//h2").text
                
        except NoSuchElementException as ex:
            print ("No Fatwa question")
            continue
            
        try:
            tags=fatwa.find_elements(By.TAG_NAME,'div')
            Fanswer=tags[len(tags)-2].text+tags[len(tags)-1].text
            
        except NoSuchElementException as ex:
            print ("No Fatwa or context")
            continue
        
        try:
            context = prep.preprocess(Fanswer)# don't forget to preprocess the question and the context to get the optimal results
        except BaseException as error:
            print('An exception in preprocessing a context for question ')
            continue
           
        try:
            if context!=None:
                prediction = qa_pipe(question=prequestion,context=context,padding=True, truncation=True)
                    
            else:
                print('Empty context fatwa')             
                continue
        except BaseException as error:
            print('An exception in inferncing the answer of qusetion ')
            continue
            
        if bestResult==None:
            bestFtitle, bestFquestion,bestFanswer,bestResult,bestURL=Ftitle, Fquestion,Fanswer,prediction,fatwaURL
                
        elif bestResult["score"]<prediction["score"]:
            bestFtitle, bestFquestion,bestFanswer,bestResult=Ftitle, Fquestion,Fanswer,prediction
         
    
    
    return {'question':msg,'bestFtitle':bestFtitle, 'bestFquestion':bestFquestion,'bestFanswer':bestFanswer,'bestResult':bestResult,'bestURL':bestURL}
#extract from a spacific site using google search (question ,site)
def extract_fatwa(msg1,msg2):
    bestResult=None
    
   
    try:
        prequestion=prep.preprocess(msg1)
    except BaseException as error:
        print('An exception in preprocessing question {} {}')
        return {'question':None,'bestFtitle':None, 'bestFquestion':None,'bestFanswer':None,'bestResult':None,'bestURL':None}
            
   
        
    results=None
     
    try:
        results = search_fatwa("site:"+msg2+" "+msg1)
        
    except Exception as ex:
        print ("Time Out to get search results")
        return {'question':None,'bestFtitle':None, 'bestFquestion':None,'bestFanswer':None,'bestResult':None,'bestURL':None}
        
    
    bestFtitle, bestFquestion,bestFanswer,bestResult,bestFtitle=None, None,None,None,None    
    print (results)   
    for fatwaURL in results:
           
        
        driver1.get(fatwaURL)
        print (fatwaURL)
        flag=0
        fatwa = None
        while fatwa is None:
            try:
                fatwa = WebDriverWait(driver1, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'fatwa')))
                print(fatwa)
            except TimeoutException as ex:
                print ("Time Out to get fatwa content")
                flag=flag+1
                if (flag>1):
                    break
        if (flag>1):
            continue
            
        try:
            Ftitle=fatwa.find_element("xpath", ".//h1").text
            print(Ftitle)
            if compute_similarity(msg1,Ftitle)<80:
                print ("Unrelated Fatwa")
                continue
        except NoSuchElementException as ex:
            print ("No Fatwa Title")
            continue
            
            
        try:
            Fquestion=fatwa.find_element("xpath", ".//h2").text
            print(Fquestion)   
        except NoSuchElementException as ex:
            print ("No Fatwa question")
            continue
            
        try:
            tags=fatwa.find_elements(By.TAG_NAME,'div')
            Fanswer=tags[len(tags)-2].text+tags[len(tags)-1].text
            print(Fanswer)
        except NoSuchElementException as ex:
            print ("No Fatwa or context")
            continue
        
        try:
            context = prep.preprocess(Fanswer)# don't forget to preprocess the question and the context to get the optimal results
        except BaseException as error:
            print('An exception in preprocessing a context for question ')
            continue
           
        try:
            if context!=None:
                prediction = qa_pipe(question=prequestion,context=context,padding=True, truncation=True)
                    
            else:
                print('Empty context fatwa')             
                continue
        except BaseException as error:
            print('An exception in inferncing the answer of qusetion ')
            continue
            
        if bestResult==None:
            bestFtitle, bestFquestion,bestFanswer,bestResult,bestURL=Ftitle, Fquestion,Fanswer,prediction,fatwaURL
                
        elif bestResult["score"]<prediction["score"]:
            bestFtitle, bestFquestion,bestFanswer,bestResult=Ftitle, Fquestion,Fanswer,prediction
         
    
    
    return {'question':msg1,'bestFtitle':bestFtitle, 'bestFquestion':bestFquestion,'bestFanswer':bestFanswer,'bestResult':bestResult,'bestURL':bestURL}

def save_fatwa(response):

    if int(response["mode"])==2:

        with open('FeedbackQQ.csv','a',encoding='utf-8-sig',newline='') as f:
            r = csv.writer(f)
            try:  
                res=response["res"]
                #r.writerow(Question1,Question2,Answer,Score)
                r.writerow([res['question1'],res['question2'],res['answer'],res['score']])
            except BaseException as error:
                print('An exception in writing a row of qusetion ',error)   
            f.close()
    elif int(response['mode'])==3 and response["res"]['bestResult']!=None:
        with open('FeedbackQA.csv','a',encoding='utf-8-sig',newline='') as f:
            r = csv.writer(f)
            try:   
                #r.writerow(Question,Title,Fquestion,Fatwa,Answer,Score,Start,End,Tag)
                res=response["res"]
                r.writerow([res["question"],res['bestFtitle'],res['bestFquestion'],res['bestFanswer'],res['bestResult']['answer'],res['bestResult']['score'],res['bestResult']['start'],res['bestResult']['end'],1])
            except BaseException as error:
                print('An exception in writing a row of qusetion ',error)   
            f.close()


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        if tag=="fatwa":
            res= retrive_fatwa(msg)
            if res['score']>=0.8:
                return {"res":res,"mode":2}
            else:          
                return {"res":extract_fatwa(msg,"https://binbaz.org.sa"),"mode":3}
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return {"res":random.choice(intent['responses']),"mode":1}
    
    return {"res":"لم افهمك جيدا...","mode":0}


if __name__ == "__main__":
    print(" يمكن التحدث والتواصل مع المفتي")
    print(" اكتب خروج لانهاء المحادثة")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit" or sentence == "خروج":
            break

        resp = get_response(sentence)
        print(bot_name,':',resp)

    try:
        driver1.close()
        driver.close()
    except BaseException as error:
        print('An exception in closing the web drivers',error) 
