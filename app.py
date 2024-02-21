from flask import Flask,jsonify
import os
from flask import request
from html.parser import HTMLParser
from PyPDF2 import PdfReader
import xml.etree.ElementTree as ET
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.callbacks import get_openai_callback

app = Flask(__name__)
OPENAI_API_KEY = "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"
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
knowledge_base = None
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        """ leo los archivos enviado """
        f = request.files['the_file']
        f1 = request.files['the_file1']
        nombre_archivo = f.filename
        f.save(nombre_archivo)
        global knowledge_base 
        knowledge_base = create_embeddings(nombre_archivo,f1)
        return "files uploaded"
@app.route("/question", methods=['POST'])
def questions():
    
    global knowledge_base 
    """ print(request.form['question']) """
    user_question = request.form['question']
    print(knowledge_base)
    a = preguntar(user_question, knowledge_base)
    print(type(a))
    
    data = {
        "data":a["respuesta"],
        "price":a["price"]
       
    }
    return jsonify(data)
    
def preguntar(user_question,knowledge_base):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    docs = knowledge_base.similarity_search(user_question, 3)
    llm = ChatOpenAI(model_name='gpt-3.5-turbo-0125')
    chain = load_qa_chain(llm, chain_type="stuff")
    respuesta = chain.run(input_documents=docs, question=user_question)
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        c= str(cb)
        """ print(c,type(c)) """
        
        
    """ print(docs) """
    """ return respuesta """
    return {"respuesta":respuesta,"price":c}

def create_embeddings(pdf,arch1):
   """ obtengo el texto de los archivos """ 
   with open(pdf,encoding='utf-8')as f:
    t =f.read()
    xml_text = str(arch1.read())
    print(xml_text)
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,
        chunk_overlap=100,
        length_function=len)
    """ genera chunks de los 2 archivos (t+xml_text) y asi genero chunks con los 2 archivos """    
    chunks = text_splitter.split_text(t+xml_text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base