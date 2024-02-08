from langchain.chains import RetrievalQA
# from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from flask import Flask, request, render_template, jsonify

from configparser import ConfigParser
config_object = ConfigParser()
config_object.read("config.ini")
openai_config = config_object["openai"]
open_ai_key = openai_config['key']

global db_chain

app = Flask(__name__)

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/set_params_session',methods=["GET","POST"])
def set_params_session():
    global dataDirectory
    dataDirectory = request.form["dataDirectory"]
    global docsearch
    docsearch = gen_and_store_embeddings()
    global qa_chain
    qa_chain = generate_qa_chain()
    
    return "True"

def document_loader():
    loader = DirectoryLoader(dataDirectory)
    documents = loader.load()
    return documents
def split_documents():
    documents = document_loader()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts
def gen_and_store_embeddings():
    texts = split_documents()
    embeddings = OpenAIEmbeddings(openai_api_key=open_ai_key)
    docsearch = Chroma.from_documents(texts, embeddings)
    return docsearch
def generate_qa_chain():
    model_name= "gpt-3.5-turbo"
    llm = ChatOpenAI(openai_api_key=open_ai_key,
                     model_name=model_name, temperature=0)
    retriever=docsearch.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    #qa_chain = load_qa_chain(llm, chain_type="stuff")
    return qa_chain

@app.route("/get")
def get_bot_response():
    query = request.args.get('msg')
    # matching_documents = docsearch.similarity_search(query)
    # response = qa_chain.run(input_documents = matching_documents,question=query)
    response = qa_chain.run(query)
   
    return response

if __name__ == "__main__":
    app.run(debug=True)