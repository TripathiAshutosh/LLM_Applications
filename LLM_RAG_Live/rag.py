import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

#load data, split_data, store in vecotrDB, retrieve, generate

path_vectorstore = "vectordb"

class RAGModel:
    def __init__(self):
        self.load_data()
        self.split_document()
        self.vector_store()
        self.retrieve()
    
    def __call__(self, request: str):
        return self.generate(request = request)

    def load_data(self):
        # from langchain.document_loaders import WebBaseLoader
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",),
            bs_kwargs={
                "parse_only": bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            },
        )
        self.docs = loader.load()
    
    def split_document(self):
        # from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        self.splits = text_splitter.split_documents(self.docs)

    def vector_store(self):
        # from langchain.embeddings import OpenAIEmbeddings
        # from langchain.vectorstores import Chroma

        self.vectorstore = Chroma.from_documents(documents=self.splits, embedding=OpenAIEmbeddings(),persist_directory=path_vectorstore)

    def retrieve(self):
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    def generate(self, request:str):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) #A temperature of 0 will produce the most deterministic responses, while a temperature of 1 will produce the most random responses.
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = hub.pull("rlm/rag-prompt")
        print(
            prompt.invoke(
                {"context": "filler context", "question": "filler question"}
            ).to_string()
        )
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        # RunnablePassthrough allows to pass inputs unchanged or with the addition of extra keys. This typically is used in conjuction with RunnableParallel to assign data to a new key in the map. 
        # RunnablePassthrough() called on it’s own, will simply take the input and pass it through.
        for chunk in rag_chain.stream(request):
            print(chunk, end="", flush=True)
        return "".join(chunk for chunk in rag_chain.stream(request))