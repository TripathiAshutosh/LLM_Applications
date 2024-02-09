import streamlit as st
from streamlit_chat import message as st_message
import streamlit.components.v1 as components
from rag import RAGModel

@st.cache_resource() # Decorator to cache functions that return global resources (e.g. database connections, ML models).
def initialize_ragmodel():
    return RAGModel()


class RAGStreamlitApplication:
    def __init__(self):
        self.ragmodel = initialize_ragmodel()

    def generate_answer(self):
        request = st.session_state.request
        with st.spinner('LLM RAG based response generation is in progress. Please wait !!!'):
            response = self.ragmodel(request=request)
            # response = "Using RAG with Huggingface transformers and the Ray retrieval implementation for faster distributed fine-tuning, \
            #  you can leverage RAG for retrieval-based generation on your own knowledge-intensive tasks."
        
        st.session_state.history.append({"message": request, "is_user": True})
        st.session_state.history.append({"message": response, "is_user": False})

    def run_app(self):
        st.title("Your Personal Gen AI Assistant!!!")

        

        with st.container(border=True):
            # st.write("This is inside the container")
            if "history" not in st.session_state:
                st.session_state.history = []
                
            for i, chat in enumerate(reversed(st.session_state.history)):
                st_message(**chat, key=str(i)) #unpacking
            # for i, chat in enumerate(st.session_state.history):
            #     st_message(**chat, key=str(i)) #unpacking
        
        st.text_input("", key="request", on_change=self.generate_answer)
        if st.button("Clear"):
                st.session_state.history = []
    
if __name__ == '__main__':
    ragqa = RAGStreamlitApplication()
    ragqa.run_app()