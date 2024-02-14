import streamlit as st
from streamlit_chat import message
import tempfile
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.title("Ask about Player Statistics in T20!")



def load_llm():
    llm = HuggingFacePipeline.from_model_id(
        model_id="google/flan-t5-base",
        task="text2text-generation",
    )
    return llm


embeddings = HuggingFaceEmbeddings(model_name='thenlper/gte-large',
                                   model_kwargs={'device': 'cpu'})

db = FAISS.load_local("faiss/cricket", embeddings)
llm = load_llm()

prompt_temp = '''
With the information provided try to answer the question. 
If you cant answer the question based on the information either say you cant find an answer or please check your question again.
This is related to cricket. So try to understand in depth about the context and answer only based on the information provided. Dont generate irrelevant answers

Context: {context}
Question: {question}
Do provide only correct answers

Correct answer:
    '''
custom_prompt_temp = PromptTemplate(template=prompt_temp,
                                    input_variables=['context', 'question'])

retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                 retriever=db.as_retriever(search_kwargs={'k': 1}),
                                                 chain_type="stuff",
                                                 return_source_documents=True,
                                                 chain_type_kwargs={"prompt": custom_prompt_temp}
                                                 )


def cricbot(query):
    answer = retrieval_qa_chain({"query": query})
    return answer["result"]


if 'user' not in st.session_state:
    st.session_state['user'] = ["Hey!!"]

if 'assistant' not in st.session_state:
    st.session_state['assistant'] = ["Hey, I am your personal bot who can answer almost everything about player Statistics!"]

container = st.container()
print(container)

with container:
    with st.form(key='cricket_form', clear_on_submit=True):
        user_input = st.text_input("", placeholder="Type here", key='input')
        submit = st.form_submit_button(label='Answer')
        if submit:
            output = cricbot(user_input)
            st.session_state['user'].append(user_input)
            st.session_state['assistant'].append(output)

if st.session_state['assistant']:
    for i in range(len(st.session_state['assistant'])):
        message(st.session_state["user"][i], is_user=True, key=str(i) + '_user')
        message(st.session_state["assistant"][i], key=str(i))




