import os
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseQAModel, BaseEmbeddingModel
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer
import streamlit as st
import replicate

st.set_page_config(page_title="1000doc_chat")

with st.sidebar:
    st.title('1000doc_Chat')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')
    os.environ['REPLICATE_API_TOKEN'] = replicate_api
    
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2-7B', 'Llama2-13B'], key='selected_model')
    if selected_model == 'Llama2-7B':
        llm = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
    elif selected_model == 'Llama2-13B':
        llm = 'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5'
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        st.write("filename:", uploaded_file.name)
        st.write(bytes_data)
    temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        
class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
    
class LlamaQAModel(BaseQAModel):
    def __init__(self, model_name="NousResearch/Llama-2-7b-chat-hf"):
        # Initialize the Llama model and tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
RAC = RetrievalAugmentationConfig(qa_model=LlamaQAModel(), embedding_model=SBertEmbeddingModel())

RA = RetrievalAugmentation(config=RAC) #to create knowledge graphs using the pipeline

with open('sample.txt', 'r') as file:
    text = file.read()
    
RA.add_documents(text)

question = prompt

answer = RA.answer_question(question=question)



        
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = RA.answer_question(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)


    
# from huggingface_hub import login
# access_token_read = "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"
# access_token_write = "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"
# login(token = access_token_write)


