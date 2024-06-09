import os
import torch
from raptor import RetrievalAugmentationConfig, RetrievalAugmentation, BaseQAModel, BaseEmbeddingModel
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

os.environ["LLAMAAPI"]="jdjwndwkdh..." #change the API


#EmbeddingClass
class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
    

#QA model    
class LlamaQAModel(BaseQAModel):
    def __init__(self, model_name="NousResearch/Llama-2-7b-chat-hf"):
        # Initialize the Llama model and tokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
    
#KnowledgeGraph Vector Data and Retrival       
RAC = RetrievalAugmentationConfig(qa_model=LlamaQAModel(), embedding_model=SBertEmbeddingModel())

RA = RetrievalAugmentation(config=RAC) #to create knowledge graphs using the pipeline

with open('sample.txt', 'r') as file:
    text = file.read()
    
RA.add_documents(text)

question = "How did Cinderella reach her happy ending?"

answer = RA.answer_question(question=question)

print("Answer: ", answer)