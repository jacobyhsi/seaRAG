import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
# import nltk
# nltk.download("stopwords")
from IPython.display import Markdown, display
from tqdm.notebook import tqdm
# current_dir = os.path.dirname(os.path.abspath(__file__))

from llama_index.core import SimpleDirectoryReader, load_index_from_storage, StorageContext, VectorStoreIndex
from llama_index.readers.file import (
    DocxReader,
    PptxReader,
    XMLReader,
    PyMuPDFReader,
)
from llama_index.llms.vllm import Vllm
# from llama_index.llms.huggingface import HuggingFaceLLM

dataset_dir = os.getcwd() + "/datasets"
storage_dir = os.getcwd() + "/storage"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

parse = True
if parse == True:
    # Parse Data
    parser = PyMuPDFReader()
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        dataset_dir, file_extractor=file_extractor
    ).load_data()

    # VectorDB
    index = VectorStoreIndex.from_documents(documents)
    index.set_index_id("vector_index")
    index.storage_context.persist("./storage")

# Load VectorDB
storage_context = StorageContext.from_defaults(persist_dir="storage")
index = load_index_from_storage(storage_context, index_id="vector_index")

# LLM
llm = Vllm(model='meta-llama/Meta-Llama-3-8B-Instruct', tensor_parallel_size=4)
# llm = Vllm(model='meta-llama/Meta-Llama-3-70B-Instruct', tensor_parallel_size=4)
# llm = Vllm(model='nvidia/Llama-3.1-Nemotron-70B-Instruct-HF', tensor_parallel_size=4)

# llm = Vllm(model='meta-llama/Meta-Llama-3-70B-Instruct', tensor_parallel_size=4)
# llm = Vllm(model='deepseek-ai/DeepSeek-R1-Distill-Llama-70B', tensor_parallel_size=4)
# llm = HuggingFaceLLM(
#     model_name='nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',  # or another Hugging Face model of your choice
#     device_map="auto"    # Automatically map model layers to available devices (GPU/CPU)
# )

chat_engine = index.as_chat_engine(chat_mode="best")
# chat_engine = index.as_chat_engine(chat_mode="best", verbose=True)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break  # Exit the loop

    streaming_response = chat_engine.stream_chat(user_input)
    
    markdown_response = f"**User:** {user_input}\n\n**LLM:** "  # Format for Markdown

    for chunk in streaming_response.response_gen:
        markdown_response += chunk  # Append chunks to build the full response

    display(Markdown(markdown_response))  # Display in Markdown format