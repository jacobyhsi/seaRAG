#!/bin/bash

# Parsing
# Create a new conda environment with Python 3.10
conda create -n rag_vllm python=3.10 -y
conda activate rag_vllm
pip install ipykernel
pip install llama-index
# pip install llama-index-readers-file
pip install accelerate
pip install nbconvert
pip install -U ipywidgets
pip install PyMuPDF
ctrl + left-click PyMuPDFReader(), change fitz to pymupdf

pip install llama-index-llms-vllm
pip install numpy==1.26.4 vllm==0.4.1
pip install flash-attn==2.5.6

# pip install llama-index-llms-huggingface
# pip install transformers


##### IGNORE #####
# Generation
# Create a new conda environment with Python 3.10
# conda create -n instrag python=3.10 -y

# # Activate the new conda environment
# conda activate instrag

# # Install numpy, vllm, and accelerate
# pip install numpy==1.26.4 vllm==0.4.1 accelerate
# pip install vllm==0.4.2

# # Install flash-attn
# # conda install -c conda-forge cudatoolkit-dev
# # export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
# # export PATH=$CUDA_HOME/bin:$PATH
# # export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# # pip install flash-attn==2.5.6 --no-build-isolation
# # pip install -U vllm==0.4.1
# # pip install -U flash-attn==2.5.6 --no-build-isolation

# # Download datasets
# pip install gdown
# gdown 1MVkdc4g9_D4REtaBFKeJ9gMun4qzdQtO
# unzip dataset.zip

# # pyserini
# conda create -n pyserini python=3.10 -y
# conda activate pyserini
# pip install pyserini
# conda install -c conda-forge openjdk
# pip install faiss-cpu
# pip install faiss-gpu

# python -m pyserini.encode \
#   input --corpus retrieval/documents.jsonl \
#         --fields text \
#         --delimiter "\n" \
#         --shard-id 0 \
#         --shard-num 1 \
#   output --embeddings retrieval/embeddings \
#   encoder --encoder castorini/tct_colbert-v2-hnp-msmarco \
#          --fields text \
#          --batch 32 \
#          --fp16

# python -m pyserini.index.faiss \
#   --input retrieval/embeddings \
#   --output retrieval/indexing \
#   --hnsw

# python retrieval/search.py

