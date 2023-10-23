FROM python:3.9.18
RUN pip install git+https://github.com/outerbounds/rag-demo \
    python-frontmatter \
    pyyaml \
    python-slugify \
    gitpython \
    pandas \
    tldextract \
    langchain==0.0.262 \
    openai \
    pinecone-client[grpc] \
    matplotlib \
    seaborn \
    pyyaml
RUN pip install --no-cache-dir torch==1.8.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-deps sentence-transformers