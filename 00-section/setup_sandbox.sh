# clear space to run big models
rm -rf /home/workspace/mambaforge/envs/scipy-full-stack-ml 
rm -rf /home/workspace/mambaforge/envs/full-stack-ML-metaflow-tutorial
rm -rf /home/workspace/mambaforge/envs/scipy-full-stack-ml 
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-nlp
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-cv
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-cv-2
rm -rf /home/workspace/mambaforge/envs/recsys
rm -rf /home/workspace/mambaforge/envs/recsys-2
rm -rf /home/workspace/mambaforge/envs/youtube-transcription

# install new packages
mamba create -n llm-rag python=3.10.12 pip -y
conda init bash
source ~/.bashrc
conda activate llm-rag
mamba install jupyter ipywidgets ipykernel pytorch torchvision torchaudio cpuonly -c pytorch -y 
/home/workspace/mambaforge/envs/llm-rag/bin/pip install transformers accelerate openai cohere ai21 transformers "pinecone-client[grpc]"
/home/workspace/mambaforge/envs/llm-rag/bin/pip install -qqq git+https://github.com/outerbounds/rag-demo \
    python-frontmatter \
    pyyaml \
    python-slugify \
    GitPython \
    sentence-transformers \
    seaborn
/home/workspace/mambaforge/envs/llm-rag/bin/pip install langchain \
    git+https://github.com/outerbounds/rag-demo \
    python-frontmatter \
    pyyaml \
    python-slugify \
    GitPython \
    sentence-transformers \
    seaborn

# reset jupyter kernel
/home/workspace/mambaforge/envs/llm-rag/bin/python -m ipykernel install --user --name llm-rag --display-name "LLMs and RAG"