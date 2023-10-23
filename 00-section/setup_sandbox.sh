# clear space to run big models
rm -rf /home/workspace/mambaforge/envs/scipy-full-stack-ml 
rm -rf /home/workspace/mambaforge/envs/full-stack-ML-metaflow-tutorial
rm -rf /home/workspace/mambaforge/envs/scipy-full-stack-ml 
rm -rf /home/workspace/mambaforge/envs/full-stack-ML-metaflow-tutorial
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-nlp
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-cv
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-cv-2
rm -rf /home/workspace/mambaforge/envs/recsys
rm -rf /home/workspace/mambaforge/envs/recsys-2
rm -rf /home/workspace/mambaforge/envs/youtube-transcription

# install new packages
mamba install pytorch torchvision torchaudio cpuonly -c pytorch -y
mamba update jupyter ipywidgets ipykernel -y
pip install transformers accelerate 
pip install -qqq git+https://github.com/outerbounds/rag-demo \
    python-frontmatter \
    pyyaml \
    python-slugify \
    GitPython \
    sentence-transformers \
    seaborn
pip install langchain

# reset jupyter kernel
python -m ipykernel install --user --name sandbox-tutorial --display-name "Sandbox Onboarding Tutorial"