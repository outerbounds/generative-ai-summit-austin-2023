# clear space to run big models
rm -rf /home/workspace/mambaforge/envs/scipy-full-stack-ml 
rm -rf /home/workspace/mambaforge/envs/full-stack-ML-metaflow-tutorial
rm -rf /home/workspace/mambaforge/pkgs/cache
rm -rf /home/workspace/mambaforge/envs/scipy-full-stack-ml 
rm -rf /home/workspace/mambaforge/envs/full-stack-ML-metaflow-tutorial
rm -rf /home/workspace/mambaforge/pkgs/cache
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-nlp
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-cv
rm -rf /home/workspace/mambaforge/envs/mf-tutorial-cv-2
rm -rf /home/workspace/mambaforge/envs/recsys
rm -rf /home/workspace/mambaforge/envs/recsys-2
rm -rf /home/workspace/mambaforge/envs/youtube-transcription
rm -rf /home/workspace/.cache/

# install new packages
mamba install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install transformers optimum accelerate
pip install pip install --upgrade jupyter ipywidgets