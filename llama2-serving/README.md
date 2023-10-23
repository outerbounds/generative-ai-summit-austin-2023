# Make fresh Ubuntu Instance with GPU
After you make the VM and ssh into it, clone/copy this directory onto your instance.

## GPU Memory requirements
| Device | GPU Memory | Enough? |
| :---: | :---: | :---: |
| A10G |  24GB | ﹖ | 
| A40 | 48GB | ✅ |

# Install & Configure Docker on Your Ubuntu VM

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

```
alias docker='sudo docker'
```

```
docker pull nvcr.io/nvidia/tritonserver:23.08-py3
```

# Setup the server

## Run the docker container
From the `llama2-serving` directory:
```
MODEL_REPO_PATH=llm
MODEL_REPO_SERVER_PATH=/models
docker run --gpus=all -it --shm-size=1g --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}:/workspace/ -v ${PWD}/${MODEL_REPO_REL_PATH}:/${MODEL_REPO_SERVER_PATH} nvcr.io/nvidia/tritonserver:23.08-py3 bash
```

## Install dependencies
``` 
python3 -m pip install app
python3 -m pip install torch
python3 -m pip install scipy
python3 -m pip install metaflow
python3 -m pip install accelerate
python3 -m pip install bitsandbytes
python3 -m pip install peft
python3 -m pip install trl
python3 -m pip install transformers
python3 -m pip install huggingface_hub
```

# Run the Triton server
```
cd /models/llm/llama2/1
tritonserver --model-repository=/models/llm --log-verbose=2
```

# Simulate the client
Open another terminal and login to the Triton SDK container
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.08-py3-sdk bash
```

## Make some requests!
```bash
python3 client.py --model_name "llama2"
```

Even better, open up Python:
```
python3
```
and then use the `chat_iter` or `batch_inference` interface:
```
from client import *
_ = batch_inference([
    ["I want to use AI technology to help humans and other animals live more fulfilling lives. How can AI help?"],
    ["Write a concise roadmap for how AI can generate abudance without runaway wealth inequality."],
    ["Write a set of fun activities I can do with my nieces."]
])
```

# Troubleshooting

On Coreweave, if you see this on `nvidia-smi` inside the container:
```
“Failed to initialize NVML: Unknown Error”
```
Exit the container runtime. 
Then, open `/etc/nvidia-container-runtime/config.toml` and set `no-cgroups = false`, then restart docker:
```
sudo systemctl restart docker
```
and continue from the beginning of the [setup the server](#set-up-the-server) section of this page.
