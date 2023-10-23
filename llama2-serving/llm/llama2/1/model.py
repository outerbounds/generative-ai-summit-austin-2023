import app
import os
import json
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import huggingface_hub
from threading import Thread
from tarfile import TarFile
from io import BytesIO
from metaflow import S3

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

METAFLOW_RUN_ID = "210840" # TODO: this should update dynamically somehow
DST_MODEL_NAME = "llama-2-7b-dolly15k"
PUBLIC_S3_OBJ = "s3://outerbounds-datasets/triton/llama2/%s_%s.tar" % (
    DST_MODEL_NAME,
    METAFLOW_RUN_ID,
)
CHECKPOINT_MODEL_PATH = "%s/model" % DST_MODEL_NAME
CHECKPOINT_TOKENIZER_PATH = "%s/tokenizer" % DST_MODEL_NAME


def extract_tar_bytes(tar_bytes, path):
    buf = BytesIO(tar_bytes)
    with TarFile(mode="r", fileobj=buf) as tar:
        tar.extractall(path=path)


def format_prompt(example):
    return f"""### INSTRUCTION: {example['instruction']}

    ### CONTEXT: {example['context']}
                            
    ### RESPONSE: {example['response']}
    """


class TritonPythonModel:
    def initialize(self, args):
        # Download the model from S3
        # with S3() as s3:
        #     obj = s3.get(PUBLIC_S3_OBJ)
        #     os.mkdir(DST_MODEL_NAME)
        #     cwd = os.getcwd()
        #     os.chdir(DST_MODEL_NAME)
        #     extract_tar_bytes(obj, UNZIP_DST)
        #     os.chdir(cwd)

        # TODO: figure out how to download the model from S3,
        # instead of this hack
        extract_tar_bytes(
            open("%s_%s.tar" % (DST_MODEL_NAME, METAFLOW_RUN_ID), "rb").read(), path="."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_TOKENIZER_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(CHECKPOINT_MODEL_PATH).to(
            "cuda"
        )
        self.task = "text-generation"
        self.max_length = 200

    def get_prompt(self, user_input: str, context: str):
        return format_prompt(
            {
                "instruction": user_input,
                "context": context,
                "response": "",
                # start the response the LLM should continue with the user input
                # these are the supervised learning labels in the dolly15k instruction tuning format
            }
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Decode the Byte Tensor into Text
            inputs = pb_utils.get_input_tensor_by_name(request, "prompt")
            inputs = inputs.as_numpy()

            context = []  # TODO: retrieve this from a RAG pipeline!

            prompts = [self.get_prompt(i[0].decode(), context) for i in inputs]
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                "cuda"
            )

            output_sequences = self.model.generate(
                **inputs,
                do_sample=True,
                max_length=self.max_length,
                temperature=0.01,
                top_p=1,
                top_k=20,
                repetition_penalty=1.1,
            )

            output = self.tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True
            )

            # Encode text as byte tensor to send in response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_text",
                        np.array([[o.encode() for o in output]]),
                    )
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self, args):
        self.generator = None
