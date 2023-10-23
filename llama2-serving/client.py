from tritonclient.utils import *
import tritonclient.http as httpclient
import time
import numpy as np


def user_text_to_inputs(
    user_prompt_list: list[str],
):
    text_obj = np.array(input_text, dtype="object")


def user_text_to_inputs(
    input_text=[
        ["Who is Lionel Messi?"],
    ]
):
    # Define input config
    text_obj = np.array(input_text, dtype="object")

    inputs = [
        httpclient.InferInput(
            "prompt", text_obj.shape, np_to_triton_dtype(text_obj.dtype)
        ).set_data_from_numpy(text_obj),
    ]

    # Define output config
    outputs = [
        httpclient.InferRequestedOutput("generated_text"),
    ]
    return inputs, outputs


def time_single_request():
    tm1 = time.perf_counter()
    with httpclient.InferenceServerClient(
        url="localhost:8000", verbose=False, concurrency=32
    ) as client:
        inputs, outputs = user_text_to_inputs()
        response = client.infer(
            "llama2", model_version="2", inputs=inputs, outputs=outputs
        )
        llm_response = (
            response.as_numpy("generated_text")[0][0]
            .decode("utf-8")
            .split("[/INST]")[1]
            .strip()
        )
    tm2 = time.perf_counter()
    print(f"Total time elapsed: {tm2-tm1:0.2f} seconds")


def chat_iter(user_prompt, model_version="1"):
    inputs, outputs = user_text_to_inputs([[user_prompt]])

    with httpclient.InferenceServerClient(
        url="localhost:8000", verbose=False, concurrency=32
    ) as client:
        response = client.infer(
            "llama2", model_version=model_version, inputs=inputs, outputs=outputs
        )

    llm_response = (
        response.as_numpy("generated_text")[0][0]
        .decode("utf-8")
        .split("### RESPONSE:")[1]
        .split("###")[0]
        .strip()
    )
    return llm_response


def batch_inference(
    user_prompts=[
        ["How did the Haitian revolution happen?"],
        ["Who is Abby Wambach?"],
        ["What is the first key decision when starting a new business?"],
    ],
    model_version="1",
    loud=True,
):
    inputs, outputs = user_text_to_inputs(user_prompts)

    with httpclient.InferenceServerClient(
        url="localhost:8000", verbose=False, concurrency=32
    ) as client:
        api_response = client.infer(
            "llama2", model_version=model_version, inputs=inputs, outputs=outputs
        )

    llm_responses = [
        response[0].decode("utf-8").split("### RESPONSE:")[1].split("###")[0].strip()
        for response in api_response.as_numpy("generated_text").T
    ]

    if loud:
        for prompt, response in zip(user_prompts, llm_responses):
            print(f"User Prompt: {prompt[0]}")
            print(f"LLM Response: {response}")
            print()

    return llm_responses


def main():
    time_single_request()


if __name__ == "__main__":
    main()