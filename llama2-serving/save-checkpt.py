import transformers

src_model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = transformers.AutoTokenizer.from_pretrained(src_model_name)
model = transformers.AutoModelForCausalLM.from_pretrained(src_model_name)
model.save_pretrained("./nous-research-llama2/model")
tokenizer.save_pretrained("./nous-research-llama2/tokenizer")
