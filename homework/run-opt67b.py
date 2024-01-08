from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
# from datasets import load_dataset

model_id = "facebook/opt-6.7b"


def defaultModel():
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    default_model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float16).cuda()
    return default_model, tokenizer

text = 'a man should be'

## Try generate
def generateTest(model, tokenizer, input):
    
    inputs = tokenizer(input, return_tensors="pt").to(0)
    out = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

## Try quantize
def quantize():
    global model_id

    # ds = load_dataset("allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
    # trainSample = ds.shuffle(seed=42).select(range(1000))
    # trainList = list(trainSample['text'])

    quantization_config = GPTQConfig(
        bits=4, # 量化精度
        group_size=128,
        desc_act=False,
        dataset=['superman is a good man']
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    quant_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map='auto')
    quant_model.save_pretrained('./models/quantized_model')

    return quant_model, tokenizer


if __name__ == "__main__":  
    # generateTest()
    quant_model, tokenizer = quantize()
    test_text =''' 
I'm superman'''
    generateTest(quant_model, tokenizer, test_text)