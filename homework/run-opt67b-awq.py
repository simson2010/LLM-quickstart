from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from transformers import AwqConfig, AutoConfig

model_path = "facebook/opt-6.7b"

quant_path = "models/opt-6_7b-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# 加载模型
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 量化模型
model.quantize(tokenizer, quant_config=quant_config)

# 修改配置文件以使其与transformers集成兼容
quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower(),
).to_dict()

# 预训练的transformers模型存储在model属性中，我们需要传递一个字典
model.model.config.quantization_config = quantization_config

# 保存模型权重
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path) 