from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from datasets import load_dataset

# dataset = load_dataset("json", data_files="dataset.json", field="data")

# dataset = dataset.map(lambda x: {
#         'input': f"<s>[INST] {x['question'].strip()} [/INST]",
#         'output': x['answer']
#         }, remove_columns=['question', 'answer'])

# print(dataset.data)
# print(dataset)


model_id = "codellama/CodeLlama-7b-Instruct-hf"
# model_id = "Transform72/PandasSolver"
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
#    load_in_8bit=True,
   bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

# user = "Pandas dataframe 'df' is already initialized. Select 5 maximal values from column 'GDP'."
# # prompt = f"<s>[INST] {user.strip()} [/INST]"
# # inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

user = """Pandas dataframe 'df' is already initialized and filled with data.
            1.Select 5 maximal values from column 'GDP' and store them to 'tmp' variable.
            2.Barplot these values.
            Here is the start of the code:
            '''
            import pandas as pd
            import matplotlib.pyplot as plt

            # 'df' has already been initialized and filled

            """
# prompt = f"<s>[INST] {user.strip()} [/INST]"
# inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

system = "Provide answers in Python"
prompt = f"<s>[INST] <<SYS>>\\n{system}\\n<</SYS>>\\n\\n{user}[/INST]"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

output = model.generate(
    inputs["input_ids"],
    max_new_tokens=200,
    do_sample=True,
    top_p=0.9,
    temperature=0.1,
)

output = output[0].to("cpu")
print(tokenizer.decode(output))
