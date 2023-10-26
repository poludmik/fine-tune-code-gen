from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import transformers
import torch

model_id = "Transform72/PandasSolver"

quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16,
)

print('hehe')

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

print('haha')
# pipeline = transformers.pipeline(model=model, quantization_config=quantization_config)

tokenizer = AutoTokenizer.from_pretrained(model_id)

print('huhu')

sample_prompt = """
PROBLEM:
You have been given a dataset that contains information about students, including their names, ages, grades, and favorite subjects. You need to perform the following tasks using Pandas:

1. Load the dataset into a Pandas DataFrame named "students_df". The dataset is provided as a CSV file named "students.csv".

2. Find the maximum and minimum ages of the students.

3. Create a pivot table that shows the average grades of students for each favorite subject. The pivot table should have the subjects as columns and the average grades as values.

4. Calculate the sum of ages for students who have the same favorite subject.
"""

print('hehe')

inputs = tokenizer(sample_prompt, return_tensors="pt", add_special_tokens=True).to("cuda")

output = model.generate(
    inputs["input_ids"],
    max_new_tokens=200,
    do_sample=True,
    top_p=0.9,
    temperature=0.1,
)

output = output[0].to("cpu")
print(tokenizer.decode(output))

# sequences = pipeline(
#     sample_prompt,
#     do_sample=True,
#     temperature=0.2,
#     top_p=0.95,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=512,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")
