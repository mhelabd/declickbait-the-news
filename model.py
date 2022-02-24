from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

print(model)

# class BaselineModel(nn.Module):
#     def __init__():
#         super().__init__()

inputs = tokenizer.encode("There's not much room under this mushroom: Mantis prays for the weather to change as it shelters from the rain\xa0", return_tensors="pt")
#outputs = model(**inputs, labels=inputs["input_ids"])
outputs = model.generate(inputs, max_length=200, do_sample=True)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)

# outputs = model.generate(inputs, max_length=200, do_sample=True)

# inputs = tokenizer.encode("Hello, my dog is cute", return_tensors='pt')
# outputs = model.generate(inputs, max_length=200, do_sample=True)

# prediction = torch.argmax(outputs.logits)


# print("Model Outputs:")
# print(prediction)    

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # initialize tokenizer and model from pretrained GPT2 model
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# sequence = "Hello my name is Neville Chamberlain"

# inputs = tokenizer.encode(sequence, return_tensors='pt')
# outputs = model.generate(inputs, max_length=200, do_sample=True)

# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(text)