from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from params import *

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

MODEL_PATH = "GPTheadlines-0.pt"
weights = torch.load(MODEL_SAVE_DIR + MODEL_PATH)
model.load_state_dict(weights)
model.eval()


# output = model.generate([[26, 27,28]])

# sequence = "Hello my name is Neville Chamberlain"

# inputs = tokenizer.encode(sequence, return_tensors='pt')
# outputs = model.generate(inputs, max_length=200, do_sample=True)

# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(text)