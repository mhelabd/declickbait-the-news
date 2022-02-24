from params import *

import torch
from torch.utils.data import Dataset

# from transformers import AutoTokenizer

import pandas as pd

from tqdm import tqdm


class ClickbaitDataset(Dataset):
	# json structure:
	# {x: {Title: "sdbs", Body: "sgsdg"},
	#  y: 3}
	def __init__(self, json_path):
		self.df = pd.read_json(json_path)
		# self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
		# self.bodies = []
		# self.titles = []
		# for _, row in tqdm(self.df.iterrows()):
		# 	self.bodies.append(self.tokenizer.encode(
		# 		row[DATA_BODY][:MAX_SEQUENCE_LENGTH], return_tensors="pt"))
		# 	self.titles.append(self.tokenizer.encode(
		# 		row[DATA_TITLE][:MAX_SEQUENCE_LENGTH], return_tensors="pt"))
		# print("Finished encoding dataset")

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		row = self.df.iloc[index]
		return row[DATA_TITLE], row[DATA_BODY], row[DATA_SCORE]
		# return self.titles[index], self.bodies[index], row[DATA_SCORE]


# for testing
if __name__ == "__main__":
	cbdataset = ClickbaitDataset(TRAIN_PATH)

	print(f'Dataset length: {len(cbdataset)}')
	print('First entry:')
	title, body, score = cbdataset[200]
	print(f'Title: {title}')
	print(f'Body: {body}')
	print(f'Score: {score}')
