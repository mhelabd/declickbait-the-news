from typing import final
from params import *

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import sys
from tqdm import tqdm



class ClickbaitDataset(Dataset):
	# json structure:
	# {x: {Title: "sdbs", Body: "sgsdg"},
	#  y: 3}
	def __init__(self, json_path):
		self.df = pd.read_json(json_path)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		row = self.df.iloc[index]
		return row[DATA_TITLE], row[DATA_BODY], row[DATA_SCORE]

class TokenizedClickbaitDataset(Dataset):
	def __init__(self, json_path, saved_dataset_path=None):
		self.df = pd.read_json(json_path)
		self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
		special_tokens = {'pad_token':PAD_TOKEN,'sep_token':SEP_TOKEN}
		#[SEP]
		self.tokenizer.add_special_tokens(special_tokens)
		
		if saved_dataset_path:
			loaded = np.load(saved_dataset_path, allow_pickle=True)
			self.sequences = loaded[0]
			self.title_masks = loaded[1]
			self.scores = loaded[2]
			print("Finished loading dataset")
		else:
			self.sequences = []
			self.title_masks = []
			for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
				sequence, title_mask = self.pad_and_encode(row[DATA_BODY], row[DATA_TITLE])
				self.sequences.append(sequence)
				self.title_masks.append(title_mask)
			print("Finished encoding dataset")
			self.scores = np.array(self.df[DATA_SCORE])
			to_save = np.array([self.sequences, self.title_masks, self.scores], dtype=object)
			np.save(TOKENIZED_DATASET_PATH, to_save)

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):
		return self.sequences[index], self.title_masks[index], self.scores[index]

	def pad_and_encode(self, body, title, block_size=MAX_SEQUENCE_LENGTH):
		encoded_body = self.tokenizer.encode(body, return_tensors="pt").squeeze(dim=0)
		encoded_title = self.tokenizer.encode(title, return_tensors="pt").squeeze(dim=0)
		encoded_sep_token =  self.tokenizer.encode(self.tokenizer.sep_token, return_tensors="pt").squeeze(dim=0)

		final_sequence = self.tokenizer.encode(self.tokenizer.pad_token)*block_size

		encoded_title = encoded_title[:int(block_size/8)]
		
		body_length = block_size - encoded_title.shape[0] - 1 #-1 because we need a sep_token between body and title
		encoded_body = encoded_body[:body_length]
		content = torch.cat([encoded_body, encoded_sep_token, encoded_title])
		final_sequence[:len(content)] = content
		final_sequence = torch.tensor(final_sequence)

		title_mask = torch.zeros_like(final_sequence)
		title_mask[body_length:] = 1
		return final_sequence, title_mask



# for testing
if __name__ == "__main__":
	dataset_type = sys.argv[1]
	dataset = None
	if dataset_type == 'tokenized':
		dataset = TokenizedClickbaitDataset(TRAIN_PATH, saved_dataset_path=TOKENIZED_DATASET_PATH)
		# dataset = TokenizedClickbaitDataset(TEST_PATH)
	else:
		dataset = ClickbaitDataset(TRAIN_PATH)

	print(f'Dataset length: {len(dataset)}')
	print('First entry:')
	print(dataset[0])

