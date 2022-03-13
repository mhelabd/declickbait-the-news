from lib2to3.pgen2 import token
from params import *

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer, BertTokenizer

import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
import argparse



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
	def __init__(
		self,
		json_path, 
		load_dataset_path=None, 
		save_dataset_path=None, 
		wanted_scores=None, 
		tokenizer="gpt"):
		self.df = pd.read_json(json_path)
		self.tokenizer_type = tokenizer
		if self.tokenizer_type == "gpt":
			self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
			#[SEP]
			special_tokens = {'pad_token':PAD_TOKEN,'sep_token':SEP_TOKEN}
			self.tokenizer.add_special_tokens(special_tokens)
		else:
			self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

		if wanted_scores != None:	
			assert load_dataset_path != None, "Wanted score only on saved datasets"
		
		if load_dataset_path:
			loaded = torch.load(load_dataset_path)
			self.sequences = loaded['sequences']
			self.title_masks = loaded['title_masks']
			self.scores = loaded['scores']
			if wanted_scores != None:
				mask = torch.abs(wanted_scores[0] - loaded['scores']) < 0.1
				for wanted_score in wanted_scores:
					curr_mask = torch.abs(wanted_score - loaded['scores']) < 0.1
					mask = curr_mask + mask
				self.sequences = self.sequences[mask.numpy()]
				self.title_masks = self.title_masks[mask.numpy()]
				self.scores = self.scores[mask.numpy()]
				assert self.scores.shape[0] != 0, "pick a number that works"
			print("Finished loading dataset")
		else:
			self.sequences = []
			self.title_masks = []
			for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
				sequence, title_mask = self.pad_and_encode(row[DATA_BODY], row[DATA_TITLE])
				self.sequences.append(sequence)
				self.title_masks.append(title_mask)
			print("Finished encoding dataset")
			self.sequences = torch.stack(self.sequences)
			self.title_masks = torch.stack(self.title_masks)
			self.scores = torch.from_numpy(self.df[DATA_SCORE].to_numpy())
			to_save = {'sequences': self.sequences, 'title_masks': self.title_masks, 'scores': self.scores}
			torch.save(to_save, save_dataset_path)

	def __len__(self):
		return self.sequences.shape[0]

	def __getitem__(self, index):
		return self.sequences[index], self.title_masks[index], self.scores[index]

	def pad_and_encode(self, body, title, block_size=MAX_SEQUENCE_LENGTH):
		encoded_body = self.tokenizer.encode(body, return_tensors="pt").squeeze(dim=0)
		encoded_title = self.tokenizer.encode(title, return_tensors="pt").squeeze(dim=0)
		if self.tokenizer_type == "gpt":
			encoded_sep_token =  self.tokenizer.encode(self.tokenizer.sep_token, return_tensors="pt").squeeze(dim=0)
			final_sequence = self.tokenizer.encode(self.tokenizer.pad_token)*block_size
		else:
			encoded_sep_token =  self.tokenizer.encode('[SEP]', return_tensors="pt").squeeze(dim=0)[1].reshape(1,)
			final_sequence = [encoded_sep_token]*block_size

		encoded_title = encoded_title[:int(block_size/8)]
		
		#block = 800
		#title = 10
		#body_length = 790

		body_length = block_size - encoded_title.shape[0] - 1 #-1 because we need a sep_token between body and title
		encoded_body = encoded_body[:body_length]
		content = torch.cat([encoded_body, encoded_sep_token, encoded_title])
		final_sequence[:len(content)] = content
		final_sequence = torch.tensor(final_sequence, dtype=torch.int)

		title_mask = torch.zeros_like(final_sequence, dtype=torch.int)
		body_length = encoded_body.shape[0]
		title_mask[body_length:] = 1
		return final_sequence, title_mask



# for testing
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--dataset", help="Dataset from: {[tr]ain, [d]ev, [te]st}", default="train")
	parser.add_argument("--load", help="load made Dataset", action="store_true")
	parser.add_argument("--bert_tokenizer", help="Use Bert Tokenizer", action="store_true")
	parser.add_argument("-sc", "--wanted_scores", help="Wanted Scores", nargs="+", default=None)

	args = parser.parse_args()
	print(args)
	
	if args.dataset[0:2].lower() == "tr":
		args.dataset = "train"
		PATH = TRAIN_PATH_T5
		TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_TRAIN_T5
	elif args.dataset[0].lower() == "d":
		args.dataset = "dev"
		PATH = DEV_PATH
		TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_DEV
	elif args.dataset[0:2].lower() == "te":
		args.dataset = "test"
		PATH = TEST_PATH
		TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_TEST

	if args.load:
		dataset = TokenizedClickbaitDataset(
			PATH, 
			load_dataset_path=TOKENIZED_DATASET_PATH, 
			wanted_scores=None if args.wanted_scores == None else [int(i) for i in args.wanted_scores],
			tokenizer= "bert" if args.bert_tokenizer else "gpt",
			)
	else:
		dataset = TokenizedClickbaitDataset(
			PATH, 
			save_dataset_path=TOKENIZED_DATASET_PATH,
			tokenizer= "bert" if args.bert_tokenizer else "gpt",
		)
	
	
	print(f'Dataset length: {len(dataset)}')
	print('First entry:')
	print(dataset[0])
