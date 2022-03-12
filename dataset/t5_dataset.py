import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoTokenizer, BertTokenizer
from params import *


import pandas as pd
import numpy as np
import sys; sys.path.append('.')
from tqdm import tqdm
import argparse


class TokenizedT5Dataset(Dataset):
	def __init__(
		self,
			json_path=None,
			load_dataset_path=None,
			save_dataset_path=None,
			wanted_scores=None,
			tokenizer=None
	):
		self.tokenizer = tokenizer

		if wanted_scores != None:
			assert load_dataset_path != None, "Wanted score only on saved datasets"

		if load_dataset_path:
			loaded = torch.load(load_dataset_path)
			self.sequences = loaded['sequences']
			self.sequence_masks = loaded['sequence_masks']
			self.summaries = loaded['summaries']
			self.summary_masks = loaded['summary_masks']
			self.titles = loaded['titles']
			self.title_masks = loaded['title_masks']
			self.scores = loaded['scores']

			if wanted_scores != None:
				mask = torch.abs(wanted_scores[0] - loaded['scores']) < 0.1
				for wanted_score in wanted_scores:
					curr_mask = torch.abs(
						wanted_score - loaded['scores']) < 0.1
					mask = curr_mask + mask
				self.sequences = self.sequences[mask.numpy()]
				self.sequence_masks = self.sequence_masks[mask.numpy()]
				self.summaries = self.summaries[mask.numpy()]
				self.summary_masks = self.summary_masks[mask.numpy()]
				self.titles = self.titles[mask.numpy()]
				self.title_masks = self.title_masks[mask.numpy()]
				self.scores = self.scores[mask.numpy()]

				assert self.scores.shape[0] != 0, "pick a number that works"
				print("Finished loading dataset")
		else:
			self.df = pd.read_json(json_path)

			sequences, sequence_masks, summaries, summary_masks, titles, title_masks = [
			], [], [], [], [], []
			for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
				sequence, sequence_mask = self.encode_and_mask(
					row[DATA_BODY])
				title, title_mask = self.encode_and_mask(
					row[DATA_TITLE], is_title=True)
				summary, summary_mask = self.encode_and_mask(
					row[DATA_SUMMARY], is_title=True)
				
				sequences.append(sequence)
				sequence_masks.append(sequence_mask)
				summaries.append(summary)
				summary_masks.append(summary_mask)
				titles.append(title)
				title_masks.append(title_mask)
			print("Finished encoding dataset")
			self.sequences = torch.stack(sequences)
			self.sequence_masks = torch.stack(sequence_masks)
			self.summaries = torch.stack(summaries)
			self.summary_masks = torch.stack(summary_masks)
			self.titles = torch.stack(titles)
			self.title_masks = torch.stack(title_masks)
			self.scores = torch.from_numpy(self.df[DATA_SCORE].to_numpy())
			to_save = {'sequences': self.sequences, 'sequence_masks': self.sequence_masks,
					   'summaries': self.summaries, 'summary_masks': self.summary_masks,
					   'titles': self.titles, 'title_masks': self.title_masks,
					   'scores': self.scores}
			torch.save(to_save, save_dataset_path)

	def __len__(self):
		return self.sequences.shape[0]

	def __getitem__(self, index):
		return self.sequences[index], self.sequence_masks[index], self.summaries[index], \
			self.summary_masks[index], self.titles[index], self.title_masks[index], self.scores[index]

	def encode_and_mask(self, text, is_title=False):
		if not is_title:
			text = "summarize: " + text
			batch_encoding = self.tokenizer.encode_plus(
				text, padding='max_length', truncation=True, return_tensors='pt')
		else:
			batch_encoding = self.tokenizer.encode_plus(
				text + self.tokenizer.eos_token, max_length=50, padding='max_length', truncation=True, return_tensors='pt')

		tokenized, mask = batch_encoding['input_ids'], batch_encoding['attention_mask']

		# tokenized[tokenized == tokenizer.pad_token_id] = -100

		return tokenized, mask


# for testing
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-d", "--dataset", help="Dataset from: {[tr]ain, [d]ev, [te]st}", default="train")
	parser.add_argument("--load", help="load made Dataset",
						action="store_true")
	parser.add_argument(
		"--tokenizer", help="[bert, gpt, t5 (default)]", default="t5")
	parser.add_argument("-sc", "--wanted_scores",
						help="Wanted Scores", nargs="+", default=None)

	args = parser.parse_args()
	print(args)

	# TRAIN TEST OR DEV
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

	# CHOOSE TOKENIZER
	if args.tokenizer.lower() == "gpt":
		tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
		# [SEP]
		special_tokens = {'pad_token': PAD_TOKEN, 'sep_token': SEP_TOKEN}
		tokenizer.add_special_tokens(special_tokens)
		TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH[:-3] + "_gpt.pt"
	elif args.tokenizer.lower() == "bert":
		tokenizer = BertTokenizer.from_pretrained(
			'bert-base-uncased', do_lower_case=True)
		TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH[:-3] + "_bert.pt"

	else:
		tokenizer = T5Tokenizer.from_pretrained('t5-small')
		TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH[:-3] + "_t5.pt"

	if args.load:
		dataset = TokenizedT5Dataset(
			load_dataset_path=TOKENIZED_DATASET_PATH,
			wanted_scores=None if args.wanted_scores == None else [
				int(i) for i in args.wanted_scores],
			tokenizer=tokenizer,
		)
	else:
		dataset = TokenizedT5Dataset(
			json_path=PATH,
			save_dataset_path=TOKENIZED_DATASET_PATH,
			tokenizer=tokenizer,
		)

	print(f'Dataset length: {len(dataset)}')
	print('First entry:')
	print(dataset[0])
