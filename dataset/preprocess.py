from params import *
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import pipeline

import sys
import os
from os.path import exists

# from pandarallel import pandarallel
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

tqdm.pandas()
# pandarallel.initialize()

article_filenames = [
	'./data/clickbait17-train-170331/instances.jsonl',
	'./data/clickbait17-test-170720/instances.jsonl',
	'./data/clickbait17-validation-170630/instances.jsonl',
	]
cbscore_filenames = [
	'./data/clickbait17-train-170331/truth.jsonl',
	'./data/clickbait17-test-170720/truth.jsonl',
	'./data/clickbait17-validation-170630/truth.jsonl',
]

summarizer = pipeline("summarization", device=0)

CSV_FILENAME = "./data/data_temp.csv"

def merge_jsons(filenames, json_filename='data.json'):
	data = pd.DataFrame()
	for f in filenames:
		print(f)
		data_f = pd.read_json(f, orient='records', lines=True)
		data = data.append(data_f, ignore_index=True)
	data.to_json(json_filename)

def make_dataset(article_json, cbscore_json, json_filename='data.json'):
	# file_exists = exists(CSV_FILENAME)
	
	# if file_exists:
	# 	os.remove(CSV_FILENAME)


	article_pd = pd.read_json(article_json)[['targetTitle', 'targetParagraphs']]
	article_pd["targetParagraphs"] = article_pd["targetParagraphs"].str.join(" ")
	# TODO: Consider other scores (truthMedian, truthMean, truthClass)
	cbscore_pd = pd.read_json(cbscore_json)[['truthMode']]

	print("Creating summaries")
	article_pd = article_pd.join(cbscore_pd)
	article_pd = article_pd[article_pd["targetParagraphs"] != ""]
	article_pd.reset_index(drop=True, inplace=True)

	article_summaries = []
	article_pd = article_pd[35536:]
	article_pd.reset_index(drop=True, inplace=True)
	for i, row in tqdm(article_pd.iterrows(), total=len(article_pd)):

		article_summaries.append(create_summary(row['targetParagraphs']))
		
		batchSize = 20

		if ((i + 1) % batchSize) == 0:
			article_summaries_df = pd.DataFrame(article_summaries)
			article_summaries_df.rename(columns={ article_summaries_df.columns[0]: "summary" }, inplace = True)

			big_boy_df = pd.concat([article_pd[i + 1 - batchSize:i+1].reset_index(), article_summaries_df], axis=1)
			# big_boy_df = article_summaries_df.join(article_pd[i + 1 - batchSize:i+1])
			big_boy_df.rename(columns={
				"targetTitle": DATA_TITLE,
				"targetParagraphs": DATA_BODY, 
				'truthMode': DATA_SCORE, 
				'summary': DATA_SUMMARY,
			}, inplace=True)

			# header = i < batchSize
			header=False
			big_boy_df.to_csv(CSV_FILENAME,
				index=False,
				header=header,
				mode='a',#append data to csv file
				chunksize=101)#size of data to append for each loop
			print("created", CSV_FILENAME)
			article_summaries = []
		

def convert_csv_to_json(json_filename):
	df = pd.read_csv(CSV_FILENAME)
	df.to_json(json_filename)
	print("created", json_filename)


def create_summary(text):
	return summarizer(text[:1000], max_length=20, min_length=5,
							do_sample=False)[0]['summary_text']


def divide_dataset(
	dataset_json,
	dev_size=0.1,
	test_size=0.1,
	train_filename=TRAIN_PATH_T5,
	dev_filename=DEV_PATH,
	test_filename=TEST_PATH,
):
	data_df=pd.read_json(dataset_json)
	train, test=train_test_split(data_df, test_size=test_size)
	train, dev=train_test_split(train, test_size=dev_size/(1-test_size))
	train.to_json(train_filename)
	dev.to_json(dev_filename)
	test.to_json(test_filename)


if __name__ == "__main__":
	article_json='./data/articles.json'
	cbscore_json='./data/cbscore.json'
	dataset_json='./data/data_temp.json'
	merge_jsons(article_filenames, json_filename=article_json)
	merge_jsons(cbscore_filenames, json_filename=cbscore_json)
	make_dataset(article_json, cbscore_json, json_filename=dataset_json)
	convert_csv_to_json(dataset_json)
	divide_dataset(dataset_json)
