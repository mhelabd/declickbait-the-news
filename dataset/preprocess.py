import numpy as np
from params import *
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from transformers import pipeline

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

tqdm.pandas()

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


def merge_jsons(filenames, json_filename='data.json'):
	data = pd.DataFrame()
	for f in filenames:
		print(f)
		data_f = pd.read_json(f, orient='records', lines=True)
		data = data.append(data_f, ignore_index=True)
	data.to_json(json_filename)

# json structure:
    # {Title: "sdbs", Body: "sgsdg", score: 0.25}


def make_dataset(article_json, cbscore_json, json_filename='data.json'):
	article_pd = pd.read_json(article_json)[['targetTitle', 'targetParagraphs']]
	article_pd["targetParagraphs"] = article_pd["targetParagraphs"].str.join(" ")
	# TODO: Consider other scores (truthMedian, truthMean, truthClass)
	cbscore_pd = pd.read_json(cbscore_json)[['truthMode']]

	print("Creating summaries")

	#Mini flag
	mini = False

	#Limited Dataset
	if mini:
		article_pd = article_pd[0:20000]

	for i, row in tqdm(article_pd.iterrows(), total=len(article_pd)):
		try:
			article_pd.at[i, 'summary'] = create_summary(row['targetParagraphs'])
		except Exception as e:
			print("skipping data point")
			article_pd.at[i, 'summary'] = np.NaN

	dataset_pd = pd.concat([article_pd, cbscore_pd], axis=1).dropna()

	dataset_pd.rename(columns={
		"targetTitle": DATA_TITLE,
		"targetParagraphs": DATA_BODY, 
		'truthMode': DATA_SCORE, 
		'summary': DATA_SUMMARY,
	}, inplace=True)

	dataset_pd.to_json(json_filename)


def create_summary(text):
	return summarizer(text[:500], max_length=20, min_length=5,
					    do_sample=False)[0]['summary_text']

def divide_dataset(
	dataset_json,
	dev_size=0.1,
	test_size=0.1,
	train_filename=TRAIN_PATH,
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
	dataset_json='./data/data.json'
	merge_jsons(article_filenames, json_filename=article_json)
	merge_jsons(cbscore_filenames, json_filename=cbscore_json)
	make_dataset(article_json, cbscore_json, json_filename=dataset_json)
	divide_dataset(dataset_json)