from collections import defaultdict
import torch
from scipy import spatial
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import spacy
import argparse

from params import *
from dataset.dataset import TokenizedClickbaitDataset
from nltk.translate.bleu_score import sentence_bleu

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pprint
import json
from rouge import Rouge


class Tester():
    def __init__(
            self,
            model,
            tokenizer,
            dataset,
            nlp,
            device='cpu', 
            save_metrics_path=METRICS_PATH_DEV, 
            save_outputs_path=OUTPUT_PATH_DEV):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.nlp = nlp
        self.device = device
        self.save_metrics_path = save_metrics_path
        self.save_outputs_path = save_outputs_path

    def decode(self, encoded):
        return self.tokenizer.decode(encoded)

    def cosine_similarity_spacey(self, real, generated):
        s1 = nlp(real)
        s2 = nlp(generated)
        return s1.similarity(s2)

    def cosine_similarity_bert(self, real, generated):
        sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
        s1 = sbert_model.encode(real)
        s2 = sbert_model.encode(generated)
        return (1 - spatial.distance.cosine(s1, s2))

    def rouge_score1(self, body, title):
        rouge = Rouge()
        rougeDict = rouge.get_scores(body, title)
        return (rougeDict[0]["rouge-1"]["f"], rougeDict[0]["rouge-2"]["f"])

    def get_metric(self, metric, values, verbose=True):
        if verbose:
            print("KEY:     ", metric)
            print("Min:  ", np.min(values))
            print("Max:  ", np.max(values))
            print("Median:  ", np.median(values))
            print("Mean:    ", np.mean(values))
            print("Std Dev: ", np.std(values))
        return {
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values),
            "mean": np.mean(values),
            "std": np.std(values)
        }

    def test(self):
        self.model.to(self.device)
        test_outputs = defaultdict(list)
        for sequence, title_mask, score in tqdm(self.dataset):
            sequence = sequence.type(torch.LongTensor).to(self.device)
            title_mask = title_mask.type(torch.LongTensor).to(self.device)
            score = score.to(self.device)
            shifted_title_mask = torch.roll(title_mask, shifts=1)
            shifted_title_mask[0] = 0
            body = torch.masked_select(
                sequence, shifted_title_mask == 0).unsqueeze(dim=0).to(self.device)
            real_title = torch.masked_select(sequence, shifted_title_mask == 1)
            real_title = real_title[real_title != self.tokenizer.encode(
                PAD_TOKEN)[0]]  # remove the pad tokens
            gen_title = self.model.generate(body).to(self.device)
            gen_title = gen_title[0][:15]
            real_title_text = self.decode(real_title)
            gen_title_text = self.decode(gen_title)
            body_text = self.decode(body.squeeze(dim=0))

            test_outputs['score'].append(score.item())
            test_outputs['real_title'].append(real_title_text)
            test_outputs['gen_title'].append(gen_title_text)
            test_outputs['body_text'].append(body_text)
            test_outputs['bleu_score'].append(sentence_bleu(
                [real_title_text.split()], gen_title_text.split(), weights=(1,)))
            test_outputs['cosine_similarity_spacey'].append(
                self.cosine_similarity_spacey(real_title_text, gen_title_text))
            test_outputs['cosine_similarity_bert_embed'].append(
                self.cosine_similarity_bert(real_title_text, gen_title_text))
            rouge_1, rouge_2 = self.rouge_score1(body_text, gen_title_text)
            test_outputs['rouge1_f1_score_gen_title'].append(rouge_1)
            test_outputs['rouge2_f1_score_gen_title'].append(rouge_2)
            rouge_1, rouge_2 = self.rouge_score1(body_text, real_title_text)
            test_outputs['rouge1_f1_score_real_title'].append(rouge_1)
            test_outputs['rouge2_f1_score_real_title'].append(rouge_2)
        metric_outputs = {}
        for metric, values in test_outputs.items():
            if type(values[0]) != str:
                metric_outputs[metric +
                    '_metrics'] = self.get_metric(metric, values, verbose=False)
        test_outputs.update(metric_outputs)
        with open(self.save_metrics_path, 'w+') as fp:
            json.dump(metric_outputs, fp)
        with open(self.save_outputs_path, 'w+') as fp:
            json.dump(test_outputs, fp)
        return test_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
          "-d", "--dataset", help="Dataset from: {[tr]ain, [d]ev, [te]st}", default="dev")
    parser.add_argument("-sc", "--wanted_scores",
                           help="Wanted Scores", nargs="+", default=None)
    parser.add_argument("--load", help="load model", action="store_true")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device')

    args = parser.parse_args()
    print(args)

    if args.dataset[0:2].lower() == "tr":
        args.dataset = "train"
        PATH = TRAIN_PATH
        METRIC_PATH = METRICS_PATH_TRAIN
        OUTPUT_PATH = OUTPUT_PATH_TRAIN
        TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_TRAIN
    elif args.dataset[0].lower() == "d":
        args.dataset = "dev"
        PATH = DEV_PATH
        METRIC_PATH = METRICS_PATH_DEV
        OUTPUT_PATH = OUTPUT_PATH_DEV
        TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_DEV
    elif args.dataset[0:2].lower() == "te":
        args.dataset = "test"
        PATH = TEST_PATH
        METRIC_PATH = METRICS_PATH_TEST
        OUTPUT_PATH = OUTPUT_PATH_TEST
        TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_TEST

    if args.wanted_scores is not None:
        METRIC_PATH = METRIC_PATH[:METRIC_PATH.find('.json')] + "scores" + ','.join(args.wanted_scores) + '.json'
        OUTPUT_PATH = OUTPUT_PATH[:OUTPUT_PATH.find('.json')] + "scores" + ','.join(args.wanted_scores) + '.json'

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    if args.load:
        MODEL_PATH = max([f for f in os.listdir(MODEL_SAVE_DIR)]) # Gets latest model
        weights = torch.load(MODEL_SAVE_DIR + MODEL_PATH)
        model.load_state_dict(weights)

    dataset = TokenizedClickbaitDataset(
        PATH,
        load_dataset_path=TOKENIZED_DATASET_PATH,
        wanted_scores=None if args.wanted_scores == None else [int(i) for i in args.wanted_scores]
    )

    model.eval()

    nlp = spacy.load('en_core_web_md')
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        # dev_dataset = TokenizedClickbaitDataset(DEV_PATH, saved_dataset_path=TOKENIZED_DATASET_PATH_DEV)
        tester = Tester(
            model, 
            tokenizer, 
            dataset, 
            nlp, 
            device=device, 
            save_metrics_path=METRIC_PATH,
            save_outputs_path=OUTPUT_PATH
        )
        tester.test()