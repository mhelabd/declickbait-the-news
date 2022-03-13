import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

import sys; sys.path.append('.')

from params import *
#BERTClassifier-2000.pt
import pprint
import json
import argparse
from dataset.dataset import TokenizedClickbaitDataset

# import spacy
# from scipy import spatial
# from nltk.translate.bleu_score import sentence_bleu
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine
# from rouge import Rouge

from dataset.dataset import TokenizedClickbaitDataset

class Tester():
    def __init__(
            self,
            model,
            tokenizer,
            dataset,
            # nlp,
            device='cpu', 
            save_metrics_path=METRICS_PATH_DEV, 
            save_outputs_path=OUTPUT_PATH_DEV):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        # self.nlp = nlp
        self.device = device
        self.save_metrics_path = save_metrics_path
        self.save_outputs_path = save_outputs_path

    def decode(self, encoded):
        return self.tokenizer.decode(encoded)

    # def cosine_similarity_spacey(self, real, generated):
    #     s1 = nlp(real)
    #     s2 = nlp(generated)
    #     return s1.similarity(s2)

    # def cosine_similarity_bert(self, real, generated):
    #     sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    #     s1 = sbert_model.encode(real)
    #     s2 = sbert_model.encode(generated)
    #     return (1 - spatial.distance.cosine(s1, s2))

    # def rouge_score1(self, body, title):
    #     rouge = Rouge()
    #     rougeDict = rouge.get_scores(body, title)
    #     return (rougeDict[0]["rouge-1"]["f"], rougeDict[0]["rouge-2"]["f"])

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
        test_outputs = pd.DataFrame()
        fn = 0
        fp = 0
        pp = 0
        nn = 0
        acc = 0
        for i, data in enumerate(tqdm(self.dataset)):
            sequence, _, scores = data   
            sequence = sequence.type(torch.LongTensor)   
            attention_mask = sequence != 102.0 # Hacky way of getting padding attention mask
            attention_mask = attention_mask.to(self.device)
            scores = scores > 0.5
            labels = scores.to(torch.int64)
            labels = F.one_hot(labels, num_classes=2)
            sequence = sequence.to(self.device)
            labels = labels.to(self.device)
            # real_title = torch.masked_select(sequence, shifted_title_mask == 1)
            # real_title = real_title[real_title != self.tokenizer.encode(
            #     PAD_TOKEN)[0]]  # remove the pad tokens
            # gen_title = self.model.generate(body).to(self.device)
            # gen_title = gen_title[0][:15]
            # real_title_text = self.decode(real_title)
            # gen_title_text = self.decode(gen_title)
            
            article_text = self.decode(sequence.squeeze(dim=0))
            output = model(sequence.unsqueeze(0), attention_mask.unsqueeze(0))
            prob = output.logits.softmax(dim=-1)[0][1]

            pred = prob >= 0.5
            acc += int(pred == scores)

            if i % 100 == 1:
                print("Accuracy")
                print(acc)
                print(acc/(i+1))

            if scores.item():
                if pred:
                    pp += 1 
                else:
                    fn += 1
            else:
                if pred:
                    fp += 1
                else:
                    nn += 1
        
            

            # rouge_1_gen_title, rouge_2_gen_title = self.rouge_score1(body_text, gen_title_text)
            # rouge_1_real_title, rouge_2_real_title = self.rouge_score1(body_text, real_title_text)

            # temp = pd.DataFrame(
            #     {
            #         'score': score.item(),
            #         'real_title': real_title_text,
            #         'gen_title':gen_title_text,
            #         'body_text':body_text,
            #         'bleu_score':sentence_bleu([real_title_text.split()], gen_title_text.split(), weights=(1,)),
            #         'cosine_similarity_spacey': self.cosine_similarity_spacey(real_title_text, gen_title_text),
            #         'cosine_similarity_bert_embed': self.cosine_similarity_bert(real_title_text, gen_title_text),
            #         'rouge1_f1_score_gen_title': rouge_1_gen_title,
            #         'rouge2_f1_score_gen_title': rouge_2_gen_title,
            #         'rouge1_f1_score_real_title': rouge_1_real_title,
            #         'rouge2_f1_score_real_title': rouge_2_real_title,
            #     }, 
            #     index=[i],
            # )
            # test_outputs = test_outputs.append(temp)
            # print(test_outputs.memory_usage(deep=True))

        metric_outputs = {
            "True Postive": pp / (pp + fn),
            "True Negative": nn / (nn + fp),
            "False Negative": fn / (fn + pp),
            "False Positive": fp / (fp + nn),
            "Overall Accuracy": (nn + pp) / (fp + fn + nn + pp)
        }

        print(metric_outputs)

        # for metric in test_outputs.keys():
        #     if type(test_outputs[metric][0]) != str:
        #         metric_outputs[metric + '_metrics'] = self.get_metric(metric, test_outputs[metric], verbose=False)
        test_outputs.update(metric_outputs)
        with open(self.save_metrics_path, 'w+') as fp:
            json.dump(metric_outputs, fp)
        test_outputs.to_json(self.save_outputs_path, index = 'true')
        return test_outputs


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
          "-d", "--dataset", help="Dataset from: {[tr]ain, [d]ev, [te]st}", default="dev")
    # parser.add_argument("-sc", "--wanted_scores",
    #                        help="Wanted Scores", nargs="+", default=None)
    # parser.add_argument("--load", help="load model", action="store_true")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using {device} device')

    args = parser.parse_args()
    print(args)

    if args.dataset[0:2].lower() == "tr":
        args.dataset = "train"
        PATH = TRAIN_PATH_T5
        METRIC_PATH = METRICS_PATH_TRAIN
        OUTPUT_PATH = OUTPUT_PATH_TRAIN
        TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_TRAIN_T5
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

    # if args.wanted_scores is not None:
    #     METRIC_PATH = METRIC_PATH[:METRIC_PATH.find('.json')] + "scores" + ','.join(args.wanted_scores) + '.json'
    #     OUTPUT_PATH = OUTPUT_PATH[:OUTPUT_PATH.find('.json')] + "scores" + ','.join(args.wanted_scores) + '.json'

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    weights = torch.load(MODEL_SAVE_DIR + "BERTClassifier-2000.pt")
    #model.load_state_dict(weights)

    # if args.load:
    #     MODEL_PATH = max([f for f in os.listdir(MODEL_SAVE_DIR)]) # Gets latest model
    #     weights = torch.load(MODEL_SAVE_DIR + MODEL_PATH)
    #     model.load_state_dict(weights)

    dataset = TokenizedClickbaitDataset(
        PATH,
        load_dataset_path=TOKENIZED_DATASET_PATH,
        tokenizer= "bert"
        # wanted_scores=None if args.wanted_scores == None else [int(i) for i in args.wanted_scores]
    )

    model.eval()

    # nlp = spacy.load('en_core_web_md')
    # sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    with torch.no_grad():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # dev_dataset = TokenizedClickbaitDataset(DEV_PATH, saved_dataset_path=TOKENIZED_DATASET_PATH_DEV)
        tester = Tester(
            model, 
            tokenizer, 
            dataset, 
            # nlp, 
            device=device, 
            save_metrics_path=METRIC_PATH,
            save_outputs_path=OUTPUT_PATH
        )
        tester.test()