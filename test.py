from tabnanny import check
import pandas as pd
import torch
from scipy import spatial
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import spacy
import argparse
from dataset.t5_dataset import TokenizedT5Dataset
from models.headline_generator_t5 import HeadlineGenerator

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
        test_outputs = pd.DataFrame()
        for i, data in enumerate(tqdm(self.dataset)):
            sequence, sequence_mask, summary, _, title, _, score = data
            sequence = sequence.to(self.device)
            sequence_mask = sequence_mask.to(self.device)
            summary = summary.to(self.device).squeeze(dim=0)
            title = title.to(self.device).squeeze(dim=0)

            gen_title = self.model.generate(
                input_ids=sequence,
                num_beams=7,
                attention_mask=sequence_mask,
                do_sample=True,
                repetition_penalty=1.2
		    )
            gen_title = gen_title.squeeze(dim=0).to(self.device)
            real_title_text = self.tokenizer.decode(title, skip_special_tokens=True)
            gen_title_text = self.tokenizer.decode(gen_title, skip_special_tokens=True)
            print("title: ", real_title_text)
            print("title: ", gen_title_text)

            body_text = self.tokenizer.decode(sequence.squeeze(dim=0), skip_special_tokens=True)
            if body_text == "" or gen_title_text == "" or real_title_text == "":
                continue
            try:
                rouge_1_gen_title, rouge_2_gen_title = self.rouge_score1(body_text, gen_title_text)
                rouge_1_real_title, rouge_2_real_title = self.rouge_score1(body_text, real_title_text)
            except:
                print("ERROR: NOT WORKING")
                print("body_text:", body_text)
                print("real_title_text:", real_title_text)
                print("gen_title_text:", gen_title_text)
                continue

            temp = pd.DataFrame(
                {
                    'score': score.item(),
                    'real_title': real_title_text,
                    'gen_title':gen_title_text,
                    'body_text':body_text,
                    'bleu_score':sentence_bleu([real_title_text.split()], gen_title_text.split(), weights=(1,)),
                    'cosine_similarity_spacey': self.cosine_similarity_spacey(real_title_text, gen_title_text),
                    'cosine_similarity_bert_embed': self.cosine_similarity_bert(real_title_text, gen_title_text),
                    'rouge1_f1_score_gen_title': rouge_1_gen_title,
                    'rouge2_f1_score_gen_title': rouge_2_gen_title,
                    'rouge1_f1_score_real_title': rouge_1_real_title,
                    'rouge2_f1_score_real_title': rouge_2_real_title,
                }, 
                index=[i],
            )
            test_outputs = test_outputs.append(temp)
            print(test_outputs.memory_usage(deep=True))

            if i > 200: break
        metric_outputs = {}
        for metric in test_outputs.keys():
            if type(test_outputs[metric][0]) != str:
                metric_outputs[metric + '_metrics'] = self.get_metric(metric, test_outputs[metric], verbose=False)
        test_outputs.update(metric_outputs)
        with open(self.save_metrics_path, 'w+') as fp:
            json.dump(metric_outputs, fp)
        test_outputs.to_json(self.save_outputs_path, index = 'true')
        print("Saved to", self.save_metrics_path)
        return test_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", help="T5 or GPT Default: T5", default="T5")
    parser.add_argument("-c", "--use_class_loss", help="Use finetinued BERT clickbait classifier as part of loss. Default: false", action="store_true")
    parser.add_argument("-s", "--use_summ_loss", help="Use BART summarized entries as part of loss. Default: false", action="store_true")
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
        TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_TRAIN_T5 if args.model_type else TOKENIZED_DATASET_PATH_TRAIN
    elif args.dataset[0].lower() == "d":
        args.dataset = "dev"
        PATH = DEV_PATH
        METRIC_PATH = METRICS_PATH_DEV
        OUTPUT_PATH = OUTPUT_PATH_DEV
        TOKENIZED_DATASET_PATH = TOKENIZED_DATASET_PATH_DEV_T5 if args.model_type else TOKENIZED_DATASET_PATH_DEV
    elif args.dataset[0:2].lower() == "te":
        args.dataset = "test"
        PATH = TEST_PATH
        METRIC_PATH = METRICS_PATH_TEST
        OUTPUT_PATH = OUTPUT_PATH_TEST
        TOKENIZED_DATASET_PATH =  TOKENIZED_DATASET_PATH_TEST_T5 if args.model_type else TOKENIZED_DATASET_PATH_TEST

    if args.model_type == "T5":
        METRIC_PATH = METRIC_PATH[:METRIC_PATH.find('.json')] + "model_" + args.model_type + "class_loss_" + str(args.use_class_loss) + "-summ_loss_" + str(args.use_summ_loss) + '.json'
        OUTPUT_PATH = OUTPUT_PATH[:OUTPUT_PATH.find('.json')] + "model_" + args.model_type + "class_loss_" + str(args.use_class_loss) + "-summ_loss_" + str(args.use_summ_loss) + '.json'

    if args.wanted_scores is not None:
        METRIC_PATH = METRIC_PATH[:METRIC_PATH.find('.json')] + "scores" + ','.join(args.wanted_scores) + '.json'
        OUTPUT_PATH = OUTPUT_PATH[:OUTPUT_PATH.find('.json')] + "scores" + ','.join(args.wanted_scores) + '.json'

    if args.model_type == "T5":
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        if args.load:
            MODEL_PATH = max([f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith(f'T5_HEADLINE-class_loss_{args.use_class_loss}-summ_loss_{args.use_summ_loss}') and not f.endswith('.part')]) # Gets latest model
            print(MODEL_PATH)
            checkpoint = torch.load(MODEL_SAVE_DIR+MODEL_PATH)['state_dict']
            checkpoint = {k[6:]: v for k, v in checkpoint.items() if k.startswith('model') }
            model.load_state_dict(checkpoint)
    else:      
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        if args.load:
            MODEL_PATH = max([f for f in os.listdir(MODEL_SAVE_DIR) if f.startswith('BERTClassifier')]) # Gets latest model
            weights = torch.load(MODEL_SAVE_DIR + MODEL_PATH)
            model.load_state_dict(weights)
    # TWO DATA FILES
    if args.model_type == "T5":
        dataset = TokenizedT5Dataset(
            PATH,
            load_dataset_path=TOKENIZED_DATASET_PATH,
            wanted_scores=None if args.wanted_scores == None else [int(i) for i in args.wanted_scores],
            tokenizer=tokenizer)
    else:
        dataset = TokenizedClickbaitDataset(
            PATH,
            load_dataset_path=TOKENIZED_DATASET_PATH,
            wanted_scores=None if args.wanted_scores == None else [int(i) for i in args.wanted_scores]
        )

    model.eval()

    nlp = spacy.load('en_core_web_md')
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    with torch.no_grad():
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