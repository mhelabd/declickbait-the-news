import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, BertForSequenceClassification, BertTokenizer
from tqdm import tqdm
import wandb
import argparse

import sys; sys.path.append('.')

from dataset.t5_dataset import TokenizedT5Dataset
from dataset.dataset import TokenizedClickbaitDataset
from params import *


class HeadlineGenerator(pl.LightningModule):
	def __init__(
		self,
		dataloader,
		epochs=40,
		lr=.05,
		eps=1e-8,
		betas=(0.7,0.999),
		warmup_steps=0,
		classifier_loss_coef=10,
		freeze_encoder=False,
		use_classifier_for_loss=False,
		use_summarizer_for_loss=False,
		bert_tokenizer=None,
		t5_tokenizer=None,
		output_dir=MODEL_SAVE_DIR,
		output_prefix="T5Generator",
		use_wandb=False
	):
		super(HeadlineGenerator, self).__init__()

		self.output_dir = output_dir
		self.output_prefix = output_prefix
		self.use_classifier_for_loss = use_classifier_for_loss
		self.use_summarizer_for_loss = use_summarizer_for_loss
		if use_classifier_for_loss:
			self.classifier_loss_coef = classifier_loss_coef
			assert bert_tokenizer != None, "Must have bert tokenizer"
			assert t5_tokenizer != None, "Must have t5 tokenizer"
			self.bert_tokenizer = bert_tokenizer
			self.t5_tokenizer = t5_tokenizer

		if use_classifier_for_loss:
			self.classifier = BertForSequenceClassification.from_pretrained(
					"bert-base-uncased")
			self.classifier.load_state_dict(torch.load(
					MODEL_SAVE_DIR + "BERTClassifier-2000.pt"))
			self.freeze_params(self.classifier)
			self.classifier.to(self.device)


		self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
		self.model.train()
		if freeze_encoder:
			self.freeze_params(self.model.get_encoder())

		self.lr = lr
		self.eps = eps
		self.betas = betas
		self.warmup_steps = warmup_steps
		self.num_training_steps = len(dataloader) * epochs

		self.loss = nn.CrossEntropyLoss(ignore_index=-100)  # INDEX TO IGNORE

		self.model.to(self.device)

		self.use_wandb = use_wandb
		if self.use_wandb:
			wandb.init()

	def freeze_params(self, model):
		for par in model.parameters():
			par.requires_grad = False

	def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
		return self.model(
			input_ids,
			attention_mask=attention_mask,
			decoder_input_ids=decoder_input_ids,
			decoder_attention_mask=decoder_attention_mask,
			labels=labels
		)

	def encode_and_mask(self, bodies, titles):
		block_size = 512
		bodies_dict = self.bert_tokenizer.batch_encode_plus(bodies, return_tensors="pt", padding='max_length', truncation=True)
		bodies = bodies_dict.input_ids.to(self.device) # (batch_size, num_tokens = 512)

		batch_size, num_tokens = bodies.shape
		title_dict = self.bert_tokenizer.batch_encode_plus(titles, return_tensors="pt", max_length=64, padding='max_length', truncation=True)
		titles = title_dict.input_ids.to(self.device) # (batch_size, num_tokens = 50)

		encoded_sep_token =  self.bert_tokenizer.encode('[SEP]', return_tensors="pt").squeeze(dim=0)[1].reshape(1,).to(self.device)
		final_sequence = encoded_sep_token* torch.ones_like(bodies) # (batch_size, num_tokens = 512)
		body_lengths = block_size - 1 - torch.argmin(titles, dim=1) # gets first zero 

		body_lengths = body_lengths.view(batch_size, 1).expand(bodies.shape)
		body_length_mask = torch.ones_like(bodies)
		batch_arange = torch.arange(0, num_tokens).to(self.device) * body_length_mask
		body_length_mask[batch_arange > body_lengths] = 0	
		bodies = bodies * body_length_mask # bodies with ensured space for title

		# title starting location
		title_loc = torch.argmin(bodies, dim=1)
		title_loc = title_loc.view(batch_size, 1).expand(bodies.shape)
		title_loc_mask = torch.zeros_like(bodies)
		title_loc_mask[batch_arange == body_lengths] = encoded_sep_token
		for i in range(batch_size):
			try:
				title_loc_mask[i, batch_arange[i] > body_lengths[i]] = titles[i].masked_select(titles[i] != 0)
			except: #if title is empty
				title_loc_mask[i, batch_arange[i] > body_lengths[i]] = 0


		final_sequence = bodies+title_loc_mask
		attention_mask = final_sequence != 0 # Hacky way of getting padding attention mask
		return final_sequence.to(self.device), attention_mask.to(self.device)
	
	def gen_logits(self, bodies, titles): # decode using t5, encode using bert
		# decode using t5
		decoded_bodies = self.t5_tokenizer.batch_decode(bodies)
		decoded_titles = self.t5_tokenizer.batch_decode(titles)


		# encode using bert and make like classifier input; sequence + eos + 
		tokenized_sequences, attenion_mask = self.encode_and_mask(decoded_bodies, decoded_titles)
		return tokenized_sequences, attenion_mask

	def _step(self, batch):
		sequence, sequence_mask, summary, summary_mask, title, title_mask, _ = batch
		sequence = sequence.to(self.device).squeeze(dim=1)
		sequence_mask = sequence_mask.to(self.device).squeeze(dim=1)
		summary = summary.to(self.device).squeeze(dim=1)
		summary_mask = summary_mask.to(self.device).squeeze(dim=1)
		title = title.to(self.device).squeeze(dim=1)
		title_mask = title_mask.to(self.device).squeeze(dim=1)

				
		outputs = self(
			input_ids=sequence,
			attention_mask=sequence_mask,
			labels=title.type(torch.LongTensor).to(self.device), 
		)
			

		title_loss = outputs.loss
		logits = outputs.logits.to(self.device)

		summary_loss = 0
		if self.use_summarizer_for_loss:
			summary[summary == 0] = -100 # 0 is pad token id
			summary_loss = self.loss(logits.view(-1, logits.size(-1)), summary.type(torch.LongTensor).view(-1).to(self.device))

		classifier_loss = 0
		if self.use_classifier_for_loss:
			generated_title = self.model.generate(
				input_ids=sequence,
				attention_mask=sequence_mask,
				do_sample=False,  # disable sampling to test if batching affects output
			)
			sequences, attention_mask = self.gen_logits(sequence, generated_title) # decode using bert, encode using t5

			output = self.classifier(sequences, token_type_ids=None, attention_mask=attention_mask)
			prob = output.logits.softmax(dim=-1)[0][1]
			#pred = prob >= 0.5
			classifier_loss = prob * self.classifier_loss_coef	
			
		total_loss = title_loss + summary_loss + classifier_loss
		
		if self.use_wandb:
			wandb.log({
				'title_loss': title_loss,
				'summary_loss': summary_loss,
				'classifier_loss': classifier_loss,
				'total_loss': total_loss})
		
		return total_loss

	def training_step(self, batch, batch_idx):
		loss = self._step(batch)
		tensorboard_logs = {"train_loss": loss}
		return {"loss": loss, "log": tensorboard_logs}

	def configure_optimizers(self):
		"Prepare optimizer and schedule (linear warmup and decay)"

		model = self.model
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
				"weight_decay": 0.001,
			},
		]
		optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=self.eps, betas=self.betas)
		self.opt = optimizer
		scheduler = get_linear_schedule_with_warmup(
			self.opt, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps
		)
		self.lr_scheduler = scheduler
		return [optimizer]

	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, using_native_amp=False):
		optimizer.step()
		optimizer.zero_grad()
		self.lr_scheduler.step()


if __name__ == "__main__":    
	# dataset, kind of model (which loss function) make all of the losses 
	# update save prefix based on this
	parser = argparse.ArgumentParser()
	parser.add_argument("-c", "--use_class_loss", help="Use finetinued BERT clickbait classifier as part of loss. Default: false", action="store_true")
	parser.add_argument("-s", "--use_summ_loss", help="Use BART summarized entries as part of loss. Default: false", action="store_true")
	parser.add_argument("-e", "--epochs", help="Number of epochs to train on. Default: 100", default=100, type=int)
	parser.add_argument("-w", "--wandb", help="Use wandb tracking. Default: False", action="store_true")

	args = parser.parse_args()
	print(args)
	
	if torch.cuda.is_available():
			device="cuda"
	else:
			device="cpu"

	train_dataset = TokenizedT5Dataset(load_dataset_path=TOKENIZED_DATASET_PATH_TRAIN_T5, tokenizer= "t5-small")
	train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=12)
	
	bert_tokenizer = None
	t5_tokenizer = None
	if args.use_class_loss: #we need both of these tokenizers for 
		bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
		t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

	checkpoint_callback = ModelCheckpoint(
			monitor="train_loss",
			filepath= MODEL_SAVE_DIR + "T5_HEADLINE-class_loss_" + str(args.use_class_loss) + "-summ_loss_" + str(args.use_summ_loss) + "-{epoch:02d}-{train_loss:.2f}",
			save_top_k=3,
			mode="min",
	)

	trainer = pl.Trainer(max_epochs=args.epochs, gpus=1, checkpoint_callback=checkpoint_callback)
	model = HeadlineGenerator(
		train_dataloader,
		epochs=args.epochs, 
		use_classifier_for_loss=args.use_class_loss,
		use_summarizer_for_loss=args.use_summ_loss,
		bert_tokenizer=bert_tokenizer,
		t5_tokenizer=t5_tokenizer,
		use_wandb=args.wandb
	)

	trainer.fit(model, train_dataloader)
