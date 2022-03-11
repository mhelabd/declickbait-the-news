import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

import sys; sys.path.append('.')

from dataset.dataset import TokenizedClickbaitDataset
from params import *

# BERT RECOMMENDED 
# Batch size: 16, 32
# Learning rate (Adam): 5e-5, 3e-5, 2e-5
# Number of epochs: 2, 3, 4
class ClassifierModule(nn.Module):
  def __init__(
    self, 
    dataloader,
    num_classes=2, 
    device="cpu", 
    lr=2e-5,
    eps=1e-8, 
    epochs=4,
    output_dir=MODEL_SAVE_DIR, 
    output_prefix="BERTClassifier",
  ):
    super(ClassifierModule, self).__init__()
    self.model = BertForSequenceClassification.from_pretrained(
      "bert-base-uncased",
      num_labels = num_classes,
      output_attentions = False, 
      output_hidden_states = False, 
    )
    self.optimizer = AdamW(self.model.parameters(), lr = lr, eps=eps)
    self.epochs = epochs
    total_steps = len(dataloader) * epochs
    self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    self.dataloader = dataloader
    self.device = device

    self.model.to(self.device)
    self.loss = nn.MSELoss()

  
  def train(self):
    self.model.train()
    for epoch in range(self.epochs):
      print('Training epoch:', epoch)
      pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))

      total_train_loss = 0
      for i, batch in pbar:
        sequences, _, scores = batch   
        sequences = sequences.type(torch.LongTensor)   
        attention_mask = sequences != 102.0 # Hacky way of getting padding attention mask
        scores = scores > 0.5
        labels = scores.to(torch.int64)
        labels = F.one_hot(labels, num_classes=2)
        sequences = sequences.to(self.device)
        labels = labels.to(self.device)

        self.model.zero_grad()

        logits = self.model(sequences, token_type_ids=None, attention_mask=attention_mask)['logits']
        loss = self.loss(logits.to(torch.float32), labels.to(torch.float32))
        pbar.set_description(f"loss: {loss}")
        
        
        total_train_loss += loss
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        if i % 1 == 0:
          torch.save (
            self.model.state_dict(),
            os.path.join(self.output_dir, f"{self.output_prefix}-{batch}.pt"),
          )

      avg_train_loss = total_train_loss / len(self.dataloader)    
      print('Avg Train Loss:', avg_train_loss)

if __name__ == "__main__":    
    train_dataset = TokenizedClickbaitDataset(TRAIN_PATH, load_dataset_path=TOKENIZED_DATASET_PATH_TRAIN, tokenizer= "bert")
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    print(f'Using {device} device')

    model = ClassifierModule(train_dataloader)
    model.train()


