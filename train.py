import torch
from torch.utils.data import DataLoader
from transformers import Trainer

from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup

from dataset.dataset import ClickbaitDataset
from params import *
from torch.nn import CrossEntropyLoss

from tqdm import tqdm, trange

class GPTTrainer():
    def __init__(
        self, 
        model, 
        dataloader, 
        tokenizer,
        warmup_steps=200, 
        batch_size=16, 
        epochs=5, 
        lr=2e-5,
        output_dir="./outputs/", 
        output_prefix="GPTheadlines",
        test_mode=False,
        save_model_on_epoch=False,
    ):
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.warmup_steps = warmup_steps    
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.test_mode = test_mode
        self.save_model_on_epoch = save_model_on_epoch
        self.optimizer = AdamW(model.parameters(), lr = lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps = self.warmup_steps, num_training_steps = -1
        )
        self.loss = CrossEntropyLoss()

    def train_2(self):
        trainer = Trainer(
            model=self.model, 
            train_dataset=self.dataloader, 
            eval_dataset=self.dataloader)
        trainer.train()
        
    def train(self): 
    
        for epoch in tqdm(range(self.epochs)):
            for batch in iter(self.dataloader):
                title, body, score = batch # ASSUMES BATCH_SIZE = 1
                tokenized_titles = self.tokenizer(list(title), padding=True, return_tensors="pt")['input_ids']
                # print(tokenized_titles.shape)
                tokenized_bodies = self.tokenizer(list(body), padding=True, return_tensors="pt")['input_ids']
                output = self.model(tokenized_bodies, labels=tokenized_bodies)
                loss = output[0]
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # if save_model_on_epoch:
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            #     )


if __name__ == "__main__":
    
    train_data = ClickbaitDataset(TRAIN_PATH)
    train_dataloader = DataLoader(train_data, batch_size=240)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    print(f'Using {device} device')

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    train = GPTTrainer(model=model, dataloader=train_dataloader, tokenizer=tokenizer, batch_size=2)
    train.train()
    
    
    # model.to(device)

    # train(model, train_dataloader)