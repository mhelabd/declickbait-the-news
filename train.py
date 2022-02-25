import torch
from torch.utils.data import DataLoader
from transformers import Trainer

from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup

from dataset.dataset import ClickbaitDataset, TokenizedClickbaitDataset
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
        device='cpu'
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
        self.device=device
        
    def train(self): 
        model.zero_grad()
        for epoch in range(self.epochs):
            print('Training epoch:', epoch)
            for _, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                # tokenized_titles = self.tokenizer(list(title), padding=True, return_tensors="pt")['input_ids']
                # # print(tokenized_titles.shape)
                # tokenized_bodies = self.tokenizer(list(body), padding=True, return_tensors="pt")['input_ids']
                sequence, sep_idx, score = batch # ASSUMES BATCH_SIZE = 1
                # sequence.to(self.device)
                # self.model.to(self.device)

                self.model.train() #TODO: try having this outside the for loops
                # sequence = self.tokenizer.encode('Hi', return_tensors="pt")
                output = self.model(sequence)
                loss = torch.squeeze(output[0], dim=1)
                # logits = output[0]
                # shift_logits = logits[..., idx:-1, :].contiguous()
                # shift_labels = labels[..., idx+1:].contiguous()
                # loss = output[0]

                # idx = batch['sum_idx'].item() # index of separator token
                # # only consider loss on reference summary just like seq2seq models
                # shift_logits = logits[..., idx:-1, :].contiguous()
                # shift_labels = labels[..., idx+1:].contiguous()
                # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                # loss = loss/args.gradient_accumulation_steps

                loss.backward(torch.ones_like(loss))
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # if save_model_on_epoch:
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            #     )


if __name__ == "__main__":
    
    # train_data = ClickbaitDataset(TRAIN_PATH)
    train_dataset = TokenizedClickbaitDataset(TRAIN_PATH, saved_dataset_path=TOKENIZED_DATASET_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    print(f'Using {device} device')

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    train = GPTTrainer(model=model, dataloader=train_dataloader, tokenizer=tokenizer, batch_size=1, device=device, epochs=1)
    train.train()
    
    inputs = tokenizer.encode("hi", return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, do_sample=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    # model.to(device)

    # train(model, train_dataloader)