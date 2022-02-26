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
        device='cpu', 
        ignore_token=PAD_TOKEN, 
        gradient_accumulation_steps=32,
        max_grad_norm=1,
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
        # Loss function ingorces index of padding
        ignore_index = self.tokenizer.encode(ignore_token)[0]
        self.loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
    def train_2(self):
        self.model.to(self.device)
        model.zero_grad()
        tr_loss = 0
        for epoch in range(self.epochs):
            print('Training epoch:', epoch)
            for step, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                sequence, sep_idx, score = batch 
                labels = torch.clone(sequence)

                sequence = sequence.to(self.device)
                labels = labels.to(self.device)

                self.model.train()
                output = self.model(sequence)

                logits = output[0] # batch_size, num_tokens, vocab_size

                #Assumes Batch Size = 1
                sep_idx = sep_idx.item()
    
                shift_logits = logits[..., sep_idx:-1, :].contiguous() #  batch_size, num_tokens[sep_idx[i]:], vocab_size
                shift_labels = labels[..., sep_idx+1:].contiguous() #  batch_size, num_tokens[body_len:]
                
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), # batch_size*num_tokens[sep_idx[i]:], vocab_size
                    shift_labels.view(-1)
                ) #  batch_size * num_tokens[body_len:]

                loss = loss/self.gradient_accumulation_steps #1/32
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                tr_loss += loss.item()
                self.optimizer.step()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    # writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    # writer.add_scalar('loss', (tr_loss - logging_loss)/args.gradient_accumulation_steps, global_step)
                    print("loss:", loss.item(), end='\n\n')
                    
                if (step + 1)/self.gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                #     generate_sample(valid_dataset, tokenizer, num=2, eval_step=False)

                if True: torch.save(model.state_dict(), f"MODEL1-{epoch}.pt")

                

    def train(self): 
        model.zero_grad()
        for epoch in range(self.epochs):
            print('Training epoch:', epoch)
            for _, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                sequence, sep_idx, score = batch # ASSUMES BATCH_SIZE = 1
                # sequence.to(self.device)
                # self.model.to(self.device)

                self.model.train() #TODO: try having this outside the for loops
                # sequence = self.tokenizer.encode('Hi', return_tensors="pt")
                output = self.model(sequence)
                loss = torch.squeeze(output[0], dim=1)

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
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"

    print(f'Using {device} device')

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model = model.to(device)
    # model = AutoModelForCausalLM.from_pretrained("gpt2")
    train = GPTTrainer(model=model, dataloader=train_dataloader, tokenizer=tokenizer, batch_size=1, device=device, epochs=1)
    train.train_2()
    
    inputs = tokenizer.encode("hi", return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, do_sample=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    # model.to(device)

    # train(model, train_dataloader)