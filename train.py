import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_linear_schedule_with_warmup
from params import *
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from dataset.dataset import ClickbaitDataset, TokenizedClickbaitDataset

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
        output_dir=MODEL_SAVE_DIR, 
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
        
    def train(self):
        self.model.to(self.device)
        model.zero_grad()
        tr_loss = 0
        loss = None
        for epoch in range(self.epochs):
            print('Training epoch:', epoch)
            for step, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
                sequences, title_masks, scores = batch   
                sequences = sequences.type(torch.LongTensor)   
                title_masks = title_masks.type(torch.LongTensor)              
                labels = torch.clone(sequences)

                sequences = sequences.to(self.device)
                title_masks = title_masks.to(self.device)
                labels = labels.to(self.device)
                self.model.train()
                try:
                    output = self.model(sequences)
                except:
                    print(sequences)
                    print(sequences.shape)
                    continue
                logits = output[0] # batch_size, num_tokens, vocab_size

                title_masks_unsqueezed = (title_masks > 0).unsqueeze(dim=2)
                shift_logits = logits.masked_fill(title_masks_unsqueezed == 0, 0)
                shifted_title_mask = torch.roll(title_masks, shifts=1, dims=1).contiguous()
                shifted_title_mask[:, 0] = 0

                shift_labels = labels.masked_fill(shifted_title_mask == 0, 0)             
                loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), # batch_size*num_tokens[sep_idx[i]:], vocab_size
                    shift_labels.view(-1)
                ) #  batch_size * num_tokens[body_len:]
                loss = loss/self.gradient_accumulation_steps #1/32
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                tr_loss += loss.item()
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    print("loss:", loss.item(), end='\n\n')
                    
                if (step + 1)/self.gradient_accumulation_steps == 1.0:
                    print('After 1st update: ', end='\n\n')
                #     generate_sample(valid_dataset, tokenizer, num=2, eval_step=False)
                
            if self.save_model_on_epoch:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, f"{self.output_prefix}-{epoch}.pt"),
                )

if __name__ == "__main__":
    
    # train_data = ClickbaitDataset(TRAIN_PATH)
    train_dataset = TokenizedClickbaitDataset(TRAIN_PATH, load_dataset_path=TOKENIZED_DATASET_PATH_TRAIN, wanted_scores=[0])
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
    train = GPTTrainer(
        model=model, 
        dataloader=train_dataloader, 
        tokenizer=tokenizer, 
        batch_size=2, 
        device=device, 
        epochs=1, 
        save_model_on_epoch=True
    )
    train.train()