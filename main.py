import torch
from potterGPT import GPTConfig, PotterGPT
from train import PotterGPTTrainer
from data import get_data
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data, val_data, vocab_size, encode, decode = get_data()

config = GPTConfig(n_layers=6, n_embd=384, block_size=256, n_head=8, vocab_size=vocab_size, dropout=0.2)

model = PotterGPT(config)
model = model.to(device)

# if os.path.exists('model/ckpt.pt'):
#    model.load_state_dict(torch.load('model/ckpt.pt'))


trainer = PotterGPTTrainer(train_data, val_data, batch_size=32, block_size=256, model=model, lr=3e-4,
                           max_iters=5000, eval_interval=1000, eval_iters=200)

trainer.train()

# Save trained model
torch.save(model.state_dict(), 'model/ckpt.pt')


# generate from the model
context = torch.LongTensor(encode("Petunia said to Harry:"), device=device).unsqueeze(0)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))