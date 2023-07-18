# create a PyTorch optimizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PotterGPTTrainer():
    def __init__(self, train_data, val_data, batch_size, block_size, model, lr, max_iters, eval_interval, eval_iters):
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.block_size = block_size
        self.model = model
        self.lr = lr
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                breakpoint()
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        for iter in range(self.max_iters):
            print(f"Epoch {iter}...")
            # every once in a while evaluate the loss on train and val sets
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss()
                print(f">> Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.get_batch('train')

            # evaluate the loss
            logits, loss = self.model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
