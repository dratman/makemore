# You give this script some words (one per line) and it will generate more things like it.
# Uses super state of the art Transformer AI tech.
# This code is intended to be hackable. Tune it to your needs.

import os
import sys
import time
from datetime import datetime
import math
import argparse
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 12 # was 4
    n_embd: int = 144 # was 64
    n_embd2: int = 144 # was 64
    n_head: int = 8 # was 4
# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):

    # Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    # Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):

    # A vanilla multi-head masked self-attention layer with a projection at the end.
    # It is possible to use torch.nn.MultiheadAttention here but I am including an
    # explicit implementation to show that this is nothing too scary.

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection

        y = self.c_proj(y)
        return y

class Block(nn.Module):
    # An unassuming Transformer block
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class Transformer(nn.Module):
    # Transformer Language Model, exactly as seen in GPT-2

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# helper functions for evaluating and sampling from the model
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
    # Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    # the sequence max_new_tokens times, feeding the predictions back into the model each time.
    # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    block_size = model.get_block_size()
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)

        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Sample from the model and pretty print the decoded samples.
# (Some comments written by Claude)

def print_samples(num=10):

# The next line of code is creating an initial tensor to start the generation process.
# --->torch.zeros(num, 1, ...): This creates a tensor filled with zeros. The shape
# of this tensor is (num, 1), where num is the number of samples to generate
# (passed as an argument to the print_samples function, defaulting to 10).
# --->dtype=torch.long: This specifies that the data type of the tensor should be a
# 64-bit integer. In PyTorch, this is often used for storing indices or labels.
# --->.to(args.device): This moves the tensor to the specified device (CPU or GPU) as
# defined in the args.device parameter. This ensures that the tensor is on the
# same device as the model, which is important for performance and compatibility.
# --->The purpose of this tensor is to serve as the initial input for the
# generation process.
# --->The tensor has shape (num, 1) because it represents num sequences, each
# starting with a single token.It's filled with zeros because zero is used as the index
# for the <START> token. This tells the model to start generating from the beginning of
# a sequence.
# --->This initial tensor will be passed to the generate function, which will then
# use the model to progressively generate new tokens, building up complete
# sequences.

    X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)

# This next line is setting up a parameter for top-k sampling, which is
# a technique used in text generation to control the randomness and quality of
# the output. This is a conditional (ternary) expression in Python.
# --->args.top_k is likely an argument passed to the script, probably set
# via command line or in a configuration file. It determines whether to use
# top-k sampling and what value of k to use. If args.top_k is not equal to -1,
# top_k is set to the value of args.top_k. This means top-k sampling will be
# used with the specified k value. If args.top_k is equal to -1, top_k is set
# to None.
# --->Top-k sampling is a method used in language models to reduce the
# likelihood of generating low-probability or nonsensical tokens. When
# generating each token:
# --->If top-k sampling is used (top_k is not None), the model will only
# consider the k most likely next tokens, effectively truncating the
# probability distribution. If top-k sampling is not used (top_k is None), the
# model will consider all possible next tokens according to their
# probabilities.
# --->Using top-k sampling can help to improve the quality and coherence of
# the generated text by limiting the model's choices to the most probable
# options. The value of k controls the trade-off between creativity (lower k)
# and predictability (higher k) in the generated output.

    top_k = args.top_k if args.top_k != -1 else None

# --->train_dataset.get_output_length(): This method call is likely returning
# the maximum or expected length of an output sequence in the training dataset.
# This could be, for example, the maximum word length if the model is
# generating words. The -1 subtraction: As the comment explains, this is
# because the generation process is already starting with a <START> token
# (represented by index 0 in the initial tensor we saw earlier).
# --->Let's elaborate on why this subtraction is necessary:
# --->In many sequence generation tasks, we start with a special <START> token
# and end with a special <END> or <STOP> token. The initial tensor (X_init)
# created earlier already includes this <START> token. Therefore, we need one
# less step than the full output length, as we're generating everything after
# the <START> token.
# --->For example, if the maximum word length in the dataset is 10 characters:
# --->The full sequence would be: <START> + 8 characters + <END>, which is 10
# tokens. But since we're already starting with <START>, we only need to
# generate 9 more tokens (8 characters + <END>).
# --->This steps variable will be used in the generate function call to specify
# how many tokens should be generated after the initial <START> token. The
# careful handling of these special tokens and sequence lengths is crucial in
# ensuring that the model generates complete and properly formatted sequences.

    steps = train_dataset.get_output_length() - 1

# --->generate(): This is a function call to generate sequences using the
# trained model. While we don't see the implementation of this function in the
# provided code snippet, it's likely a custom function defined elsewhere in the
# codebase. Arguments to generate():
# --->model: The trained language model being used for generation. X_init: The
# initial tensor we saw earlier, containing the start tokens. steps: The number
# of generation steps, which we just discussed in the previous explanation.
# top_k: The parameter for top-k sampling, which we discussed earlier.
# do_sample=True: This likely indicates that the generation should use sampling
# rather than always choosing the most probable token (which would be called
# greedy decoding).
# --->.to('cpu'): After generation, the resulting tensor is moved to the CPU.
# This is often done to make further processing easier, especially if the
# generation was performed on a GPU.
# --->The generate() function is doing the heavy lifting here. It's using the
# trained model to probabilistically generate sequences of tokens, starting
# from the initial <START> tokens in X_init, and continuing for steps number of
# steps. The resulting X_samp is likely a tensor with shape (num, steps+1),
# where:
# --->num is the number of samples generated (10 by default in this function)
# steps+1 is the length of each generated sequence (including the initial
# <START> token)
# --->Each row in X_samp represents one generated sequence, with each element
# being a token index corresponding to the model's vocabulary. This generated
# sample will then be processed and decoded in the subsequent code to produce
# human-readable output.

    X_samp = generate(model, X_init, steps, top_k=top_k, do_sample=True).to('cpu')

    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        # get the i'th row of sampled integers, as python list
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token
        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        # separately track samples that we have and have not seen before
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-'*80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print("---")
            print(word)
    print('-'*80)

# This function, evaluate, is designed to assess the performance of the model on a given dataset.
# "@torch.inference_mode()" is a decorator that puts PyTorch into inference mode.
# It disables gradient computation, which reduces memory usage and speeds up
# computation when you're not training.
# This function is typically used to assess how well the model is performing
# on a validation or test set during or after training. The mean loss gives
# an indication of the model's overall performance on the dataset.

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    # Put the model in evaluation mode, which can affect
    # certain layers like Dropout or BatchNorm.
    model.eval()

    # Create a DataLoader to efficiently iterate over the dataset in batches.
    # It shuffles the data and uses the specified batch size.
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)

    losses = []

    # Loop over batches from the DataLoader; move each tensor in the batch to the
    # specified device (CPU/GPU), and unpack the input (X) and target (Y) from the batch.
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # Run the model on the input X and compute the loss against the target Y.
        logits, loss = model(X, Y)

        # Add the loss value for this batch to the list of losses.
        losses.append(loss.item())

        # Break the loop if we've processed the maximum number of batches specified.
        if max_batches is not None and i >= max_batches:
            break
    # Compute the mean loss across all batches.
    mean_loss = torch.tensor(losses).mean().item()

    model.train() # reset model back to training mode
    return mean_loss
# ------------------------------
# helper functions for creating the training and test Datasets that emit words
class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length
        self.stoi = {ch:i+1 for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def get_output_length(self):
        return self.max_word_length + 1 # <START> token followed by words

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations
        return x, y

def split_text(text, max_length):
    lines = text.splitlines()
    result = []
    for line in lines:
        while len(line) > max_length:
            # Split the line at the maximum length
            result.append(line[:max_length])
            line = line[max_length:]
        result.append(line)
    return result

def create_datasets(input_file):

    # preprocessing of the input text file
    with open(input_file, 'r') as f:
        data = f.read()
#    words = data.splitlines()
    max_length = 100  # Set maximum line length
    words = split_text(data, max_length) # Split into lines with a maximum length specified.

    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    max_word_length = max(len(w) for w in words)
    print(f"number of examples in the dataset: {len(words)}")
    print(f"max word length: {max_word_length}")
    print(f"number of unique characters in the vocabulary: {len(chars)}")
    print("vocabulary:")
    print(''.join(chars))

    # partition the input data into a training and the test set
    test_set_size = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-test_set_size]]
    test_words = [words[i] for i in rp[-test_set_size:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset

class InfiniteDataLoader:
    """
    this is really hacky and I'm not proud of it, but there doesn't seem to be
    a better way in PyTorch to just create an infinite dataloader?
    """

    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration: # this will technically only happen after 1e10 samples... (i.e. basically never)
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch
# ------------------------------
if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # system/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--device', type=str, default='cpu', help="device to use for compute, examples: cpu|cuda|cuda:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    # model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--n-layer', type=int, default=4, help="number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-4, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.01, help="weight decay")
    args = parser.parse_args()
    print(vars(args))

    # system inits
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.work_dir)

    # init datasets
    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()
    print(f"dataset determined that: {vocab_size=}, {block_size=}")

    # init model
    config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                       n_layer=args.n_layer, n_head=args.n_head,
                       n_embd=args.n_embd, n_embd2=args.n_embd2)
    if args.type == 'transformer':
        model = Transformer(config)
    else:
        raise ValueError(f'model type {args.type} is not recognized')
    model.to(args.device)
    print(f"model #params: {sum(p.numel() for p in model.parameters())}")
    if args.resume or args.sample_only: # note: if we sample-only then we also assume we are resuming
        print("resuming from existing model in the workdir")
        model.load_state_dict(torch.load(os.path.join(args.work_dir, 'model.pt')))
    if args.sample_only:
        print_samples(num=50)
        sys.exit()

    # init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99), eps=1e-8)

    # init dataloader
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

    # training loop
    best_loss = None

    step = 0
    while True:
        t0 = time.time()
        # get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        # feed into the model
        logits, loss = model(X, Y)
        # calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()

        optimizer.step()
        # wait for all CUDA work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()
        # logging

        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time {(t1-t0)*1000:.2f}ms")
        # evaluate the model

        if step > 0 and step % 500 == 0:
            # print date and time of loss report
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime(">>>>>   %Y-%m-%d %H:%M")
            print(formatted_datetime)
            # prepare to print loss report
            train_loss = evaluate(model, train_dataset, batch_size=100, max_batches=10)
            test_loss  = evaluate(model, test_dataset,  batch_size=100, max_batches=10)

            writer.add_scalar("Loss/train", train_loss, step)
            writer.add_scalar("Loss/test", test_loss, step)
            writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved [Save anyway! See below]
            if best_loss is None or test_loss < best_loss:
                      best_loss = test_loss
            # Save model unconditionally
            out_path = os.path.join(args.work_dir, "model.pt")
            torch.save(model.state_dict(), out_path)

        # sample from the model

        if step > 0 and step % 200 == 0:
            print_samples(num=10)

        step += 1

        # termination conditions

        if args.max_steps >= 0 and step >= args.max_steps:
            break
