import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 02:27:31 2020

@author: desmo
"""

CONTEXT_SIZE = 8
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
ngram = 8
ngrams = []
for i in range(len(test_sentence) - (ngram-1)):
    ng = []
    for k in range(ngram-1):
        ng.append(test_sentence[k+i])
    ngrams.append((ng,test_sentence[i+ngram-1]))
trigrams = [([test_sentence[i], test_sentence[i + 1], test_sentence[i + 2], test_sentence[i + 3], test_sentence[i + 4], test_sentence[i + 5], test_sentence[i + 6], test_sentence[i + 7]], test_sentence[i + 8]) for i in range(len(test_sentence) - 8)]
print(ngrams[:3])
print(trigrams[:-1])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

device = torch.device("cuda")

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
a = []
for epoch in range(10):
    total_loss = 0
    
    for context, target in trigrams:
        #print(torch.tensor([word_to_ix[target]], dtype=torch.long))
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs.to(device))

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long).to(device))
        

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    print(total_loss)
print(losses)  # The loss decreased every iteration over the training data!
criterion = nn.NLLLoss()

def evaluate(trigrams):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    for context, target in trigrams:
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        output = model(context_idxs.to(device))
        loss = criterion(output, torch.tensor([word_to_ix[target]], dtype=torch.long).to(device))
        total_loss += loss.item()

    # ntokens = len(corpus.dictionary)
    # if args.model != 'Transformer':
    #     hidden = model.init_hidden(eval_batch_size)
    # with torch.no_grad():
    #     for i in range(0, data_source.size(0) - 1, args.bptt):
    #         data, targets = get_batch(data_source, i)
    #         if args.model == 'Transformer':
    #             output = model(data)
    #             output = output.view(-1, ntokens)
    #         else:
    #             output, hidden = model(data, hidden)
    #             hidden = repackage_hidden(hidden)
    #         total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(trigrams) - 1)
print(evaluate(trigrams))