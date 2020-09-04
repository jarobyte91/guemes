import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import ceil

def pad(x, l, padding_char = 0, end_char = 2):
    if len(x) < l:
        return x + [padding_char for i in range(l - len(x))]
    else:
        return x[:l-1] + [end_char]

class CharLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocabulary_size, layers):
        super(CharLSTM, self).__init__()
        self.embeddings = nn.Embedding(vocabulary_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, layers)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, layers)
        self.decoder_to_vocabulary = nn.Linear(hidden_dim, vocabulary_size)
        self.vocabulary_size = vocabulary_size
    
    def forward(self, sentence, current_token):
        embeddings = self.embeddings(sentence).transpose(0, 1)
        sequence, hidden = self.encoder(embeddings)
        current_token_embeddings = self.embeddings(current_token).transpose(0, 1)
        chars, final_hidden = self.decoder(current_token_embeddings, hidden)
        chars = self.decoder_to_vocabulary(chars).transpose(0, 1)
        return chars.log_softmax(-1), final_hidden
    
    def fit(self, X, Y, 
            lr = 0.1, 
            epochs = 20, 
            plot = True, 
            padding_char = 0, 
            end_char = 2, 
            save_path = "checkpoint.pt", 
            verbose = True):
        loss_function = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr)
        X_max_length = max([len(x) for x in X])
        Y_max_length = max([len(y) for y in Y])        
        X = torch.tensor([pad(x, X_max_length, padding_char, end_char) for x in X])
        Y = torch.tensor([pad(y, X_max_length, padding_char, end_char) for y in Y])
        losses = []
        if verbose:
            process = tqdm(range(epochs))
        else:
            process = range(epochs)
        for i in process:
            outputs, last_hidden = self.forward(X, Y[:, :-1])
            outputs = outputs.transpose(1, 2).squeeze()
            loss = loss_function(outputs, Y[:, 1:])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        if plot:
            print("Final loss:", losses[-1])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training process")
            plt.axhline(0, c = "black", ls = "--")
            plt.plot(losses)
            plt.show()
        torch.save(self.state_dict(), save_path)
        return losses
                   
    def fit_batch(self, X, Y, 
                  lr = 0.1, 
                  epochs = 20, 
                  batch_size = 10, 
                  plot = True, 
                  padding_char = 0, 
                  end_char = 2, 
                  save_path = "checkpoint.pt", 
                  verbose = True):
        loss_function = nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr)
        losses = []
        # generate batches
        X_batch = []
        Y_batch = []
        for i in range(ceil(len(X) / batch_size)):
            x, y = X[i:i + batch_size], Y[i:i + batch_size]
            x_max_length, y_max_length = max([len(s) for s in x]), max([len(s) for s in y])
            x, y = [pad(s, x_max_length, padding_char, end_char) for s in x], [pad(s, y_max_length) for s in y]
            x, y = torch.tensor(x), torch.tensor(y)
            X_batch.append(x)
            Y_batch.append(y)
        # do the training
        if verbose:
            process = tqdm(range(epochs))
        else:
            process = range(epochs)
        for i in process:
            epoch_loss = []
            for x, y in zip(X_batch, Y_batch):
                outputs, last_hidden = self.forward(x, y[:, :-1])
                outputs = outputs.transpose(1, 2).squeeze()
                loss = loss_function(outputs, y[:, 1:])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss.append(loss.item())
            losses.append(np.array(epoch_loss).mean())
        if plot:
            print("Final loss:", losses[-1])
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training process")
            plt.axhline(0, c = "black", ls = "--")
            plt.plot(losses)
            plt.show()
        torch.save(self.state_dict(), save_path)
        return losses
        
    def predict(self, X, Y, padding_char = 0, end_char = 2):
        with torch.no_grad():
            self.eval()
            X_max_length = max([len(x) for x in X])
            Y_max_length = max([len(y) for y in Y])
            X = torch.tensor([pad(x, X_max_length, padding_char, end_char) for x in X])
            Y = torch.tensor([pad(y, Y_max_length, padding_char, end_char) for y in Y])
            outputs, last_hidden = self(X, Y[:, :-1])
            outputs = outputs.transpose(1, 2).squeeze()
        return outputs
    
    def predict_batch(self, X, Y, batch_size = 10, padding_char = 0, end_char = 2):
        predictions = []
        for b in range(ceil(len(X) / batch_size)):
            X_batch = X[b * batch_size:(b + 1) * batch_size]
            Y_batch = Y[b * batch_size:(b + 1) * batch_size]
            X_max_length = max([len(x) for x in X_batch])
            Y_max_length = max([len(y) for y in Y_batch])
            predictions.append(self.predict(X_batch, Y_batch, padding_char, end_char))
        return predictions
        
    def spellcheck(self, sentence, max_length = 400):
        sentence = torch.tensor(sentence)
        sentence_embeddings = self.embeddings(sentence).unsqueeze(1)
        sequence, hidden = self.encoder(sentence_embeddings)
        output = [1]
        for i in range(max_length):
            current_token = self.embeddings(torch.tensor([output[-1]])).unsqueeze(1)
            next_token, hidden = self.decoder(current_token, hidden)
            next_token = self.decoder_to_vocabulary(next_token).argmax(-1).item()
            output.append(next_token)
            if next_token == 2 or next_token == 0:
                break
        return output
