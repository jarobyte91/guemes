from char_lstm import CharLSTM
from multiprocessing import Pool
import pandas as pd
from os import cpu_count
from random import randint
from timeit import default_timer
import datetime

data = pd.read_pickle("../data/tidy/noisy_opus_sample.pkl")
vocabulary_size = 136

print("starting experiments")

def experiment(order):
    print("experiment {} started".format(order))
    embedding_dim = randint(2, 8)
    hidden_dim = randint(2, 8)
    layers = randint(1, 4)
    lr = 10 ** randint(-5, -1)
    epochs = 2
    batch_size = randint(2, 10)
    start = default_timer()
    
    model = CharLSTM(embedding_dim, hidden_dim, vocabulary_size, layers)
    losses = model.fit_batch(X = data["X"], 
                             Y = data["Y"], 
                             lr = lr,  
                             plot = False, 
                             epochs = epochs, 
                             batch_size = batch_size, 
                             verbose = False,
                             save_path = "checkpoints/{}_{}_{}_{}_{}.pt".format(embedding_dim, 
                                                                                hidden_dim, 
                                                                                layers, 
                                                                                str(lr).replace(".", ""), 
                                                                                epochs))
    training_time = default_timer() - start
    print("experiment {} finished".format(order))
    return {"experiment":order, 
            "embedding_dim":embedding_dim, 
            "hidden_dim":hidden_dim, 
            "layers":layers, 
            "lr":lr, 
            "training_epochs":epochs, 
            "batch_size":batch_size, 
            "training_time":training_time, 
            "losses":[list(enumerate(losses))]}

sample_size = 6
p = Pool(cpu_count() - 1)

print("creating results dataframe")

results = pd.DataFrame(columns = ["experiment",
                                  "embedding_dim", 
                                  "hidden_dim", 
                                  "layers", 
                                  "lr", 
                                  "training_epochs", 
                                  "batch_size", 
                                  "training_time", 
                                  "epoch",
                                  "loss"])\

print("writing data")

for e in p.imap_unordered(experiment, range(sample_size)):
    e = pd.DataFrame(e)\
    .explode("losses")\
    .assign(epoch = lambda df: df["losses"].map(lambda x: x[0]),
            loss = lambda df: df["losses"].map(lambda x: x[1]))\
    .drop("losses", axis = 1)
    
    results = results.append(e)
    results.to_csv("experiments/{}.csv".format(date.today()), index = False)

