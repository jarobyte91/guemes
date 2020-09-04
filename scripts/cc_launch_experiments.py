from char_lstm import CharLSTM
from multiprocessing import Pool
import pandas as pd
from os import cpu_count, mkdir
from random import randint
from timeit import default_timer
import datetime

print("start")

start = default_timer()

data = pd.read_pickle("../data/tidy/noisy_opus_sample.pkl")
vocabulary_size = 136

folder = "experiments/{}".format(datetime.datetime.now()).replace(".", "")
mkdir(folder)

print("starting experiment")

embedding_dim = randint(2, 1024)
hidden_dim = randint(2, 2048)
layers = randint(1, 10)
lr = 10 ** randint(-5, -1)
epochs = 100
batch_size = randint(2, 100)

model = CharLSTM(embedding_dim, hidden_dim, vocabulary_size, layers)
losses = model.fit(X = data["X"], 
                   Y = data["Y"], 
                   lr = lr,  
                   plot = False, 
                   epochs = epochs, 
                   verbose = False,
                   save_path = "{}/{}_{}_{}_{}_{}.pt".format(folder,
                                                             embedding_dim, 
                                                             hidden_dim, 
                                                             layers, 
                                                             str(lr).replace(".", ""), 
                                                             epochs))

training_time = default_timer() - start

print("experiment {} finished".format(order))

results = pd.Series({"experiment":order, 
                     "embedding_dim":embedding_dim, 
                     "hidden_dim":hidden_dim, 
                     "layers":layers, 
                     "lr":lr, 
                     "training_epochs":epochs, 
                     "batch_size":batch_size, 
                     "training_time":training_time, 
                     "losses":[list(enumerate(losses))]})\
            .explode("losses")\
            .assign(epoch = lambda df: df["losses"].map(lambda x: x[0]),
                    loss = lambda df: df["losses"].map(lambda x: x[1]))\
            .drop("losses", axis = 1)

results.to_csv(folder + "/log.csv", index = False)

