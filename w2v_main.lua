require("io")
require("os")
require("paths")
require("torch")
dofile("w2v_torch.lua")

config = {}
config.corpus = "text8_newline.txt" 
config.window = 5 
config.dim = 100 
config.alpha = 0.75
config.table_size = 1e8
config.neg_samples = 5 
config.minfreq = 5 
config.lr = 0.5 
config.min_lr = 0.4
config.epochs = 1

m = Word2Vec(config)
m:build_datastructures(config.corpus)
--m:build_table()

for k = 1, config.epochs do
    m.lr = config.lr
    m:train_data(config.corpus)
end

m:writebinary()

