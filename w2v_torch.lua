
require("sys")
require("nn")

local Word2Vec = torch.class("Word2Vec")

function Word2Vec:__init(config)
    self.tensortype = torch.getdefaulttensortype()
    self.gpu = config.gpu -- 1 if train on gpu, otherwise cpu
    self.stream = config.stream -- 1 if stream from hard drive, 0 otherwise
    self.neg_samples = config.neg_samples
    self.minfreq = config.minfreq
    self.dim = config.dim
    self.criterion = nn.ClassNLLCriterion() -- logistic loss
    self.word = torch.IntTensor(1) 
    self.contexts = torch.IntTensor(1+self.neg_samples) 
    self.labels = torch.zeros(1+self.neg_samples); self.labels[1] = 1 
    self.window = config.window 
    self.lr = config.lr 
    self.min_lr = config.min_lr
    self.alpha = config.alpha
    self.table_size = config.table_size 
    self.vocab = {}
    self.index2word = {}
    self.word2index = {}
    self.total_count = 0
    self.numhid2 = 200
    self.numwords = self.window - 1
end

dataset = {}
function dataset:size() return 200 end

function Word2Vec:build_datastructures(corpus)

    print("Creating Datastructures")
    local start = sys.clock()
    print("Opening file: "..corpus)
    local f = io.open(corpus, "r")
    local n = 1
    print("Creating dictionary")
    for line in f:lines() do
        for _, word in ipairs(self:split(line)) do
           self.total_count = self.total_count + 1
           if self.vocab[word] == nil then
               self.vocab[word] = 1  
           else
               self.vocab[word] = self.vocab[word] + 1
           end
        end
        n = n + 1
    end
    f:close()
    print("Dictionary creation done")
    freq_words=0

    print("Reducing vocab size")
    for word, count in pairs(self.vocab) do
        if count >= self.minfreq then
            freq_words=freq_words+1
            self.index2word[#self.index2word+1] = word
            self.word2index[word] = #self.index2word        
        else
           self.vocab[word] = nil
        end
    end
    print("Frequency reduced")
    print("Num words after reduction: "..freq_words)
    
    self.vocab_size = #self.index2word

    -- self.word_vecs = nn.LookupTable(self.vocab_size, self.dim)
    -- self.context_vecs = nn.LookupTable(self.vocab_size, self.dim)
    -- self.word_vecs:reset(0.25); self.context_vecs:reset(0.25)
    -- self.w2v = nn.Sequential()
    -- self.w2v:add(nn.ParallelTable())
    -- self.w2v.modules[1]:add(self.context_vecs)
    -- self.w2v.modules[1]:add(self.word_vecs)
    -- self.w2v:add(nn.MM(false, true))
    -- self.w2v:add(nn.Sigmoid())
    self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)
    --modified version for CBOW implementation
    self.model = nn.Sequential();                           -- model description
    self.word_vecs = nn.LookupTable(self.vocab_size, self.dim);
    self.model:add( self.word_vecs ); 
    self.model:add( nn.Reshape(self.numwords*self.dim));  
    self.model:add( nn.Linear(self.numwords*self.dim, self.numhid2)); 
    self.model:add( nn.Sigmoid() );                  
    self.model:add( nn.Linear(self.numhid2, self.vocab_size) );    
    self.model:add( nn.LogSoftMax() ); 

    -- Minimize the negative log-likelihood
    self.criterion = nn.ClassNLLCriterion();
    self.trainer = nn.StochasticGradient(self.model, self.criterion);
    self.trainer.learningRate = self.lr;
    self.trainer.maxIteration = 2;
    --maintaining word order
    self.trainer.shuffleIndices = false;

    
    self.model.vocab = self.word2index;
    self.model.vocab_ByIndex = self.index2word;

end


function Word2Vec:train_data_test(corpus)
    input_batch = torch.IntTensor(1,self.window-1)
    target_batch = torch.IntTensor(1)

    print("Training Data...")
    local start = sys.clock()
    local c = 0
    f = io.open(corpus, "r")
    print("Iterating over dataset")
    local batch_count = 1
    for line in f:lines() do
        sentence = self:split(line)
        for i, word in ipairs(sentence) do
            word_idx = self.word2index[word]
            if word_idx ~= nil then -- word exists in vocab
                local reduced_window = 2
                self.word[1] = word_idx 
                local cnt = 1
                local adjacent_words = torch.IntTensor(self.window - 1)
        local found = 0
                for j = i - reduced_window, i + reduced_window do 
            --print(j .. " j val")
                    local context = sentence[j]
                    if i == j and context ~= nil then
            --print(context)
            --print(self.word2index[context])
            --print("here1")
                        target_batch[1] = self.word2index[context]
            --print("here")
                    end
                    if context ~= nil and j ~= i then
                        context_idx = self.word2index[context]

                        if context_idx ~= nil then
                found = found + 1
                            adjacent_words[cnt] = context_idx
                            cnt = cnt + 1
                            c = c + 1
                            self.lr = math.max(self.min_lr, self.lr + self.decay) 
                            if c % 100000 ==0 then
                                print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
                            end
                        end
                    end
                end
        if found == 4 then
            --sending input batch and target batch for training using stochastic gradient
                input_batch = adjacent_words
            --print(input_batch)
            --print(target_batch)
            --os.exit(0)
            --print(batch_count)
                dataset[batch_count] = { input_batch, target_batch }
            batch_count  = batch_count + 1
            --print(dataset)
            found = 0
        end
        if batch_count % 201 == 0 then
            print("Processing Batch... ")
            --print(dataset)
            self.trainer:train(dataset);    
            batch_count = 1
        end
            end
        end
    end
    print("Training Complete")
end


function Word2Vec:normalize(m)
    m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
        m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end


function Word2Vec:writebinary()
    print("writing to disk")
    file = torch.DiskFile('yoonkim_4mb.txt', 'w'):ascii()
    if file == nil then
    print("cannot open file")
    os.exit(0)
    end
    wstr=self.vocab_size.." 200\n"

    --file:writeInt(self.vocab_size)
    --file:writeString(" 100\n")
    file:writeString(wstr)

    for word, count in pairs(self.vocab) do
        if self.word_vecs_norm == nil then
            self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
        end
        --print("*********"..word)
        file:writeString(word.." ")
        w = self.word_vecs_norm[self.word2index[word]]
        --file:wri
        --print(#w)
        wstr=""
        for i=1,self.dim do -- fill up the Storage
            wstr=wstr..w[i].." "
        --print(wstr)
            --file:writeDouble(w[i])
            --file:writeString(" ")
        -- for i in #w do
        --     print(i)
        end
    --print(wstr)
        file:writeString(wstr.."\n")
        --file:writeObject(w)

    end
    print("closing file")
    file:close() 
end

function Word2Vec:split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        t[i] = str; i = i + 1
    end
    return t
end




