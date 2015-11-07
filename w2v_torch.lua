
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
end

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

    self.word_vecs = nn.LookupTable(self.vocab_size, self.dim)
    self.context_vecs = nn.LookupTable(self.vocab_size, self.dim)
    self.word_vecs:reset(0.25); self.context_vecs:reset(0.25)
    self.w2v = nn.Sequential()
    self.w2v:add(nn.ParallelTable())
    self.w2v.modules[1]:add(self.context_vecs)
    self.w2v.modules[1]:add(self.word_vecs)
    self.w2v:add(nn.MM(false, true))
    self.w2v:add(nn.Sigmoid())
    self.decay = (self.min_lr-self.lr)/(self.total_count*self.window)

    local start = sys.clock()
    local total_count_pow = 0
    print("creating table ")
    for _, count in pairs(self.vocab) do
        total_count_pow = total_count_pow + count^self.alpha
    end   
    self.table = torch.IntTensor(self.table_size)
    local word_index = 1
    local word_prob = self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
    for idx = 1, self.table_size do
        self.table[idx] = word_index
        if idx / self.table_size > word_prob then
            word_index = word_index + 1
            --print(self.index2word[word_index]);
            if self.index2word[word_index] ~= nil then
               word_prob = word_prob + self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
            end
        end
        if word_index > self.vocab_size then
            word_index = word_index - 1
        end
    end
    print("table creation done")
    
end


function Word2Vec:train_context_word(contexts,word)
    local p = self.w2v:forward({contexts, word})
    local loss = self.criterion:forward(p, self.labels)
    local dl_dp = self.criterion:backward(p, self.labels)
    self.w2v:zeroGradParameters()
    self.w2v:backward({contexts, word}, dl_dp)
    self.w2v:updateParameters(self.lr)
end

function Word2Vec:sample_contexts(context)
    self.contexts[1] = context
    local i = 0
    while i < self.neg_samples do
        neg_context = self.table[torch.random(self.table_size)]
	if context ~= neg_context then
	    self.contexts[i+2] = neg_context
	    i = i + 1
	end
    end
end

function Word2Vec:train_data(corpus)
    print("Training Data...")
    local start = sys.clock()
    local c = 0
    f = io.open(corpus, "r")
    print("Iterating over dataset")
    for line in f:lines() do
        sentence = self:split(line)
        for i, word in ipairs(sentence) do
	        word_idx = self.word2index[word]
            if word_idx ~= nil then -- word exists in vocab
                local reduced_window = 2
                self.word[1] = word_idx 
                for j = i - reduced_window, i + reduced_window do 
                    local context = sentence[j]
                    if context ~= nil and j ~= i then
                        context_idx = self.word2index[context]
                        if context_idx ~= nil then
  		                    self:sample_contexts(context_idx) 
                            self:train_context_word(self.word,self.contexts) 
                            c = c + 1
                            self.lr = math.max(self.min_lr, self.lr + self.decay) 
                            if c % 100000 ==0 then
                                print(string.format("%d words trained in %.2f seconds. Learning rate: %.4f", c, sys.clock() - start, self.lr))
                            end
                        end
                    end
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
    file = torch.DiskFile('foo.txt', 'w'):ascii()

    wstr=self.vocab_size.." 100\n"

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
        for i=1,100 do -- fill up the Storage
        	wstr=wstr..w[i].." "
            --file:writeDouble(w[i])
            --file:writeString(" ")
        -- for i in #w do
        --     print(i)
        end
        file:writeString(wstr.."\n")
        --file:writeObject(w)

    end
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


