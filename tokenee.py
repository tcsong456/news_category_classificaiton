import numpy as np

class Vocab:
    def __init__(self,
                 list_of_tokens,
                 bos_token,
                 eos_token,
                 unk_token,
                 pad_token,
                 min_freq,
                 lower):
        self.list_of_tokens = list_of_tokens
        self.lower = lower
        self.min_freq = min_freq
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word_embeddings = None
        self.stoi,self.itos,self.freq = {},{},{}
        for i,special_token in enumerate([bos_token,eos_token,unk_token,pad_token]):
            self.stoi[special_token] = i
            self.itos[i] = special_token
        
    def build(self):
        print('building vocab!')
        for token in self.list_of_tokens:
            if self.lower:
                token = token.lower()
            
            if token not in self.freq:
                self.freq[token] = 1
            else:
                self.freq[token] += 1
        
        self.freq = dict(sorted(self.freq.items(),key=lambda x:x[1],reverse=True))
        
        tokens = []
        for token in self.freq:
            if self.freq[token] >= self.min_freq:
                tokens.append(token)
        
        for token in tokens:
            self.stoi[token] = self.__len__()
            self.itos[self.__len__()] = token
    
    def from_pretrained(self,pretrained_vectors):
        vec_size = len(list(pretrained_vectors.values())[0])  
        self.word_embeddings = np.zeros([len(self),vec_size])
        for token,ind in self.stoi.items():
            vec = pretrained_vectors.get(token)
            if vec is not None:
                self.word_embeddings[ind] = vec
        
    def __len__(self):
        return len(self.itos)

class Tokenizer:
    def __init__(self,
                 token_fn,
                 is_sentence,
                 max_len,
                 vocab=None):
        self.token_fn = token_fn
        self.is_sentence = is_sentence
        self.max_len = max_len
        self.vocab = vocab
    
        if not self.is_sentence:
            from nltk.tokenize import sent_tokenize
            self.sent_tokenize_fn = sent_tokenize
    
    def _vocab_fn(self,tokens):
        if self.vocab.bos_token:
            tokens = [self.vocab.bos_token] + tokens
        
        if self.vocab.eos_token:
            tokens += [self.vocab.eos_token]
        
        return tokens
    
    def tokenize(self,text):
        if self.is_sentence:
            tokens = self.token_fn(text)
            if self.vocab:
                tokens = self._vocab_fn(tokens)
        else:
            tokens = []
            sentences = self.sent_tokenize_fn(text)
            for sent in sentences:
                _token = self.token_fn(sent)
                if self.vocab:
                    _token = self._vocab_fn(_token)
                tokens += _token
        
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        
        if self.vocab:
            if self.vocab.pad_token and len(tokens) < self.max_len:
                tokens += [self.vocab.pad_token] * (self.max_len - len(tokens))
            
            if self.vocab.lower:
                tokens = [token.lower() for token in tokens]
        
        return tokens
    
    def transform(self,tokens):
        if self.vocab:
            return [self.vocab.stoi[token] if token in self.vocab.stoi else self.vocab.stoi[self.vocab.unk_token] for token in tokens]
    
    def inverse_transform(self,indices):
        if self.vocab:
            return [self.vocab.itos[index] for index in indices]
    
    def tokenize_and_transform(self,tokens):
        if self.vocab:
            return self.transform(self.tokenize(tokens))

#%%

