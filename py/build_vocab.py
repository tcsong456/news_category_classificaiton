import argparse
import pickle
import os
from tokenee import Tokenizer,Vocab
import nltk
nltk.download('punkt')

TOKENIZER = ('treebank','mecab')

def argparser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--tokenizer',type=str,default='treebank')
    arg('--is_sentence',action='store_true')
    arg('--max_seq_len',type=int,default=1024)
    arg('--input',type=str)
    arg('--corpus',type=str,default='corpus/corpus_train.txt')
    arg('--vocab',type=str,default='corpus/voacb_train.pkl')
    arg('--bos_token',type=str,default='<bos>')
    arg('--eos_token',type=str,default='<eos>')
    arg('--unk_token',type=str,default='<unk>')
    arg('--pad_token',type=str,default='<pad>')
    arg('--min_freq',type=int,default=3)
    arg('--mode',type=str,default='skipgram',choices=['skipgram','cbow'])
    arg('--lower',action='store_true')
    arg('--pretrained_vectors',action='store_true')
    args = parser.parse_args()
    return args

def load_pretrained(file):
    word_vec = {}
    with open(file,'r',encoding='-utf-8') as f:
        for line in f:
            tokens = line.rstrip().split()
            try:
                word,vector = tokens[0],list(map(float,tokens[1:]))
                word_vec[word] = vector
            except UnicodeEncodeError:
                pass
        print(f'{len(word_vec)} number of vectors loaded')
    
    return word_vec

def fasttext_vec(corpus,output_file,mode='skipgram'):
    import fasttext
    assert mode in ['skipgram','cbow']
    model = fasttext.train_unsupervised(corpus, model=mode)
    words = model.words
    with open(output_file,'a') as f:
        for word in words:
            word_vec = model.get_word_vector(word)
            word_vec = ' '.join([word] + [str(w) for w in word_vec])
            try:
                f.write(word_vec+'\n')
            except UnicodeEncodeError:
                pass

if __name__ == '__main__':
    args = argparser()
    if args.tokenizer == TOKENIZER[0]:
        from nltk.tokenize import word_tokenize
        tokenize_fn = word_tokenize
    elif args.tokenizer == TOKENIZER[1]:
        from konlpy.tag import Mecab
        tokenize_fn = Mecab().morphs
    
    tokenizer = Tokenizer(token_fn=tokenize_fn,
                          is_sentence=args.is_sentence,
                          max_len=args.max_seq_len)
    
    list_of_tokens = []
    with open(args.corpus,'r',encoding='-utf-8',errors='ignore') as f:
        for line in f:
            text = ' '.join(line.split('\t')[1:]).strip()
            list_of_tokens += tokenizer.tokenize(text)

    vocab = Vocab(list_of_tokens=list_of_tokens,
                  bos_token=args.bos_token,
                  eos_token=args.eos_token,
                  unk_token=args.unk_token,
                  pad_token=args.pad_token,
                  min_freq=args.min_freq,
                  lower=args.lower)
    vocab.build()

    if args.pretrained_vectors:
        output = args.mode + '_vec.txt'
        fasttext_vec(args.input,output,args.mode)
        pretrained_vectors = load_pretrained(output)
        vocab.from_pretrained(pretrained_vectors)
    print(f'vocab has a len of {len(vocab)}')
    
    with open(args.vocab,'wb') as f:
        pickle.dump(vocab,f)
    print(f'vocab saved to {args.vocab}')
    
#%%

