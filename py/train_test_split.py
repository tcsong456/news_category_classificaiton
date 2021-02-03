import argparse
import numpy as np
import pandas as pd

def write_file(file,df):
    with open(file,'a') as f:
        for _,(cat,desc) in df.iterrows():
            line = f'{cat}\t{desc}'
            f.write(line+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ratio',type=float,default=0.7,
                        help='ratio of training samples')
    parser.add_argument('--corpus',type=str,
                        help='the corpus to be splitted')
    args = parser.parse_args()
    train_ratio = args.train_ratio
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError('train ratio must be between 0 and 1')
    else:
        eval_ratio = 1.0 - train_ratio
        category,description = [],[]
        with open(args.corpus,'r') as f:
            for line in f:
                _line = line.split('\t')
                cat,desc = _line[0].strip(),' '.join(_line[1:]).strip()
                category.append(cat),description.append(desc)
        text = np.vstack([np.array(category),np.array(description)]).transpose()
        corpus = pd.DataFrame(text,columns=['lable','text'])
        rand_corpus = corpus.sample(frac=1).reset_index(drop=True)
        n_samples = len(corpus)
        n_train = np.ceil(n_samples*train_ratio).astype(int)
        train = rand_corpus.iloc[:n_train]
        eval = rand_corpus.iloc[n_train:]
        
        write_file('corpus_train.txt',train)
        write_file('corpus_eval.txt',eval)


