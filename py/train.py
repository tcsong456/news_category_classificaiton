import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from tokenee import Tokenizer
from create_corpus import Corpus
from model import CBOWClassifier,LSTMClassifier
import logzero
import logging
import os
import warnings
warnings.filterwarnings('ignore')
from azureml.core.run import Run

TOKENIZER = ('treebank','mecab')
MODE = ('lstm','cbow')

def parseargs():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model_name',type=str,default='news_clf_model.pt',
        help='the name of the model')
    arg('--cuda',type=str,default='true',
        help='wether to use gpu or not')
    arg('--batch_size_train',type=int,default=8,
        help='batch size used for training')
    arg('--batch_size_eval',type=int,default=8,
        help='batch size used for validation')
    arg('--tokenizer',type=str,default='treebank',
        choices=['treebank','mecab'],
        help='the tokenizer to use')
    arg('--vocab',type=str,
        help='the vocabulary to be used')
    arg('--is_sentence',type=str,default='false',
        help='if the paragraph has already been transformed into sentences')
    arg('--max_seq_len',type=int,default=1024,
        help='max number of tokens allowed')
    arg('--train_corpus',type=str,
        help='the path to train corpus')
    arg('--eval_corpus',type=str,
        help='the path to eval corpus')
    arg('--mode',type=str,default='skipgram',
        choices=['lstm','cbow'],
        help='the mode to be used for model')
    arg('--hidden_size',type=int,default=128,
        help='number of cells at hidden layer')
    arg('--num_layers',type=int,default=2,
        help='number of stacked lstm layers')
    arg('--dropout',type=float,default=0.5,
        help='dropout rate')
    arg('--embedding_size',type=int,default=100,
        help='embedding dim for word vectors')
    arg('--embedding_trainable',type=str,default='true',
        help='whether the embedding vector is trainable')
    arg('--use_word_embedding',type=str,default='true',
        help='whethe to use provided word embedding')
    arg('--bidirectional',action='store_true',
        help='if bi-directional lstm is used')
    arg('--learning_rate',type=float,default=0.0001,
        help='learning rate of optimizer')
    arg('--epochs',type=int,default=50,
        help='number of epoches for training')
    arg('--save_path',type=str,
        help='the path to save the model')
    args = parser.parse_args()
    return args

def costume_logger(name):
    formatter = logging.Formatter('%(asctime)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    logger = logzero.setup_logger(name=name,
                                  formatter=formatter,
                                  level=logging.INFO)
    return logger
 
run = Run.get_context()
def single_train(args,
                 epoch,
                 model,
                 loss_fn,
                 optimizer,
                 train_loader,
                 ):
    n_samples = len(train_loader.dataset)
    rounds = np.ceil(n_samples / args.batch_size_train)
    tq = tqdm(train_loader,total=rounds)
    total_losses,total_acc = 0,0
    model.train()
    for texts,targets in tq:
        preds = model(texts)
        loss = loss_fn(preds,targets)
        total_losses += loss.item()
        acc = (np.argmax(preds.cpu().data,axis=-1) == targets.cpu().data).sum()
        total_acc += acc.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_acc = total_acc / n_samples
    avg_loss = total_losses / rounds 
    run.log('train_avg_acc',np.round(avg_acc,3))
    run.parent.log('train_avg_acc',np.round(avg_acc,3))
    logger.info(f'train: epoch:{epoch},avg loss:{avg_loss:.5f},avg acc:{avg_acc:.3f}')

def single_eval(args,
                epoch,
                model,
                loss_fn,
                eval_loader
                 ):
    n_samples = len(eval_loader.dataset)
    rounds = np.ceil(n_samples / args.batch_size_eval)
    tq = tqdm(eval_loader,total=rounds)
    total_acc,total_losses = 0,0
    model.eval()
    with torch.no_grad():
        for texts,targets in tq:
            preds = model(texts)
            acc = (np.argmax(preds.cpu().data,axis=-1) == targets.cpu().data).sum()
            total_acc += acc.item()
            loss = loss_fn(preds,targets)
            total_losses += loss.item()
            
        avg_loss = total_losses / rounds
        avg_acc = total_acc / n_samples
        run.log('eval_avg_acc',np.round(avg_acc,3))
        run.parent.log('eval_avg_acc',np.round(avg_acc,3))
        logger.info(f'eval: epoch:{epoch},avg loss:{avg_loss:.5f},avg acc:{avg_acc:.3f}')

if __name__ == '__main__':
    logger = costume_logger('news_clf')
    args = parseargs()
    args_dict = vars(args)
    for key,value in args_dict:
        run.log(key,value)
        run.parent.log(key,value)
    vocab = args.vocab
        
    args.tokenizer = args.tokenizer.lower()
    if args.tokenizer == TOKENIZER[0]:
        from nltk.tokenize import word_tokenize
        tokenize_fn = word_tokenize
    elif args.tokenizer == TOKENIZER[1]:
        from konlpy.tag import Mecab
        tokenize_fn = Mecab.morphs()
    else:
        raise ValueError(f'{args.tokenizer} is not supported!')
    
    tokenizer = Tokenizer(token_fn=tokenize_fn,
                          is_sentence=args.is_sentence,
                          max_len=args.max_seq_len,
                          vocab=vocab)
    train_dataset = Corpus(corpus_path=args.train_corpus,
                           tokenizer=tokenizer,
                           cuda=args.cuda)
    eval_dataset = Corpus(corpus_path=args.eval_corpus,
                          tokenizer=tokenizer,
                          cuda=args.cuda)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size_train,
                              shuffle=True)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size_eval,
                             shuffle=False)
    
    args.mode = args.mode.lower()
    if args.mode == MODE[0]:
        model = LSTMClassifier(input_size=len(vocab),
                               hidden_size=args.hidden_size,
                               output_size=len(train_dataset.ltoi),
                               num_layers=args.num_layers,
                               dropout=args.dropout,
                               embedding_size=args.embedding_size,
                               embedding_trainable=args.embedding_trainable,
                               bidirectional=args.bidirectional,
                               embedding_weight=vocab.word_embeddings if args.use_word_embedding=='true' else None)
    elif args.mode == MODE[1]:
        model = CBOWClassifier(input_size=len(vocab),
                               embedding_size=args.embedding_size,
                               hidden_size=args.hidden_size,
                               output_size=len(eval_dataset.ltoi),
                               embedding_trainable=args.embedding_trainable,
                               dropout=args.dropout,
                               embedding_weight=vocab.word_embeddings if args.use_word_embedding=='true' else None)
    else:
        raise ValueError(f'{args.mode} is not supported')
    
    loss_fn = nn.NLLLoss()
    optimizer = Adam(filter(lambda x:x.requires_grad,model.parameters()),lr=args.learning_rate)
    
    if args.cuda:
        model = model.cuda()
    
    for epoch in range(args.epochs):
        print('training starts!')
        single_train(args=args,
                     epoch=epoch,
                     model=model,
                     loss_fn=loss_fn,
                     optimizer=optimizer,
                     train_loader=train_loader)
        single_eval(args=args,
                    epoch=epoch,
                    model=model,
                    loss_fn=loss_fn,
                    eval_loader=eval_loader)
    
    torch.save(model,os.path.join(args.save_path,args.model_name))        
            
            
    

    #%%
