import argparse
import pandas as pd
import re
import sys

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',type=str,default='corpus/News_Category_Dataset_v2.json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argparser()
    data = pd.read_json(args.corpus,lines=True)
    data = data.loc[:,['category','headline','short_description']]
    
    corpus = data['headline'].str.strip() + '.' + data['short_description'].str.strip()
    labels = data['category'].str.strip()
    
    for i,(text,label) in enumerate(zip(corpus,labels)):
        line = '{}\t{}'.format(label,re.sub(r'\n',' ',text))
        try:
            sys.stdout.write(line+'\n')
        except UnicodeEncodeError:
            pass


