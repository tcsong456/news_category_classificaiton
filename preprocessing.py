import argparse
import sys
import re

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',type=str,default='corpus/corpus.txt')
    args = parser.parse_args()
    return args

def cleaning(text):
    text = re.sub(r'[\U00010000-\U0010ffff][\u20000000-\u2fffffff][\U0001f000-\U0001ffff]', '', text) # Clean emoji
    text = re.sub(r'<.*?>', '', text) # Clean HTML tag
    text = re.sub(r'http\S+', '<url>', text) # url -> <url> token
    text = re.sub(r'[\w._-]+[@]\w+[.]\w+', '<email>', text) # email -> <email> token
    text = re.sub(r'\d+[-.]\d{3,4}[-.]\d{3,4}', '<pnum>', text) # phone number -> <pnum> token
    text = re.sub(r'[!]{2,}', '!', text) # multiple !s -> !
    text = re.sub(r'[!]{2,}', '?', text) # multiple ?s -> ?
    text = re.sub(r'[-=+,#:^$@*\"※~&%ㆍ』┘\\‘|\(\)\[\]\`\'…》]','', text) # Clean special symbols

    return text

if __name__ == '__main__':
    args = argparser()
    with open(args.corpus,'r') as f:
        for line in f:
            lines = line.split('\t')
            label,text = lines[0],' '.join(lines[1:])
            text = cleaning(text)
            if len(text) > 0:
                line = '{}\t{}'.format(label,re.sub(r'\n',' ',text))
                sys.stdout.write(line+'\n')
    
    #%%


