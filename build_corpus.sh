echo 'building corpus'
if [ ! -d "corpus" ];then
  mkdir corpus
fi
if [ ! -r "News_Category_Dataset_v2.json.zip" ];then
  echo "download the data first"
  exit 1
fi
if [ ! -f "corpus/News_Category_Dataset_v2.json" ];then
  unzip News_Category_Dataset_v2.json.zip -d corpus
fi
python build_corpus.py --corpus corpus/News_Category_Dataset_v2.json > corpus/corpus.txt
wc -l corpus/corpus.txt
head -n 5 corpus/corpus.txt
