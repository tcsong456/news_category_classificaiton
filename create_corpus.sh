echo 'building corpus'
if [ ! -f "News_Category_Dataset_v2.json" ];then
  echo "manully download the data from https://www.kaggle.com/rmisra/news-category-dataset and unzip the data"
fi
python py/build_corpus.py --corpus News_Category_Datset_v2.json > corpus.txt
wc -l corpus.txt
head -n 5 corpus.txt
