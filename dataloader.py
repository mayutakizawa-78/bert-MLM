import glob
import os
import io
import string
import re
import random
import spacy
import torchtext
from torchtext.vocab import Vectors
from torchtext.legacy import data
import urllib
import zipfile
import tarfile


def get_IMDb_DataLoaders_and_TEXT(max_length=256, batch_size=24):
    """IMDbのDataLoaderとTEXTオブジェクトを取得する。 """
    
    # IMDbデータセットをダウンロード。30秒ほどでダウンロードできます
    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "./data/aclImdb_v1.tar.gz"
    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
    # './data/aclImdb_v1.tar.gz'の解凍　1分ほどかかります

    # tarファイルを読み込み
    tar = tarfile.open('./data/aclImdb_v1.tar.gz')
    tar.extractall('./data/')  # 解凍
    tar.close()  # ファイルをクローズ
    # フォルダ「data」内にフォルダ「aclImdb」というものができます。


    # 訓練データのtsvファイルを作成します
    f = open('./data/IMDb_train.tsv', 'w')

    path = './data/aclImdb/train/pos/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # タブがあれば消しておきます
            text = text.replace('\t', " ")

            text = text+'\t'+'1'+'\t'+'\n'
            f.write(text)

    path = './data/aclImdb/train/neg/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # タブがあれば消しておきます
            text = text.replace('\t', " ")

            text = text+'\t'+'0'+'\t'+'\n'
            f.write(text)

    f.close()

   # テストデータの作成
    f = open('./data/IMDb_test.tsv', 'w')

    path = './data/aclImdb/test/pos/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # タブがあれば消しておきます
            text = text.replace('\t', " ")

            text = text+'\t'+'1'+'\t'+'\n'
            f.write(text)

    path = './data/aclImdb/test/neg/'
    for fname in glob.glob(os.path.join(path, '*.txt')):
        with io.open(fname, 'r', encoding="utf-8") as ff:
            text = ff.readline()

            # タブがあれば消しておきます
            text = text.replace('\t', " ")

            text = text+'\t'+'0'+'\t'+'\n'
            f.write(text)
    f.close()

    def preprocessing_text(text):
        # 改行コードを消去
        text = re.sub('<br />', '', text)

        # カンマ、ピリオド以外の記号をスペースに置換
        for p in string.punctuation:
            if (p == ".") or (p == ","):
                continue
            else:
                text = text.replace(p, " ")

        # ピリオドなどの前後にはスペースを入れておく
        text = text.replace(".", " . ")
        text = text.replace(",", " , ")
        return text

    # 分かち書き（今回はデータが英語で、簡易的にスペースで区切る）
    def tokenizer_punctuation(text):
        return text.strip().split()


    # 前処理と分かち書きをまとめた関数を定義
    def tokenizer_with_preprocessing(text):
        text = preprocessing_text(text)
        ret = tokenizer_punctuation(text)
        return ret


    # データを読み込んだときに、読み込んだ内容に対して行う処理を定義します
    # max_length
    TEXT = data.Field(sequential=True, tokenize=tokenizer_with_preprocessing, use_vocab=True,
                                lower=True, include_lengths=True, batch_first=True, fix_length=max_length, init_token="<cls>", eos_token="<eos>")
    LABEL = data.Field(sequential=False, use_vocab=False)

    # フォルダ「data」から各tsvファイルを読み込みます
    train_val_ds, test_ds = data.TabularDataset.splits(
        path='./data/', train='IMDb_train.tsv',
        test='IMDb_test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

    # data.Datasetのsplit関数で訓練データとvalidationデータを分ける
    train_ds, val_ds = train_val_ds.split(split_ratio=0.8, random_state=random.seed(1234))
    
    # torchtextで単語ベクトルとして英語学習済みモデルを読み込みます
    filename='wiki-news-300d-1M.vec' # 680MBほど
    fpath = 'data/%s'%filename
    zipfpath = 'data/%s.zip'%filename
    
    if not os.path.exists(zipfpath):
        url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/%s.zip'%filename
        urllib.request.urlretrieve(url, zipfpath)
        
    # ./data/wiki-news-300d-1M.vec.zipを解凍する
    zip = zipfile.ZipFile(zipfpath)
    zip.extractall("./data/")  # ZIPを解凍
    zip.close()  # ZIPファイルをクローズ
    # ./data/wiki-news-300d-1M.vecができる
    
    english_fasttext_vectors = Vectors(name=fpath)

    # ベクトル化したバージョンのボキャブラリーを作成します
    TEXT.build_vocab(train_ds, vectors=english_fasttext_vectors, min_freq=10)

    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
    train_dl = data.Iterator(
        train_ds, batch_size=batch_size, train=True)

    val_dl = data.Iterator(
        val_ds, batch_size=batch_size, train=False, sort=False)

    test_dl = data.Iterator(
        test_ds, batch_size=batch_size, train=False, sort=False)

    return train_dl, val_dl, test_dl, TEXT
