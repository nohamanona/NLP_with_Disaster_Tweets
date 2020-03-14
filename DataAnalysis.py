import pandas as pd
import numpy as np
import stanfordnlp
import itertools
import collections
import warnings
from pyvis.network import Network

#ファイル読み込み
train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

train.head()
test.head()

#データサイズを確認
print('train shape:',train.shape, '\n')
print('test shape:',test.shape, '\n')

def lackdata(df):
    null_val = df.isnull().sum()
    percent = 100*df.isnull().sum()/len(df)
    lackdata = pd.concat([null_val, percent], axis = 1)
    lackdata_table = lackdata.rename(columns = {0:'lack', 1:'%'})
    return lackdata_table

#欠損数を確認
print('lack data of train.csv\n')
print(lackdata(train),'\n')
print('lack data of test.csv\n')
print(lackdata(test))

#keywordとlocationを簡単に分析
#各コラムのワード出現数をファイル出力
print('各コラムのワード出現数をcsv出力\n')
keyword_train = train['keyword'].value_counts(sort = False)
keyword_train.to_csv('keyword_train.csv')
keyword_test = test['keyword'].value_counts(sort = False)
keyword_test.to_csv('keyword_test.csv')

location_train = train['location'].value_counts(sort = False)
location_train.to_csv('location_train.csv')
location_test = test['location'].value_counts(sort = False)
location_test.to_csv('location_test.csv')

#textを分析
#記号が残るとあとで分析できないため消しておく
#twitterなのでハッシュタグ"#"だけ2文字目以降を残す
for k in range(len(train.index)):
    train_sp = train['text'].iloc[k].split()
    for l in range(len(train_sp)):
        if train_sp[l][0] == '#' or train_sp[l][0] == '@':
            train_sp[l] = train_sp[l][1:]
        if train_sp[l].isalpha() == False:
            train_sp[l] = ''
    train_join = ' '.join(train_sp)
    if len(train_join.split()) == 0:
        train_join = 'DUMMY'
    train['text'].iloc[k] = train_join

#trainの'text'をNLP
print('trainデータをNLP&共起ネットワークで解析\n')

print('stanfordnlpで文を単語に分解\n')
sentence_list = []
nlp = stanfordnlp.Pipeline()

warnings.simplefilter('ignore', UserWarning)

nlpdatalist = []
wordlist = []
for sl in range(len(train.index)):
    if sl % 100 == 0:
        print('progress:', sl, '/', len(train.index))
    if len(train['text'].iloc[sl]) < 1:
        continue
    doc = nlp(train['text'].iloc[sl])
    for wl in doc.sentences:
        for wrd in wl.words:
            wordlist.append(wrd.lemma)
    nlpdatalist.append(wordlist)
    wordlist = []


print('itertools&collectionsで単語間の相関を分析\n')
sentence_comb = [list(itertools.combinations(se, 2)) for se in nlpdatalist]
sentence_comb = [[tuple(sorted(wd)) for wd in se] for se in sentence_comb]
target_comb = []
for se in sentence_comb:
    target_comb.extend(se)
ct = collections.Counter(target_comb)
ct.most_common()[:10]
pd.DataFrame([{'first' : i[0][0], 'second' : i[0][1], 'count' : i[1]} for i in ct.most_common()]).to_csv('kyoki.csv', index=False)



print('pyvisで得られた相関を基に共起ネットワークを作成\n')
def kyoki_word_network():
    got_net = Network(height="1000px", width="95%", bgcolor="#FFFFFF", font_color="black", notebook=True)
    # set the physics layout of the network
    #got_net.barnes_hut()
    got_net.force_atlas_2based()
    got_data = pd.read_csv("kyoki.csv")[:150]
    sources = got_data['first']#count
    targets = got_data['second']#first
    weights = got_data['count']#second
    edge_data = zip(sources, targets, weights)
    
    for e in edge_data:
        src = e[0]
        dst = e[1]
        w = e[2]
        
        got_net.add_node(src, src, title=src)
        got_net.add_node(dst, dst, title=dst)
        got_net.add_edge(src, dst, value=w)
    
    neighbor_map = got_net.get_adj_list()
    
    #add neighbor data to node hover data
    for node in got_net.nodes:
        node["title"] += " Neighbors:<br>" + "<br>".join(neighbor_map[node["id"]])
        node["value"] = len(neighbor_map[node["id"]])
    
    got_net.show_buttons(filter_=['physics'])
    return got_net

got_net = kyoki_word_network()
got_net.show("kyoki.html")
