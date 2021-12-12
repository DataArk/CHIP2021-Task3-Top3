import os
import re
import elasticsearch
import pandas as pd


es_client = elasticsearch.Elasticsearch(hosts=[{"host": "192.168.237.236",
                                                "port": "9200"}])

# 创建索引

index_ = {
    'mappings': {
            'properties': {
                'entity_name': {
                    "type": "text",
#                     "analyzer": "ik_max_word",
#                     "search_analyzer": "ik_smart"
                },
                "surface_name": {
                    "type": "text",
#                     "analyzer": "ik_max_word",
#                     "search_analyzer": "ik_smart"
                }
            }
        }
    }

if es_client.indices.exists(index='icd_diagnose_test_20210601') is not True:
    result = es_client.indices.create(index='icd_diagnose_test_20210601', body=index_)
    
import pandas as pd
icd_df = pd.read_excel(
    './国际疾病分类 ICD-10 北京临床版v601.xlsx',
    header=None, 
    names=['icd_code', 'name'])

icd_df['name'] = icd_df['name'].apply(lambda x: re.sub('"', '', x))
icd_df = icd_df.rename(columns={'name': 'entity_name'})
icd_df['icd_code_length'] = icd_df['icd_code'].apply(lambda x: len(x))
icd_df.sort_values('icd_code_length', ascending = False, inplace=True)
icd_df = icd_df.groupby('entity_name').head(1)
icd_df = icd_df[icd_df['entity_name'] != 'N']
icd_df['icd_code_length'] = icd_df['icd_code'].apply(lambda x: len(x))
icd_df['surface_name'] = icd_df['entity_name']
records = list()

for i, record in enumerate(icd_df.astype(object)
                           .fillna('')
                           .to_dict('records')):
    
    records.append({'index':{'_index':'icd_diagnose_test_20210601'}})
    records.append(record)
    if i % 1000 == 0:
        es_client.bulk(body=records, index="icd_diagnose_test_20210601")
        records = []
_ = es_client.bulk(body=records, index="icd_diagnose_test_20210601")

import os
import re
import tqdm
import torch
import elasticsearch

from tqdm import tqdm
from elasticsearch import Elasticsearch


class DiseaseSearchEngine:
    def __init__(self, ):
        self.es = Elasticsearch(hosts=[{"host": "39.99.190.185", 
                                        "port": "9200"}])

    def search(self, _query: str, size=20):
        dsl = {
            "query": {
                "match": {
                    "surface_name": {
                        'query': _query,
#                         "analyzer": "ik_smart"
                    }
                  }
                },
            "sort": [
                {
                    "_score": {
                        "order": "desc"
                    }
                }
            ]
        }

        result = self.es.search(index='icd_diagnose_test_20210601', body=dsl, size=size)
        return result 
    
icd_name_set = set(icd_df['entity_name'].unique())
name2icd_dict = dict()

for icd_code, entity_name in zip(icd_df['icd_code'], icd_df['entity_name']):
    name2icd_dict[entity_name] = icd_code

train_data_df = pd.read_csv(
    './train.txt',
    sep='\t', 
    header=None, 
    names=['text', 'normalized_result'])

history_list = []
for text_, normalized_result_ in zip(train_data_df['text'], 
                                     train_data_df['normalized_result']):
    for term_ in normalized_result_.split('##'):
        if re.sub('"', '', term_) in name2icd_dict:
            history_list.append([text_, re.sub('"', '', term_), name2icd_dict[re.sub('"', '', term_)]])
            
train_history_df = pd.DataFrame(history_list, columns=['surface_name', 'entity_name', 'icd_code'])

records = list()

for i, record in enumerate(train_history_df.astype(object)
                           .fillna('')
                           .to_dict('records')):
    
    records.append({'index':{'_index':'icd_diagnose_test_20210601'}})
    records.append(record)
    if i % 1000 == 0:
        es_client.bulk(body=records, index="icd_diagnose_test_20210601")
        records = []
        
_ = es_client.bulk(body=records, index="icd_diagnose_test_20210601")

import pandas as pd
import os
import re
import json
from tqdm import tqdm
import pickle

i_to_num_dict = {'i':'1', 'ii':'2', 'iii':'3', 'iv':'4', 'v':'5', 'vi':'6', 'vii':'7', 'viii':'8'}

def match_itomun(substring):
    abbr = re.search('^v?i+v?', substring.groupdict()['pat'])
    if not abbr:
        abbr = re.search('v?i+v?$', substring.groupdict()['pat'])
    if not abbr:
        return substring.group()
    else:
        abbr = abbr.group()
        matched = re.sub(abbr, i_to_num_dict[abbr], substring.groupdict()['pat'])
        return matched

def i_to_num(string):
    if 'i' in string:
        string = re.sub('(?P<pat>[a-zA-Z]+)', match_itomun, string)
    return string

digit_map = {"Ⅳ":"iv", "Ⅲ":"iii", "Ⅱ":"ii", "Ⅰ":"i", "一":"1", "二":"2", "三":"3", "四":"4", "五":"5", "六":"6"}
def clean_digit(string):
    # Ⅳ Ⅲ Ⅱ Ⅰ
    # IV III II I
    # 4 3 2 1
    # 四 三 二 一
    new_string = ""
    for ch in string:
        if ch.upper() in digit_map:
            new_string = new_string + digit_map[ch.upper()]
        else:
            new_string = new_string + ch
    return new_string

greek_lower = [chr(ch) for ch in range(945, 970) if ch != 962]
greek_upper = [chr(ch) for ch in range(913, 937) if ch != 930]
greek_englist = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
                 "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega"]
greek_map = {ch:greek_englist[idx % 24] for idx, ch in enumerate(greek_lower + greek_upper)}
def clean_greek(string):
    new_string = ""
    for ch in string:
        if ch in greek_map:
            new_string = new_string + greek_map[ch]
        else:
            new_string = new_string + ch
    return new_string

prefix_suffix_src = ["部位未特指的", "未特指的", "原因不明的", "意图不确定的", "不可归类在他处", "其他特指的疾患"]
prefix_suffix_tgt = ["部未指", "未指", "不明", "意不", "不归他", "他特指"]
def clean_prefix_suffix(string):
    for idx, replace_str in enumerate(prefix_suffix_src):
        string = string.replace(replace_str, prefix_suffix_tgt[idx])
    return string

other_map = {'＋': '+',
 'pci': '经皮冠状动脉介入治疗',
 'cad': '冠状动脉性心脏病',
 'sle': '系统性红斑狼疮',
 'loa': '左枕前胎位',
 'mp': '支原体',
 'ou': '双眼',
 'mt': '恶性肿瘤',
 'paget': '佩吉特',
 'tpsa': '肿瘤标志物',
 'tc': '血清总胆固醇',
 'pbc': '原发性胆汁型肝硬化',
 'fgr': '胎儿生长受限',
 'barrett': '巴氏',
 'tia': '短暂性脑缺血发作',
 'bowen': '鲍恩',
 'as': '强直性脊柱炎',
 'dic': '弥散性血管内凝血',
 'hcc': '肝细胞癌',
 'ggo': '肺部阴影',
 'cushing': '库欣',
 'ln': '狼疮性肾炎',
 'prl': '泌乳素',
 'copd': '慢性阻塞性肺疾病',
 'mia': '微浸润性腺癌',
 'cea': '癌胚抗原',
 'hpv': '人乳头瘤病毒感染',
 'carcinoma': '恶性上皮肿瘤',
 'iud': '具有子宫内避孕装置',
 'aecopd': '急性加重期慢性阻塞性肺疾病',
 'gvhd': '移植物抗宿主病',
 'crohn': '克罗恩',
 'dixon': '直肠切除术',
 'tsh': '促甲状腺激素',
 'ptca': '冠状动脉腔内血管成形术',
 'ivf': '人工妊娠',
 'rop': '早产儿视网膜病',
 'avnrt': '房室结折返性心动过速',
 'cg': '慢性胃炎',
 'avn': '成人股骨头缺血性坏死',
 'rca': '右冠状动脉',
 'nt': '颈部透明度厚度',
 'nerd': '非糜烂性胃食管反流病',
 'sonk': '自发性膝关节骨坏死',
 'cabg': '冠状动脉搭桥',
 'burrkitt': '伯基特',
 'chd': '冠状动脉粥样硬化性心脏病',
 'hf': '心力衰竭',
 'chdhf': '冠心病心力衰竭',
 'ep': '癫痫',
 'simmond': '西蒙',
 'mgd': '睑板腺功能障碍',
 'fl': '滤泡性淋巴瘤',
 'teson': '特尔松',
 'ra': '类风湿性关节炎',
 'gd': '毒性弥漫性甲状腺肿',
 'poland': '波兰',
 'eb': '疱疹病毒',
 'msi': '微卫星不稳定',
 'pnet': '原始性神经外胚瘤',
 'lutembacher': '卢滕巴赫',
 'acl': '膝关节前交叉韧带',
 'he': '人附睾蛋白',
 'vkh': '伏格特-小柳-原田',
 'le': '红斑狼疮',
 'nyha': '纽约心脏病协会',
 'kt': '克利佩尔-特农纳',
 'rhcc': '复发性肝癌',
 'ige': '免疫球蛋白E',
 'poncet': '篷塞',
 'lst': '大肠侧向发育型肿瘤',
 'cgn': '慢性肾小球肾炎',
 'fsgs': '局灶节段性肾小球硬化',
 'gdm': '妊娠期糖尿病',
 'rsa': '右骶前',
 'htn': '高血压',
 'ncr': '接近完全缓解',
 'hunt': '亨特',
 'ddd': '退变性椎间盘病',
 'alzheimer': '阿尔茨海默',
 'nsclc': '非小细胞肺腺癌',
 'evens': '伊文氏',
 'mikulicz': '米库利奇',
 'ev': '肠病毒',
 'igd': '免疫球蛋白D',
 'chf': '充血性心力衰竭',
 'od': '右眼',
 'ipi': '国际预后指数',
 'dieulafoy': '迪厄拉富瓦',
 'lad': '左前降支',
 'ao': '主动脉',
 'hoffa': '霍法',
 'tunner': '特纳',
 'pagtes': '佩吉特',
 'killip': '基利普',
 'addison': '艾迪生',
 'rett': '雷特',
 'wernicke': '韦尼克',
 'castelman': '卡斯尔曼',
 'goldenhar': '戈尔登哈尔',
 'ufh': '普通肝素',
 'ddh': '发育性髋关节发育不良',
 'stevens': '史蒂文斯',
 'johnson': '约翰逊',
 'athmas': '哮喘',
 'rfa': '射频消融',
 'kippip': '基利普',
 'pancreaticcancer': '胰腺恶性肿瘤',
 'srs': '立体定向放射外科',
 'ama': '抗线粒体抗体',
 'cgd': '慢性肉芽肿病',
 'bmt': '骨髓移植',
 'sd': '脐带血流比值',
 'arnold': '阿诺德',
 'tb': '结核感染',
 'dvt': '下肢深静脉血栓形成',
 'sturge': '斯特奇',
 'weber': '韦伯',
 'smt': '黏膜下肿瘤',
 'ca': '恶性肿瘤',
 'smtca': '粘膜下恶性肿瘤',
 'nse': '神经元特异性烯醇化酶',
 'psvt': '阵发性室上性心动过速',
 'gaucher': '戈谢',
 'fai': '髋关节撞击综合征',
 'lop': '左枕后位',
 'lot': '左枕横位',
 'pcos': '多囊卵巢综合征',
 'sweet': '急性发热性嗜中性皮病',
 'graves': '格雷夫斯',
 'cdh': '先天性髋关节脱位',
 'enneking': '恩内金',
 'leep': '利普',
 'itp': '特发性血小板减少性紫癜',
 'wbc': '白细胞',
 'malt': '粘膜相关淋巴样组织',
 'naoh': '氢氧化钠',
 'fd': '功能性消化不良',
 'ck': '肌酸激酶',
 'hl': '霍奇金淋巴瘤',
 'chb': '慢性乙型肝炎',
 'est': '内镜下十二指肠乳头括约肌切开术',
 'enbd': '内镜下鼻胆管引流术',
 'carolis': '卡罗利斯',
 'lam': '淋巴管肌瘤病',
 'ptcd': '经皮肝穿刺胆道引流术',
 'alk': '间变性淋巴瘤激酶',
 'hunter': '亨特',
 'pof': '卵巢早衰',
 'ems': '子宫内膜异位症',
 'asd': '房间隔缺损',
 'vsd': '室间隔缺损',
 'pda': '动脉导管未闭',
 'stills': '斯蒂尔',
 'ecog': '东部癌症协作组',
 'castlemen': '卡斯尔曼',
 'cgvhd': '慢性移植物抗宿主病',
 'ards': '急性呼吸窘迫综合征',
 'op': '骨质疏松',
 'lsa': '左骶前',
 'afp': '甲胎蛋白',
 'sclc': '小细胞癌',
 'ecg': '心电图',
 'pdl': '细胞程序性死亡配体',
 'mss': '微卫星稳定',
 'masson': '马松',
 'ms': '多发性硬化',
 'tg': '甘油三酯',
 'cmt': '腓骨肌萎缩',
 'ph': '氢离子浓度指数',
 'dlbcl': '弥漫大B细胞淋巴瘤',
 'turner': '特纳',
 'aml': '急性骨髓系白血病',
 'pta': '经皮血管腔内血管成形术',
 'alpers': '阿尔珀斯',
 'tat': '破伤风抗毒素',
 'cavc': '完全性房室间隔缺损',
 'coa': '主动脉缩窄',
 'ggt': '谷氨酰转肽酶',
 'edss': '扩展残疾状态量表',
 'vin': '外阴上皮内瘤变',
 'vini': '外阴上皮内瘤变1',
 'vinii': '外阴上皮内瘤变2',
 'viniii': '外阴上皮内瘤变3',
 'ebv': '疱疹病毒',
 'dcis': '乳腺导管原位癌',
 'gu': '胃溃疡',
 'terson': '特尔松',
 'oa': '骨关节炎',
 'cin': '宫颈上皮内瘤变'
}

def match(substring):
    abbr = re.search('[a-z]+', substring.groupdict()['pat']).group()
    matched = re.sub(abbr, other_map[abbr], substring.groupdict()['pat'])
    return matched

def clean_other(string):
    # oa
    # "＋"="+"
    # aoux not replace ou
    for item in list(other_map.keys()):
        if item == "＋":
            string = re.sub(item, other_map[item], ' '+string+' ')
        else:
            string = re.sub('(?P<pat>[^a-zA-Z]'+item+'[^a-zA-Z])', match, ' '+string+' ')
    return string.strip(' ')

def clean_index(string):
    # 1. 2.
    new_string = ""
    idx = 0
    while idx < len(string):
        ch = string[idx]
        if "0" <= ch <= "9" and idx < len(string) - 1 and string[idx + 1] == ".":
            new_string += " "
            idx += 1
        else:
            new_string += ch
        idx += 1
    return new_string

def clean(string):
    string = string.replace("\"", " ").lower()
    string = clean_index(string)
    string = clean_prefix_suffix(string)
    string = clean_greek(string)
    string = clean_digit(string)
    string = clean_other(string)
    string = i_to_num(string)
    string = clean_other(string)
    return string.lower()

prefix_suffix_src_x = ["恶性","癌", "慢支", "化疗", "皮肤", "胃口", "节育器",
                        "左甲","右甲","腮裂","白内障","小便","停经","积血"]

prefix_suffix_tgt_x = ["恶性肿瘤","癌恶性肿瘤","慢性支气管炎","化学治疗","皮肤和皮下组织", "食欲","避孕环",
                        "左甲状腺","右甲状腺","鳃裂","白内障眼","尿","孕","积血肿"]

def extend_x(string):
    for idx, replace_str in enumerate(prefix_suffix_src_x):
        string = string.replace(replace_str, prefix_suffix_tgt_x[idx])
    return string

import pandas as pd

icd_df2 = pd.read_excel('./国际疾病分类 ICD-10 北京临床版v601.xlsx',header=None, names=['icd_code', 'name'])
icd_df2 = icd_df2.rename(columns={'name': 'entity_name'})
icd_df2['icd_code_length'] = icd_df2['icd_code'].apply(lambda x: len(x))
icd_df2.sort_values('icd_code_length', ascending = False, inplace=True)
icd_df2 = icd_df2.groupby('entity_name').head(1)
icd_df2 = icd_df2[icd_df2['entity_name'] != 'N']
icd_df2['entity_name'] = icd_df2['entity_name'].apply(lambda x: re.sub('"', '', x))
icd_df2['icd_code_length'] = icd_df2['icd_code'].apply(lambda x: len(x))
icd_df2['surface_name'] = icd_df2['entity_name']

icd_df2['surface_name'] = icd_df2['surface_name'].apply(lambda x: clean(x))

def get_same_label(df_):
    if df_['surface_name'] == df_['entity_name']:
        return True
    else:
        return False

icd_df2['is_same'] = icd_df2.apply(lambda x: get_same_label(x), axis=1)

icd_df2 = icd_df2[icd_df2['is_same'] != True].loc[:,['icd_code', 'entity_name', 'surface_name']]

records = list()

for i, record in enumerate(icd_df2.astype(object)
                           .fillna('')
                           .to_dict('records')):
    
    records.append({'index':{'_index':'icd_diagnose_test_20210601'}})
    records.append(record)
    if i % 1000 == 0:
        es_client.bulk(body=records, index="icd_diagnose_test_20210601")
        records = []
_ = es_client.bulk(body=records, index="icd_diagnose_test_20210601")

train_data_df = pd.read_csv('./train.txt', sep='\t', header=None, names=['text', 'normalized_result'])

extra_list = []
for text_, normalized_result_ in zip(train_data_df['text'], 
                                     train_data_df['normalized_result']):
    for term_ in normalized_result_.split('##'):
        if term_ not in name2icd_dict:
            if term_ == 'O':
                continue
            extra_list.append([clean(text_), term_, '0'])
            extra_list.append([clean(term_), term_, '0'])
            
train_extra_df = pd.DataFrame(extra_list, columns=['surface_name', 'entity_name', 'icd_code'])

records = list()

for i, record in enumerate(train_extra_df.astype(object)
                           .fillna('')
                           .to_dict('records')):
    
    records.append({'index':{'_index':'icd_diagnose_test_20210601'}})
    records.append(record)
    if i % 1000 == 0:
        es_client.bulk(body=records, index="icd_diagnose_test_20210601")
        records = []
        
_ = es_client.bulk(body=records, index="icd_diagnose_test_20210601")

import queue


class Match(object):

    def __init__(self, start, end, keyword):
        self.start = start
        self.end = end
        self.keyword = keyword

    def __str__(self):
        return "{0}:{1}={2}".format(self.start, self.end, self.keyword)

    __repr__ = __str__


class State(object):

    def __init__(self, word, deepth):
        self.success = {}
        self.failure = None
        self.emits = dict()
        self.deepth = deepth

    def add_word(self, word):
        if word in self.success:
            return self.success.get(word)
        else:
            state = State(word, self.deepth + 1)
            self.success[word] = state
        return state

    def add_one_emit(self, keyword, value):
        self.emits[keyword] = value

    def add_emits(self, emits):
        if not isinstance(emits, dict):
            raise Exception("keywords need a dict")
        self.emits.update(emits)

    def set_failure(self, state):
        self.failure = state

    def get_transitions(self):
        return self.success.keys()

    def next_state(self, word):
        return self.success.get(word)


class Trie(object):

    def __init__(self, words=None):

        self.root = State("", 0)
        self.root.set_failure(self.root)
        self.is_create_failure = False
        if words:
            self.create_trie(words)

    def create_trie(self, words):
        if isinstance(words, (list, set)):
            for keyword in words:
                self.add_keyword(keyword, '')
            self.create_failure()
        elif isinstance(words, dict):
            for keyword, value in words.items():
                self.add_keyword(keyword, value)
            self.create_failure()
        else:
            raise ValueError('错误的数据类型')

    def add_keyword(self, keyword, value):
        current_state = self.root
        word_list = list(keyword)

        for word in word_list:
            current_state = current_state.add_word(word)
        current_state.add_one_emit(keyword, value)

    def create_failure(self):
        root = self.root
        state_queue = queue.Queue()

        for k, v in self.root.success.items():
            state_queue.put(v)
            v.set_failure(root)

        while (not state_queue.empty()):
            current_state = state_queue.get()
            transitions = current_state.get_transitions()
        
            for word in transitions:
                target_state = current_state.next_state(word)

                state_queue.put(target_state)
                trace_state = current_state.failure

                while trace_state.next_state(word) is None and trace_state.deepth != 0:
                    trace_state = trace_state.failure

                if trace_state.next_state(word) is not None:
                    target_state.set_failure(trace_state.next_state(word))
                    target_state.add_emits(trace_state.next_state(word).emits)
                else:
                    target_state.set_failure(trace_state)
        self.is_create_failure = True

    def get_state(self, current_state, word):
        new_current_state = current_state.next_state(word)

        while new_current_state is None and current_state.deepth != 0:
            current_state = current_state.failure
            new_current_state = current_state.next_state(word)

        return new_current_state

    def match(self, text, allow_over_laps=True):
        matchs = []
        if not self.is_create_failure:
            self.create_failure()

        position = 0
        current_state = self.root
        for word in list(text):
            position += 1
            current_state = self.get_state(current_state, word)
            if not current_state:
                current_state = self.root
                continue
            for mw in current_state.emits:
                m = Match(position - len(mw), position, mw)
                matchs.append(m.keyword)
        return matchs
    
trie_recall_model = Trie(icd_name_set)

