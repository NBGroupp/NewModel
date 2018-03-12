from data_util import *

# Tokenizer:
# cut_word_tokenizer: 结巴分词全模式
# cut_word_when_tokenize: 结巴分词普通模式
# cut_char_tokenizer: 按字分词
# filter_name_place_tokenizer: 结巴普通模式分词，pyltp替换人名地名
# filter_name_place_char_tokenizer: ptlyp替换人名地名后按字分词


#-------DATA FILE PATH--------#

# 建立字典语料
VOCAB_DATA_PATH = '/media/gyh/Files/语料库/corpuses/wiki'
# 使用外来字典文件，不受上方字典最大长度限制
VOCAB_FILE = '/home/gyh/srtp/NewModel/data//vocab.100000'
# 建立数据语料
TRAIN_DATA_PATH = '/media/gyh/Files/语料库/corpuses/all.txt'
# 建立形近字表语料
NEAR_WORDS_DATA_PATH = '/media/gyh/Files/语料库/corpuses/形近字表(1).txt'

#-----BUILD VOCAB OPTIONS-----#

# 字典最大长度，-1 则不作限制
VOCAB_SIZE = 10000
# 建立字典时处理语料方式，见上面的Tokenizer
VOCAB_TOKENIZER = filter_name_place_char_tokenizer

#---BUILD TRAIN DATA OPTION---#

# 建立数据时处理语料方式，见上面的Tokenizer
TRAIN_DATA_TOKENIZER = filter_name_place_char_tokenizer
# 定长K大小，若为-1则不定长
K = -1
# 容许建立数据的语料中一个句子里面的UNK TOKEN比例
MAX_UNK_PERCENT_IN_SENTENCE = 0.1
# 带UNK数据在总数据中的比例
UNK_PERCENT_IN_TOTAL_DATA = 1
# 是否将运行数据写入文件
OPERATE_IN_FILE = True

# 混错比例
LOW_ERROR_RATIO = 0
UP_ERROR_RATIO = 0.1
# 是否向量化
VECTORIZE = True
