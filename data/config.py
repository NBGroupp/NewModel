from data_util import *

# Tokenizer:
# cut_word_tokenizer: 结巴分词全模式
# cut_word_when_tokenize: 结巴分词普通模式
# cut_char_tokenizer: 按字分词
# filter_name_place_tokenizer: 结巴普通模式分词，pyltp替换人名地名


#-------DATA FILE PATH--------#

# 建立字典语料
VOCAB_DATA_PATH = './corpuses/aihanyu.gz'
# 建立数据语料
TRAIN_DATA_PATH = './corpuses/aihanyu.gz'

#-----BUILD VOCAB OPTIONS-----#

# 字典最大长度，-1 则不作限制
VOCAB_SIZE = 5000
# 建立字典时处理语料方式，见上面的Tokenizer
VOCAB_TOKENIZER = cut_char_tokenizer

#---BUILD TRAIN DATA OPTION---#

# 建立数据时处理语料方式，见上面的Tokenizer
TRAIN_DATA_TOKENIZER = cut_char_tokenizer
# 定长K大小，若为-1则不定长
K = -1
# 容许建立数据的语料中一个句子里面的UNK TOKEN比例
MAX_UNK_PERCENT_IN_SENTENCE = 0
# 带UNK数据在总数据中的比例
UNK_PERCENT_IN_TOTAL_DATA = 0
