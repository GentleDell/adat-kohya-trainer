from helper import (
    load_tokens,
    sklearn_tfidf,
    visualize_top_freq,
    randomly_check_labeling_quality,
)

PATH = "./data/"
TOP_K = 10
RANDOM_CHECK_PERCENT = 50
SEED = 42

SKIP_TOKEN = ["1girl", "solo"]

img_tokens_list = load_tokens(PATH, SKIP_TOKEN)

# 提取token的列表
token_list = [" ".join(item[1:]) for item in img_tokens_list]

# 提取各图TF-IDF关键词
tf_idf, words, freq = sklearn_tfidf(token_list)

# 展示TOP K的关键词，用于评估整体标注质量
# 目前使用的自动标注器不是专用于服装的，因此会有很多其他细节的高频标注词
visualize_top_freq(freq, TOP_K, words)
input()

try:
    randomly_check_labeling_quality(img_tokens_list, RANDOM_CHECK_PERCENT, SEED)
except KeyboardInterrupt:
    print("stop checking")

# for idx in range(tf_idf.shape[0]):
#     top_tokens = words[tf_idf[idx].argsort()[-TOP_K:]]
#     print(top_tokens)

pass
