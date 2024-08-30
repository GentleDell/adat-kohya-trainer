import os
from os.path import join as pjoin
from glob import iglob, glob
from textwrap import TextWrapper
import time
from IPython import display

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
from PIL import Image


def load_tokens(path, skip_tokens=[]):
    img_token_list = []
    iter_token_files = iglob(pjoin(path, "*.txt"))
    for file in iter_token_files:
        with open(file, "r") as f:
            tokens = f.readline()
        tokens = tokens.split(", ")
        tokens = list(filter(lambda x: x not in skip_tokens, tokens))
        img_token_list.append([file, *tokens])

    img_files = glob(pjoin(path, "*.png")) + glob(pjoin(path, "*.jpg"))
    assert len(img_token_list) == len(
        img_files
    ), "num of label does not match the num of images"

    return img_token_list


def handle_as_dataframe(token_list):
    img_token_df = pd.DataFrame(token_list)
    print(img_token_df)


def sklearn_tfidf(token_list):
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(token_list)
    # 获取词袋中所有文本关键词
    words = vectorizer.get_feature_names_out()

    # 类调用
    transformer = TfidfTransformer()
    # 将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)

    tfidf_array = tfidf.toarray()
    freq_array = X.toarray()

    return tfidf_array, words, freq_array


# top_k freq of the whole dataset
def visualize_top_freq(complete_freq, top_k, all_words):
    all_freq = complete_freq.sum(axis=0) / complete_freq.shape[0]
    top_k_idx = all_freq.argsort()[-top_k:]
    top_k_words = all_words[top_k_idx]
    top_k_value = all_freq[top_k_idx]
    plt.figure()
    plt.title(f"TOP {top_k} mostly mentioned words")
    plt.bar(top_k_words, top_k_value)
    plt.xticks(rotation=30)
    plt.xlabel("words (tokens)")
    plt.ylabel("normalized frequency")
    plt.grid()
    plt.show()


# 简单的搜索功能
def search_img_given_token():
    pass


def randomly_check_labeling_quality(img_tokens_list, percent=20, seed=0):
    assert (percent < 100) and (
        percent > 0
    ), "checking percentage should be less than 100"

    rng = np.random.default_rng(seed)

    num_img = len(img_tokens_list)
    selected_indices = rng.choice(
        num_img, size=int(num_img * percent / 100), replace=False
    )

    checking_list = {}
    tw = TextWrapper()
    tw.width = 25
    for idx in selected_indices:
        filepath = os.path.splitext(img_tokens_list[idx][0])[0]
        token = img_tokens_list[idx][1:]
        # Image lib can handle data format and suffix mismatch
        image = np.asarray(Image.open(filepath + ".png"))
        checking_list["filepath"] = {"image": image, "token": token}

        text = ",".join(token)

        plt.figure()
        plt.imshow(image)
        plt.text(
            image.shape[1] + 10,
            0,
            "\n".join(tw.wrap(text)),
            horizontalalignment="left",
            verticalalignment="top",
            multialignment="left",
        )
        plt.axis("off")
        # display the figure
        display.display(plt.gcf())

        print(f"Checking Image: {filepath}")

        time.sleep(0.5)
        input("press Enter to continue... OR use cell stop button to stop \n")
        display.clear_output(wait=True)
