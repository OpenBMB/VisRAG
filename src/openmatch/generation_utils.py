import pandas as pd
import re
import Levenshtein
import editdistance
import sys
import json
import numpy as np
import os
import string 
from collections import Counter
from PIL import Image


WORD_NUMBER_MAP = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                   "five": 5, "six": 6, "seven": 7, "eight": 8,
                   "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
                   "thirteen": 13, "fourteen": 14, "fifteen": 15,
                   "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}

def get_flatten_table(csv_file_path):
    # used for ChartQA
    df = pd.read_csv(csv_file_path)

    # 初始化一个空的字符串来存储结果
    formatted_string = "Table:"

    # 遍历每一列
    for column in df.columns:
        formatted_string += f" {column}"
        for value in df[column]:
            formatted_string += f" | {value}"
        formatted_string += " &"

    # 移除最后一个多余的“ & ”
    formatted_string = formatted_string.rstrip(" &")

    return formatted_string

def preprocess_text(text):
    # 替换换行符和制表符并去除前后空白
    text = text.replace('\n', ' ').replace('\t', ' ').strip()

    # 定义标点符号和模式
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile(r"(\d)(\,)(\d)")

    # 定义缩略词映射
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
        "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", 
        "dont": "don't", "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", 
        "havent": "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", 
        "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", 
        "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", 
        "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", 
        "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", 
        "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", 
        "shed've": "she'd've", "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", 
        "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", 
        "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd", 
        "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll", "someones": "someone's", 
        "somethingd": "something'd", "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll", 
        "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've", "therere": "there're", 
        "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've", "theyll": "they'll", 
        "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've", 
        "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", 
        "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", 
        "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", 
        "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", 
        "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", 
        "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", 
        "youll": "you'll", "youre": "you're", "youve": "you've"
    }

    # 定义数字映射和冠词
    manual_map = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    articles = ['a', 'an', 'the']

    # 处理标点符号
    for p in punct:
        if (p + ' ' in text or ' ' + p in text) or (re.search(comma_strip, text) is not None):
            text = text.replace(p, '')
        else:
            text = text.replace(p, ' ')
    text = period_strip.sub("", text, re.UNICODE)

    # 处理数字和冠词
    words = text.lower().split()
    processed_words = []
    for word in words:
        word = manual_map.get(word, word)
        if word not in articles:
            processed_words.append(word)

    # 处理缩略词
    for idx, word in enumerate(processed_words):
        if word in contractions:
            processed_words[idx] = contractions[word]

    return ' '.join(processed_words)

def is_numeric_data(text):
    try:
        float(text)
        return True
    except:
        return False
    
def is_within_5_percent(responds, answer):
    # used for relaxed accuracy
    # 计算差距的百分比
    answer = float(answer)
    responds = float(responds)
    diff_percentage = abs((responds - answer) / answer) * 100
    
    # 判断是否不超过 5%
    return diff_percentage <= 5


def NLS(pred, truths):
    """计算标准化Levenshtein相似度"""
    if len(pred) == 0:
        return 0

    if pred == 'none':
        return 0

    answers_similarity = [1 - editdistance.eval(truth, pred) / max(len(truth), len(pred)) for truth in truths]
    
    max_similarity = max(answers_similarity)

    anls = max_similarity if max_similarity >= 0.5 else 0 # 0.5 is threshold
    return anls

def normalize_answer(s, question):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    def yesno(text):
        if 'yes' == text[:3] or 'no' == text[:2]:
            text = text.split()[0]
        return text
    def replace_text(text):
        return text.replace('this is ', '').replace('it is ', '').replace('&', ',').replace('and', ',').replace('percent', '').replace('organisation', 'organization').replace('because of', '').replace('because', '').replace('due to', '').replace('hours', 'hrs').replace('minites', 'min')
    def word2number(text):
        words = text.split()
        return ' '.join([str(WORD_NUMBER_MAP[word]) if word in WORD_NUMBER_MAP else word for word in words])
    def remove_unit(text, question):
        if 'how many' in question:
            idx = question.find('how many')
            unit = question[idx+len('how many'):].split()[0]
            text = text.replace(unit, '')
        if 'which' in question:
            idx = question.find('which')
            unit = question[idx+len('which'):].split()[0]
            text = text.replace(unit, '')
        return text
    return word2number(white_space_fix(yesno(remove_articles(remove_punc(remove_unit(replace_text(lower(s)), question))))))

def horizontal_concat(images):
    """
    将传入的PIL图像列表等比例缩放至相同高度，然后进行水平拼接。
    
    参数:
        images (list of PIL.Image): 需要拼接的PIL图像列表。
    
    返回:
        PIL.Image: 拼接后的图像。
    """
    if not images:
        raise ValueError("Image list is empty")

    # 找到所有图片中最大的高度
    max_height = max(i.height for i in images)
    
    # 将所有图片等比例缩放至相同高度
    resized_images = []
    for img in images:
        # 计算缩放比例
        ratio = max_height / img.height
        new_width = int(img.width * ratio)
        resized_image = img.resize((new_width, max_height), Image.Resampling.BICUBIC)
        resized_images.append(resized_image)
    
    # 计算拼接后的总宽度
    total_width = sum(img.width for img in resized_images)

    # 创建新图像
    new_image = Image.new('RGB', (total_width, max_height))

    # 拼接图像
    x_offset = 0
    for img in resized_images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image

def vertical_concat(images):
    """
    将传入的PIL图像列表等比例缩放至相同宽度，然后进行垂直拼接。
    
    参数:
        images (list of PIL.Image): 需要拼接的PIL图像列表。
    
    返回:
        PIL.Image: 拼接后的图像。
    """
    if not images:
        raise ValueError("Image list is empty")
    
    # 找到所有图片中最大的宽度
    max_width = max(i.width for i in images)
    
    # 将所有图片等比例缩放至相同宽度
    resized_images = []
    for img in images:
        # 计算缩放比例
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        resized_image = img.resize((max_width, new_height), Image.Resampling.BICUBIC)
        resized_images.append(resized_image)
    
    # 计算拼接后的总高度
    total_height = sum(img.height for img in resized_images)

    # 创建新图像
    new_image = Image.new('RGB', (max_width, total_height))

    # 拼接图像
    y_offset = 0
    for img in resized_images:
        new_image.paste(img, (0, y_offset))
        y_offset += img.height

    return new_image
