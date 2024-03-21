import argparse
import re
import inflect


def to_digit(digit):
    i = inflect.engine()
    if digit.isdigit():
        output = i.number_to_words(digit)
    else:
        output = digit
    return output


# 分词，小写，获得词源
def process_token(str):
    # 把&转化为有意义的and
    str = str.replace("&", "and")
    # 去掉非数字字母空格符号，比如标点
    str = re.sub(r'[^\w\s]', ' ', str)
    # 处理下划线
    str = str.replace('_', ' ')
    # 处理驼峰命名的词，切分成多个单词
    str = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', str)
    tokens = str.split()
    # 将数字转化为英文单词
    tokens = [to_digit(x) for x in tokens]
    # 转化为小写
    tokens = [x.lower() for x in tokens]

    return ' '.join(tokens)


def get_args():
    parser = argparse.ArgumentParser(
        """Get file summary of each project""")

    parser.add_argument("--pred_path", type=str, default="./predictions/pred.txt")
    parser.add_argument("--target_path", type=str, default="./predictions/target.txt")
    parser.add_argument("--pred_output", type=str, default="./predictions/pred_token.txt")
    parser.add_argument("--target_output", type=str, default="./predictions/target_token.txt")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_args()
    pred_results = []
    target_results = []
    with open(config.pred_path, 'r', encoding='utf-8') as pred, open(config.target_path, 'r', encoding='utf-8') as target:
        # 逐行读取数据
        for line in pred:
            line = process_token(line.strip())
            pred_results.append(line)
        for line in target:
            line = process_token(line.strip())
            target_results.append(line)
    with open(config.pred_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(pred_results))
    with open(config.target_output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_results))