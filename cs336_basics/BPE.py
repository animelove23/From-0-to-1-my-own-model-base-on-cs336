from .pretokenization_example import pretoken_and_statistic
from collections import Counter

def bpe_tokenization(input_path, vocab_size, special_tokens):
    vocab = {i:bytes([i]) for i in range(256)}#初始化 vocabulary， 将 0~255 每个可能的 byte value 转成长度为 1 的 bytes 对象
    for i in special_tokens:
        vocab[len(vocab)] = i.encode("utf-8")
    merge_groups = pretoken_and_statistic(input_path)
    merge_way= []
    while vocab_size > len(vocab):#开始合并
        item = list(merge_groups.items())
        counter = Counter()
        for k,v in item:#先查找频率最高的token对
            for j in range(len(k)-1):
                counter[(k[j],k[j+1])] += v# counter里面记录的是token对，后面才会把token对合并，为了方便于记录merge way
        if not counter:
            break
        key, value = max(
            counter.items(),
            key=lambda item: (item[1], item[0]),
        )
        merge_way.append(key)# 合并方法集
        vocab[len(vocab)] = key[0]+key[1]
        new_merge_groups = Counter()
        for k,v in item:#对频率最高的token对进行合并
            new_key = []
            j=0
            while j<len(k):
                element = k[j]
                if j+1 < len(k) and (k[j],k[j+1])== key:
                    element = key[0] + key[1]
                    j+=2
                    new_key.append(element)
                else:
                    new_key.append(element)
                    j += 1
            new_tuple = tuple(new_key)
            new_merge_groups[new_tuple] = v
        merge_groups = new_merge_groups

    return vocab, merge_way




if __name__ == "__main__":
    a,b = bpe_tokenization("/home/lianggon/cs336/data/TinyStoriesV2-GPT4-train.txt",10000,"<|endoftext|>")
    with open("result.txt","w") as f:
        for k,v in a.items():
            f.write(f"{v}\n")

