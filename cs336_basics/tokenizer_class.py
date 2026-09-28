import pickle
from token import tok_name

import regex as re
import typing
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        return cls(vocab_filepath, merges_filepath, special_tokens=special_tokens)
    def encode(self,text: str):
        def normal_text(text,byte_to_id):
            PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
            text = re.findall(PAT, text)
            utf_text=[]
            for i in text:
                utf_text.append([bytes([byte]) for byte in i.encode('utf-8')])
            for i,e in self.merges:
                for index,a in enumerate(utf_text):
                    new_utf_text =[]
                    long = 0
                    while long < len(a):
                        if long!=len(a)-1 and a[long]==i and a[long+1] == e:
                            new_utf_text.append(a[long]+a[long+1])
                            long+=1
                        else:
                            new_utf_text.append(a[long])
                        long += 1
                    utf_text[index] = new_utf_text
            byte_to_id = byte_to_id
            token_id = [byte_to_id[token] for i in utf_text for token in i]
            return token_id
        if self.special_tokens is None:
            return normal_text(text)
        byte_to_id = {token: id for id, token in self.vocab.items()}
        pattern = "(" + "|".join(
            re.escape(token) for token in sorted(
                self.special_tokens,
                key=len,
                reverse=True,
            )
        ) + ")"
        parts = re.split(pattern, text)
        token_id = []
        for i in parts:
            if i in self.special_tokens:
                token_id.append(byte_to_id[i.encode('utf-8')])
            else:
                token_id.extend(normal_text(i,byte_to_id))
        return token_id
    def encode_iterable(self, iterable: typing.Iterable[str]) -> typing.Iterator[int]:
        for i in iterable:
            yield from self.encode(i)
    def decode(self, ids: list[int]) -> str:
        word = b''
        for i in ids:
            word.join(self.vocab[i])
        return word.decode("utf-8", errors="replace")

