# counter 返回的是一个字典，用update可以合并他们
#findall 在运用在大批数据量的文本当中时，会出现堵塞内存的情况，因此应该使用finditer和counter合作，这样就不会出现占用大量闪存写入写出时的问题
#坏处是代码没有很直观，pretokenization和 文本分段结合在了一起
#在需要同时遍历多个可迭代对象时，使用zip，如 for i，j in zip(两个数组)
#tuple可以将可迭代对象变成tuple
#遍历或用索引对字节整体操作都只能得到数字，如b"abcd", 但是如果直接对分开的字节取的话就是字节本身
# 如果想要保持bytes形式就要用切片或者从新用bytes（[]）， encode和decode是对于处在byte或者str状态下的对象
#+= 的时间复杂度对于str来讲是o n方 而join 则是on
#print(ord('你')) 把单个unicode字符转换成整数表达
#print(chr(20320)) 换回来
# unicode 是一个编码标准，Unicode字符对应成整数表达形式
# 但是计算机存储的是byte 一个byte有8bits ， 为此我们采用utf 8 编码形式， 就是根据unicode code point 的大小来判断要把整数的二进制形式放进哪个 bytes 模板
# SiLU = x/1-e**(-x)  sigmoid =  1/1-e**(-x）把目标压到0-1， Silu就是乘了一个x GLU = 𝜎(𝑊1𝑥) ⊙ 𝑊2𝑥 ,
#𝐴 ∈ ℝ𝑚×𝑛 and 𝐵 ∈ ℝ𝑛×𝑝, the matrix-matrix product 𝐴𝐵 requires 2𝑚𝑛𝑝 FLOPs., 这里，对mp矩阵中的每个元素，都需要n次乘法和n-1次加法，有mp个元素，FLOPs近似于2mnp


# 作业中，GPT-2 XL架构的总参数是，1.64B, 由 48 层的transformer 组成，
# 每层 中有两个 RMSNorm，每个 RMSNorm 只有一个长度为 d_model = 1600 的可训练参数 g，因此共有 2 × 1600 个参数。Multi-Head Attention 中包含 W_Q、W_K、W_V 和 W_O 四个权重矩阵，
# 每个矩阵大小都是 1600 × 1600，因此共有 4 × 1600 × 1600 个参数。SwiGLU FFN 中包含三个权重矩阵 W_1、W_2、W_3，
# 因此共有 3 × 1600 × 4288 个参数。所以一个 Transformer Block 的参数量为 2 × 1600 + 4 × 1600 × 1600 + 3 × 1600 × 4288 = 30,825,600
#模型还有 token embedding，参数量为 50,257 × 1600 = 80,411,200；最后还有一个 RMSNorm，参数量为 1600
#1,479,628,800 + 80,411,200 + 1,600 + 80,411,200 = 1,640,452,800