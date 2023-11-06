[toc]

# 学习资料
突然发现看过学过的东西很容易忘，好记性不如烂笔头，整理一下日常学习笔记，忘了好来翻一翻

## 1. ML

## 2. DL

### 2.1 Basic

#### 2.1.1 Transformer

* [Attention is all you need! ](https://arxiv.org/pdf/1706.03762.pdf)

architechture:
![Alt text](/figs/transformer.png)

相较于RNN，原话：“This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.” 说人话就是，RNN是sequential，不能并行计算，序列长度很长的话，就会要求很大内存。

“allowing modeling of dependencies without regard to their distance in the input or output sequences” 还能消除长距离依赖

self-attention: "Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."  结合一个序列的不同位置来计算representation of the sequence

### 2.2 NLP

#### 2.2.1 Bert

### 2.3 CV

## 3. leetcode