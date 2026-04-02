# 第二章：字节对编码（BPE）分词器（Tokenizer）

在作业的第一部分，我们将训练并实现一个字节级字节对编码（BPE）分词器（tokenizer）\[Sennrich et al., 2016; Wang et al., 2019]。具体而言，我们会把任意（Unicode）字符串表示成一个字节序列，并在该字节序列上训练 BPE 分词器。之后，我们会用这个分词器把文本（字符串）编码为 token（一个整数序列），以便用于语言建模。

## 2.1 Unicode 标准

Unicode 是一种文本编码标准，用于把字符映射到整数代码点（code point）。截至 Unicode 16.0（2024 年 9 月发布），该标准在 168 种文字系统（scripts）中定义了 154,998 个字符。例如，字符 “s” 的代码点是 115（通常记为 U+0073，其中 U+ 是惯例前缀，0073 是 115 的十六进制表示），字符 “୤” 的代码点是 29275。在 Python 中，可以用 `ord()` 把单个 Unicode 字符转换为整数表示；`chr()` 则把整数 Unicode 代码点转换为包含对应字符的字符串。

```python
>>> ord('୤')
29275
>>> chr(29275)
'୤'
```

**问题（unicode1）：理解Unicode（1分）**

(a) `chr(0)`返回什么Unicode字符？

交付物：一句话回答。

(b) 这个字符的字符串表示（`__repr__()`）与它的打印表示有什么不同？

交付物：一句话回答。

(c) 当这个字符出现在文本中时会发生什么？在Python解释器中尝试以下操作可能会有所帮助，看看它是否符合您的预期：

```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

交付物：一句话回答。

## 2.2 Unicode 编码

虽然 Unicode 标准定义了从字符到代码点（整数）的映射，但直接在 Unicode 代码点上训练分词器并不现实：词表会过大（约 150K 项）且非常稀疏（许多字符很少出现）。因此我们改用 Unicode 编码（encoding），把 Unicode 字符转换为字节序列。Unicode 标准定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网最主流的编码（超过 98% 的网页）。

要把 Unicode 字符串编码为 UTF-8，可以使用 Python 的 `encode()`。要查看 Python `bytes` 对象对应的字节值，可以对其迭代（例如调用 `list()`）。最后，用 `decode()` 把 UTF-8 字节串解码回 Unicode 字符串。

```python
>>> test_string = "hello! ŶƶƎƄƒ !"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! ŶƶƎƄƒ !
```

把 Unicode 代码点转换成字节序列（例如使用 UTF-8）后，本质上就是把一串代码点（0 到 154,997 范围内的整数）变换为一串字节值（0 到 255 范围内的整数）。256 大小的字节词表更容易处理。采用字节级 tokenization 时，我们不必担心词表外（out-of-vocabulary, OOV）token，因为任何输入文本都能表示为 0 到 255 的整数序列。

**问题（unicode2）：Unicode编码（3分）**

(a) 与UTF-16或UTF-32相比，优先选择在UTF-8编码的字节上训练分词器的一些原因是什么？比较这些编码对各种输入字符串的输出可能会有所帮助。

交付物：一到两句话回答。

(b) 考虑以下（不正确的）函数，该函数旨在将UTF-8字节字符串解码为Unicode字符串。为什么这个函数是不正确的？提供一个产生不正确结果的输入字节字符串的示例。

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

交付物：一个示例输入字节字符串，其中`decode_utf8_bytes_to_str_wrong`产生不正确的输出，并附带一句话解释为什么该函数不正确。

(c) 给出一个不能解码为任何Unicode字符的两字节序列。

交付物：一个示例，附带一句话解释。

## 2.3 子词（Subword）Tokenization

尽管字节级 tokenization 能缓解词级分词器的 OOV 问题，但把文本切分成字节会产生非常长的输入序列，从而减慢模型训练：一个包含 10 个单词的句子，在词级语言模型里可能只有 10 个 token，但在字符级/字节级模型中可能会变成 50 个甚至更多 token（取决于单词长度）。处理更长的序列意味着每一步需要更多计算；同时，基于字节序列的语言建模也更困难，因为序列更长会在数据中引入更长程的依赖。

子词 tokenization 介于词级分词器与字节级分词器之间。注意：字节级分词器的词表只有 256 个条目（字节值为 0 到 255）。子词分词器用更大的词表换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在训练文本中频繁出现，把它作为一个词表条目，就能把这个原本长度为 3 的 token 序列压缩为单个 token。

我们如何选择这些要加入词表的子词单元？Sennrich等人\[2016]提出使用字节对编码（BPE；Gage，1994）：这是一种压缩算法，会迭代地把出现频率最高的一对字节合并为一个新的 token（使用一个新的、未使用的索引表示）。该算法通过不断向词表添加这些合并得到的子词 token 来最大化对输入序列的压缩——如果某个词在训练文本中出现足够多次，它最终可能会被表示为单个子词 token。

使用 BPE 构建词表的子词分词器通常称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器：词表中的条目要么是单个字节，要么是合并后的字节序列，从而兼顾了 OOV 处理与更短、更可训练的输入序列长度。构建 BPE 分词器词表的过程称为 “training”（训练）BPE 分词器。

## 2.4 BPE 分词器训练

BPE 分词器训练过程包括三个主要步骤。

**词表初始化（Vocabulary initialization）** 分词器词表是从字节串 token 到整数 ID 的一一映射。由于我们训练的是字节级 BPE 分词器，初始词表就是所有可能的字节。字节一共有 256 种取值，因此初始词表大小为 256。

**预分词（Pre-tokenization）** 有了词表之后，从原理上讲，你可以统计文本中相邻字节对出现的频率，并从最频繁的字节对开始合并。然而这会非常昂贵：每做一次合并都需要再把整个语料库扫一遍去统计。此外，如果直接在全语料上跨任意位置合并字节，还可能得到只在标点上不同的 token（例如 `dog!` vs. `dog.`），它们会被映射到完全不同的 token ID，尽管它们在语义上可能非常接近。

为此，我们会先对语料做预分词。可以把它理解为一种粗粒度的切分，用于更高效地统计相邻符号对出现的频率。例如，单词 `'text'` 可能是一个出现 10 次的 pre-token。在统计 `'t'` 和 `'e'` 的相邻次数时，我们只需要知道 `'text'` 里 `'t'` 与 `'e'` 相邻，并把计数加 10，而无需每次都在整段语料中逐字符遍历。由于我们训练的是字节级 BPE，每个 pre-token 都表示为 UTF-8 字节序列。

Sennrich等人\[2016]的原始BPE实现通过简单地按空格分割（即`s.split(" ")`）进行预分词。相比之下，我们将使用基于正则表达式的预分词器（由GPT-2使用；Radford等人，2019）来自github.com/openai/tiktoken/pull/234/files：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

交互式地使用此预分词器切分一些文本，可能会有助于更好地理解其行为：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```

然而，在代码中使用时，您应该使用`re.finditer`来避免在构建从 pre-token 到其计数的映射时，把所有 pre-token 都存到内存中。

**计算 BPE merges** 现在我们已经把输入文本转换为 pre-token，并把每个 pre-token 表示为 UTF-8 字节序列，就可以计算 BPE merges（即训练 BPE 分词器）。

从高层来看，BPE 算法会迭代地统计每一种相邻字节对的频次，并找出频次最高的一对（“A”，“B”）。随后把所有出现的这对（“A”，“B”）合并，用一个新的 token “AB” 替换之，并把新 token 加入词表。因此，训练结束后的词表大小等于初始词表大小（这里为 256）加上训练过程中执行的 merge 次数。为提升训练效率，我们不考虑跨越 pre-token 边界的字节对。<sup>2</sup> 当多个字节对频次并列时，按字典序（lexicographical order）选择更大的那一对作为确定性的 tie-breaker。例如，如果（“A”，“B”）、（“A”，“C”）、（“B”，“ZZ”）和（“BA”，“A”）并列最高频，我们会合并（“BA”，“A”）：

```python
>>> max([("A","B"), ("A","C"), ("B","ZZ"), ("BA","A")])
('BA', 'A')
```

**特殊 token（Special tokens）** 有些字符串（例如 `<|endoftext|>`）用于编码元数据（例如文档边界）。在编码文本时，通常希望把某些字符串当作“特殊 token”，使它们永远不会被拆分成多个 token（即始终作为一个 token 保留）。例如，序列结束标记 `<|endoftext|>` 应始终被编码为一个整数 ID，这样我们在生成时才知道何时停止。这些特殊 token 必须加入词表，从而拥有固定的 token ID。

Sennrich等人\[2016]的算法1包含一个低效的 BPE 分词器训练实现（基本上遵循我们上面概述的步骤）。作为第一个练习，实现并测试该函数可能有助于检验你的理解。

**示例（bpe\_example）：BPE 训练示例**

这是来自Sennrich等人\[2016]的风格化示例。考虑一个由以下文本组成的语料库：

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

并且词表中包含一个特殊 token `<|endoftext|>`。

**词表** 我们用特殊 token `<|endoftext|>` 和 256 个字节值初始化词表。

**预分词** 为简单起见并专注于 merge 过程，我们在这个示例中假设预分词只是按空格切分。当我们进行预分词并计数时，最终得到频率表：

```python
{low: 5, lower: 2, widest: 3, newest: 6}
```

<sup>2</sup> 请注意，原始BPE公式\[Sennrich等人，2016]指定包含词尾标记。在训练字节级BPE模型时，我们不添加词尾标记，因为所有字节（包括空格和标点符号）都包含在模型的词表中。由于我们显式表示空格和标点符号，学习到的 BPE merges 会自然反映这些词边界。

将其表示为`dict[tuple[bytes], int]`很方便，例如`{(l,o,w): 5 …}`。请注意，即使是单个字节在Python中也是一个`bytes`对象。Python中没有`byte`类型来表示单个字节，就像没有`char`类型来表示单个字符一样。

**Merges** 我们首先查看每个连续的字节对，并把它们出现的 pre-token 的频次加总起来：`{lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}`。对 `('es')` 和 `('st')` 并列，因此我们取字典序更大的那对 `('st')`。随后合并各 pre-token，得到 `{ (l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6 }`。

在第二轮中，我们看到`(e, st)`是最常见的对（计数为9），我们将合并为`{ (l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6 }`。继续这样，我们最终得到的合并序列将是`['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']`。

如果我们进行6次合并，我们有`['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']`，我们的词表条目将是`[<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]`。

使用这套词表与 merges，单词 `newest` 会被分词为 `[ne, west]`。

## 2.5 实验：BPE 分词器训练

让我们在TinyStories数据集上训练一个字节级 BPE 分词器。查找/下载数据集的说明可以在第1节中找到。在开始之前，我们建议查看TinyStories数据集，以了解数据中包含的内容。

**并行化预分词** 你会发现主要瓶颈在预分词步骤。可以使用内置库 `multiprocessing` 并行化以加速。具体来说，建议在并行预分词实现里对语料进行分块，并确保每个分块边界都落在某个特殊 token 的开头。你可以直接使用下面链接里的 starter code 来获得分块边界，然后据此把工作分配到各个进程：

<https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py>

这种分块总是有效的，因为我们永远不想跨文档边界进行合并。对于作业的目的，您总是可以以这种方式分割。不要担心接收到一个不包含`<|endoftext|>`的非常大的语料库的边缘情况。

**预分词前移除特殊 token** 在用正则表达式模式（`re.finditer`）进行预分词之前，你应当先从语料（或并行实现中的 chunk）里移除所有特殊 token，并确保按特殊 token 进行切分，使得 merges 不会跨越它们所划定的边界。例如，若语料（或 chunk）为 `[Doc 1]<|endoftext|>[Doc 2]`，应当以特殊 token `<|endoftext|>` 为分隔符，把 `[Doc 1]` 与 `[Doc 2]` 分开预分词，从而保证不会跨文档边界发生合并。可以使用 `re.split`，并以 `"|".join(special_tokens)` 作为分隔符（注意需要谨慎使用 `re.escape`，因为 `|` 可能出现在特殊 token 中）。测试 `test_train_bpe_special_tokens` 会检查这一点。

**优化合并步骤** 上面风格化示例中BPE训练的朴素实现很慢，因为每次合并时，它都会迭代所有字节对以识别最频繁的对。然而，每次合并后唯一变化的对计数是那些与合并对重叠的对。因此，可以通过索引所有对的计数并增量更新这些计数，而不是显式迭代每个字节对来计数对频率，来提高BPE训练速度。通过这种缓存过程，您可以获得显著的加速，尽管我们注意到BPE训练的合并部分在Python中是不可并行化的。

**低资源/缩小提示：分析**

您应该使用分析工具如`cProfile`或`scalene`来识别实现中的瓶颈，并专注于优化这些瓶颈。

**低资源/缩小提示："缩小"**

与其直接在完整的TinyStories数据集上训练分词器，我们建议您先在数据的一小部分上训练：一个“调试数据集”。例如，可以在TinyStories验证集上训练分词器（22K文档，而不是2.12M）。这体现了一个通用策略：在可能的情况下先“缩小规模”以加速开发（更小的数据集、更小的模型等）。选择调试数据集的大小或超参数配置需要权衡：既要足够大到能复现与完整配置相同的瓶颈（保证优化可泛化），又要足够小以免运行过慢。

**问题（train\_bpe）：BPE 分词器训练（15分）**

**交付物：** 编写一个函数，给定输入文本文件的路径，训练一个（字节级）BPE 分词器。您的BPE训练函数应该处理（至少）以下输入参数：

- `input_path: str`：BPE 分词器训练数据的文本文件路径。
- `vocab_size: int`：定义最大最终词表大小的正整数（包括初始字节词表、合并产生的词表项以及任何特殊 token）。
- `special_tokens: list[str]`：要添加到词表的字符串列表。这些特殊 token 不会以其他方式影响 BPE 训练。

您的BPE训练函数应该返回生成的词表和 merges：

- `vocab: dict[int, bytes]`：分词器词表，从 int（词表中的 token ID）到 bytes（token 的字节串）的映射。
- `merges: list[tuple[bytes, bytes]]`：训练产生的BPE合并列表。每个列表项是一个bytes元组`<token1>, <token2>`，表示`<token1>`与`<token2>`合并。合并应按创建顺序排序。

要针对我们提供的测试测试您的BPE训练函数，您首先需要在`adapters.run_train_bpe`实现测试适配器。然后，运行`uv run pytest tests/test_train_bpe.py`。您的实现应该能够通过所有测试。（可选，这可能是一个大的时间投资），您可以使用某些系统语言实现训练方法的关键部分，例如C++（考虑使用cppyy）或Rust（使用PyO3）。如果您这样做，请注意哪些操作需要复制与直接从Python内存读取，并确保留下构建说明，或者确保它仅使用pyproject.toml构建。另请注意，GPT-2正则表达式在大多数正则表达式引擎中支持不佳，并且在大多数支持的引擎中速度太慢。我们已经验证Oniguruma速度合理并支持负向前瞻，但Python中的regex包（如果有的话）甚至更快。

**问题（train\_bpe\_tinystories）：在TinyStories上训练 BPE（2分）**

(a) 在TinyStories数据集上训练一个字节级 BPE 分词器，使用最大词表大小为10,000。确保将TinyStories的`<|endoftext|>`特殊 token 添加到词表中。将生成的词表与 merges 序列化到磁盘以供进一步检查。训练花费了多少小时和内存？词表中最长的 token 是什么？这是否有意义？

**资源要求：** ≤30分钟（无GPU），≤30GB RAM

**提示** 您应该能够使用预分词期间的多处理和以下两个事实在2分钟内完成BPE训练：

(a) `<|endoftext|>` token 在数据文件中分隔文档。
(b) `<|endoftext|>` token 在应用 BPE merges 之前作为特殊情况处理。

**交付物：** 一到两句话回答。

(b) 分析您的代码。分词器训练过程的哪部分花费最多时间？

**交付物：** 一到两句话回答。

接下来，我们将尝试在OpenWebText数据集上训练一个字节级 BPE 分词器。与之前一样，我们建议查看数据集以更好地理解其内容。

**问题（train\_bpe\_expts\_owt）：在OpenWebText上训练BPE（2分）**

(a) 在OpenWebText数据集上训练一个字节级 BPE 分词器，使用最大词表大小为32,000。将生成的词表与 merges 序列化到磁盘以供进一步检查。词表中最长的 token 是什么？这是否有意义？

**资源要求：** ≤12小时（无GPU），≤100GB RAM

**交付物：** 一到两句话回答。

(b) 比较和对比在TinyStories与OpenWebText上训练得到的分词器。

**交付物：** 一到两句话回答。

## 2.6 BPE 分词器：编码与解码

在作业的前一部分，我们实现了一个函数，用于在输入文本上训练 BPE 分词器，从而得到分词器词表以及 BPE merges 列表。现在，我们将实现一个 BPE 分词器：加载给定的词表与 merges 列表，并用它们把文本编码为 token IDs，以及从 token IDs 解码回文本。

### 2.6.1 编码文本（Encoding）

通过 BPE 编码文本的过程反映了我们如何训练 BPE 词表。有几个主要步骤。

**步骤1：预分词** 我们首先对序列进行预分词，并将每个 pre-token 表示为 UTF-8 字节序列，就像在 BPE 训练中所做的那样。我们会在每个 pre-token 内部把这些字节合并为词表元素；各 pre-token 独立处理（不会跨 pre-token 边界进行 merges）。

**步骤2：应用 merges** 然后，我们取 BPE 训练阶段产生的 merges 列表，并按其创建顺序依次应用到这些 pre-token 上。

**示例（bpe\_encoding）：BPE 编码示例**

例如，假设输入字符串为 `'the cat ate'`，词表为 `{0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}`，学到的 merges 为 `[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]`。首先，预分词器会把字符串切分成 `['the', ' cat', ' ate']`。然后我们逐个处理每个 pre-token 并应用 BPE merges。

第一个 pre-token `'the'` 初始表示为 `[b't', b'h', b'e']`。查看 merges 列表，我们找到第一个可应用的 merge 是 `(b't', b'h')`，据此把该 pre-token 变换为 `[b'th', b'e']`。然后回到 merges 列表，找到下一个可应用的 merge 为 `(b'th', b'e')`，从而得到 `[b'the']`。继续检查 merges 列表后不再有可应用的 merge（因为整个 pre-token 已合并为一个 token），因此应用 merges 的过程结束。对应的整数序列为 `[9]`。

对其余 pre-token 重复该过程：pre-token `' cat'` 在应用 merges 后表示为 `[b' c', b'a', b't']`，对应整数序列 `[7, 1, 5]`；pre-token `' ate'` 在应用 merges 后为 `[b' at', b'e']`，对应整数序列 `[10, 3]`。因此，对输入字符串的编码结果为 `[9, 7, 1, 5, 10, 3]`。

**特殊 token** 你的分词器在编码文本时应当能够正确处理用户定义的特殊 token（在构造分词器时提供）。

**内存考虑** 假设我们要对一个无法完全放入内存的大文本文件进行 tokenize。为了高效地 tokenize 该大文件（或任何数据流），需要把它切成可管理的 chunk 并依次处理，使得内存复杂度保持为常数而不是随文本大小线性增长。在切分时必须确保 token 不会跨越 chunk 边界，否则结果会与“把整段文本一次性读入内存再 tokenize”的朴素方法不同。

### 2.6.2 解码文本（Decoding）

要把整数 token ID 序列解码回原始文本，可以在词表中查找每个 ID 对应的条目（一个字节序列），把这些字节串拼接起来，再将其解码为 Unicode 字符串。注意：输入 ID 并不保证一定能映射到有效的 Unicode 字符串（用户可以输入任意整数序列）。若输入 token ID 产生的字节序列无法解码为有效的 Unicode 字符串，应当用官方的 Unicode 替换字符 U+FFFD<sup>3</sup> 来替换畸形字节。`bytes.decode` 的 `errors` 参数用于控制 Unicode 解码错误的处理方式，使用 `errors='replace'` 会自动用替换标记替换畸形数据。

<sup>3</sup> See en.wikipedia.org/wiki/Specials\_(Unicode\_block)#Replacement\_character for more information about the Unicode replacement character.

**问题（tokenizer）：实现分词器（15分）**

**交付物：** 实现一个`Tokenizer`类，给定词表与 merges 列表，将文本编码为整数 token IDs，并将整数 token IDs 解码为文本。你的分词器还应支持用户提供的特殊 token（如果它们尚未在词表中，则把它们追加到词表中）。我们建议以下接口：

- `def __init__(self, vocab, merges, special_tokens=None)`：从给定的词表、merges 列表以及（可选）特殊 token 列表构造分词器。此函数应接受以下参数：
  - `vocab: dict[int, bytes]`
  - `merges: list[tuple[bytes, bytes]]`
  - `special_tokens: list[str] | None = None`
- `def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)`：类方法，从序列化的词表与 merges 列表（与BPE训练代码输出格式一致）以及（可选）特殊 token 列表构造并返回`Tokenizer`。此方法应接受以下附加参数：
  - `vocab_filepath: str`
  - `merges_filepath: str`
  - `special_tokens: list[str] | None = None`
- `def encode(self, text: str) -> list[int]`：将输入文本编码为 token ID 序列。
- `def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`：给定字符串的可迭代对象（例如Python文件句柄），返回一个 generator，惰性地产生 token IDs。这对于无法直接加载到内存的大文件进行内存高效的 tokenize 是必需的。
- `def decode(self, ids: list[int]) -> str`：将 token ID 序列解码为文本。

要使用我们提供的测试来验证`Tokenizer`，你需要先在`adapters.get_tokenizer`里实现测试适配器。然后运行`uv run pytest tests/test_tokenizer.py`。你的实现应当能够通过所有测试。

## 2.7 实验

**问题（tokenizer\_experiments）：分词器实验（4分）**

(a) 从TinyStories和OpenWebText中采样10个文档。使用你之前训练的TinyStories与OpenWebText分词器（词表大小分别为10K与32K），把这些样本文档编码为整数 token IDs。每个分词器的压缩率（bytes/token）是多少？

**交付物：** 一到两句话回答。

(b) 如果用TinyStories分词器对OpenWebText样本进行 tokenize 会发生什么？比较压缩率和/或定性描述发生了什么。

**交付物：** 一到两句话回答。

(c) 估计你的分词器吞吐量（例如 bytes/second）。对 Pile 数

据集（825GB 文本）进行 tokenize 需要多长时间？

**交付物：** 一到两句话回答。

(d) 使用你的TinyStories与OpenWebText分词器，把各自的训练与开发数据集编码为整数 token ID 序列。我们稍后会用它来训练语言模型。我们建议把 token IDs 序列序列化为 dtype 为 uint16 的 NumPy 数组。为什么 uint16 是一个合适的选择？

**交付物：** 一到两句话回答。
