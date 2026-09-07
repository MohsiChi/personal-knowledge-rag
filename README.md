# Personal Knowledge RAG

给**结构化知识库**做的混合检索层：dense + BM25 双路召回 → RRF 融合 → 带出处的结果与完整检索留痕。
输入契约兼容 [knowledge-wiki](https://github.com/MohsiChi/knowledge-wiki) 的 `wiki/` 目录约定，
但本项目是独立仓库，不修改也不依赖那个 skill。

> 📓 完整开发过程（含走错的弯路、自己引入又自己抓到的 bug、每条结论的实测依据）见
> **[docs/dev-log.md](docs/dev-log.md)**。设计取舍见本文第八节。

> **状态：v0 雏形。** 管线完整、可评测、58 项测试全绿；但检索质量**尚未调优**，
> 已知的失败案例与原因都写在下面的「实测结论」与「已知限制」里，没有藏。

---

## 一、这个项目解决什么问题

knowledge-wiki 解决的是**知识组织**：人往 `raw/` 放材料，AI 维护 `wiki/` 的结构
（concepts / connections / questions / index）。它的"检索"方式是 AI 读 `index.md` 再顺着链接翻页面——
几十个页面时没问题，几百个页面时 `index.md` 自己就读不完，而"哪些概念和我现在这个问题相关"
这种语义召回它根本做不了。

本项目补的就是这一层：**知识检索**。

### 为什么结构化 wiki 是比原始 PDF 更好的 RAG 语料

经典 RAG 的效果上限由切块质量决定：把 PDF 按字数切会破坏语义（跨页引用断裂、表格公式丢失、
章节层级抹平）。而 `wiki/concepts/*.md` 这类页面是已经被 AI 做过概念提取的产物——
**每一页就是一个语义自洽的检索单元**，还自带 frontmatter（title / tags / status / 阅读等级）。
`wiki/connections/` 更提供了显式的概念关系，是纯向量 RAG 结构上没有的信息。

所以定位不是"又一个 PDF 问答机器人"，而是：**在高质量结构化语料上做可评测的混合检索。**

---

## 二、架构

```
输入契约：兼容 knowledge-wiki 的 wiki/ 目录（可同时挂多个库 -> 联邦检索）
  concepts/   connections/   questions/   papers/
        ↓
① ingest      按 markdown 标题层级切块；frontmatter -> metadata；
              代码围栏内的 # 不当标题；默认排除 status:stub 与 index 层
        ↓
② embed       Embedder 抽象：hash（离线确定性）| fastembed（本地 ONNX，真实语义）
        ↓
③ index       NumpyVectorIndex（暴力余弦）+ BM25Index（中文 unigram+bigram 分词）
        ↓
④ retrieve    前置 metadata 过滤 -> 双路召回 -> RRF 融合 -> ScoredChunk + RetrievalTrace
        ↓
⑤ evaluate    Recall@k / Precision@k / MRR / NDCG@k，支持 dense|sparse|hybrid 三档 ablation
```

**刻意不做**（v0）：raw PDF 层、cross-encoder reranker、LLM 生成、query 分类路由、
connections 图扩展、Web UI、索引持久化。理由见「设计取舍」。

---

## 三、安装与运行

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt          # 核心：numpy / rank-bm25 / PyYAML / pytest
.venv\Scripts\pip install -r requirements-embed.txt    # 真实 embedding（fastembed，本地 ONNX）
.venv\Scripts\python -m pytest                         # 62 passed
```

仓库自带一份合成示例语料 `sample_wiki/`（内容是检索领域的概念页，由作者撰写，不含任何真实课程材料），
所以 clone 下来**不配置也能跑**：

```bash
python -m pkr stats                                   # 语料统计
python -m pkr search "为什么 RRF 用排名而不是分数" --trace
python -m pkr eval --per-query                        # 在 smoke 集上跑三档 ablation
```

### 接自己的语料

复制 `config.example.json` 为 `config.local.json`（**已被 gitignore**），填入你的 wiki 根路径：

```json
{
  "wiki_roots": [
    { "name": "CS61B", "path": "D:/.../CS61B/research-wiki" },
    { "name": "web",   "path": "D:/.../没学的课" }
  ],
  "embedder": "fastembed"
}
```

`wiki_roots[i].path` 指向含 `wiki/` 子目录的那一层（即 knowledge-wiki 的 `research-wiki/`）；
直接指向 `wiki/` 本身也能容错。多个库会各自索引并打上 `source_wiki`，检索时合并，
因此支持跨学科联邦检索与按库过滤（`--wiki CS61B`）。

> **隐私设计**：语料路径只存在于本地 `config.local.json`，仓库里只有合成示例。
> 课程笔记属个人/课程材料，不应进公开仓库。

---

## 四、Embedding：本地推理，不调外部 API

`fastembed` 走**本地 ONNX 推理**：不需要 API key、不产生费用、语料不出本机。
首次运行会下载模型（multilingual-MiniLM 约 220MB），之后离线可用。

默认模型是 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`（dim 384），
**不是**中文专用的 `BAAI/bge-small-zh-v1.5`。这是实测决定的，不是推测：

| 查询 | bge-small-zh-v1.5 | multilingual-MiniLM-L12-v2 |
|---|---|---|
| "什么是 BM25"（三篇候选，其中一篇字面含 BM25） | 分数 0.4395 / 0.4502 / 0.4347，**几乎无区分力**，含 BM25 的那篇排第 2 | **0.4914 vs 0.0488 / 0.0929**，含 BM25 的排第 1 |
| "稀疏检索和稠密检索应该怎么取舍" | 无关文档排到第 2，相关文档排最后 | 无关文档排最后 |

原因是课程笔记是**中英混排**（中文叙述 + `BM25` / `AVL` / `B+ Tree` 这类英文术语），
中文专用模型的跨语言对齐在这种语料上明显偏弱。

> ⚠️ 上述对比只是 **3 文档探针**，属方向性信号，不是质量结论。最终选型应由评测集决定。
> 两个候选模型的 fastembed 元数据都注明 `Prefixes for queries/documents: not necessary`，
> 因此本实现不给 query 加指令前缀——这是查库得到的，不是猜的。
> **评测数字必须绑定 fastembed 版本**（本项目实测为 0.8.0）：该版本对此模型改用 mean pooling，
> 换版本会让历史数字不可比。`pkr eval` 的输出与 `--out` JSON 里都会记录版本。

### 三档 embedder

| 档位 | 是什么 | 用途 |
|---|---|---|
| `hash` | 确定性哈希词袋投影，零依赖、离线、瞬时 | 跑测试、验证管线正确性、无网 demo |
| `fastembed` | 本地 ONNX 神经模型 | 真实检索与真实评测 |

**hash 档跑出的任何指标都不能当作检索质量结论**——它没有语义泛化能力。
`pkr eval` 在 hash 档下会主动打印这条警告。

---

## 五、评测方法

```bash
# 1. 写 query 清单（一行一条；或 id<TAB>type<TAB>query）
#    原则：写你复习时真的会问的问题，不要为了凑数编
# 2. 生成候选池工作表
python evaluation/label.py pool --queries my_queries.txt --pool-size 20 --out evaluation/pool.tsv
# 3. 在 pool.tsv 的 label 列填 y（相关）/ p（部分相关）/ n（不相关，留空同 n）
# 4. 回收成评测集
python evaluation/label.py collect --pool evaluation/pool.tsv --out evaluation/dataset.json
# 5. 跑 ablation
python -m pkr eval --dataset evaluation/dataset.json --per-query
```

**候选池法（pooling）**：让你逐页判断"这个 query 和全部 N 个 chunk 相关吗"是 O(N×M) 的苦力。
改为先由三种 mode 各取 top-P 求并集作为候选池，你只在池里勾 y/p/n。

> **必须知道的偏差**：所有 mode 都没召回的页面永远进不了候选池，因而默认被视为不相关，
> 这会系统性**高估**各方法的绝对分数。但它**不影响方法之间的相对比较**，
> 因为所有方法共用同一个池——这正是 TREC 采用 pooling 的理由。
>
> **标注纪律**：尽量盲标（只看 chunk 内容，不看 `routes` 列）；任何在看到检索结果之后的标注补正，
> 都必须像 `evaluation/dataset.smoke.json` 的 `_labeling_correction` 那样写明理由与时间。
> 事后改标注可以被用来刷分，所以必须留痕。

指标口径：分级相关性 relevant=2 / partial=1 / 未标注=0；Recall、Precision、MRR 把 partial 也计为命中；
NDCG@k 用指数增益 `(2^gain − 1)/log2(rank + 1)`；无任何相关文档的 query 不计入平均。

---

## 六、实测结论（真实数字，含失败的）

### 6.1 smoke 集（22 chunk 合成语料，5 条 query，k=5）

`evaluation/dataset.smoke.json`，2026-09-06 实测：

| embedder | Method | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---|---|---:|---:|---:|---:|
| fastembed 0.8.0 | dense | 0.6238 | 0.4400 | 0.8000 | 0.5500 |
| | sparse | 0.5952 | 0.3600 | **1.0000** | 0.5195 |
| | **hybrid** | **0.7405** | **0.4800** | 0.9000 | **0.5958** |
| hash | dense | 0.5286 | 0.3200 | 0.9000 | 0.4793 |
| | sparse | 0.5952 | 0.3600 | 1.0000 | 0.5195 |
| | hybrid | 0.5286 | 0.3200 | 1.0000 | 0.5284 |

注意 hash 档下 hybrid 的 Recall@5（0.5286）**低于** sparse（0.5952），
而 fastembed 档下 hybrid 明显最好（0.7405）。**同一套管线、同一份标注，只换 embedder，结论就反过来了**——
这就是"hash 档数字不能当质量结论"的实证。

### 6.2 真实语料（471 chunk，4 个学科库）

语料规模实测：`CS61B 78 / database 12 / web 162 / DS-final 219`，
按层 `concept 371 / connection 46 / question 54`（已排除 stub 与 index 层）。

查询「AVL 树和 B+ 树的区别」，语料里确实存在最切题的页面
（`CS61B/wiki/questions/balanced-trees-why.md`，标题即"为什么需要这么多平衡树变体？
2-3 Tree, Red-Black Tree, AVL Tree, LLRB Tree, B-Tree, B+Tree…"）：

| mode | 该页最好名次 |
|---|---:|
| dense | **1** |
| sparse | 25 |
| hybrid（RRF, k=60） | **11** |

**hybrid 把 dense 的第 1 名拖到了第 11 名。** 原因是稀疏路在这个语料上很弱
（中文查询 vs 英文术语正文），RRF 把这路弱信号也计入了融合。

这与 6.1 的 smoke 结论**相反**：小语料上 hybrid 最好，真实语料上 hybrid 反而更差。
所以「混合检索一定更好」是不能假设的，**RRF 的 k 值、甚至是否融合，都必须按语料实测决定**。

---

## 七、已知限制（如实披露）

1. **检索质量未调优。** 6.2 显示默认 `rrf_k=60` 在真实语料上会拖累 dense 的强项。
   需要按评测集扫 k 值，或引入按路质量自适应的融合权重。
2. **没有"我不知道"的能力。** 查询「CSS 盒模型是怎样的」时语料里**完全没有**相关内容
   （全库 0 命中），系统仍然自信地返回了鸽巢原理——因为"盒子"字面撞上了。
   根因之一是 RRF 分数只有相对意义（0.028~0.031 挤在一起），无法设绝对阈值；
   要做 no-answer 判定必须用 dense 的原始余弦分数，而不是融合分数。
3. **稀疏路对中英混排语料偏弱。** BM25 分词是零依赖的 unigram+bigram，
   不如 jieba 等分词器；且中文查询匹配英文术语正文时几乎没有信号。
4. **BM25 在小语料上会退化。** N=2 且查询词同时出现在两篇时，rank-bm25 的 epsilon 修正
   会产生**负 IDF**，长度归一化方向反转（短文档反而更低）。本项目**刻意不修改 BM25 打分**，
   以保持与标准实现可比；结论是评测语料不能太小。已有回归测试记录此行为。
5. **标注集尚未建立。** 目前只有 5 条 smoke query（标在合成语料上，仅验证管线）。
   真实语料上的 30~50 条人工标注集还没做，因此**本项目目前没有任何可信的质量结论**。
6. **`raw/` 层未索引。** 真实语料的 `raw/` 下有 39 个 PDF（课件等），v0 只索引 `wiki/` 的 markdown。
   少了原始证据层，"概念层解释 + 原始层举证"的设计还没成立。
7. **无索引持久化。** 每次运行重建索引（471 chunk + fastembed 约 20~30 秒）。
   这是刻意的：持久化会引入**静默陈旧**（语料改了索引没更新，且无从察觉）。

---

## 八、设计取舍

| 取舍 | 决定 | 理由 |
|---|---|---|
| 向量索引 | 暴力 numpy 余弦，不用 FAISS | 几百到几千 chunk 时暴力检索是毫秒级；ANN 引入近似召回、调参与增量更新复杂度，此规模属过度设计。留 `VectorIndex` 抽象，撞到延迟预算再换 |
| reranker | **不进 v0** | 加 cross-encoder 要拖 GB 级依赖，而它是否有效**必须先由评测证明**。先建评测、跑三档 baseline，有 headroom 再加 |
| LLM 生成 | **不进 v0** | 本项目价值主张是检索质量；生成会把评测复杂度翻倍（要评 faithfulness / answer relevance）并引入 API 成本 |
| BM25 打分 | 不修改库行为 | 改了就与标准实现不可比，评测数字失去参照意义 |
| 过滤时机 | **前置**（排名前过滤） | 后置过滤会让 top_k 被筛空、指标虚低且无法解释 |
| RRF | 自己实现 | 公式只有三行，引框架反而把可调参数藏起来 |
| LangChain | 不使用 | 检索核心自己写，才能讲清每一阶段的设计 |

---

## 九、Roadmap

1. **建真实评测集**（30~50 条人工标注）——在此之前不做任何调参，否则是在猜
2. 扫 `rrf_k`，或改为按路质量自适应融合（解决 6.2 的 hybrid 退化）
3. dense 原始分数阈值 → no-answer 判定（解决限制 2）
4. `raw/` PDF 层接入（parser + 切块），补齐"概念层解释 + 原始层举证"
5. connections 图扩展：wikilink 解析即可，不上 Neo4j
6. reranker（**仅当** 1 的评测显示有 headroom）
7. 索引持久化 + 增量更新（需先解决陈旧检测，例如语料内容哈希）

---

## 十、测试

```bash
.venv\Scripts\python -m pytest          # 62 passed
```

覆盖：markdown 切块（含代码围栏不被切断、超长块二次切分）、frontmatter 解析（含非法 YAML 容错）、
stub/index 排除、中文分词（含 bigram 不跨非中文间隙）、RRF（与手算值对照、k 的语义、并列决胜确定性）、
四个指标（全部与手算值对照）、向量索引（过滤、维度校验）、BM25（IDF 退化、词重叠判据）、
检索编排（前置过滤、trace、三档 mode）、评测管线（标注 id 必须真实存在、全 NaN 时非零退出）、
标注工具（含"行尾多打一个 tab"的容错回归）、
CLI 全局选项解析（`--config` 写在子命令前后都必须生效）。

最后一条是针对一个真实发生过的**静默 bug** 加的回归：argparse 的 `parents` 共享 dest 时，
子解析器的 `default=None` 会覆盖父解析器已解析的值，于是 `pkr --config X stats` 会静默忽略 X、
退回 `config.local.json`——用户可能对着错误的语料跑完整套评测而毫不知情。
修法是 `default=argparse.SUPPRESS` + 在 `main()` 里统一补默认值。
这类"不报错但结果错"的缺陷比崩溃危险得多，所以必须有测试钉住。

测试全部离线、确定性、不下载模型，且**从不读取 `config.local.json`**（`tests/conftest.py` 用固定夹具指向
仓库自带的 `sample_wiki`）。这条隔离是踩坑后加的：一旦测试依赖本地语料配置，
换台机器或改一下 `config.local.json` 就会既变慢又莫名失败。
全量 62 项约 15 秒。

fastembed 真实路径由人工验证（2026-09-06，fastembed 0.8.0，471 chunk 真实语料跑通 stats/search/eval），
刻意不进自动化测试——否则测试套件会依赖 220MB 模型下载与网络可用性。