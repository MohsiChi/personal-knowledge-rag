# 开发日志（Dev Log）

> 规则：每个工作会话结束前追加一条。记录"做了什么、为什么、证据、下一步"。
> 这份日志是完整的工程过程记录，包括走错的弯路与自己引入又自己抓到的 bug。
>
> 隐私约定：本仓库公开，因此日志中不出现真实语料的本地路径（它们在 gitignored 的
> `config.local.json` 里），只以语料规模与学科分布指代。

---

## Session 1 — 2026-09-06｜项目定位、外部方案裁定与 v0 实现

**起因**：姊妹项目 `aigc-motivated-ADs-workflow` 的架构演进路线里有一项
"RAG：资料切块检索（chromadb）喂给 Script Agent"，一直未实现。本轮把它独立出来做，
但定位改了：不是给广告流水线补检索，而是**给结构化个人知识库做检索层**——
因为手上已有 4 个用 knowledge-wiki 约定维护的真实研究 wiki，那是比原始 PDF 质量高得多的 RAG 语料。

外部（网页端）给过一份很详细的 Phase 1 规格，逐条裁定后采纳了一部分、修正了一部分、拒绝了一部分（见下）。

### 对外部方案的裁定

**采纳**：hybrid（dense + BM25 + RRF）作为主干；结构化切块；metadata schema（`source_type` /
`knowledge_layer` / `parent_document` / `chunk_index`）；`retrieval_trace`；评测框架与 ablation 表；
不上 Neo4j（markdown wikilink 本身就是图）；不过度依赖 LangChain、检索核心自己写。

**修正**：

1. **向量索引不用 FAISS，用暴力 numpy 余弦。** 外部建议"第一版可以用本地 FAISS"。
   查证后确认 faiss-cpu 在本平台可装（有 cp313+win 轮子），所以不是可行性问题，而是**必要性问题**：
   几百到几千 chunk 时暴力检索是毫秒级，ANN 引入近似召回、调参与增量更新复杂度，属过度设计。
   留 `VectorIndex` 抽象，撞到延迟预算再换实现。
2. **reranker 不进 v0。** 外部把它列为 Phase 1 必需项，但没有任何证据表明它在本语料上有收益，
   而 cross-encoder 要拖 GB 级依赖。改为：**先建评测、跑三档 baseline，ablation 显示有 headroom 才加。**
   这与本项目一贯的纪律一致——不凭假设塞组件。
3. **v0 不含 LLM 生成。** 外部架构图画到 "LLM Answer with Citation"，但详细规格止于
   "LLM-ready Retrieval Output"，自相矛盾。裁定为不做生成：本项目价值主张是检索质量，
   生成会把评测复杂度翻倍（要评 faithfulness / answer relevance）并引入 API 成本。
4. **载体是独立新仓库，不是塞进 knowledge-wiki。** 外部建议把 `rag/`、`api/` 放进 knowledge-wiki 仓库，
   但那个仓库是纯 markdown/shell/json 的 Claude Code skill（16 个文件、零 Python），
   塞进 Python 运行时与模型依赖会改变它的性质，也直接违反它自己写的"不破坏现有 skill 功能"约束。
   改为：新仓库 + 把 `wiki/` 目录结构作为**输入契约**，两边互相导流。

**拒绝**：把检索内核注册成姊妹项目的一个 Tool。理由是语料域不搭（课程知识 vs 商品文案），
且这属于**叙事驱动的耦合而非业务驱动**——为了看起来"两个项目咬合"而耦合两个无关系统，
一句追问就能打穿。两个项目各自独立，最多在设计思路层面同源；等真有第二个消费者再抽公共内核，
否则是过早抽象。

### 语料勘察

4 个真实研究 wiki（数据结构 / 数据库 / Web / 期末复习），实测规模：

| wiki | concepts | connections | questions |
|---|---:|---:|---:|
| 数据结构（CS61B） | 26 | 4 | 4 |
| 数据库 | 5 | 2 | 2 |
| Web | 18 | 2 | 0 |
| 期末复习 | 12 | 4 | 10 |
| **合计** | **61** | **12** | **16** |

`raw/` 另有 39 个 PDF（课件等），v0 未索引（见已知限制 6）。

勘察时发现三个会影响解析器设计的真实细节，都是看了实际文件才发现的，不是推测：

- frontmatter 是 YAML（`title` / `date` / `tags` / **`status`**），部分页面是 `status: stub` 的空壳占位；
- `wiki/` 根下的 `index.md` 是清单页而非知识页，它会匹配几乎所有查询；
- 正文含表格与代码围栏，切块时若不跟踪围栏状态，会把代码块从中间切断。

### 做了什么

| 模块 | 职责 |
|---|---|
| `pkr/ingest.py` | 扫描 wiki 根、按目录判定 `knowledge_layer`、解析 frontmatter、按 markdown 标题层级切块（围栏感知 + 超长块按空行二次切分） |
| `pkr/text.py` | 中文感知分词：拉丁/数字连续段成词，每个连续 CJK 段出 unigram + **段内** bigram（不跨非中文间隙，避免造出假词） |
| `pkr/embed.py` | `Embedder` 抽象 + `HashEmbedder`（离线确定性）+ `FastEmbedder`（本地 ONNX） |
| `pkr/index.py` | `VectorIndex` 抽象 + `NumpyVectorIndex`（暴力余弦）+ `BM25Index` |
| `pkr/fuse.py` | 自己实现的 RRF，并列按 id 字典序决胜 |
| `pkr/retrieve.py` | **前置** metadata 过滤 → 双路召回 → 融合 → `ScoredChunk` + `RetrievalTrace`；支持 dense/sparse/hybrid 三档 |
| `pkr/metrics.py` | Recall@k / Precision@k / MRR / NDCG@k（分级增益，partial 计命中但不计入 NDCG 的高增益） |
| `pkr/cli.py` | `stats` / `search` / `eval` 三个子命令 |
| `evaluation/label.py` | 候选池法标注工具：`pool` 生成 TSV 工作表，`collect` 回收成 dataset.json |
| `sample_wiki/` | 7 页合成示例语料（作者自撰的检索领域概念页），使 clone 后零配置可跑 |
| `tests/` | 9 个文件 62 项，全离线确定性 |

### 关键决策

- **默认 embedding 模型是实测选的，不是推测的。** 起初打算用 `intfloat/multilingual-e5-small`，
  查证发现它**不在 fastembed 0.8.0 的支持列表里**（我原先的说法是错的）。
  改用三文档探针实测两个候选后定为 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`：
  查"什么是 BM25"时，中文专用的 `BAAI/bge-small-zh-v1.5` 三篇候选分数为
  0.4395 / 0.4502 / 0.4347（**几乎零区分力**，字面含 BM25 的那篇只排第二），
  多语言模型为 0.4914 vs 0.0488 / 0.0929。根因是语料**中英混排**（中文叙述 + 英文术语），
  中文专用模型的跨语言对齐在这种语料上明显偏弱。
  该对比只是 3 文档探针，属方向性信号而非质量结论，已如实写进 README 与 `config.py` 注释。
- **query 不加指令前缀**：两个候选模型的 fastembed 元数据都注明
  `Prefixes for queries/documents: not necessary`，因此不加。这是查库得到的，不是猜的。
- **评测数字必须绑定 fastembed 版本**：0.8.0 起该模型改用 mean pooling（库自己会 warn），
  换版本会让历史数字不可比。`pkr eval` 的输出与 `--out` JSON 都记录版本、模型名与维度。
- **不做索引持久化**：语料小、重建快（471 chunk + fastembed 约 20~30 秒），
  而持久化会引入**静默陈旧**——语料改了索引没更新，检索结果与语料不一致且无从察觉。
  宁可每次重建。
- **过滤前置而非后置**：后置过滤会让 top_k 被筛空、指标虚低且无法解释。
- **不修改 BM25 的打分行为**：改了就与标准实现不可比，评测数字失去参照意义（见事故 1）。
- **测试必须与本地环境配置隔离**：`tests/conftest.py` 用固定夹具指向 `sample_wiki`，
  永不调用 `load_settings()`（见事故 2）。
- **隐私边界**：语料路径只存在于 gitignored 的 `config.local.json`，仓库内只有合成示例。
  课程笔记属个人/课程材料，不进公开仓库。提交前逐条验过 `config.local.json` / `.venv` /
  `__pycache__` / `.pytest_cache` 均被忽略，且无任何真实语料路径泄漏进将提交的文件。

### 事故与修复（本会话自身，如实记录）

1. **BM25 用"分数 > 0"当入选判据 —— 小语料下整路召回为空。**
   `rank-bm25` 的 IDF 是 `log((N-df+0.5)/(df+0.5))`：N=2、df=1 时等于 `log(1)=0`；
   且当全部 IDF 非正时，它的 epsilon 修正（`epsilon * average_idf`，而 `average_idf` 是对**全部**
   IDF 求均值）会给出**负值**，长度归一化方向随之反转（短文档反而更低）。
   我原来按"分数为 0 即完全不匹配"过滤，结果把有真实词重叠的文档也全滤掉了。
   **修复**：入选判据改为**词重叠**，排序用 `(score 降, overlap 降, id 升)`。
   同时保留一条记录性测试钉住这个退化行为，并在 README 写明"评测语料不能太小"。
2. **`config.local.json` 破坏测试隔离（我自己造成的）。**
   为了让人能直接对真实语料跑检索而建了它，结果测试通过 `load_settings()` 读到真实语料，
   从 58 绿变 2 红、单次运行从约 10 秒涨到 90 秒以上（smoke 集的 `sample::` chunk id
   在真实语料里根本不存在）。
   **修复**：新增 `tests/conftest.py`，用 session 级固定夹具指向 `sample_wiki` + hash embedder；
   走子进程的测试显式传 `--config config.example.json`。恢复 58 绿、7 秒。
3. **argparse `parents` 导致 `--config` 被静默忽略（本会话最坏的一个）。**
   为了让 `--embedder` 在子命令前后都能用而加了 `parents=[common]`，
   但父解析器与子解析器共享同一 dest，子解析器的 `default=None` 会**覆盖**父解析器已解析的值。
   于是 `pkr --config X stats` 静默退回 `config.local.json`——
   **可能对着错误的语料跑完整套评测而毫不知情**。是我在冒烟时从输出里
   `wiki 根 = ['CS61B', ...]` 发现"这不是我传的配置"才抓到的。
   **修复**：`default=argparse.SUPPRESS` + 在 `main()` 里统一补默认值；
   新增 4 条回归测试钉死两种写法。这类"不报错但结果错"的缺陷比崩溃危险得多。
4. **`FastEmbedder.dim` 探测写错括号**：`int(len(next(iter(...)))[0])` 对 int 取下标，
   `TypeError: 'int' object is not subscriptable`。修为 `int(len(next(iter(...))))`。
   顺带把维度探测结果与库版本一起暴露为 `describe` 属性，供评测输出记录。
5. **smoke 集标注漏项，被评测结果暴露。** 首次评测 q005 三种 mode 全部 0 分，
   诊断发现 `dense-retrieval.md::1`（标题即"与稀疏检索的根本区别"）在两路都排第 1、
   `sparse-vs-dense.md::0` 进了 hybrid top5，但两者都未被标注。
   **修复**：补为 partial，并在数据文件里写入 `_labeling_correction` 说明
   "这是看到结果之后修改标注，此类操作可以被用来刷分，因此必须留痕"。
   同时定下标注纪律：真实语料上应尽量盲标（只看内容、不看 `routes` 列）。
6. **两处我自己的手误**：NDCG 手算常数写成 0.742078（实际 0.7420981，代码是对的、断言错了）；
   标注工具的测试脚手架在模拟"行尾多打一个 tab"时忘了先清空 label 列，
   导致容错分支根本没被触发（测试假绿）。两处都已修正。
7. **`cmd_eval` 里留了一行死代码**（`base = dict(...keys())`，从未被使用且会崩）。
   同一会话里还在 `fuse.py` 留过一个未被调用的 `rrf_ranks`。
   两处都清掉了——死代码比缺一个便利函数更糟。

### 证据

- **测试**：`pytest` **62 passed**，约 15 秒，全离线确定性、不下载模型、从不读 `config.local.json`。
  覆盖：切块（围栏不被切断 / 超长二次切分）、frontmatter（含非法 YAML 容错）、stub 与 index 排除、
  中文分词（bigram 不跨非中文间隙）、RRF（与手算值对照 / k 的语义 / 并列决胜确定性）、
  四个指标（全部与手算值对照）、向量索引（过滤 / 维度校验）、BM25（IDF 退化 / 词重叠判据）、
  检索编排（前置过滤 / trace / 三档 mode）、评测管线（标注 id 必须真实存在 / 全 NaN 时非零退出）、
  标注工具（含行尾多 tab 的容错回归）、CLI 全局选项（两个位置都必须生效）。
- **真实语料跑通**：471 chunk（CS61B 78 / database 12 / web 162 / DS-final 219；
  concept 371 / connection 46 / question 54），stub 与 index 层按设计排除。
- **真实 embedding 路径已验证**：fastembed 0.8.0，模型约 220MB，首次加载 41.5 秒、之后走缓存；
  dim=384；`stats` / `search` / `eval` 三个子命令在真实语料上全部跑通。
- **smoke 集 ablation**（22 chunk 合成语料，5 条 query，k=5）：

  | embedder | Method | Recall@5 | Precision@5 | MRR | NDCG@5 |
  |---|---|---:|---:|---:|---:|
  | fastembed 0.8.0 | dense | 0.6238 | 0.4400 | 0.8000 | 0.5500 |
  | | sparse | 0.5952 | 0.3600 | **1.0000** | 0.5195 |
  | | **hybrid** | **0.7405** | **0.4800** | 0.9000 | **0.5958** |
  | hash | dense | 0.5286 | 0.3200 | 0.9000 | 0.4793 |
  | | sparse | 0.5952 | 0.3600 | 1.0000 | 0.5195 |
  | | hybrid | 0.5286 | 0.3200 | 1.0000 | 0.5284 |

  hash 档下 hybrid 的 Recall@5（0.5286）**低于** sparse（0.5952），fastembed 档下 hybrid 明显最好。
  **同一套管线、同一份标注，只换 embedder，结论就反转**——这是"hash 档数字不能当质量结论"的实证。
- **真实语料上的反例（最有价值的一条）**：查询「AVL 树和 B+ 树的区别」，
  语料中确实存在最切题的页面（`questions/balanced-trees-why.md`，标题即
  "为什么需要这么多平衡树变体？2-3 Tree, Red-Black Tree, AVL Tree, LLRB Tree, B-Tree, B+Tree…"，
  `status: open`，已确认在索引中，4 个 chunk）：

  | mode | 该页最好名次 |
  |---|---:|
  | dense | **1** |
  | sparse | 25 |
  | hybrid（RRF, k=60） | **11** |

  **hybrid 把 dense 的第 1 名拖到了第 11 名**，因为稀疏路在中英混排语料上很弱（中文查询 vs 英文术语正文），
  RRF 把这路弱信号也计入了融合。这与 smoke 集结论相反，说明
  **"混合检索一定更好"不能假设，`rrf_k` 与是否融合都必须按语料实测决定。**
- **无 no-answer 能力的实证**：查询「CSS 盒模型是怎样的」，全库对该词 **0 命中**，
  系统仍自信返回了鸽巢原理页面（"盒子"字面撞上）。根因之一是 RRF 分数只有相对意义
  （实测 0.028~0.031 挤成一团），无法设绝对阈值；要做 no-answer 判定必须用 dense 的原始余弦分。
- **依赖可安装性全部查证过**（Windows + Python 3.13）：faiss-cpu 1.15.0 有 cp313+win 轮子、
  rank-bm25 0.2.2 纯 py、sqlite-vec 0.1.9 py3-none-win、chromadb 1.5.9 cp39-abi3、
  fastembed 0.8.0 纯 py、onnxruntime 1.29.0 有 cp313+win 轮子；
  `pip install --dry-run fastembed rank-bm25 sqlite-vec` 解析通过（未实际安装）。

### 已知限制

全部写在 README「已知限制」一节（7 条），此处不重复。要点：检索质量未调优、
无 no-answer 阈值、稀疏路对中英混排偏弱、BM25 小语料退化、**真实评测集尚未建立**
（因此本项目目前没有任何可信的质量结论）、`raw/` PDF 层未索引、无索引持久化。

### 下一步

- [ ] **建真实评测集**（30~50 条人工标注）——在此之前不做任何调参，否则是在猜
- [ ] 扫 `rrf_k`，或改为按路质量自适应融合（解决 hybrid 拖累 dense 的问题）
- [ ] dense 原始分数阈值 → no-answer 判定
- [ ] `raw/` PDF 层接入，补齐"概念层解释 + 原始层举证"
- [ ] connections 图扩展（wikilink 解析即可，不上 Neo4j）
- [ ] reranker（**仅当**评测显示有 headroom）
- [ ] 索引持久化 + 增量更新（需先解决陈旧检测，例如语料内容哈希）
- [ ] commit 与 push 由用户执行

### 待用户决策

1. 评测集的 query 清单需由语料主人本人撰写（写复习时真的会问的问题），标注也只能由本人做——
   相关性判断无法外包。工具已就绪（`evaluation/label.py`）。
2. 是否把 README「设计取舍」一节拆成独立的 ADR 文档（与姊妹项目的 `docs/02-架构设计.md` 体例对齐）。
   目前不拆，避免与 README 重复维护同一批决策。