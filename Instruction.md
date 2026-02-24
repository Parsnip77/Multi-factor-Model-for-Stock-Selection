# 项目说明 (Instruction)

本文档用简明语言说明本仓库中**所有文件与代码的用途、使用方式**，并展示项目架构。随项目推进会持续更新。

---

## 一、项目概述

本项目是一个**多因子选股模型**的量化交易示例，使用 Python 实现从数据获取、因子构建到回测的完整流程。

**当前阶段：数据层（ETL）+ 因子计算层**  
用 Tushare Pro 拉取沪深300成分股的日线行情与基本面指标，写入本地 SQLite 数据库；在此基础上复现《101 Formulaic Alphas》中的经典因子，供后续回测使用。

---

## 二、项目架构（文件与目录）

```
项目根目录/
├── data/
│   ├── .gitkeep            # 保留空目录用于 git 追踪（数据文件本身不上传）
│   └── stock_data.db       # SQLite 数据库（运行 download_data() 后生成）
├── src/
│   ├── __init__.py         # 标识 src 为 Python 包
│   ├── config.py           # 全局配置（Token、日期范围、路径等）
│   ├── data_loader.py      # DataEngine 类：数据下载与读取
│   └── alphas.py           # Alpha101 类：因子计算（101 Formulaic Alphas）
├── notebooks/
│   └── explore.ipynb       # Jupyter Notebook：数据探索与可视化
├── .gitignore              # 版本控制忽略规则
├── requirements.txt        # Python 依赖列表
├── prompt.md               # 项目需求与规范（仅本地查阅）
└── Instruction.md          # 本说明文档
```

| 文件/目录 | 用途 |
|-----------|------|
| `data/` | 存放 SQLite 数据库文件（不上传至 git） |
| `src/config.py` | 全局参数配置，包含 Tushare Token（不上传至 git） |
| `src/data_loader.py` | `DataEngine` 类：数据下载、缓存、读取 |
| `src/alphas.py` | `Alpha101` 类：复现《101 Formulaic Alphas》中的 5 个因子 |
| `notebooks/explore.ipynb` | 数据探索 + Alpha 因子计算示例 |
| `.gitignore` | 忽略 Token、数据库、本地文档等敏感/冗余文件 |
| `requirements.txt` | `pip install -r requirements.txt` 所需依赖 |

---

## 三、各文件使用说明

### 1. requirements.txt

- **用途**：定义项目 Python 依赖。
- **使用**：
  ```bash
  pip install -r requirements.txt
  ```
  安装 tushare、pandas（`sqlite3` 为 Python 标准库，无需单独安装）。

---

### 2. src/config.py

- **用途**：集中管理所有全局参数，避免在业务代码中硬编码。
- **主要配置项**：

  | 变量 | 说明 | 默认值 |
  |------|------|--------|
  | `TUSHARE_TOKEN` | Tushare Pro Token（必填） | `"your_tushare_token"` |
  | `START_DATE` | 数据开始日期（YYYYMMDD） | `"20220101"` |
  | `END_DATE` | 数据结束日期（YYYYMMDD） | `"20250222"` |
  | `UNIVERSE_INDEX` | 股票池指数代码 | `"000300.SH"` |
  | `DB_PATH` | 数据库路径（相对于 src/） | `"../data/stock_data.db"` |
  | `SLEEP_PER_CALL` | 每次 API 调用间隔（秒） | `0.2` |

- **注意**：此文件含 Token，已在 `.gitignore` 中排除，**切勿提交至 git**。

---

### 3. src/data_loader.py

- **用途**：项目唯一的数据 I/O 层，封装为 `DataEngine` 类。

- **数据库表结构**：

  | 表名 | 字段 | 说明 |
  |------|------|------|
  | `daily_price` | code, date, open, high, low, close, vol, **amount** | 日线行情，主键 (code, date) |
  | `daily_basic` | code, date, pe, pb, total_mv | 每日基本面指标，主键 (code, date) |
  | `stock_info` | code, name, industry | 股票静态信息（行业分类），主键 code |
  | **`adj_factor`** | code, date, adj_factor | 复权因子（前/后复权计算用），主键 (code, date) |

  > **VWAP 计算**：`amount`（千元）× 1000 ÷（`vol`（手）× 100）= `amount × 10 / vol` 元/股。`vol` 与 `amount` 在任何情况下**不复权**。

  > **模式迁移**：若数据库由旧版下载（缺少 `amount` 列或 `adj_factor` 表），调用 `init_db()` 可自动添加缺失列/表。缺少 `amount` 的历史行不会自动补充，需删除对应股票的 `daily_price` 数据后重新下载。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__()` | 初始化 Tushare Pro API，解析数据库路径 |
  | `init_db()` | 建表 + 模式迁移（幂等，可重复调用） |
  | `download_data()` | 下载 price/basic/adj_factor；两套独立缓存，可单独补充缺失的 adj_factor |
  | `fetch_latest_adj_factor(codes)` | 即时调用 Tushare API 获取最新复权因子，供前复权使用 |
  | `load_data()` | 从 SQLite 读取，返回结构化字典（见下） |

- **`load_data()` 返回值**：
  ```python
  {
    "df_price"    : DataFrame  # MultiIndex (date, code)，列：open, high, low, close, vol, amount
    "df_mv"       : DataFrame  # MultiIndex (date, code)，列：total_mv
    "df_industry" : DataFrame  # index = code，列：name, industry
    "df_adj"      : DataFrame  # MultiIndex (date, code)，列：adj_factor
  }
  ```

- **缓存机制**：`download_data()` 对 `daily_price` 和 `adj_factor` 维护**各自独立**的缓存集合，允许在 price 已缓存的情况下单独补充 adj_factor（反之亦然）。如需完全重新下载，删除 `data/stock_data.db` 后重新运行即可。

---

### 4. src/alphas.py

- **用途**：因子计算模块，复现《101 Formulaic Alphas》（Kakushadze, 2015）中的 Alpha 因子。纯计算逻辑，不涉及 I/O；输入为 `DataEngine.load_data()` 返回的数据字典，输出为原始因子值（保留 NaN/inf，清洗由后续模块处理）。

- **已实现因子**：

  | 因子 | 公式 | 说明 |
  |------|------|------|
  | `alpha006` | `-1 * correlation(open, volume, 10)` | 开盘价与成交量的滚动相关性取反 |
  | `alpha012` | `sign(delta(volume, 1)) * (-1 * delta(close, 1))` | 成交量方向乘以收盘价变动反向 |
  | `alpha038` | `(-1 * rank(ts_rank(close, 10))) * rank(close/open)` | 近期高位且高涨幅的股票做空 |
  | `alpha041` | `sqrt(high * low) - vwap` | 高低价几何均值与成交均价之差（vwap ≈ (H+L+C)/3） |
  | `alpha101` | `(close - open) / (high - low + 0.001)` | 日内动量：价格区间归一化的涨跌幅 |

- **已实现辅助函数**：`_rank`、`_delay`、`_delta`、`_corr`、`_cov`、`_stddev`、`_sum`、`_product`、`_ts_min`、`_ts_max`、`_ts_argmax`、`_ts_argmin`、`_ts_rank`、`_scale`、`_decay_linear`、`_sign`、`_log`、`_abs`、`_signed_power`

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(data_dict)` | 接收数据字典，将 MultiIndex DataFrame 拆解为宽表形式 |
  | `alpha006()` … `alpha101()` | 各返回 wide-form DataFrame（行 = 日期，列 = 股票代码） |
  | `get_all_alphas()` | 计算所有已实现因子，返回 MultiIndex (date, code) × alpha 列的 DataFrame |

- **使用示例**：
  ```python
  from alphas import Alpha101
  data = DataEngine().load_data()

  alpha = Alpha101(data)
  df = alpha.get_all_alphas()   # MultiIndex (date, code) × ['alpha006', ...]
  ```

---

### 5. notebooks/explore.ipynb

- **用途**：两部分演示：  
  - **Part 1**：数据库完整性校验（1.1 OHLCV+amount、1.2 adj_factor 历史浏览、1.3 市值、1.4 行业分布、1.5 缺失值检查）。  
  - **Part 2**：Alpha 因子计算（单独宽表、合并长表、描述统计、NaN 覆盖率、最新截面）。
- **使用**：
  ```bash
  jupyter notebook notebooks/explore.ipynb
  ```
  或在 VS Code / Cursor 中直接打开并逐单元格运行。
- **前提**：已运行 `download_data()`，`data/stock_data.db` 已生成。

---

### 6. data/stock_data.db

- **用途**：本地 SQLite 数据库，存储**四张表**：`daily_price`（含 amount）、`daily_basic`、`stock_info`、`adj_factor`。
- **生成方式**：由 `DataEngine.init_db()` + `download_data()` 自动生成，无需手动创建。
- **模式迁移**：重新调用 `init_db()` 可对旧版数据库自动补全 `amount` 列与 `adj_factor` 表。
- **git 状态**：已在 `.gitignore` 中排除（文件较大，且可随时重新生成）。

---

## 四、推荐使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 填写 Token（编辑 src/config.py，将 'your_tushare_token' 替换为真实 Token）

# 3. 建库并下载数据（每股约 3 次接口：daily + daily_basic + adj_factor）
python - <<EOF
import sys; sys.path.insert(0, 'src')
from data_loader import DataEngine
engine = DataEngine()
engine.init_db()        # 建表 + 模式迁移（幂等）
engine.download_data()  # 下载并缓存；中断后重跑可断点续传
EOF

# 4. 打开 Notebook 探索数据及因子示例
jupyter notebook notebooks/explore.ipynb
```

> **耗时估算**：约 300 只股票 × 每股 3 次接口 + 限频 sleep，预计 15 ~ 40 分钟。  
> 因子计算（`get_all_alphas()`）为纯内存运算，通常数秒内完成。

---

## 五、后续更新说明

- **新增脚本或目录**：在第二节「项目架构」目录树与表格中补充。
- **新增功能模块**（因子计算、回测等）：在第三节补充对应条目。
- **数据库表结构变动**：更新第三节中的「数据库表结构」表格。
- **使用流程变化**：更新第四节。

---

> `prompt.md` 仅供本地查阅，已在 `.gitignore` 中排除，不会推送至远程仓库。  
> 本说明文件（Instruction.md）已纳入版本控制，会随项目进展持续更新并推送至远程仓库。  
> 远程仓库的公开说明文档（README.md）将在项目后期单独编写。
