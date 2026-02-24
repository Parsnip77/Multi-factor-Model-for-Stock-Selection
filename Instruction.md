# 项目说明 (Instruction)

本文档用简明语言说明本仓库中**所有文件与代码的用途、使用方式**，并展示项目架构。随项目推进会持续更新。

---

## 一、项目概述

本项目是一个**多因子选股模型**的量化交易示例，使用 Python 实现从数据获取、因子构建到回测的完整流程。

**当前阶段：数据层（ETL）+ 因子计算层 + 因子预处理层（第一阶段完成）+ 因子评估层（第二阶段完成）**  
用 Tushare Pro 拉取沪深300成分股的日线行情与基本面指标，写入本地 SQLite 数据库；在此基础上复现《101 Formulaic Alphas》中的经典因子；再通过预处理模块对原始因子执行去极值、标准化、中性化，最终由 `data_preparation_main.py` 串联全流程，将四张核对齐的宽表导出为 Parquet 文件，供后续回测使用。

---

## 二、项目架构（文件与目录）

```
项目根目录/
├── data/
│   ├── .gitkeep                # 保留空目录用于 git 追踪（数据文件本身不上传）
│   ├── stock_data.db           # SQLite 数据库（运行 download_data() 后生成）
│   ├── prices.parquet          # 原始行情 + 复权因子（由总脚本生成）
│   ├── meta.parquet            # 市值、行业、PE、PB（由总脚本生成）
│   ├── factors_raw.parquet     # 原始 Alpha 因子（由总脚本生成）
│   └── factors_clean.parquet   # 清洗后 Alpha 因子（由总脚本生成）
├── src/
│   ├── __init__.py             # 标识 src 为 Python 包
│   ├── config.py               # 全局配置（Token、日期范围、路径等）
│   ├── data_loader.py          # DataEngine 类：数据下载与读取
│   ├── alphas.py               # Alpha101 类：因子计算（101 Formulaic Alphas）
│   ├── preprocessor.py         # FactorCleaner 类：因子预处理（清洗）
│   ├── targets.py              # calc_forward_return：未来收益率标签生成
│   └── ic_analyzer.py          # calc_ic / calc_ic_metrics / plot_ic：因子 IC 评估
├── notebooks/
│   └── explore.ipynb           # Jupyter Notebook：数据探索与可视化
├── plots/                      # IC 分析图表输出目录（由 analyze_main.py 自动创建）
├── data_preparation_main.py    # 第一阶段总脚本：串联所有组件，导出 Parquet
├── analyze_main.py             # 第二阶段总脚本：因子 IC 评估与有效因子筛选
├── .gitignore                  # 版本控制忽略规则
├── requirements.txt            # Python 依赖列表
├── prompt.md                   # 项目需求与规范（仅本地查阅）
└── Instruction.md              # 本说明文档
```

| 文件/目录 | 用途 |
|-----------|------|
| `data/` | 存放 SQLite 数据库文件（不上传至 git） |
| `src/config.py` | 全局参数配置，包含 Tushare Token（不上传至 git） |
| `src/data_loader.py` | `DataEngine` 类：数据下载、缓存、读取 |
| `src/alphas.py` | `Alpha101` 类：复现《101 Formulaic Alphas》中的 5 个因子 |
| `src/preprocessor.py` | `FactorCleaner` 类：对原始因子执行去极值、标准化、中性化 |
| `src/targets.py` | `calc_forward_return(prices_df, d)`：计算 d 日未来收益率标签 |
| `src/ic_analyzer.py` | `calc_ic` / `calc_ic_metrics` / `plot_ic`：截面 Spearman IC 评估 |
| `data_preparation_main.py` | 第一阶段总脚本：串联 DataEngine → Alpha101 → FactorCleaner，导出四张 Parquet |
| `analyze_main.py` | 第二阶段总脚本：载入 Parquet，循环单因子 IC 检验，筛选有效 alpha |
| `plots/` | IC 分析图表输出目录（`analyze_main.py` 运行时自动创建），每个因子生成 `{factor}_ic.png` |
| `data/*.parquet` | 导出的宽表数据，共享主键 (trade_date, ts_code) |
| `notebooks/explore.ipynb` | 数据探索 + Alpha 因子计算 + 因子清洗示例 |
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
  | `alpha041` | `sqrt(high * low) - vwap` | 高低价几何均值与成交均价之差（精确 vwap） |
  | `alpha101` | `(close - open) / (high - low + 0.001)` | 日内动量：价格区间归一化的涨跌幅 |

- **VWAP 计算**：`amount` 字段可用时使用精确公式 `amount × 10 / vol`；否则回退到典型价格 `(H+L+C)/3`。

- **复权模式**（`adj_type` 参数，默认 `"forward"`）：

  | 模式 | 公式 | 说明 |
  |------|------|------|
  | `"forward"` | `P × adj_factor / adj_factor_latest` | 前复权，最新价等于原始价 |
  | `"backward"` | `P × adj_factor` | 后复权，完整体现历史涨幅 |
  | `"raw"` | 不调整 | 直接使用数据库原始价格 |

  > `vol` 和 `amount` 在任何复权模式下均**不调整**。  
  > `adj_factor_latest` 默认取 `df_adj` 中的最后一行；也可通过 `latest_adj` 参数传入 `fetch_latest_adj_factor()` 的返回值以获得更精确的当日因子。

- **已实现辅助函数**：`_rank`、`_delay`、`_delta`、`_corr`、`_cov`、`_stddev`、`_sum`、`_product`、`_ts_min`、`_ts_max`、`_ts_argmax`、`_ts_argmin`、`_ts_rank`、`_scale`、`_decay_linear`、`_sign`、`_log`、`_abs`、`_signed_power`

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(data_dict, adj_type, latest_adj)` | 接收数据字典；自动计算精确 vwap 并应用复权 |
  | `alpha006()` … `alpha101()` | 各返回 wide-form DataFrame（行 = 日期，列 = 股票代码） |
  | `get_all_alphas()` | 计算所有已实现因子，返回 MultiIndex (date, code) × alpha 列的 DataFrame |

- **使用示例**：
  ```python
  from alphas import Alpha101
  data = DataEngine().load_data()

  alpha = Alpha101(data)                         # 前复权（默认）
  alpha = Alpha101(data, adj_type='backward')    # 后复权
  alpha = Alpha101(data, adj_type='raw')         # 不复权

  # 使用 API 获取最新复权因子（更精确的前复权）
  latest_adj = engine.fetch_latest_adj_factor(codes)
  alpha = Alpha101(data, adj_type='forward', latest_adj=latest_adj)

  df = alpha.get_all_alphas()   # MultiIndex (date, code) × ['alpha006', ...]
  ```

---

### 5. src/preprocessor.py

- **用途**：因子预处理模块，将 `Alpha101.get_all_alphas()` 输出的**原始因子**清洗为**可直接输入模型的因子**，封装为 `FactorCleaner` 类。

- **核心方法**：

  | 方法 | 说明 |
  |------|------|
  | `__init__(data_dict)` | 从数据字典中提取对数市值（宽表）和行业 dummy 矩阵，作为中性化回归的基准 |
  | `winsorize(factor_df, method, limits)` | 截面去极值；`method='mad'`（默认）使用中位数绝对偏差，`method='sigma'` 使用均值±标准差 |
  | `standardize(factor_df)` | 截面 Z-score 标准化：\(Z = (X - \mu) / \sigma\) |
  | `neutralize(factor_df)` | 截面 OLS 回归：`Factor = β · [log_mv, industry_dummies] + ε`，返回残差 ε |
  | `process_all(raw_alphas_df)` | 完整五步流水线（见下），返回 `df_clean_factors` |

- **`process_all` 流水线**（按顺序）：

  | 步骤 | 操作 | 目的 |
  |------|------|------|
  | 1. 异常值初筛 | ±inf → NaN | 防止均值/方差计算出现数学错误 |
  | 2. 去极值 | MAD-based 截断（Median ± 3×1.4826×MAD） | 压缩尾部风险，保护后续 OLS 回归 |
  | 3. 初步标准化 | Z-score | 统一量纲，提升数值稳定性 |
  | 4. 中性化 | OLS 残差（剥离市值 + 行业偏好） | 保留纯 Alpha，消除系统性偏差 |
  | 5. 二次标准化 | Z-score on residuals | 使残差在不同日期间具有可比性 |
  | 6. 填补 NaN | 将缺失值替换为 0 | 统一维度，方便后续矩阵运算 |

- **输入**：
  ```python
  data_dict     # DataEngine.load_data() 的返回值（需含 df_mv 和 df_industry）
  raw_alphas_df # Alpha101.get_all_alphas() 的返回值
                # MultiIndex (date, code) × alpha 列，保留 NaN/inf
  ```

- **输出**：
  ```python
  df_clean_factors  # MultiIndex (date, code) × alpha 列
                    # 与输入相同结构，NaN 已填补为 0
  ```

- **使用示例**：
  ```python
  from preprocessor import FactorCleaner

  cleaner = FactorCleaner(data)            # data = engine.load_data()
  df_clean = cleaner.process_all(df_alphas)  # df_alphas = alpha.get_all_alphas()
  ```

---

### 6. notebooks/explore.ipynb

- **用途**：三部分演示：  
  - **Part 1**：数据库完整性校验（1.1 OHLCV+amount、1.2 adj_factor 历史浏览、1.3 市值、1.4 行业分布、1.5 缺失值检查）。  
  - **Part 2**：Alpha 因子计算（前复权价格 + 精确 vwap；单独宽表、合并长表、描述统计、NaN 覆盖率、最新截面）。  
  - **Part 3**：因子清洗（`FactorCleaner.process_all()` 输出的清洗因子；描述统计、零填充率、最新截面对比）。
- **使用**：
  ```bash
  jupyter notebook notebooks/explore.ipynb
  ```
  或在 VS Code / Cursor 中直接打开并逐单元格运行。
- **前提**：已运行 `download_data()`，`data/stock_data.db` 已生成。

---

### 7. data_preparation_main.py

- **用途**：第一阶段端到端总脚本，串联 `DataEngine → Alpha101 → FactorCleaner`，将全流程处理结果导出为四张 Parquet 文件。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 检查 `data/stock_data.db` 是否存在，缺失则报错退出 |
  | 2 | 调用 `DataEngine.load_data()` 载入全量数据 |
  | 3 | 调用 `Alpha101.get_all_alphas()` 计算原始 Alpha 因子（前复权） |
  | 4 | 调用 `FactorCleaner.process_all()` 执行五步清洗流水线 |
  | 5 | 将结果导出为四张 Parquet 文件（见下） |

- **输出文件**（均存放于 `./data/`，主键逻辑对齐：`trade_date × ts_code`）：

  | 文件 | 列 | 说明 |
  |------|----|------|
  | `prices.parquet` | trade_date, ts_code, open, high, low, close, volume, adj_factor | 原始（未复权）行情 + 复权因子 |
  | `meta.parquet` | trade_date, ts_code, total_mv, industry, pe, pb | 每日基本面 + 静态行业分类 |
  | `factors_raw.parquet` | trade_date, ts_code, alpha006, … | 原始因子值（保留 NaN） |
  | `factors_clean.parquet` | trade_date, ts_code, alpha006, … | 清洗后因子值（NaN 填补为 0） |

- **使用**：
  ```bash
  python data_preparation_main.py
  ```
  脚本会逐步打印每个阶段的状态和关键统计信息。

---

### 8. src/targets.py

- **用途**：标签生成模块，计算未来 d 日收益率，为 IC 分析提供 target。

- **核心函数**：

  | 函数 | 说明 |
  |------|------|
  | `calc_forward_return(prices_df, d)` | 输入平表 prices_df（含 trade_date / ts_code / close）和前向天数 d，计算 `(close_{T+d} - close_T) / close_T`，返回 MultiIndex `(trade_date, ts_code)` × `forward_return` 的长表 |

- **使用示例**：
  ```python
  import pandas as pd
  from targets import calc_forward_return

  prices_df = pd.read_parquet("data/prices.parquet")
  target_df = calc_forward_return(prices_df, d=5)  # 5-day forward return
  ```

---

### 9. src/ic_analyzer.py

- **用途**：因子 IC（信息系数）评估模块，衡量因子值与未来收益率的截面相关性。

- **核心函数**：

  | 函数 | 说明 |
  |------|------|
  | `calc_ic(factors_df, target_df)` | 按 `(trade_date, ts_code)` merge 合并；`groupby('trade_date')` 计算截面 Spearman 相关；返回 `ic_series`（index = trade_date） |
  | `calc_ic_metrics(ic_series)` | 计算 IC 均值、标准差、ICIR（= 均值 / 标准差），返回字典 |
  | `plot_ic(ic_series, factor_name, show)` | 绘制 IC 时间序列柱状图 + 累计 IC 折线图双子图，返回 `Figure` 对象 |

- **使用示例**：
  ```python
  from ic_analyzer import calc_ic, calc_ic_metrics, plot_ic

  ic_series = calc_ic(single_factor_df, target_df)
  metrics   = calc_ic_metrics(ic_series)  # {'ic_mean': ..., 'ic_std': ..., 'icir': ...}
  plot_ic(ic_series, factor_name="alpha006")
  ```

---

### 10. analyze_main.py

- **用途**：第二阶段端到端总脚本，串联 targets → ic_analyzer，逐因子输出 IC 评估报告，并筛选有效因子。

- **执行流程**：

  | 步骤 | 操作 |
  |------|------|
  | 1 | 载入 `prices.parquet` 与 `factors_clean.parquet` |
  | 2 | 调用 `calc_forward_return(prices_df, d=5)` 生成 target |
  | 3 | 遍历每个 alpha 列，依次计算 IC 时间序列、IC metrics，展示图表 |
  | 4 | 筛选满足 `abs(IC mean) > 0.02` 且 `abs(ICIR) > 0.5` 的因子并输出列表 |

- **使用**：
  ```bash
  python analyze_main.py
  ```

- **可调配置**（脚本顶部常量）：

  | 变量 | 默认值 | 说明 |
  |------|--------|------|
  | `FORWARD_DAYS` | `1` | 未来收益率天数 |
  | `IC_MEAN_THRESHOLD` | `0.02` | IC 均值绝对值阈值 |
  | `ICIR_THRESHOLD` | `0.50` | ICIR 绝对值阈值 |
  | `SHOW_PLOTS` | `True` | 是否交互展示 IC 图表 |

---

### 11. data/stock_data.db

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

# 4. 运行第一阶段总脚本（因子计算 + 清洗 + 导出 Parquet）
python data_preparation_main.py

# 5. 运行第二阶段总脚本（因子 IC 评估 + 有效因子筛选）
python analyze_main.py

# 6. （可选）打开 Notebook 交互探索数据及因子
jupyter notebook notebooks/explore.ipynb
```

> **耗时估算**：数据下载约 15 ~ 40 分钟（300 只股票 × 3 次接口 + 限频 sleep）。  
> `data_preparation_main.py` 为纯内存运算，通常在 1 分钟内完成。  
> `analyze_main.py` 为纯内存运算，通常在 1 分钟内完成（含绘图）。

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
