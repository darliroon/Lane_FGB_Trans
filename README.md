# lane_trans

把车道中心线里的**小环**消掉：输入一个 FlatGeobuf（`input.fgb`），输出一个 FlatGeobuf（`output.fgb`）。
消环的算法叫**内核**（kernel），目前有 `v0` 和 `v1`；换内核只改一个参数，文件接口不变。

```bash
lane-trans data/start.fgb output.fgb                  # 默认内核 v0
lane-trans data/start.fgb output.fgb --kernel v1
```

```python
from lane_trans import convert

report = convert("data/start.fgb", "output.fgb", kernel="v0")
print(report.loops_before, "->", report.loops_after)   # 11 -> 0
```

## 安装

需要 Python ≥ 3.10（在 3.13 上验证过）。

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # 核心 + Flask + pytest
```

只装核心用 `pip install -e .`；跑 `scripts/` 里的出图脚本另需 `pip install -e ".[viz]"`。

## 输入与输出

| | 要求 / 内容 |
|---|---|
| 输入 | LineString 图层，**投影坐标系**（单位米，例如 EPSG:3857），带整数列 `Id`。MultiLineString、经纬度坐标、缺 `Id` 会报错并说明原因。空图层允许。 |
| 输出 | LineString 图层，坐标系同输入。列：`Id`（1..n）、`src_ids`（该链来自哪些输入 `Id`，逗号分隔）、`modified`（几何是否被平均过）。 |

输出**不带**输入的其它属性列（`RoadId`、`LaneType`、`TurnType` 等）：合并后的一条链可能来自多条车道，没有唯一来源。
需要这些属性时，用 `src_ids` 回查输入。

FlatGeobuf 按空间索引顺序存放要素，读回来的 `Id` 顺序不是 1..n，这是格式本身的行为。

## 内核

| 名称 | 每个小环删哪条链 | 文件 |
|---|---|---|
| `v0`（默认） | 先找“横插连接线”（两端都被路口上另外两条链贯通得更好），没有再删“备选路径与它最像”的链 | [kernels/v0.py](src/lane_trans/kernels/v0.py) |
| `v1`、`v1:weakest_tree` | 两端“直行贯通”程度较差的那条 | [kernels/v1.py](src/lane_trans/kernels/v1.py) |
| `v1:shortest` | 最短的链 | 同上 |
| `v1:longest` | 最长的链 | 同上 |
| `v1:hausdorff` | 备选路径与它 Hausdorff 距离最小的链 | 同上 |

所有内核共用同一个前端（吸附悬挂端点、删自环、平行链取平均），只在“怎么剪断剩下的环”上不同。算法细节见 [docs/algorithms.md](docs/algorithms.md)。

在 `data/start.fgb`（33 条车道，11 个环）对 `data/target.fgb`（人工修改结果）上的对比，覆盖率按 0.5 m 缓冲计：

| 内核 | result 落在 target 内 | target 落在 result 内 | Hausdorff |
|---|---|---|---|
| `v0` | 0.905 | 0.907 | 1.43 m |
| `v1` | 0.876 | 0.902 | 1.78 m |
| `v1:shortest` | 0.817 | 0.859 | 3.31 m |
| `v1:longest` | 0.925 | 0.885 | 3.31 m |
| `v1:hausdorff` | 0.901 | 0.859 | 6.28 m |

五个内核都把环数降到 0，并保留全部 13 个悬挂端点。**这只是一个约 100 m 路段的样本**，不能当作通用排名；
默认内核是 `v0`，因为它在这份数据上两个方向的覆盖率都 ≥ 0.9 且 Hausdorff 最小。想改默认值，改 [api.py](src/lane_trans/api.py) 里的 `DEFAULT_KERNEL`。

**v0 是冻结的基线。** `tests/test_kernels.py::test_v0_output_matches_the_frozen_v0_result` 把它在 `start.fgb` 上的输出锁在
[tests/fixtures/v0_result.fgb](tests/fixtures/v0_result.fgb)。要改变 v0 的行为，请新增内核而不是改它；确实要改时，同一次提交里更新这个文件并写明原因。

### 阈值

`convert()` 和命令行都能覆盖，只对当次调用生效：

| 参数 | 默认 | 含义 |
|---|---|---|
| `max_loop_len` / `--max-loop-len` | 200 m | 双连通分量总长超过它的环视为真实环（环岛等），不动 |
| `connector_margin` / `--connector-margin` | 0.10 | 仅 `v0`。连接线判据的余量 |
| `parallel_ratio` / `--parallel-ratio` | 1.5 | 平行链里比最短的长出这个倍数的，直接丢弃而不是取平均 |
| `max_loop_width` / `--max-loop-width` | 不限制 | 宽于它的环、自环链、平行链都视为真实道路（环岛、双向分离的两条车道线），不动。宽度 = 最小外接矩形的短边 |

## 新增一个内核

内核是一个函数 `kernel(C, origin, params, on_step=None) -> list[str]`，原地修改链图 `C`，返回每次改动的日志行。
共用前端已经封装在 `pipeline.simplify`，你只需要提供“剪哪条链”的 `cut`：

```python
# src/lane_trans/kernels/v2.py
from ..chains import fmt, small_loops
from ..pipeline import simplify
from .registry import register


def cut(C, origin, params):
    """Delete one chain of one small loop; return its log line, or None if no small loop is left."""
    H, comps = small_loops(C, params.max_loop_len)
    ...


@register("v2")
def run(C, origin, params, on_step=None):
    return simplify(C, origin, params, cut, on_step)
```

然后在 [kernels/\_\_init\_\_.py](src/lane_trans/kernels/__init__.py) 的 import 行加上 `v2`。命令行的 `--kernel` 选项、web 接口和 `tests/test_kernels.py`
里按全部内核参数化的用例（环全部消除、悬挂端点保留、空输入）都会自动覆盖它。

## 目录结构

```
lane_trans/
├── src/lane_trans/
│   ├── api.py          convert()：读入校验 → 内核 → 写出
│   ├── cli.py          命令行（lane-trans）
│   ├── graph.py        FGB → 顶点图；吸附悬挂端点
│   ├── chains.py       链图：度 ≠ 2 的节点之间的折线
│   ├── rules.py        共用前端：删自环、平行链取平均
│   ├── pipeline.py     消环主循环（内核只提供 cut）
│   ├── geometry.py     折线工具
│   ├── metrics.py      统计量、与 target 的覆盖率
│   ├── params.py       阈值
│   ├── kernels/        registry.py · v0.py · v1.py
│   └── web/            浏览器画图 → 转换（lane-trans-web）
├── tests/              pytest；fixtures/ 放冻结的 v0 结果
├── scripts/            出图与分析脚本；exploration/ 是早期探索
├── docs/               algorithms.md、figures/
└── data/               start.fgb（输入样例）、target.fgb（人工结果，用于评估）
```

## 测试

```bash
pytest -q
```

`tests/test_web.py` 需要 Flask，没装时自动跳过。

## 脚本与出图

在仓库根目录运行（需要 `.[viz]`）。写进 `docs/figures/` 的图被文档引用并随仓库提交；其余输出在 `outputs/`，已被 git 忽略。

| 脚本 | 产出 |
|---|---|
| `scripts/compare_v0_v1.py` | v0 与 v1 四个变体对 target 的总览图，并打印指标 |
| `scripts/compare_v0_v1_detail.py` | 两个有分歧的环的放大对比，加 `connector_margin` 敏感性扫描 |
| `scripts/explain_weakest_tree.py` | v1 最弱生成树边的逐步讲解（4 张图） |
| `scripts/visualize_loops.py [--kernel v1]` | 一次运行的逐步动画（GIF + 总览） |
| `scripts/exploration/` | 早期探索：车道级对比、顶点图导出、Steiner 树可行性试验 |

## 网页

```bash
lane-trans-web            # http://127.0.0.1:8765
```

页面上画一张图，点转换，看优化后的图。接口 `POST /api/convert` 另接受可选字段 `"kernel"`（默认 `v0`），页面本身暂时没有内核选择框。
