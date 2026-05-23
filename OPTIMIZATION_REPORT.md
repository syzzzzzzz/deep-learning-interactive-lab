# 性能优化与内存减耗报告

## 优化范围

本次优化聚焦 Streamlit 交互页面、主入口导航、matplotlib figure 生命周期、重复计算缓存和 Plotly 渲染路径。项目不是 git 仓库，因此改动按文件列出。

## 改动文件

### `main.py`

- 为 `module_catalog()` 添加 `functools.lru_cache(maxsize=1)`，避免首页每次 rerun 都扫描目录、构建 `ModuleInfo` 列表。
- 新增 `cached_route_map()` 和 `build_route_map()`，复用路由表构建结果；对外 `route_map()` 返回副本，避免调用方意外修改缓存对象。
- 为 `is_streamlit_app()` 添加缓存，减少重复读取模块文件判断 Streamlit 页面。
- 为 `render_feature_cards()` 添加缓存，避免重复拼接静态 HTML。
- 保留导航点击时的 `st.rerun()`，这是 query 参数切换页面所需的单次 rerun，不属于无效重复调用。

效果验证：

- `module_catalog()` 首次构建约 `1.287 ms`。
- 缓存命中约 `0.000700 ms`。
- `route_map()` 返回 116 条路由，缓存副本返回约 `0.029200 ms`。

### `sitecustomize.py`

- 在统一 matplotlib 配置中包装 `plt.show()`，显示结束后执行 `plt.close("all")`。
- 作用于通过项目启动器运行的教学脚本，降低连续运行大量 `plt.show()` 教学脚本时的 figure 累积风险。
- 包装逻辑带幂等标记，避免重复包裹 `plt.show()`。

效果验证：

- 显式导入 `sitecustomize` 后，`plt.show()` 前 figure 数为 `1`，调用后为 `0`。

### `part1_foundations/classical_ml.py`

- 新增 `render_matplotlib(fig)` helper，统一 `st.pyplot()` 后执行 `plt.close(fig)`。
- 将页面内直接 `st.pyplot(fig, ...)` 的 matplotlib 图渲染改为统一 helper。
- 保留原有缓存数据函数：`linear_regression_data`、`logistic_data`、`tree_data`、`kmeans_data`、`svm_data`、`knn_data`。

效果：

- 页面切换算法、调整控件后不再把旧 figure 长期留在 matplotlib 全局状态中。

### `part1_foundations/math_primer.py`

- 新增 `render_matplotlib(fig)` helper。
- 将采样直方图的 `st.pyplot(fig_mpl, ...)` 改为渲染后关闭 figure。

效果：

- 高频滑动采样数量时降低 matplotlib figure 泄漏风险。

### `part1_foundations/machine_learning_basics.py`

- 新增 `render_matplotlib(fig)` helper。
- 将阈值 tradeoff 的 matplotlib 图渲染后关闭。
- 将通用散点 helper 从 `go.Scatter` 改为 `go.Scattergl`，让较多点的散点图走 WebGL 渲染。

效果：

- `Scattergl` 静态检查命中 1 处。
- 阈值评估页面不再保留旧 matplotlib figure。

### `part5_toolbox/tuning_challenge.py`

- 为 `learning_curves(...)` 添加 `@st.cache_data(show_spinner=False)`。
- 拆出 `search_landscape_data(scenario_name, batch_size, epochs)` 并添加 `@st.cache_data(show_spinner=False)`，缓存 22x22 搜索地形的数组结果。
- `search_landscape()` 只负责把缓存后的数组组装成 Plotly figure，避免把较重的 figure 对象长期放入缓存。
- 删除“记录本轮实验”按钮后的显式 `st.rerun()`；按钮点击本身会触发 Streamlit 正常 rerun。

效果验证：

- 该文件内 `st.rerun()` 调用数为 `0`。
- 新增缓存函数：`learning_curves`、`search_landscape_data`。

### `part6_universal_framework/06_streamlit_demo.py`

- `run_experiment()` 中将 `history_steps`、`losses`、`accuracies` 转为更小 dtype，减少缓存对象体积。
- 决策边界散点从 `go.Scatter` 改为 `go.Scattergl`。
- 为 `synthetic_image()`、`convolve_image()`、`attention_matrix()` 添加 `@st.cache_data(show_spinner=False)`。
- 保留训练主路径 `run_experiment()` 的缓存。

效果验证：

- `Scattergl` 静态检查命中 2 处。
- 新增缓存函数：`synthetic_image`、`convolve_image`、`attention_matrix`。

## 验证结果

- 语法编译通过：
  - `main.py`
  - `sitecustomize.py`
  - `part1_foundations/classical_ml.py`
  - `part1_foundations/math_primer.py`
  - `part1_foundations/machine_learning_basics.py`
  - `part5_toolbox/tuning_challenge.py`
  - `part6_universal_framework/06_streamlit_demo.py`
- 主入口缓存基准通过。
- `sitecustomize` 的 `plt.show()` 自动关闭逻辑验证通过。
- 静态检查确认：
  - `tuning_challenge.py` 无 `st.rerun()`。
  - `06_streamlit_demo.py` 有 2 处 `Scattergl`。
  - `machine_learning_basics.py` 有 1 处 `Scattergl`。

## 剩余建议

- 仍有大量非 Streamlit 教学脚本直接使用 `plt.show()`；通过 `sitecustomize.py` 已做统一兜底，但如果后续要完全独立运行单文件脚本，建议逐步改成 `savefig/show` 后显式 `plt.close(fig)`。
- 若需要更精确的性能数据，建议后续用 Streamlit 的页面级 smoke test 记录首次加载、切换控件、缓存命中的端到端耗时。
