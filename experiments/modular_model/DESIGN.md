# 模块化模型开发计划

本文档定义 pypto-lib 最终希望提供的模型开发体验，以及从当前实现逐步达到该目标的实施计划。
在相关前端契约正式确认以前，文中的 API 名称均为设计示意。

## 目标

适配一个新模型，主要工作应当是组合可复用的语义模块并提供模型配置，而不是复制 kernel、
创建按 shape 命名的函数、手工绑定 tiling 参数，或者重新开发整套运行时编排。

目标分层如下：

```text
pypto-serving：请求调度、batch/token bucket、artifact dispatch
        ↓
pypto-lib：接近 Torch 的 Module、基础算子、参数和模型组合接口
        ↓
pypto-lib：shape/layout/分布式属性传播、profile 和 schedule 选择
        ↓
pypto-lib：展开为只包含显式 JIT 调用和 constexpr 绑定的 program
        ↓
pypto：接近 Triton 的 kernel DSL、constexpr specialization、lowering 和编译缓存
        ↓
PyPTO orchestration、InCore kernel 和分布式任务图
        ↓
NPU runtime
```

pypto-lib 在模型组合层借鉴 Torch，在 workload/profile 管理上借鉴 vLLM；PyPTO 在 kernel 模板层
借鉴 Triton，同时保留自身多级程序表示和 NPU 任务编排能力。

## 组件归属边界

`pypto` 只提供 kernel 及其以下的通用编译能力：Tensor/kernel DSL、`pl.constexpr`、JIT 函数依赖、
lowering、codegen、目标信息和 kernel artifact cache。它不认识 `Module`、RMSNorm、模型配置、
checkpoint、schedule registry 或 serving workload。

kernel 以上的能力均由 pypto-lib 提供：

- `Module`、Parameter schema、基础算子库和模型库；
- shape、dtype、layout、动态维度和分布式契约传播；
- target-aware schedule registry、fallback 和可选 autotune；
- decode/prefill profile、模型变体规划和编译 bundle；
- 将 Module/算子图展开成普通 PyPTO JIT 调用图的 elaboration 层。

pypto-serving 负责在线请求、KV cache、batch/token bucket 和已编译 bundle 的选择。生产 serving
不应在一次普通请求的 guard miss 后临时编译未知模型变体。

这一边界有一个直接约束：**不能为了支持 Module 调用而让 PyPTO 前端理解
`module(...)` 或 `module.forward(...)`。** pypto-lib 必须在调用 PyPTO 编译器前，将 Module 树
规范化为 PyPTO 已经理解的 JIT 函数、Tensor 契约和显式 constexpr 参数。

## 最终的模型开发接口

模型开发者应当只编写语义组合：

```python
from pypto_lib import nn


class DeepSeekDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = nn.DeepSeekAttention(config)
        self.post_norm = nn.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = nn.DeepSeekMoE(config)

    def forward(self, hidden_states, positions, kv_cache):
        residual = hidden_states
        hidden_states = self.input_norm(hidden_states)
        hidden_states = self.attention(hidden_states, positions, kv_cache)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
```

编译和权重加载应当是模型级操作：

```python
from pypto_lib import CompileProfile, compile_model


model = DeepSeekV4(config)
bundle = compile_model(
    model,
    target="a2a3",
    profiles=[
        CompileProfile.decode(dynamic_tokens=(1, 4096), batch_buckets=(1, 2, 4, 8)),
        CompileProfile.prefill(dynamic_tokens=(1, 32768)),
    ],
)
bundle.load_state_dict(weights)
output = bundle.dispatch(input_ids, positions, kv_cache)
```

模型开发接口不应暴露：

- tile size、pipeline stage 或 core 划分；
- `pl.constexpr` 绑定或显式 `.specialize()`；
- `rms_norm_4096` 这类按 shape 创建的 Python 别名；
- 内部生成的函数符号和编译缓存键；
- 普通中间输出 buffer 的手工分配；
- 生产模型代码中的 Golden Harness `TensorSpec`。

## 最终的 kernel 开发接口

Kernel 开发者仍然保留显式的编译期控制：

```python
@pl.jit.inline
def rms_norm_kernel(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Tensor,
    *,
    HIDDEN_SIZE: pl.constexpr,
    EPS: pl.constexpr,
    HIDDEN_TILE: pl.constexpr,
    TOKEN_TILE: pl.constexpr,
    PIPELINE_STAGE: pl.constexpr,
):
    ...
```

前端负责在调用点绑定 constexpr，将其从运行时 ABI 中移除，并纳入编译缓存 identity。
同一个 program 中允许存在同一源码函数的多个 specialization。

pypto-lib 的语义 Module 对模型开发者隐藏 kernel 接口：

```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.ParameterSpec([hidden_size], dtype=pl.BF16)

    def forward(self, x):
        schedule = schedules.select(
            "rms_norm",
            target=self.compile_context.target,
            dtype=x.dtype,
            hidden_size=self.hidden_size,
            tokens=x.shape[0],
        )
        out = self.compile_context.empty_like(x)
        return rms_norm_kernel(
            x,
            self.weight,
            out,
            HIDDEN_SIZE=self.hidden_size,
            EPS=self.eps,
            **schedule.constexprs(),
        )
```

上面的代码展示语义 Module 的效果，不要求 `forward()` 直接在 PyPTO tracing 中执行。
pypto-lib 的 elaboration 可以把它规范化为 `BoundOperator`/`ProgramSpec`，最终输出只包含普通
PyPTO JIT 调用和显式 constexpr 绑定的 program。无论内部表示如何，必须保持相同的
specialization 和缓存语义。

## 最终的 schedule 接口

Tiling 应注册到独立、target-aware 的 schedule registry，而不是写在模型文件中：

```python
from pypto_lib import schedules


@schedules.register("rms_norm", target="a2a3")
def rms_norm_a2a3(meta):
    if meta.hidden_size <= 4096:
        return schedules.Config(HIDDEN_TILE=128, TOKEN_TILE=8, PIPELINE_STAGE=2)
    return schedules.Config(HIDDEN_TILE=256, TOKEN_TILE=16, PIPELINE_STAGE=3)
```

每个算子都应为其声明支持的 shape domain 提供一个 correctness-first 的 fallback schedule。
已调优的 schedule 可以覆盖已知 target 和 shape 范围。Autotune 是可选能力，其缓存键必须明确且
有界，例如由 target、dtype、hidden size 和 token 范围组成。

## 设计原则

1. **语义与 schedule 分离。** 模型配置描述“计算什么”，target 配置决定“如何实现”。
2. **kernel 边界的编译期参数保持显式。** 生成代码不能被隐藏的 Python 全局变量控制。
3. **模型边界隐藏编译器机制。** Specialization 对象、内部符号和缓存管理属于框架职责。
4. **Module 组合必须结构化。** 子模块、参数、buffer 和名称形成稳定、可检查、可加载权重的树。
5. **契约只声明一次。** Shape、dtype、layout、动态维度和分布式信息应沿调用关系传播，
   不应在每个 wrapper 中重复。
6. **保留专家逃生通道。** Kernel 开发者可以显式选择 schedule 或直接调用底层 JIT 函数，
   用于正确性和性能调试。
7. **真实硬件验证。** Simulator 用于验证 lowering 和正确性可移植性；运行时 ABI、调度和性能结论
   必须经过真实 NPU。

## vLLM、Torch 和 Triton 中的 specialization

“specialization”在这条技术链上不是一个动作，而是不同层次的多次决策：

```text
vLLM workload/profile dispatch
        ↓  选择 batch/token bucket、compiled graph、CUDA Graph
TorchDynamo graph guards
        ↓  选择或重新捕获 FX graph，决定静态维与动态维
TorchInductor graph lowering
        ↓  分区、融合并选择 kernel/template
Triton JIT specialization/autotune
        ↓  constexpr、类型、对齐、target、config 形成 kernel variant
已编译 binary cache / CUDA Graph replay
```

### vLLM：对 workload 和已编译执行路径做 specialization

vLLM 知道请求处于 decode 还是 prefill、batch 中有多少 token，以及可用的 capture size。它通过
compile sizes/ranges、padding bucket、piecewise/full CUDA Graph 和 dispatcher 选择执行路径。
这是一层 serving policy；vLLM 本身不定义 Triton `constexpr` 如何进入 kernel cache key。

因此，vLLM 中“batch size 8 的图”和 Triton 中“`BLOCK_SIZE=128` 的 kernel”是两种不同的变体。
前者解决在线 workload 如何落到固定执行资源，后者解决一段 kernel 如何生成机器码。

### Torch：以 graph 和 guard 为单位做 specialization

`torch.compile` 由 TorchDynamo 捕获 Python frame 并生成 FX graph，同时为输入类型、shape、Python
全局状态和控制流假设安装 guards。guard 命中时复用已编译 graph；guard 失败时可能重新编译，或者
将变化的维度推广为 dynamic。TorchInductor 再对 graph 分区、融合，并选择外部算子或生成 kernel。

`nn.Module` 本身不是最终的 specialization unit。它提供可遍历的代码和参数结构；真正被缓存的是
“捕获的 graph + guards + compiler options/target”产生的编译结果。Torch 的便利性来自 Module、
graph capture、guard 和后端 lowering 的分层，而不是 Module 自动消除了 specialization。

### Triton：以 kernel 编译签名做 specialization

Triton JIT 接收 tensor/scalar 实参和 `tl.constexpr`。会改变控制流、索引、tile 或资源分配的
constexpr 值，以及实参类型、对齐属性、target 和编译 options，共同决定 kernel variant 和缓存。
`triton.autotune` 又在其显式 `key` 变化时评测候选 `Config`，选择 tile、warp 和 stage 等参数。

这说明语义 shape 和 tiling 不应混成一个参数：hidden size 可能是算子语义常量，`BLOCK_SIZE`
是 schedule 选择；二者最终都可绑定为 kernel constexpr，但所有权和变化原因不同。

### CUDA Graph：固定资源实例，不等于编译 specialization

CUDA Graph capture 依赖固定 shape/地址等运行时条件。vLLM 可以对同一个已编译 graph 捕获多个
batch bucket，也可以不启用 graph capture。因此 graph replay key 和 compiler/kernel cache key
应保持正交。NPU runtime 的固定资源 replay 也应遵循同样原则。

## 我们的目标 specialization 形态

目标不是复制 Torch 的“请求触发 guard miss 后即时重编译”，而是保留其分层，同时采用更适合
NPU serving 的 AOT-first 策略：开发模式可以 lazy compile；生产模式启动前根据有限 profile 生成
bundle，在线请求只能 dispatch 到已准备的变体，未覆盖的 workload 给出明确错误。

各层分别持有自己的 key：

| 层次 | key 的主要内容 | 所有者 |
| --- | --- | --- |
| Serving dispatch | decode/prefill、batch/token bucket、capture/replay 条件 | pypto-serving |
| Model graph/profile | 模型配置指纹、量化、TP/EP、layout、动态 shape domain、target | pypto-lib |
| Operator schedule | 算子语义签名、dtype/layout、shape range、target、调优版本 | pypto-lib |
| Kernel artifact | JIT 源码、tensor 编译签名、constexpr、target、compiler options | pypto |
| Runtime instance | artifact、资源/地址和 replay size | runtime/pypto-serving |

specialization 生命周期如下：

1. pypto-lib 规范化模型配置、参数 schema 和分布式契约；
2. pypto-lib 根据 serving profile 规划有限的模型 graph 变体；
3. pypto-lib 将 Module 树展开为语义算子图，传播 tensor 契约；
4. pypto-lib 为每个算子选择 schedule，并将语义常量与 tiling 都显式绑定到 kernel；
5. pypto-lib 生成不含 Module 语义的普通 PyPTO JIT program；
6. pypto 按 kernel 编译签名 specialization、lower、codegen 并缓存 artifact；
7. pypto-lib 打包 artifact、权重 manifest 和 dispatch manifest，serving 按 workload 选择。

值的处理原则：

- 会改变代码结构、layout 或本地资源分配的值应 specialization，例如 hidden size、head dim、
  quant mode、TP/EP、tile 和 pipeline stage；
- kernel 能覆盖一个范围时，token extent 等长轴应尽量保持 runtime dynamic；
- runtime replay 必须固定的 batch/token 规模应由 serving profile 分 bucket，而不是伪装成模型常量；
- 请求数据、weight 内容、物理 device ID 不进入 kernel specialization；
- “任意新 shape 自动生成 kernel”由 pypto-lib 的 correctness-first fallback schedule 保证覆盖，
  PyPTO 只负责编译 pypto-lib 已经明确选择的 kernel specialization，并不理解该 shape 的算子语义。

## 当前基础

当前原型已经建立 kernel specialization 层：

- `pl.constexpr` 可以声明为 keyword-only JIT 参数；
- 调用点以及顶层 `compile`、`lower` 和即时执行可以直接绑定 constexpr；
- constexpr 仅存在于编译期，并参与缓存 identity；
- 同一个 program 可以调用同一源码函数的多个 specialization；
- 编译期参数可以从入口传递到 inline dependency；
- 旧的显式 `.specialize()` 写法继续作为兼容接口；
- 公共 DeepSeek V4 RMSNorm 模板已经在同一个 program 内验证 4096 和 5120 hidden size，
  并通过 simulator 和真实 A2/A3 NPU。

这些能力属于 kernel 开发接口，是完成模型开发接口所需的基础，但还不是最终的模型模块化体验。

第一版 pypto-lib 垂直切片已经覆盖 Qwen 的 `RMSNorm → Linear → residual`：

- `pypto_lib.nn.Module` 提供确定性的子模块和 Parameter schema 树；
- `RMSNorm`、`Linear` 和 tensor residual add 记录不含 tiling 的语义图；
- target-aware registry 为语义节点选择 schedule；
- lowerer 将图展开为一个普通入口和两个带显式 `pl.constexpr` 的 PyPTO inline kernel；
- Qwen 模型文件不包含 tile 参数，并可用 Golden Harness 在真实 A2/A3 NPU 上执行。

当前 lowerer 只接受这一种图模式。它验证了组件边界和完整数据流，不代表任意 Module 图已经可
lowering；下一步应扩展通用 operator lowering protocol，而不是在 PyPTO 中增加 Module 语义。

## 实施计划

### 阶段一：pypto-lib Module 和 elaboration 契约

在 pypto-lib 增加最小化 `nn.Module`、`forward()`、结构化子模块注册和 elaboration。Module 调用先
规范化为 `BoundOperator`/`ProgramSpec`，再展开为 PyPTO 已支持的裸 JIT 函数调用；不扩展 PyPTO
前端去解析 `module(...)` 或 `module.forward(...)`。

验收标准：

- 一个 Module 可以包含并调用另一个 Module；
- 同一 Module 类型的两个不同配置实例生成不同且确定的 graph profile；
- 结构 identity 相同的实例复用 graph 和 kernel artifact；
- Module 配置不可变，或者能够产生确定性的缓存 identity；
- RMSNorm 模型 wrapper 中不再出现 constexpr 或 tiling 参数；
- 现有裸函数 JIT program 继续正常编译。

### 阶段二：pypto-lib Schedule registry 和 fallback 策略

在 pypto-lib 增加由语义算子元数据驱动的 target-aware schedule lookup。将 RMSNorm tiling 从模型 wrapper 移出，
提供通用 fallback 和当前已经验证的 A2/A3 schedule。

验收标准：

- 模型代码不选择任何 tile 参数；
- target、dtype 和 shape 元数据能够确定性选择 schedule；
- 不支持的 shape 在前端报告 schedule domain 错误，而不是落到后端 buffer 或 legalization 错误；
- schedule 选择参与缓存 identity；
- 保留用于调优和调试的显式 schedule override。

### 阶段三：pypto-lib Parameter 和 state schema

增加 `nn.ParameterSpec`、持久 buffer 描述、层次化名称和模型级 state loading 契约。
语义 Module 定义与运行时存储、设备放置保持分离。

验收标准：

- 可以通过 Module 树发现 parameter 和 buffer；
- 提供确定性的 `state_dict` 兼容名称映射，可加载已有模型 checkpoint；
- 缺失、额外、shape 不匹配、dtype 不匹配和 layout 不匹配的权重在前端产生明确错误；
- shared/tied weight 保持 alias 关系；
- 分布式 sharding 元数据不包含硬编码物理设备 ID。

### 阶段四：pypto-lib graph 组合和 buffer planning 契约

pypto-lib 将 Module 树展开为 PyPTO orchestration 和函数依赖图，并推导中间 tensor 契约、生命周期
和 alias 约束；PyPTO/runtime 的低层 memory planner 根据这些约束分配物理 buffer。KV cache 等用户
持有的可变状态仍保持显式。

验收标准：

- 模型 `forward()` 可以返回 tensor，无需手工分配普通中间输出；
- shape、dtype、layout、动态维度和分布式元数据可以跨 Module 传播；
- memory planner 能够看到临时 tensor 生命周期；
- 外部可变状态必须显式声明，不能被误认为可释放的临时值；
- RMSNorm 加 Linear 的组合参考模块通过 simulator 和真实 NPU 验证。

### 阶段五：profile、动态 shape 和 specialization 策略

pypto-lib 明确哪些维度保持 runtime dynamic、哪些值产生模型或 schedule 变体，并生成有限的
profile manifest；PyPTO 负责 kernel artifact cache，serving 负责 runtime dispatch/replay cache。
每层缓存都必须有界、可观察且能解释 miss 原因。

验收标准：

- decode 和 prefill token 范围按照公开策略复用 artifact；
- 缓存增长有上限或可以显式配置；
- 日志可以解释 cache hit、产生 specialization 的原因以及被拒绝的 shape domain；
- 动态 extent 不会意外变成 tiling specialization；
- 结构 identity 相同的多个模型实例可以安全共享编译 artifact。

### 阶段六：参考模型迁移

先迁移一个 DeepSeek V4 垂直切片，再推广抽象：

```text
RMSNorm → projection → attention 或 MLP → residual
```

用这个切片验证配置归属、权重命名、schedule 选择、动态 token、输出分配、分布式元数据和真实 NPU
执行。只有该切片稳定后，才能将 API 推广到完整 model zoo。

验收标准：

- 同一组语义 Module 至少服务两个 DeepSeek V4 变体；
- 修改 hidden size 或 token domain 时只修改配置，不修改 kernel 源码；
- 模型 wrapper 不包含重复 kernel 实现或 target tiling 表；
- golden 精度和真实设备行为与现有实现一致；
- 迁移后模型特有源码量下降，真实设备 wall time 不回退。

## 初始阶段不做什么

- Training、autograd、optimizer 和完整 `torch.nn.Module` 兼容；
- compiled `forward()` 内的任意 Python 执行；
- 为每个未知 shape 自动找到最优 schedule；
- 向 kernel 开发者隐藏底层 JIT 和 schedule 控制；
- 在没有显式语义契约时自动推断分布式 sharding；
- 在一个垂直参考切片证明可行以前迁移所有现有模型。

## 兼容和迁移策略

迁移过程采用增量方式：

1. 保持现有裸 `@pl.jit` 函数和显式 `.specialize()` 有效。
2. 先在 pypto-lib 中让 Module 成为现有 JIT kernel 模板的 wrapper，并在编译前展开。
3. 在不修改算法的前提下，将 tiling 从模型 wrapper 移到 schedule。
4. 在 pypto-lib 增加 Parameter schema，同时为 serving 集成保留显式传权重模式。
5. 迁移一个参考切片，对比生成 program、精度和真实设备 wall time。
6. 只有在 Module 和 schedule 路径稳定后，才废弃 recipe factory 和按 shape 创建的别名。

## 待确认的设计问题

以下问题需要通过小型可执行原型确定：

- 第一版 `pypto_lib.nn.Module` 是持有运行时 Parameter 对象，还是只持有 Parameter schema；
- pypto-lib Module attribute call 应规范化成怎样的 `BoundOperator`/`ProgramSpec`；
- Module structural identity 如何序列化进缓存键；
- pypto-lib 如何表达输出和生命周期约束，以及 PyPTO/runtime 如何完成物理分配；
- schedule fallback domain 和优先级如何表示；
- autotune 结果在 serving 进程中如何持久化并限制规模；
- tensor parallel 和 expert parallel 的语义 sharding 元数据如何由 pypto-lib 表达并 lowering；
- `forward()` 支持哪些 Python 语法，以及不支持的语法如何尽早报错。

## 成功标准

当一个受支持的新模型变体可以通过以下步骤完成适配时，目标才算实现：

1. 映射模型配置和 checkpoint 名称；
2. 组合已有语义 Module；
3. 只为真正新增的计算增加算子；
4. 只为尚未覆盖的 target 或 shape domain 增加、调优 schedule；
5. 使用标准 harness 验证精度和性能。

复制已有模型目录再修改 kernel 常量，不属于可以接受的模型适配流程。

## 参考设计

- [PyTorch `torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)：
  模型树、子模块、Parameter、buffer 和 state 管理。
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)：
  带 guard 的 graph compilation 和多个编译结果缓存。
- [vLLM `torch.compile` integration](https://docs.vllm.ai/en/stable/design/torch_compile/)：
  serving workload、compile ranges/sizes、piecewise compilation 和 graph dispatch。
- [vLLM CUDA Graph design](https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md)：
  capture mode、batch descriptor 和 runtime dispatcher。
- [Triton `triton.jit`](https://triton-lang.org/main/python-api/generated/triton.jit.html)：
  kernel 级 JIT 和 constexpr specialization。
- [Triton `triton.autotune`](https://triton-lang.org/main/python-api/generated/triton.autotune.html)：
  显式候选配置、key、测量和结果缓存。
