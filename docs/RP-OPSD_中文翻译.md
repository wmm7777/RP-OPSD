# RP-OPSD：面向多模态大语言模型的分辨率特权 On-Policy 自蒸馏

> **来源**：arXiv:2607.24447v1 [cs.CV], 2026-07-27
> **作者**：Qihui Zhu¹*, Yuchen Wang¹*, Zijian Wen¹, Tao Zhang¹, Mengjie Zhang¹, Yang Liu², Shuangwu Chen¹†, Siying Wu¹, Jian Yang¹, Xiaofeng Jiang¹
> ¹中国科学技术大学 ²长鑫存储技术有限公司
> **代码**：https://github.com/sansanyuchen/RP-OPSD
> \* 共同第一作者；† 通讯作者。
> **译者注**：本文为便于复现的中文翻译，公式、表格、关键数值均保留原文格式，图注转述其含义。如需引用请以原文为准。

---

## 摘要（Abstract）

On-Policy 自蒸馏（OPSD）利用仅教师可获取的特权信息，在学生生成的轨迹上提供密集的 token 级监督。然而，现有方法通常依赖于已验证的解题轨迹、外部模型生成的解释或人工标注的视觉证据，这限制了其在多模态大语言模型上的可扩展应用。为解决这一问题，我们利用同一图像的高分辨率与低分辨率视图之间的信息差异，提出 **RP-OPSD**（面向多模态大语言模型的分辨率特权 On-Policy 自蒸馏）。在训练中，学生策略从四分之一原始分辨率的图像生成 on-policy 轨迹，而教师策略使用原始分辨率图像提供监督。通过最小化二者在学生轨迹上输出分布之间的散度，学生学习教师在高分辨率输入下的预测行为，从而增强其低分辨率能力，并将所学改进迁移到原始分辨率推理。RP-OPSD 既不需要额外人工标注，也不需要外部模型生成解题轨迹，仅需图像-问题对。在 Qwen3.5-9B 上的实验表明，RP-OPSD 在原始分辨率下平均性能相对提升 **5.45%**，并比 OPSD 实现 **1.78×** 训练加速。这些结果表明，分辨率差异可以作为一种简单且可扩展的特权信息来源，为多模态大语言模型提供有效且高效的 on-policy 自蒸馏方法。

**代码** — https://github.com/sansanyuchen/RP-OPSD

---

## 1. 引言（Introduction）

近年来，多模态大语言模型（MLLMs）在视觉问答、文档理解、图表分析和多模态问题求解方面取得了显著进展（Liu et al. 2023; Wang et al. 2024a）。近期开源系统如 Qwen3-VL（Bai et al. 2025）和 InternVL3.5（Wang et al. 2025）进一步强化了多模态推理和高效视觉处理能力。然而，高效的后训练仍然充满挑战。现有方法主要包括监督微调、带可验证奖励的强化学习和知识蒸馏（Hinton, Vinyals, and Dean 2015; Shao et al. 2024）。监督微调从高质量示范中学习，但其固定的训练轨迹可能与推理时生成的轨迹不同，导致分布偏移和曝光偏差（Agarwal et al. 2024）。强化学习使用模型生成的轨迹，但稀疏的序列级奖励只能提供有限的 token 级反馈，而为每个 prompt 采样多个响应成本高昂（Shao et al. 2024; Huang et al. 2025; Shen et al. 2025）。知识蒸馏从教师输出分布提供密集监督，但对离线轨迹的依赖导致类似的训练-推理不匹配问题（Agarwal et al. 2024）。

On-Policy 蒸馏（OPD）允许学生策略生成自己的轨迹，同时教师策略在学生访问的状态上提供 token 级监督（Agarwal et al. 2024; Lu and Thinking Machines Lab 2025）。通过在这些轨迹上匹配二者的输出分布，OPD 结合了 on-policy 学习和密集监督的优势。然而，传统 OPD 通常依赖一个独立的更强教师模型，并要求兼容的输出空间或模型架构，限制了其在大规模模型后训练中的应用。On-Policy 自蒸馏（OPSD）进一步去除了对外部教师模型的需求（Zhao et al. 2026; Shenfeld et al. 2026; Hübotter et al. 2026）。它在不同输入条件下使用同一模型同时充当学生和教师。学生在无额外信息下生成轨迹，而教师使用已验证的解、专家示范或丰富的环境反馈作为训练时的特权信息来提供 token 级监督。如此，OPSD 结合了 on-policy 轨迹、密集反馈和自监督，使模型能够利用自身现有能力进行改进。

尽管 OPSD 在需要多步推理的语言任务中已展现出潜力（Zhao et al. 2026; Shenfeld et al. 2026; Hübotter et al. 2026），将其扩展到多模态大语言模型仍然具有挑战性，因为必须为教师构建有效的特权信息。现有方法通常使用参考答案、已验证的解题轨迹或上下文示例，这些都很适合具有明确定义答案的文本任务。然而，多模态模型的错误可能源于遗漏对象、丢失视觉细节、定位不准确或跨模态连接薄弱，通常需要额外的区域标注、多模态解释或证据定位。OmniOPSD 使用外部模型生成多模态解题证据（Cheng et al. 2026），而 Vision-OPD 通过裁剪相关图像区域为教师构建局部特权视图（Yuan et al. 2026）。虽然这些方法证明了在不同视觉条件下进行师生自蒸馏的可行性，但它们依赖外部生成、目标识别、区域分割或局部裁剪，增加了数据构建和质量控制的成本。此外，额外的上下文并不总能提供有效的教师信号，因为 OPD 的性能还可能取决于教师选择、学生能力和监督上下文（Ma et al. 2026）。因此，构建简单且有效的特权信息仍然是 OPSD 扩展到多模态大语言模型的关键挑战。

这些局限性促使我们寻找一种更简单的特权信息来源，并要求其具备三个理想属性：应在教师与学生之间创造有意义的能力差距；应保留其输入的语义内容；且既不需要外部模型也不需要额外标注。**图像分辨率天然满足这些要求**。给定同一图像和问题，多模态大语言模型可能从原始分辨率图像给出正确答案，但从其降采样版本给出错误答案。我们将此现象称为**分辨率诱导的能力差距（resolution-induced capability gap）**。如图 1 所示，当输入图像的宽和高均缩减为原始尺寸的一半时，Qwen3.5-9B Base（Qwen Team 2026）在五个基准上的平均性能下降 6.21 分。在 V\*Bench（Wu and Xie 2024）上，性能下降达 15.19 分。这一差距源于原始分辨率图像保留了关于局部纹理、文本和小目标的更完整信息，而降采样在不同程度上削弱了这些视觉线索。因此，我们提出使用原始分辨率教师来监督低分辨率学生生成的轨迹。这种监督首先提升了模型在有限视觉证据下的能力。由于教师和学生共享相同的模型参数，所得改进可进一步迁移到原始分辨率推理。

基于这一观察和假设，我们提出 **RP-OPSD**，一个面向多模态大语言模型的分辨率特权 on-policy 自蒸馏框架。给定一张原始图像，我们在不同视觉条件下从同一多模态模型实例化两个策略。学生策略以低分辨率图像及其对应问题为输入，从当前策略采样 on-policy 响应。教师策略则使用原始分辨率图像作为特权视觉视图，在学生生成的轨迹上提供监督。RP-OPSD 并非仅将分辨率诱导的能力差距视为低分辨率鲁棒性问题，而是利用这一差距构建 OPSD 所需的不对称师生条件。由于这种不对称直接由同一输入的两种分辨率创建，优化仅需图像-查询对，不依赖标注答案、外部生成的推理轨迹或局部视觉证据。这一设计大幅降低了构建特权信息的成本，可直接应用于现有多模态数据。我们在 Qwen3.5-4B 和 Qwen3.5-9B 上跨多个广泛使用的多模态基准评测了 RP-OPSD。在半分辨率评测下，RP-OPSD 将 Qwen3.5-9B 的平均性能提升 6.09 分。与代表性后训练方法和最先进的多模态 OPSD 基线相比，RP-OPSD 在原始分辨率设置下获得了最大的平均性能增益，在 Qwen3.5-4B 和 Qwen3.5-9B 上分别实现 **6.28%** 和 **5.45%** 的平均相对提升。这些跨分辨率的一致改进共同支持了我们的假设。此外，在 Qwen3.5-9B 上的效率分析表明，RP-OPSD 比 OPSD 实现 **1.78×** 训练加速。

总结而言，我们的主要贡献如下：

- 我们提出 RP-OPSD，一个分辨率特权 on-policy 自蒸馏框架，其中原始分辨率教师通过 token 级分布匹配，在学生的 rollout 上监督低分辨率学生。
- RP-OPSD 直接从图像-查询对构建特权监督，无需额外标注或外部模型，实现简单且可扩展的多模态 OPSD。
- 跨两个模型规模和多个多模态基准的全面实验验证了 RP-OPSD 的有效性和高效性。

---

## 图 1（Figure 1）：分辨率诱导的能力差距

> **图 1**：Qwen3.5-9B Base 在五个多模态基准上由分辨率诱导的能力差距。将图像宽和高均缩减为一半，导致平均性能下降 6.21 分。

| 指标 | VisualProbe | V\*Bench | HR-Bench 4K | MMStar | POPE |
|---|---|---|---|---|---|
| 1/4 像素（低分辨率） | 34.81 | 69.63 | 79.25 | 80.27 | 87.83 |
| 原始分辨率 | 41.85 | 84.82 | 84.75 | 82.07 | 89.36 |

> 解读：低分辨率（1/4 像素，即宽高各减半）下各项分数均低于原始分辨率，平均下降 6.21 分；V\*Bench 下降最剧烈（15.19 分）。

---

## 2. 相关工作（Related Works）

### 多模态大语言模型（Multimodal Large Language Models）

多模态大语言模型通常通过视觉编码器和模态连接器将视觉表示映射到大语言模型的语义空间，并通过多模态预训练和指令微调获得视觉理解和跨模态推理能力。近期代表性模型包括 Qwen3-VL（Bai et al. 2025）、InternVL3.5（Wang et al. 2025）和 GLM-4.5V（Team et al. 2026），它们通过大规模、多样化的多模态预训练，以及视觉-语言对齐、跨模态特征融合、动态分辨率处理和多阶段后训练的持续进步，在视觉问答、文档理解、细粒度视觉感知和复杂多模态推理方面取得了显著进展。然而，其性能对输入图像质量和分辨率仍然敏感。降低图像分辨率会削弱文本、小目标和局部纹理等细粒度视觉线索，导致对同一图像-问题对的理解和推理变差。这种敏感性表明，不同输入分辨率不仅影响模型性能，还可能在同一模型内创造天然的能力差距，为建立多模态自蒸馏的师生条件提供了新途径。

### On-Policy 蒸馏（On-Policy Distillation）

传统知识蒸馏在固定轨迹上对齐教师和学生的输出分布（Hinton, Vinyals, and Dean 2015），这可能导致训练与推理之间的状态分布不匹配。On-Policy 蒸馏（OPD）通过允许学生生成自己的轨迹、教师在学生实际访问的状态上提供 token 级监督来解决这一问题（Agarwal et al. 2024; Lu and Thinking Machines Lab 2025）。OPD 结合了 on-policy 采样和密集反馈，但通常依赖一个独立的更强教师模型。On-Policy 自蒸馏（OPSD）进一步将同一模型在不同上下文下实例化为教师和学生（Zhao et al. 2026; Shenfeld et al. 2026; Hübotter et al. 2026）。教师使用已验证的解、专家示范或丰富的环境反馈作为特权信息来监督学生生成的轨迹，去除了对外部教师模型的需求。

在多模态设置中，OmniOPSD 使用外部模型生成的多模态解释作为教师端特权信息（Cheng et al. 2026），而 Vision-OPD 使用裁剪的证据区域来监督以完整图像为条件的学生（Yuan et al. 2026）。这些方法需要额外的解释生成或区域构建。相比之下，RP-OPSD 直接使用同一图像的原始分辨率和降采样视图来创建不对称的师生条件，不需要外部生成、区域标注或问题合成，提供了一种更简单、更可扩展的特权信息构建方式。

---

## 图 2（Figure 2）：RP-OPSD 概览

> **图 2**：RP-OPSD 概览。首先使用低分辨率学生从降采样图像生成 on-policy 轨迹；然后，原始分辨率教师评估相同的生成前缀，并利用更丰富的视觉证据提供 token 级分布目标。我们使用偏差校正的教师 Top-K 反向 KL 目标优化学生，并通过指数移动平均（EMA）更新教师，实现无需外部教师或额外标注的自蒸馏。

**流程要点**：
- **特权输入**：原始分辨率图像 $x^H$（仅教师可见）
- **问题** $q$：例如"钟表上显示的时间是什么？"
- **学生策略** $\pi_\theta$：输入低分辨率图像 $x^L$（1/4 像素），生成 on-policy 轨迹 $y$，输出学生分布 $p_t^S(\cdot\mid x^L, q, y_{<t})$
- **教师策略** $\bar{\pi_\theta}$（EMA）：输入原始分辨率 $x^H$，在同一前缀上输出教师分布 $p_t^T(\cdot\mid x^H, q, y_{<t})$
- **蒸馏损失**：对齐学生与教师分布；教师端 stop-gradient，学生端梯度回传
- **EMA 更新**：教师参数由学生参数的指数移动平均更新

---

## 3. 方法（Method）

我们提出 RP-OPSD，一个面向多模态大语言模型的分辨率特权 on-policy 自蒸馏框架。如图 2 所示，低分辨率学生首先生成 on-policy 轨迹，然后高分辨率教师在相同的生成前缀上提供 token 级分布监督，学生通过在教师选定的支撑集上匹配教师分布进行优化。

### 分辨率特权形式化（Resolution-Privileged Formulation）

给定原始分辨率图像 $x^H$ 和视觉问题 $q$，我们构造低分辨率视图：

$$x^L = R_{1/2}(x^H), \tag{1}$$

其中 $R_{1/2}$ 将图像宽和高各缩减一半。因此，$x^L$ 包含约四分之一的原始像素。两个视图包含相同的场景和视野，无裁剪、区域标注或额外位置 prompt。这种对齐设置使我们能够利用 $x^H$ 中可用的视觉细节作为从 $x^L$ 学习的特权信息。

RP-OPSD 使用具有相同模型架构的低分辨率学生和高分辨率教师：

$$\pi_\theta^L(\cdot\mid x^L, q), \quad \pi_\phi^H(\cdot\mid x^H, q). \tag{2}$$

两个模型从同一预训练检查点初始化，$\theta_0 = \phi_0$。因此，教师的优势来自其能获取更高分辨率视觉证据，而非更大的模型。训练完成后，EMA 教师和 rollout 副本被丢弃，优化后的模型可以在无需额外教师分支的情况下以任一分辨率进行评测。

### 分辨率特权 On-Policy 自蒸馏（Resolution-Privileged On-Policy Self-Distillation）

在固定或教师生成的响应上蒸馏可能在学生推理时很少访问的前缀上监督学生。遵循 on-policy 蒸馏原则（Agarwal et al. 2024; Lu and Thinking Machines Lab 2025; Yuan et al. 2026），RP-OPSD 改为在从当前低分辨率策略采样的轨迹上进行蒸馏。对于每个输入 $(x^L, x^H, q)$，rollout 策略生成 $G$ 个响应：

$$y^{(g)} \sim \pi_{\theta^-}^L(\cdot\mid x^L, q), \quad g=1,\dots,G, \tag{3}$$

其中 $\theta^-$ 表示在每个 rollout 批次前与学生同步的 rollout 策略。每个批次用于一次学生更新，然后进行下一次同步。

在 token 位置 $t$，学生和教师评估同一学生生成的前缀：

$$p_{g,t}(v) = \pi_\theta^L\!\left(v\mid x^L, q, y^{(g)}_{<t}\right), \tag{4}$$

$$r_{g,t}(v) = \mathrm{sg}\!\left[\pi_\phi^H\!\left(v\mid x^H, q, y^{(g)}_{<t}\right)\right], \tag{5}$$

其中 $v \in V$ 是词表中的一个 token，$\mathrm{sg}[\cdot]$ 截断通过教师的梯度。教师不生成单独的目标响应，而是在学生访问的状态上提供下一 token 分布。由于两个模型接收相同的问题和文本前缀，它们的预测差异主要来自可用视觉分辨率的不同。这一设计在转移高分辨率视觉知识的同时，保持训练状态与学生当前行为对齐。

我们将教师维护为学生的指数移动平均（EMA），遵循权重平均教师范式（Tarvainen and Valpola 2017）。每次成功的学生更新后，教师参数更新为：

$$\phi_{s+1} = (1-\rho)\phi_s + \rho\,\theta_{s+1}, \tag{6}$$

其中 $s$ 为优化步数，$\rho$ 为 EMA 更新率。教师始终在不带梯度下评估。EMA 提供缓慢变化的训练目标，同时允许教师跟随学生的改进。

### 偏差校正的教师 Top-K 反向 KL 蒸馏（Bias-Corrected Teacher-Top-K Reverse KL Distillation）

在每个响应 token 上匹配全词表分布代价高昂。因此，我们将比较限制在高分辨率教师偏好的 token 上。在每个 token 位置，定义教师选定的支撑集为：

$$S^g_{K,t} = \mathrm{TopK}_{v\in V}\, r_{g,t}(v). \tag{7}$$

我们在相同的 token 索引上收集学生和教师的概率。这些值仍为其原始全词表 softmax 分布上的概率：我们不在 $S^g_{K,t}$ 内重新归一化，也不添加单独的尾桶。从教师选择支撑集确保比较覆盖了特权视觉输入最强烈支持的 token。

将反向 KL 朴素地限制在教师选定的 Top-K 支撑上会引入截断偏差，因为保留的概率质量之和不为 1。遵循 MOPD（Ma et al. 2026）中的 top-k 蒸馏目标，我们定义偏差校正的教师 Top-K 反向 KL 为：

$$d_K(p_{g,t}\|r_{g,t}) = \sum_{v\in S^g_{K,t}} \left[ p_{g,t}(v)\log\frac{p_{g,t}(v)}{r_{g,t}(v)} - p_{g,t}(v) + r_{g,t}(v) \right]. \tag{8}$$

校正项 $-p_{g,t}(v)+r_{g,t}(v)$ 修正了这一 Top-K 截断偏差，确保当学生在保留支撑上与教师匹配时梯度为零。这在不比较全词表的情况下保持了所需的优化行为。

在此同步单次更新设置下，我们不应用额外的 rollout 策略重要性重加权。设 $m_{g,t}\in\{0,1\}$ 为响应掩码，用于排除 prompt 和 padding token。最终的 RP-OPSD 目标为：

$$\mathcal{L}_{\mathrm{RP\text{-}OPSD}} = \mathbb{E}\left[\frac{\sum_{g,t} m_{g,t}\, d_K(p_{g,t}\|r_{g,t})}{\sum_{g,t} m_{g,t}}\right]. \tag{9}$$

期望在训练样本和从 rollout 策略采样的响应上取值。该目标在低分辨率学生访问的状态上提供密集的分布级监督。

RP-OPSD 不使用答案级奖励、监督目标响应或单独的参考策略。多次 rollout 用于覆盖多样化的学生状态，而非执行基于组的奖励优化。

**算法 1** 总结了 RP-OPSD 的完整训练流程。

---

### 算法 1：RP-OPSD 训练流程

> **Algorithm 1**: RP-OPSD 训练流程

```
输入：训练集 D = {(x^H_i, q_i)}；预训练参数 θ_0；rollout 数 G；支撑集大小 K；EMA 更新率 ρ
输出：训练后的学生策略 π^L_θ

 1: 初始化学生、教师和 rollout 参数：θ ← θ_0，φ ← θ_0，θ⁻ ← θ_0
 2: for 每个小批次 B ⊂ D do
 3:   for 每个 (x^H, q) ∈ B do
 4:     构造低分辨率视图 x^L ← R_{1/2}(x^H)
 5:     采样 G 个响应 y^(g) ∼ π^L_{θ⁻}(· | x^L, q)
 6:     for 每个响应 y^(g) 和 token 位置 t do
 7:       用 (x^L, q, y^(g)_{<t}) 计算学生分布 p_{g,t}
 8:       用 (x^H, q, y^(g)_{<t}) 计算截断梯度的教师分布 r_{g,t}
 9:       选择教师 Top-K 支撑 S^K_{g,t} ← TopK(r_{g,t})
10:       用式(8)计算偏差校正的教师 Top-K 反向 KL d_K(p_{g,t}‖r_{g,t})
11:     end for
12:   end for
13:   用式(9)计算 L_{RP-OPSD}
14:   更新学生：θ ← θ − η∇_θ L_{RP-OPSD}
15:   更新教师：φ ← (1−ρ)φ + ρθ
16:   同步 rollout 策略：θ⁻ ← θ
17: end for
18: return π^L_θ
```

---

## 4. 实验（Experiments）

### 实验设置（Experimental Setup）

**训练设置。** 我们将 RP-OPSD 应用于 Qwen3.5-4B 和 Qwen3.5-9B（Qwen Team 2026）。训练集包含 5.2K 样本，取自 Vision-SR1（Li et al. 2026）、VLM-CapCurriculum Perception（Wu et al. 2026）、ZwZ-RL-VQA（Wei et al. 2026）和 Vision-OPD（Yuan et al. 2026），详见补充材料的训练数据组成。教师接收原始图像，而学生接收的图像通过 Lanczos 插值将宽和高各缩减一半。所有参数（包括视觉编码器）均可训练。我们优化偏差校正的教师 Top-100 反向 KL 目标，并以 EMA 更新率 0.05 更新教师。对于每个输入，我们以温度 1.0、top-p 1.0、不使用 top-k 截断、最大生成长度 1,024 token 采样 8 个响应。每个 rollout 批次用于一次学生更新，之后 rollout 策略与更新后的学生同步。所有模型训练 1 个 epoch，对应 55 个优化步，批次大小为 96，学习率为 $2\times10^{-6}$。前 10 步使用线性预热，数据和生成种子均设为 42。训练在 8× H20 GPU 上进行。

**基准。** 我们考虑两组基准。第一组评测细粒度视觉感知，包括 V\*Bench（Wu and Xie 2024）、HR-Bench 4K/8K（Wang et al. 2024b）、MME-RealWorld EN/CN（Zhang et al. 2025）和 VisualProbe（Lai et al. 2025）。第二组评测训练分布外的泛化能力，包括 MMVP（Tong et al. 2024b）、MMStar（Chen et al. 2024）和 POPE（Li et al. 2023）。消融研究额外报告 CV-Bench（Tong et al. 2024a）。所有模型使用原始分辨率图像评测。我们尽可能使用基于规则的解析；其余响应由 Qwen3.5-9B（Qwen Team 2026）LLM 判官以其默认解码设置评估。

**基线。** 我们将 RP-OPSD 与 Base、SFT、GRPO、OPSD 和 Vision-OPD 进行比较。
- **Base** 表示原始 Qwen3.5 模型（无后训练），衡量训练带来的整体增益。
- **SFT** 使用原始图像和真实答案，最小化助手 token 上的交叉熵。
- **GRPO**（Shao et al. 2024）对每张原始分辨率图像采样 8 个响应，使用二元正确性奖励计算组相对优势。
- **OPSD**（Zhao et al. 2026）向学生和教师呈现同一原始图像，但用从真值构造的答案提示增强教师 prompt。教师随后在学生的 on-policy 响应上提供 token 级蒸馏信号。
- SFT、GRPO、OPSD 和 RP-OPSD 使用相同的 5.2K 训练样本。
- **Vision-OPD**（Yuan et al. 2026）执行区域到全局的自蒸馏，教师接收证据裁剪区域，学生接收带边界框标注的完整图像。我们使用其官方实现复现 Vision-OPD。

### 主结果（Main Results）

如表 1 所示，RP-OPSD 在两个模型规模上均取得最高平均分。在 9B 模型上，RP-OPSD 达到 80.43，分别比 Base、Vision-OPD 和 OPSD 高出 4.16、1.44 和 0.63 分。在 4B 模型上，对应提升分别为 4.63、1.03 和 0.61 分。这些结果表明 RP-OPSD 的增益可跨模型规模泛化。

与 Base 相比，9B RP-OPSD 在 V\*Bench、VisualProbe、MME-RW EN/CN 和 HR-Bench 4K/8K 上分别提升 6.28、15.12、5.51/5.17 和 1.75/2.62 分。4B 模型在 MME-RW EN/CN 和 VisualProbe 上分别提升 12.36、8.62 和 11.15 分。同时，9B 模型在 MMVP、MMStar 和 POPE 上分别提升 0.33、0.60 和 0.07 分，表明细粒度感知的提升不以一般视觉能力为代价。相对于 OPSD，RP-OPSD 将平均分提升 0.63 分，在 VisualProbe 和 MMStar 上分别获得 7.00 和 1.27 分的增益。这一结果表明分辨率特权蒸馏有效增强了细粒度视觉搜索和一般视觉感知。

#### 表 1：Qwen3.5-9B 和 Qwen3.5-4B 主结果

> **表 1**：Qwen3.5-9B 和 Qwen3.5-4B 主结果。所有方法使用原始分辨率图像评测。Avg. 为九项报告指标的非加权均值。粗体表示各模型规模内的最佳结果。

| 模型 | 方法 | V\* | HR-4K | HR-8K | MME-RW EN | MME-RW CN | VisualProbe | MMVP | MMStar | POPE | Avg. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-9B** | Base | 84.82 | 84.75 | 81.50 | 71.40 | 67.67 | 41.85 | 83.00 | 82.07 | 89.36 | 76.27 |
| | SFT | 91.10 | 87.88 | 83.62 | 73.25 | 71.54 | 51.25 | 83.67 | 78.93 | 89.74 | 79.00 |
| | GRPO | 88.48 | 84.50 | 81.50 | 75.72 | 71.81 | 50.38 | 84.33 | 80.73 | 89.09 | 78.50 |
| | OPSD | 91.10 | 86.88 | 84.25 | 78.12 | 74.43 | 49.97 | 83.00 | 81.40 | 89.08 | 79.80 |
| | Vision-OPD | 85.86 | 86.62 | 85.12 | 73.40 | 70.46 | 56.84 | 81.33 | 81.53 | 89.79 | 78.99 |
| | **RP-OPSD** | **91.10** | 86.50 | 84.12 | **76.91** | **72.84** | **56.97** | 83.33 | **82.67** | 89.43 | **80.43** |
| **Qwen3.5-4B** | Base | 84.29 | 84.38 | 80.13 | 64.20 | 63.80 | 43.22 | 76.67 | 78.53 | 88.28 | 73.72 |
| | SFT | 87.96 | 85.75 | 80.00 | 71.24 | 69.17 | 48.53 | 78.00 | 68.60 | 89.43 | 75.41 |
| | GRPO | 84.82 | 81.88 | 78.38 | 72.15 | 70.27 | 52.23 | 82.33 | 72.27 | 85.63 | 75.55 |
| | OPSD | 86.39 | 85.50 | 80.12 | 76.75 | 74.72 | 48.78 | 82.00 | 76.67 | 88.70 | 77.74 |
| | Vision-OPD | 87.96 | 82.62 | 81.75 | 74.70 | 70.46 | 54.95 | 77.00 | 77.40 | 89.10 | 77.33 |
| | **RP-OPSD** | **87.96** | **85.50** | **82.00** | **76.56** | **72.42** | **54.37** | 78.33 | **79.07** | 88.98 | **78.35** |

### 低分辨率能力分析（Low-Resolution Capability Analysis）

如表 2 所示，RP-OPSD 在低分辨率评测下改进了全部五个基准，将平均分从 71.45 提升至 77.54。最大增益出现在 VisualProbe 和 V\*Bench 上，RP-OPSD 分别比基础模型提升 14.90 和 12.05 分。这些基准强调小视觉细节和定向视觉搜索，表明来自原始分辨率教师的监督使学生能更好地利用有限的视觉证据。

结合表 1 的原始分辨率结果，这些发现支持了我们的核心假设：RP-OPSD 直接增强了模型在低分辨率学生视图下的能力，所得能力改进迁移到原始分辨率推理。因此，这些增益反映了模型本身的改进。

#### 表 2：半分辨率评测下的 Qwen3.5-9B 与 RP-OPSD 性能

> **表 2**：Qwen3.5-9B 和 RP-OPSD 在半分辨率评测下的性能。Avg. 为五项报告指标的非加权均值，∆ 表示相对 Base 的提升。所有指标值越高越好。粗体表示每行中更好的分数。

| 基准 | Qwen3.5-9B | RP-OPSD | ∆ |
|---|---|---|---|
| VisualProbe | 34.81 | **49.71** | +14.90 |
| V\*Bench | 69.63 | **81.68** | +12.05 |
| HR-Bench 4K | 79.25 | **82.00** | +2.75 |
| POPE | 87.83 | **88.39** | +0.56 |
| CV-Bench | 85.75 | **85.94** | +0.19 |
| **Avg.** | 71.45 | **77.54** | **+6.09** |

### 消融研究与分析（Ablation Studies and Analysis）

#### 蒸馏目标的选择（Choice of Distillation Objective）

表 3 比较了 GSD（Agarwal et al. 2024）、前向 KL、标准反向 KL 和偏差校正的教师 Top-100 反向 KL，以考察 token 级蒸馏目标和 Top-K 截断偏差校正如何影响性能。我们使用 Qwen3.5-9B，GSD 设 $\alpha=0.5$，其他训练和评测设置保持不变。

偏差校正的教师 Top-100 反向 KL 目标取得最高平均分 82.73，分别比标准反向 KL 和 GSD 高出 1.14 和 0.37 分。相对于标准反向 KL，它在 V\*Bench、HR-Bench 8K 和 VisualProbe 上分别提升 2.09、1.37 和 3.18 分。这些结果支持在蒸馏限制到教师选定词表支撑时校正 Top-K 截断偏差。因此，我们在后续实验中采用该目标。

#### 表 3：蒸馏目标消融

> **表 3**：蒸馏目标消融。所有目标使用教师选定的 Top-100 支撑；RKL 表示反向 KL。Avg. 为八项指标的非加权均值。粗体标记每列最佳结果；涉及默认设置的并列中仅默认设置加粗。

| 设置 | V\* | HR-4K | HR-8K | VisualProbe | MMVP | CV-Bench | MMStar | POPE | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| GSD Top-100 | 91.10 | 86.38 | 82.38 | 55.89 | 83.33 | 87.91 | 82.47 | 89.42 | 82.36 |
| 前向 Top-100 KL | 90.05 | 85.62 | 81.88 | 53.84 | 82.33 | 87.61 | 83.40 | 89.39 | 81.77 |
| 反向 Top-100 KL | 89.01 | 87.25 | 82.75 | 53.79 | 82.33 | 87.86 | 80.80 | 88.93 | 81.59 |
| **偏差校正 RKL（本文）** | **91.10** | 86.50 | **84.12** | **56.97** | **83.33** | 87.73 | **82.67** | **89.43** | **82.73** |

#### 教师更新策略（Teacher Update Strategy）

表 4 比较 EMA 教师与固定在初始参数上的教师，以考察 EMA 是否提供更有效的蒸馏目标。两种变体使用相同的初始化、视觉输入、蒸馏目标和训练预算。

EMA 教师平均达到 82.73，比冻结教师高 0.51 分，在 HR-Bench 8K、VisualProbe 和 MMStar 上分别提升 1.12、4.58 和 0.67 分。尽管冻结教师在 V\*Bench、MMVP 和 CV-Bench 上略好，但 EMA 提供了更强的平均和细粒度性能。

#### 表 4：教师更新策略消融

> **表 4**：教师更新策略消融。Avg. 为八项指标的非加权均值。粗体标记每列最佳结果；涉及默认设置的并列中仅默认设置加粗。

| 设置 | V\* | HR-4K | HR-8K | VisualProbe | MMVP | CV-Bench | MMStar | POPE | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| 冻结初始教师 | **91.62** | 86.50 | 83.00 | 52.39 | **84.33** | **88.23** | 82.00 | **89.69** | 82.22 |
| **EMA 教师（本文）** | 91.10 | 86.50 | **84.12** | **56.97** | 83.33 | 87.73 | **82.67** | 89.43 | **82.73** |

#### 学生输入分辨率（Student Input Resolution）

表 5 仅改变学生分辨率；教师和评测保持原始分辨率图像。1/2→1 设置将两个维度各缩减一半后恢复原始画布尺寸，使我们能将丢失的图像细节与输入画布变化区分开。

两个维度各减半获得最佳平均分 82.73。将缩减倍数增至三倍和四倍会使平均分下降 0.80 和 2.56 分，表明过度降采样削弱了可用于迁移的视觉证据。1/2→1 设置仅达到 81.85，低于直接半分辨率训练，确认恢复画布不能恢复丢失的视觉细节。

#### 表 5：学生训练分辨率消融

> **表 5**：学生训练分辨率消融。教师和评测使用原始分辨率图像。Avg. 为八项指标的非加权均值。粗体标记每列最佳结果；涉及默认设置的并列中仅默认设置加粗。

| 设置 | V\* | HR-4K | HR-8K | VisualProbe | MMVP | CV-Bench | MMStar | POPE | Avg. |
|---|---|---|---|---|---|---|---|---|---|
| 1/3 宽/高 | **91.62** | **86.75** | 83.12 | 55.38 | 82.67 | 86.55 | 80.93 | 88.40 | 81.93 |
| 1/4 宽/高 | 89.01 | 85.12 | 82.75 | 55.21 | 79.00 | 85.85 | 79.87 | 84.57 | 80.17 |
| 1/2→1 画布 | 91.10 | 86.88 | 83.25 | 51.82 | 82.00 | 87.66 | 82.27 | **89.82** | 81.85 |
| **1/2 宽/高（本文）** | 91.10 | 86.50 | **84.12** | **56.97** | **83.33** | **87.73** | **82.67** | 89.43 | **82.73** |

#### 训练效率（Training Efficiency）

表 6 在相同数据、步数、批次大小、rollout 和硬件下比较 OPSD 和 RP-OPSD，以隔离学生输入分辨率的影响。

设 $n^H$ 为原始分辨率的视觉 token 数；将两个图像维度减半得 $n^L = n^H/4$。设 $C_g(n)$ 和 $C_f(n)$ 分别表示一次 rollout 和一次全序列前向传播。在 $G$ 次 rollout 和 $3C_f(n)$ 的前向-反向近似下，OPSD 成本为：

$$F_{\mathrm{OPSD}} \simeq G\!\left[C_g(n^H) + 4C_f(n^H)\right]. \tag{10}$$

而 RP-OPSD 成本为：

$$F_{\mathrm{RP}} \simeq G\!\left[C_g(n^H/4) + 3C_f(n^H/4) + C_f(n^H)\right]. \tag{11}$$

设 $\Delta C_g = C_g(n^H) - C_g(n^H/4)$，类似地定义 $\Delta C_f$。二者的差距为：

$$F_{\mathrm{OPSD}} - F_{\mathrm{RP}} \simeq G(\Delta C_g + 3\Delta C_f) > 0. \tag{12}$$

因子 3 近似学生前向-反向计算。由于 $n^L = n^H/4$，线性视觉项降至四分之一，二次自注意力项降至十六分之一。由于在我们的设置中文本 token 远少于视觉 token，我们在 FLOPs 估计中省略其贡献。共享的原始分辨率教师在式(12)中相消，仅留下 rollout 和学生更新的节省。该估计保持非视觉教师上下文固定并省略系统开销，而实际墙钟测量已将其包含。

#### 表 6：训练效率

> **表 6**：在相同数据、批次大小（96）、每输入 rollout 数（8）和硬件（8× NVIDIA H20 GPU）下的训练效率。加速比为 $T_{\mathrm{OPSD}}/T_{\mathrm{method}}$；Full 和 Half 表示输入分辨率。

| 方法 | 学生 | 教师 | 时间（h） | 加速比 | 主表 Avg. |
|---|---|---|---|---|---|
| OPSD | Full | Full+hint | 13.93 | 1.00× | 79.80 |
| **RP-OPSD** | Half | Full | **7.83** | **1.78×** | **80.43** |

RP-OPSD 将训练时间从 13.93 小时降至 7.83 小时（1.78×），同时将主表平均分从 79.80 提升至 80.43。因此，更低分辨率的学生输入在不牺牲原始分辨率性能的前提下降低了训练成本。报告的加速比仅覆盖训练，不包括评测。

---

## 5. 结论（Conclusion）

我们提出了 RP-OPSD，一个 on-policy 自蒸馏框架，利用同一图像的原始和低分辨率视图所诱导的能力差距作为多模态大语言模型的特权监督。低分辨率学生生成 on-policy 轨迹，而原始分辨率 EMA 教师利用更丰富的视觉证据在相同前缀上提供 token 级分布目标。RP-OPSD 仅需图像-查询对，无需额外答案标注、外部教师模型、生成解题轨迹或局部视觉证据。在 Qwen3.5-4B 和 Qwen3.5-9B 上，RP-OPSD 在对比的后训练方法中取得了最佳平均性能，相对其基础模型分别提升 6.28% 和 5.45%。半分辨率评测下的增益进一步支持了改进的低分辨率能力向原始分辨率推理的迁移。RP-OPSD 还比 OPSD 实现 1.78× 训练加速，表明分辨率差异提供了简单、有效且高效的特权信息。

---

## 参考文献（References）

- Agarwal, R.; Vieillard, N.; Zhou, Y.; Stanczyk, P.; Ramos Garea, S.; Geist, M.; and Bachem, O. 2024. On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes. In *ICLR*, 2024, 21246–21263.
- Bai, S.; et al. 2025. Qwen3-VL Technical Report. arXiv:2511.21631.
- Chen, L.; et al. 2024. Are We on the Right Way for Evaluating Large Vision-Language Models? arXiv:2403.20330.
- Cheng, Z.; et al. 2026. OmniOPSD: Rationale-Privileged On-Policy Self-Distillation for Affective Computing. arXiv:2606.15920.
- Hinton, G.; Vinyals, O.; and Dean, J. 2015. Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Huang, W.; et al. 2025. Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models. arXiv:2503.06749.
- Hübotter, J.; et al. 2026. Reinforcement Learning via Self-Distillation. arXiv:2601.20802.
- Lai, X.; et al. 2025. Mini-o3: Scaling Up Reasoning Patterns and Interaction Turns for Visual Search. arXiv:2509.07969.
- Li, Y.; et al. 2023. Evaluating Object Hallucination in Large Vision-Language Models. In *EMNLP* 2023, 292–305.
- Li, Z.; et al. 2026. Self-Rewarding Vision-Language Model via Reasoning Decomposition. arXiv:2508.19652.
- Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual Instruction Tuning. arXiv:2304.08485.
- Lu, K.; and Thinking Machines Lab. 2025. On-Policy Distillation. https://thinkingmachines.ai/blog/on-policy-distillation/.
- Ma, W.; et al. 2026. MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training. arXiv:2606.30406.
- Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. https://qwen.ai/blog?id=qwen3.5.
- Shao, Z.; et al. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
- Shen, H.; et al. 2025. VLM-R1: A Stable and Generalizable R1-Style Large Vision-Language Model. arXiv:2504.07615.
- Shenfeld, I.; Damani, M.; Hübotter, J.; and Agrawal, P. 2026. Self-Distillation Enables Continual Learning. arXiv:2601.19897.
- Tarvainen, A.; and Valpola, H. 2017. Mean Teachers Are Better Role Models: Weight-Averaged Consistency Targets Improve Semi-Supervised Deep Learning Results. In *NeurIPS* 2017, 1195–1204.
- Team, V.; et al. 2026. GLM-4.5V and GLM-4.1V-Thinking: Towards Versatile Multimodal Reasoning with Scalable Reinforcement Learning. arXiv:2507.01006.
- Tong, S.; et al. 2024a. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. In *NeurIPS* 2024, 87310–87356.
- Tong, S.; et al. 2024b. Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs. In *CVPR* 2024, 9568–9578.
- Wang, P.; et al. 2024a. Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution. arXiv:2409.12191.
- Wang, W.; et al. 2024b. Divide, Conquer and Combine: A Training-Free Framework for High-Resolution Image Perception in Multimodal Large Language Models. arXiv:2408.15556.
- Wang, W.; et al. 2025. InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Efficiency. arXiv:2508.18265.
- Wei, L.; et al. 2026. Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception. arXiv:2602.11858.
- Wu, J.; et al. 2026. From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models. arXiv:2605.20177.
- Wu, P.; and Xie, S. 2024. V*: Guided Visual Search as a Core Mechanism in Multimodal LLMs. In *CVPR* 2024, 13084–13094.
- Yuan, Q.; et al. 2026. Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation. arXiv:2605.18740.
- Zhang, Y.-F.; et al. 2025. MME-RealWorld: Could Your Multimodal LLM Challenge High-Resolution Real-World Scenarios That Are Difficult for Humans? In *ICLR* 2025, 89655–89701.
- Zhao, S.; et al. 2026. Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models. arXiv:2601.18734.

---

## 附录 A：训练数据（Training Data）

我们从 Vision-SR1、VLM-CapCurriculum Perception、ZwZ-RL-VQA 和公开的 Vision-OPD 训练集构造了 5,295 个单图像训练样本。表 1 报告了来源组成。

#### 补充表 1：RP-OPSD 训练集来源组成

| 来源 | 样本数 |
|---|---|
| ZwZ-RL-VQA | 3,034 |
| Vision-OPD | 1,060 |
| Vision-SR1 | 887 |
| VLM-CapCurriculum Perception | 314 |
| **总计** | **5,295** |

---

## 附录 B：详细训练设置（Detailed Training Settings）

我们从同一 Qwen3.5 检查点初始化学生和教师。学生从半分辨率图像生成 on-policy 响应。教师随后在观察原始分辨率图像和相同文本 prompt 的同时，对相同的响应前缀打分。梯度仅通过学生传播。每次 actor 更新后，教师参数更新为：

$$\phi_{s+1} = (1-\rho)\phi_s + \rho\,\theta_{s+1}, \quad \rho = 0.05. \tag{1}$$

所有学生参数（包括视觉编码器）保持可训练。

我们使用正文中描述的偏差校正反向 KL 目标蒸馏教师选定的 Top-100 token 分布。对于每个输入，学生采样 8 个响应。训练不使用序列级奖励，也不使用外部判官。表 2 和表 3 列出了 4B 和 9B 实验共享的完整配置。

训练环境为 FSDP 和 vLLM，Python 3.12，PyTorch 2.10.0，Transformers 5.5.0，vLLM 0.18.0，Ray 2.53.0。Rollout 使用 8 个 worker，张量并行度为 1。感知 chat template 前置一个空 `<reasoning>` 块；它不请求也不监督推理轨迹。9B 运行在 8 块 H20 GPU 上需 7.83 小时。

#### 补充表 2：模型与生成配置

| 配置 | 值 |
|---|---|
| 主干 | Qwen3.5-4B / Qwen3.5-9B |
| 可训练参数 | 全部，包括视觉编码器 |
| 学生图像 | 物理 1/2 宽和高 |
| 教师图像 | 原始分辨率 |
| 插值 | Lanczos |
| 教师更新 | EMA, $\rho=0.05$ |
| 蒸馏目标 | 偏差校正反向 KL |
| 蒸馏支撑 | 教师选定 Top-100 |
| 每输入 rollout 数 | 8 |
| 最大响应长度 | 1,024 tokens |
| 采样温度 | 1.0 |
| Top-p / Top-k | 1.0 / −1 |
| 最大 prompt 长度 | 8,192 tokens |
| 最大序列长度 | 9,216 tokens |

#### 补充表 3：RP-OPSD 的优化与系统设置

| 配置 | 值 |
|---|---|
| 全局批次大小 | 96 |
| PPO mini-batch 大小 | 96 |
| 每GPU micro-batch | 1 |
| 动态批处理 | 禁用 |
| 优化器 | AdamW |
| 学习率 | $2\times10^{-6}$ |
| Adam 系数 | (0.9, 0.999) |
| 权重衰减 | 0.01 |
| 梯度裁剪 | 1.0 |
| 线性预热 | 10 步 |
| 训练计划 | 1 epoch / 55 步 |
| 检查点 | 最终步 |
| 训练精度 | BF16 |
| 梯度检查点 | 启用 |
| Actor 参数卸载 | 启用 |
| 优化器卸载 | 启用 |
| 数据 / 生成种子 | 42 / 42 |
| 硬件 | 8× NVIDIA H20 |

---

## 附录 C：案例分析（Case Study）

我们在两个原始分辨率 VisualProbe 案例上比较 Qwen3.5-9B Base 和 RP-OPSD。

### 补充图 1：集装箱上的细粒度 OCR

> **补充图 1**：集装箱上的细粒度 OCR。目标裁剪使定位要求明确，预测比较显示 RP-OPSD 能分辨被查询的小标签。
> - (a) 完整图像，7952 × 5304
> - (b) 目标区域裁剪

| 项目 | 内容 |
|---|---|
| **问题** | 集装箱门底部白色矩形上写的是什么？ |
| **真值** | MCI |
| **Qwen3.5-9B Base（错误）** | 预测：MSKU 728 702 7 / 22G1。分析：基础模型复制了查询区域附近的显眼集装箱代码，而非读取小白色标签。 |
| **RP-OPSD（正确）** | 预测：MCI。分析：RP-OPSD 定位到开门底部附近的白色矩形，并正确转录了小目标文本。 |

### 补充图 2：集装箱码头的远距离数字识别

> **补充图 2**：集装箱码头的远距离数字识别。RP-OPSD 将空间引用定位到正确的起重机并读取其小编号，而基础模型选择了附近的标签。
> - (a) 完整图像，7587 × 4410
> - (b) 目标区域裁剪

| 项目 | 内容 |
|---|---|
| **问题** | 最右侧红色集装箱起重机上的数字是什么？ |
| **真值** | 85 |
| **Qwen3.5-9B Base（错误）** | 预测：R95。分析：基础模型选择了附近的起重机标识符，未能将"最右侧"这一短语定位到目标起重机。 |
| **RP-OPSD（正确）** | 预测：85。分析：RP-OPSD 识别出最右侧的起重机，并读取了其上部结构上的小白色编号。 |
