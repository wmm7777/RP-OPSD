# Cross-Model Resolution-Privileged On-Policy Distillation

This supplementary experiment tests whether resolution-privileged on-policy
distillation transfers knowledge across model scales. A frozen Qwen3.5-9B
teacher supervises a Qwen3.5-4B student. The experiment uses the same
Dataset4.0 data and the same resolution gap as RP-OPSD, but replaces the EMA
self-teacher with a larger fixed teacher.

## Training protocol

### Student rollout

The Qwen3.5-4B student receives images resized to half of the original width
and height (one-quarter of the original pixels). For each question, it samples
8 responses on policy with `temperature=1.0`, `top_p=1.0`, `top_k=-1`, and a
maximum response length of 1,024 tokens.

### Teacher scoring

The frozen Qwen3.5-9B teacher receives the same question and the corresponding
original-resolution image. It does not generate a separate answer. Instead,
it evaluates each student-generated trajectory and returns the conditional
token distribution at every response position.

### Token-level objective

At each position, the objective uses the 100 tokens with the highest teacher
probabilities. On that teacher-selected support, it applies the bias-corrected
truncated reverse KL:

$$
\mathcal{L}_{\mathrm{distill}}
=
\sum_{v \in \operatorname{TopK}(p_t)}
\left[
p_s(v)\log\frac{p_s(v)}{p_t(v)} - p_s(v) + p_t(v)
\right].
$$

The experiment uses `alpha=1.0`, does not add a tail bucket, and does not
renormalize probabilities over the top-100 support. The optimization objective
contains only this token-level distillation loss; no additional GRPO policy
loss or SFT loss is added.

### Parameter updates

Only the Qwen3.5-4B student is updated. The Qwen3.5-9B teacher remains fixed
throughout training. The student's visual encoder is not frozen, so both its
vision tower and language model participate in optimization.

### Training setup

| Setting | Value |
|---|---|
| Dataset | Dataset4.0, 5,295 examples |
| Student | Qwen3.5-4B |
| Teacher | Qwen3.5-9B, frozen |
| Student view | Half width and height |
| Teacher view | Original resolution |
| Rollouts per question | 8 |
| Batch size | 96 |
| Learning rate | 2e-6 |
| Warmup | 10 steps |
| Training length | 1 epoch, 55 steps |
| Distillation support | Teacher top-100 tokens |

The overall data flow is:

```text
half-resolution image -> 4B student samples a response
                                  |
                                  | same response trajectory
                                  v
original-resolution image -> frozen 9B teacher scores every token
                                  |
                                  v
                 bias-corrected top-100 reverse KL
                                  |
                                  v
                         update 4B student
```

The key distinction from answer-level distillation is that the 9B teacher
does not provide a generated target answer. The 4B student instead learns the
teacher's token-level prediction behavior along its own on-policy trajectories.

## Results

All models are evaluated with original-resolution images.

| Benchmark | Qwen3.5-4B Base | 9B Teacher → 4B Student | Change |
|---|---:|---:|---:|
| V\*Bench | 84.29 | **90.58** | **+6.29** |
| HR-Bench 4K | **84.38** | 84.25 | -0.13 |
| HR-Bench 8K | 80.13 | **83.00** | **+2.87** |
| VisualProbe | 43.22 | **49.90** | **+6.68** |
| MMVP | 76.67 | **77.67** | **+1.00** |
| MMStar | 78.53 | **80.07** | **+1.54** |
| POPE | 88.28 | **89.31** | **+1.03** |
| **7-benchmark average** | **76.50** | **79.25** | **+2.75** |

The cross-model student improves on 6 of 7 benchmarks. The largest gains
appear on VisualProbe (+6.68) and V\*Bench (+6.29), while HR-Bench 4K remains
effectively stable (-0.13). These results indicate that resolution-privileged
on-policy supervision can transfer fine-grained visual behavior from a larger
fixed teacher to a smaller student.

This is a seven-benchmark supplementary experiment. Its average should not be
compared directly with the nine-benchmark average in the paper's main table.
This update documents the protocol and results without changing the canonical
Qwen3.5-9B training configuration in the repository.
