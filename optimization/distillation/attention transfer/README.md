# FitNets: Hints for Thin Deep Nets — Figure 1 Explanation

This document explains the two-stage training procedure shown in **Figure 1** of the paper **“FitNets: Hints for Thin Deep Nets”**.

The central idea is slightly different from ordinary Knowledge Distillation:

> **Normal KD:** teacher teaches student only through the final output.  
> **FitNets:** teacher first teaches the student an **intermediate representation**, then teaches it through the final output.

This intermediate supervision is what the paper calls a **hint**.

---

# 1. First understand the notation

In the figure:

- $T$ = teacher network
- $S$ = student / FitNet
- $W_T$ = all teacher parameters
- $W_S$ = all student parameters
- $h$ = chosen **hint layer** in the teacher
- $g$ = chosen **guided layer** in the student

So:

$$
W_T^h
$$

means the parameters up to the teacher's hint layer.

And

$$
W_S^g
$$

means the parameters up to the student's guided layer.

The paper groups them as:

$$
W_{\text{Hint}}
$$

and

$$
W_{\text{Guided}}.
$$

The important correspondence is:

$$
\boxed{
\text{Teacher hint layer}
\quad\longrightarrow\quad
\text{Student guided layer}
}
$$

---

# 2. Panel (a): Teacher and Student Networks

Look at the left side.

### Teacher

The teacher is already trained:

$$
x
\rightarrow
W_T^1
\rightarrow
W_T^2
\rightarrow \cdots
\rightarrow
W_T^h
\rightarrow \cdots
\rightarrow
W_T^L
$$

The green region:

$$
W_{\text{Hint}}
$$

contains the teacher parameters up to layer $h$.

At layer $h$, the teacher generates an intermediate feature representation:

$$
u_h(x)
$$

You can think of it as a tensor such as:

$$
u_h(x)\in\mathbb{R}^{H_T\times W_T\times C_T}.
$$

This representation contains useful information that the teacher learned.

For example, in a CNN:

- early layers → edges/textures
- middle layers → shapes/parts
- deeper layers → semantic concepts

FitNets asks:

> Instead of only copying the teacher's final prediction, why not also teach the student to construct a similar internal representation?

That's the motivation for the hint.

---

# 3. What is the "hint"?

Suppose the teacher produces:

$$
u_h(x)
$$

at layer $h$.

The student produces:

$$
v_g(x)
$$

at its guided layer $g$.

FitNets wants:

$$
v_g(x)
\approx
u_h(x).
$$

But there is a problem.

The student is **thin**, so its feature dimensions may be very different.

For example:

Teacher:

$$
u_h(x)
\in
\mathbb{R}^{14\times14\times256}
$$

Student:

$$
v_g(x)
\in
\mathbb{R}^{14\times14\times64}
$$

You cannot directly calculate

$$
\|u_h-v_g\|^2
$$

because

$$
256 \neq 64.
$$

This is why the blue component in panel (b) exists.

---

# 4. The blue $W_r$: the regressor

FitNets adds a small transformation:

$$
r(\cdot;W_r)
$$

called a **regressor**.

Its job is:

$$
v_g(x)
\xrightarrow{r}
r(v_g(x))
$$

so that:

$$
r(v_g(x))
$$

has approximately the same dimensionality as the teacher's hint.

For example:

$$
14\times14\times64
\rightarrow
\boxed{regressor}
\rightarrow
14\times14\times256.
$$

Then we can compare:

$$
u_h(x)
\qquad\text{vs}\qquad
r(v_g(x)).
$$

Conceptually:

```text
Teacher

x ───────────────► Hint layer
                    │
                    ▼
                 u_h(x)
                    │
                    │ target
                    ▼

Student

x ─► ... ─► Guided layer ─► Regressor
               │                │
               ▼                ▼
             v_g(x) ───────► r(v_g(x))
```

---

# 5. Panel (b): Hints Training

This is **Stage 1**.

The equation shown is approximately:

$$
W_{\text{Guided}}^*
=
\arg\min_{W_{\text{Guided}}}
\mathcal L_{HT}
(W_{\text{Guided}},W_r)
$$

where $HT$ means **Hint Training**.

More explicitly, the loss is essentially:

$$
\boxed{
\mathcal L_{HT}
=
\frac{1}{2}
\left\|
u_h(x)
-
r(v_g(x);W_r)
\right\|^2
}
$$

or over $N$ samples:

$$
\mathcal L_{HT}
=
\frac{1}{2N}
\sum_{i=1}^{N}
\left\|
u_h(x_i)
-
r(v_g(x_i);W_r)
\right\|^2.
$$

---

# 6. What is actually being optimized here?

This is crucial.

### Teacher

The teacher is **frozen**.

So:

$$
W_{\text{Hint}}
$$

doesn't change.

The teacher simply generates targets:

$$
u_h(x).
$$

### Student

The lower part of the student:

$$
W_{\text{Guided}}
$$

is trained.

### Regressor

$$
W_r
$$

is also trained.

So during hint training:

$$
\boxed{
W_{\text{Hint}}=\text{fixed}
}
$$

while

$$
\boxed{
W_{\text{Guided}},W_r=\text{trainable}
}
$$

The goal is:

$$
r(v_g(x))
\rightarrow u_h(x).
$$

---

# 7. Why train only the first part of the student?

This is one of the clever ideas in FitNets.

Instead of asking a very deep thin student to learn the entire task from random initialization:

$$
x
\rightarrow
100\text{ layers}
\rightarrow
y
$$

we first teach the initial portion:

$$
x
\rightarrow
\boxed{W_S^1\cdots W_S^g}
$$

to generate a meaningful representation.

So after Stage 1:

$$
W_{\text{Guided}}
\rightarrow
W_{\text{Guided}}^*.
$$

The $^*$ means:

> optimized parameters obtained after hint training.

This gives the student a much better initialization.

---

# 8. An intuitive example

Imagine the teacher sees a cat.

At an intermediate layer, the teacher may have learned features representing:

```text
ears
fur texture
eyes
head shape
whiskers
```

Call this representation:

$$
z_T.
$$

The student initially produces basically random features:

$$
z_S.
$$

FitNet first forces:

$$
r(z_S)\approx z_T.
$$

Therefore the student learns:

> "Before worrying about whether the image is ultimately classified as cat, dog, or bird, first learn to represent the image in a way similar to the teacher."

That's the **hint**.

---

# 9. After Hint Training

Suppose Stage 1 gives:

$$
W_{\text{Guided}}^*.
$$

The student now looks like:

```text
Input
  │
  ▼
┌──────────────────┐
│ W_Guided*        │  ← learned from teacher's hint
└──────────────────┘
  │
  ▼
Remaining layers
  │
  ▼
Output
```

The remainder of the student may still be randomly initialized.

Now comes Stage 2.

---

# 10. Panel (c): Knowledge Distillation

Panel (c) shows:

$$
W_S^*
=
\arg\min_{W_S}
\mathcal L_{DK}(W_S)
$$

where $DK$ corresponds to the distillation / knowledge-distillation objective.

Now FitNet performs the more conventional Knowledge Distillation process.

The teacher produces final logits:

$$
z_T.
$$

The student produces:

$$
z_S.
$$

Instead of training the student using only the hard label:

$$
y=[0,0,1,0,\dots],
$$

the student also learns from the teacher's distribution:

$$
p_T
=
\operatorname{softmax}
\left(
\frac{z_T}{\tau}
\right).
$$

Student:

$$
p_S
=
\operatorname{softmax}
\left(
\frac{z_S}{\tau}
\right).
$$

Then the student is encouraged to reproduce the teacher's output behavior.

A generic modern formulation is:

$$
\mathcal L
=
\alpha
\mathcal L_{\text{hard}}
+
(1-\alpha)
\mathcal L_{\text{soft}}.
$$

For example:

$$
\mathcal L_{\text{hard}}
=
CE(y,p_S)
$$

and

$$
\mathcal L_{\text{soft}}
=
\tau^2
KL(p_T^\tau\|p_S^\tau).
$$

So the training signal now comes from the teacher's **output**, rather than its internal hint.

---

# 11. Is $W_{\text{Guided}}^*$ frozen during KD?

No.

This is an important detail.

After hint training:

$$
W_{\text{Guided}}^*
$$

is used as an **initialization**.

Then during full Knowledge Distillation:

$$
\boxed{
\text{the entire student network is optimized}
}
$$

including the previously guided layers.

So:

$$
W_{\text{Guided}}^*
$$

does **not** mean those weights stay fixed forever.

It means:

> "start full student training from these good weights."

Conceptually:

$$
W_{\text{Guided}}^*
\xrightarrow{\text{initialization}}
W_S
\xrightarrow{\text{KD training}}
W_S^*.
$$

---

# 12. Therefore FitNet is a two-stage optimization

You can summarize the whole paper with these two equations.

## Stage 1 — Hint training

Teacher intermediate representation:

$$
z_T^h=f_T^h(x)
$$

Student intermediate representation:

$$
z_S^g=f_S^g(x)
$$

Transform student feature:

$$
\hat z_S^g=r(z_S^g;W_r).
$$

Minimize:

$$
\boxed{
\mathcal L_{HT}
=
\left\|
z_T^h-\hat z_S^g
\right\|_2^2
}
$$

giving:

$$
W_{\text{Guided}}^*.
$$

---

## Stage 2 — Knowledge Distillation

Initialize the student using:

$$
W_{\text{Guided}}^*.
$$

Then use teacher outputs:

$$
p_T(x)
$$

as targets for:

$$
p_S(x).
$$

Optimize:

$$
\boxed{
W_S^*
=
\arg\min_{W_S}
\mathcal L_{KD}
}
$$

to train the whole student.

---

# 13. Why is this particularly useful for a thin/deep student?

This is the main research question behind the paper.

Suppose:

### Teacher

```text
Wide
████████████████
████████████████
████████████████
```

Maybe:

$$
20 \text{ layers}\times256\text{ channels}.
$$

### Student

```text
Thin
████
████
████
████
████
████
████
████
```

Maybe:

$$
40\text{ layers}\times64\text{ channels}.
$$

The student can actually be **deeper than the teacher** while having far fewer parameters.

But training such a thin/deep network from scratch can be difficult.

The teacher's intermediate representation acts as a training guide:

$$
\text{input}
\rightarrow
\underbrace{\text{guided representation}}_{\text{teacher hint}}
\rightarrow
\text{output}.
$$

Instead of solving one huge optimization problem, the student gets an intermediate target.

---

# 14. Relation to normal Knowledge Distillation

This is the easiest way to understand why FitNet was important.

### Classical KD

Teacher:

$$
x
\rightarrow
T
\rightarrow
\boxed{p_T}
$$

Student:

$$
x
\rightarrow
S
\rightarrow
\boxed{p_S}
$$

Match:

$$
p_S\approx p_T.
$$

Only **output knowledge** is transferred.

---

### FitNet

First:

$$
\boxed{
\text{Intermediate feature knowledge}
}
$$

is transferred:

$$
r(z_S^g)\approx z_T^h.
$$

Then:

$$
\boxed{
\text{Output knowledge}
}
$$

is transferred:

$$
p_S\approx p_T.
$$

Therefore:

$$
\boxed{
\text{FitNet}
=
\text{Intermediate representation transfer}
+
\text{Output distillation}
}
$$

---

# 15. How to read the three panels together

The complete figure can effectively be read left → right.

### (a) Select layers

```text
TEACHER                    STUDENT

x                          x
│                          │
▼                          ▼
...                        ...
│                          │
▼                          ▼
Hint layer                 Guided layer
h                          g
│                          │
▼                          ▼
z_T^h                      z_S^g
```

---

### (b) Hint training

Add regressor:

```text
Teacher hint

z_T^h
  │
  │ target
  ▼
[ MSE loss ]
  ▲
  │
r(z_S^g)
  ▲
  │
Student guided layer
```

Optimize:

$$
W_{\text{Guided}}, W_r.
$$

Get:

$$
W_{\text{Guided}}^*.
$$

---

### (c) Full distillation

Remove the temporary hint-training machinery.

Use:

$$
W_{\text{Guided}}^*
$$

as initialization.

Then:

```text
Teacher                  Student

 x                         x
 │                         │
 ▼                         ▼
 W_T                       W_S
 │                         │
 ▼                         ▼
 logits_T ──────────────► logits_S
          KD loss
```

Optimize all:

$$
W_S.
$$

Result:

$$
W_S^*.
$$

---

# 16. The regressor $W_r$ is temporary

This detail is also important if you think about **Efficient ML / deployment**.

The blue regressor:

$$
W_r
$$

is primarily a **training-time component**.

It exists so the student's feature map can be compared to the teacher's.

After hint training, you don't need it as part of the deployed student architecture.

So:

$$
\boxed{
\text{no inference overhead from hint training}
}
$$

in the intended setup.

This is particularly attractive for model compression.

---

# 17. A useful mental model

Think of university teaching.

### Normal supervised learning

Professor only says:

> "Final answer should be 42."

Student tries to figure everything out.

---

### Knowledge Distillation

Professor says:

> "Final answer is 42, but my confidence over the possible answers is approximately  
> 42: 0.75, 40: 0.15, 44: 0.08..."

The student learns richer information.

---

### FitNet

Professor additionally says:

> "Halfway through solving this problem, you should arrive at this intermediate representation."

So training becomes:

$$
\boxed{
\text{learn how to think halfway}
}
$$

followed by:

$$
\boxed{
\text{learn how to produce the final answer}
}
$$

That is why the term **hint** is appropriate.

---

# 18. Why this paper matters historically

FitNets introduced an important extension of Knowledge Distillation:

$$
\text{logit distillation}
\quad\rightarrow\quad
\text{representation distillation}.
$$

This idea became foundational for a large family of later methods that transfer:

- hidden features,
- activation maps,
- attention maps,
- relations between examples,
- Gram matrices,
- feature directions,
- semantic structure.

For example, **“Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer”** follows this broader philosophy.

FitNet transfers roughly:

$$
\boxed{\text{feature representation}}
$$

while Attention Transfer transfers something more like:

$$
\boxed{\text{spatial attention derived from features}}.
$$

So there is an important conceptual evolution:

$$
\text{Hinton KD}
\rightarrow
\boxed{\text{FitNet}}
\rightarrow
\text{Attention Transfer}
\rightarrow
\text{many modern feature/representation KD methods}.
$$

For Efficient ML/model-compression studies, **FitNets is therefore one of the key papers to understand before moving to modern feature-based knowledge distillation.**
