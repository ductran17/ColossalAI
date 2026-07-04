# Guide: Reverse Engineering Hybrid Auto Planner for Thesis

## Objective

Analyze the source code of the Hybrid Auto Planner and generate technical documentation suitable for a Master's thesis.

The output is **NOT** API documentation.

Instead, explain the system from a software architecture and algorithmic perspective.

---

# Expected writing style

Academic writing.

Avoid describing implementation details line-by-line.

Do not explain every function.

Focus on:

- architecture
- algorithms
- data flow
- design decisions
- interaction between modules

The reader should understand how the planner works without reading the source code.

---

# For every module

For each module (Profiler, Topology, Cost Model, Search, Orchestrator, ...), answer the following questions.

## 1. Purpose

What problem does this module solve?

Why is it needed?

What limitation of previous modules does it address?

Maximum:
1–2 paragraphs.

---

## 2. Inputs

Describe the logical inputs.

For example:

- model configuration
- cluster profile
- topology
- world size
- micro batch size

Do NOT simply list Python arguments.

---

## 3. Outputs

Describe what the module produces.

Example:

- estimated training time

- communication classification

- optimal (tp, pp, dp)

---

## 4. Internal workflow

Describe the algorithm.

Prefer numbered steps.

Example:

1.
Collect cluster profile.

2.
Generate candidate strategies.

3.
Prune impossible configurations.

4.
Estimate execution cost.

5.
Select minimum cost.

Avoid talking about Python syntax.

---

## 5. Important algorithms

If the module implements algorithms,
identify them.

Examples:

- exhaustive enumeration

- heuristic pruning

- alpha-beta communication model

- ring allreduce model

- linear regression

- topology classification

Explain WHY these algorithms are chosen.

---

## 6. Key data structures

Describe important objects.

Not every class.

Only major structures.

Example:

ClusterProfile

contains

- alpha_intra
- beta_intra
- ...

Strategy

contains

- tp
- pp
- dp

---

## 7. Interaction with other modules

Explain

who calls this module

and

which module consumes its output.

Example

Profiler

↓

Cost Model

↓

Search

↓

Training Launcher

---

## 8. Contribution

Identify what is original.

Separate

existing framework

vs

new implementation.

Clearly classify every component into one of:

- Existing ColossalAI functionality

- Existing PyTorch/NCCL functionality

- Proposed in this thesis

This is extremely important.

---

# Architecture extraction

Generate one subsection describing

overall execution flow.

Produce a sequence similar to:

User launches training

↓

Profiler

↓

Topology

↓

Search

↓

Cost Model

↓

HybridParallelPlugin

↓

Training

---

# Ignore

Do NOT explain

- logging

- argparse

- command line parsing

- exception handling

- helper functions

unless they are algorithmically important.

---

# Figures

Whenever possible, suggest figures that could appear in the thesis.

For each figure provide

Title

Purpose

Elements

Example

Figure:
Hybrid Auto Planner execution pipeline

Contains

Profiler

↓

Topology

↓

Search

↓

HybridParallelPlugin

---

# Pseudocode

If a module contains an algorithm,

rewrite it as pseudocode.

Never copy Python.

Use algorithmic style.

Example

Algorithm SearchBestStrategy()

enumerate candidates

for each candidate

    if invalid

        continue

    estimate cost

return minimum

---

# Complexity

Whenever applicable,

estimate

Time Complexity

Space Complexity

using Big-O.

---

# Writing target

The generated explanation should be suitable for

Chapter 3

"System Architecture and Module Design"

of a Master's thesis.

Do not produce developer documentation.

Do not produce code comments.

Explain the implementation as an engineering system.