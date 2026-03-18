# Machine Learning Systems Course Homework (2026 Spring)

## Submission Deadlines

| Assignment | Topic                     |        Deadline         |
| :--------: | :------------------------ | :---------------------: |
|     1      | Automatic Differentiation | **Mar 29, 2026, 23:59** |
|     2      | CUDA Programming          | **Apr 12, 2026, 23:59** |
|     3      | Distributed Training      | **Apr 26, 2026, 23:59** |

---

## Assignment 1 — Automatic Differentiation (100 pt)

Implement a reverse-mode automatic differentiation system on a computational graph (TensorFlow v1 style).

- **Task 1 (20 pt):** Implement `compute` for all ops
- **Task 2 (25 pt):** Implement `Evaluator.run` (forward pass with topological sort)
- **Task 3 (25 pt):** Implement `gradient` for all ops
- **Task 4 (30 pt):** Implement the `gradients` function (backward graph construction)

**Language:** Python (no GPU required)

**Submit:** `auto_diff.py`, `tests/test_customized_cases.py`, `journal.txt`, `feedback.txt`

---

## Assignment 2 — CUDA Programming (100 pt)

Implement a 2D cumulative sum (prefix sum) kernel on NVIDIA GPUs using CUDA C++.

- **Correctness (60 pt):** Pass all tests (`make small`, `make medium`, `make large`, `make final_test`)
- **Performance (40 pt):** Based on latency in `make final_test` (< 1ms = full marks)

**Language:** CUDA C++  
**Hardware:** NVIDIA GPU

**Submit:** `cumsum.cu`, `cumsum_host.cu`, terminal screenshot, `journal&feedback.txt`

---

## Assignment 3 — Distributed Training (100 pt)

Implement communication protocols for Data Parallel and Tensor Model Parallel training using MPI and NumPy.

- **Part 1 (10 pt):** Data split for data parallel training
- **Part 2 (20 pt):** Layer initialization (`get_info`)
- **Part 3 (15 pt):** Naive model parallel forward communication
- **Part 4 (15 pt):** Megatron-style model parallel forward communication
- **Part 5 (20 pt):** Naive model parallel backward communication
- **Part 6 (10 pt):** Megatron-style model parallel backward communication
- **Part 7 (10 pt):** Gradient communication for data parallel training

**Language:** Python (MPI + NumPy, CPU only)

**Submit:** `func_impl.py`, `data_parallel_preprocess.py`, `journal.txt`, `feedback.txt`
