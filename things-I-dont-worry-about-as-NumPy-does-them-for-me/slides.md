---
theme: default
colorSchema: dark
title: Things I don't worry about as NumPy does them for me
info: |
  A mental model for the things NumPy is quietly doing on your behalf.
author: Kai Striega
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
mdc: true
layout: cover
---

# Things I don't worry about as NumPy does them for me

A mental model for the things NumPy is quietly doing on your behalf

Kai Striega

---

# This is a talk about NumPy

<v-clicks>

> The fundamental package for scientific computing with Python.
- NumPy provides performant, powerful N-dimensional arrays and numerical computing tools on those arrays.
- NumPy is heavily used with more than one billion monthly downloads between PyPI and conda.
- Some of NumPy's feats:
    - The [first image of a black hole](https://numpy.org/case-studies/blackhole-image/).
    - The [detection of gravitational waves](https://numpy.org/case-studies/gw-discov/).
    - [Protein Structure Prediction](https://www.nature.com/articles/s41586-021-03819-2) (2024 Nobel Prize in Chemistry).
    - Humanity's [first flight on Mars](https://github.com/readme/featured/nasa-ingenuity-helicopter).
    - Was published in [Nature](https://www.nature.com/articles/s41586-020-2649-2), a journal that almost never publishes software.

</v-clicks>

---

# `whoami`? (and why should you listen to me?)

<v-clicks>

- Hi! I'm Kai Striega.
- I build and productionise Mathematical Optimisation models at Endgame Analytics.
- I was a maintainer of SciPy for 7 years.
    - SciPy implements numerical algorithms using NumPy
    - Downloaded 385 million times last month.
- Very active in the Australian Python community.
    - Local Meetups: PythonWA, MelbPy, SydPy.
    - PyCon AU's Scientific Python Track.

</v-clicks>

---

# Why we're here

<v-clicks>

- In the years I spent maintaining SciPy, I read a *lot* of NumPy code written by very smart people.
- The performance problems they hit weren't because they didn't know NumPy.
- They had the right syntax, the right idioms, the right intuitions.
- They had the **wrong mental model** of what NumPy was doing underneath.

</v-clicks>

---

# The villain

```python {1|2|3}{lines:true}
images = load_images()                 # (1000, 512, 512, 3) float64
images = images.transpose(0, 3, 1, 2)
flat = images.reshape(1000, -1)        # ← 6.3 GB copy
```

<v-clicks>

- Three innocent-looking lines.
- One of them just allocated and moved 6.3 gigabytes.
- **The cost isn't where you think it is.**

</v-clicks>

---

# The promise

By the end of this talk, you'll look at this expression:

```python
result = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
```

...and know exactly what NumPy just did to memory.

Three ideas, not tricks.

---
layout: section
---

# Idea 1

## The work isn't where you think it is

---

# The familiar comparison

```python {1|3-4|6-7}
data = np.arange(1_000_000, dtype=np.float64)

# Pure Python
%timeit  [x ** 2 for x in data]    # ~72 ms

# NumPy
%timeit data ** 2                  # ~0.25 ms
```

<v-clicks>

- Yes, NumPy is faster.
- *But that's not the lesson*.

</v-clicks>

---

# Watch what's actually happening

```python {1-5|7-11|13-17}
def counter(frame, event, arg):
    global python_lines_visited
    if event == 'line':
        python_lines_visited += 1
    return counter

python_lines_visited = 0
sys.settrace(counter)
result = [x ** 2 for x in data]
sys.settrace(None)
# python_lines_visited: 1,000,001

python_lines_visited = 0
sys.settrace(counter)
result = data ** 2
sys.settrace(None)
# python_lines_visited:  1
```

---

# A million versus one

<v-clicks>

- Python's bytecode interpreter ran:
  - a **million** times in the first version.
  - **once** in the second.
- NumPy isn't accelerating Python.
- It's *relocating the work* somewhere Python never touches.

</v-clicks>

---

# Where did the work go?

Simplified from `numpy/_core/src/umath/loops.c.src`:

```c
static void
DOUBLE_square(char **args, npy_intp const *dimensions, ...)
{
    npy_intp n = dimensions[0];
    double *in = (double *)args[0];
    double *out = (double *)args[1];
    for (npy_intp i = 0; i < n; i++) {
        out[i] = in[i] * in[i];
    }
}
```

<v-clicks>

- *This* is the loop that ran.
- In C.
- Over the whole array.

</v-clicks>

---

# When the relocation breaks

```python {1-2|4-5|7-8}
ints = np.arange(1_000_000, dtype=np.int64)
objs = np.arange(1_000_000, dtype=object)

# INT64_square kernel runs in C
%timeit ints ** 2     # ~0.28 ms     

# Python's __pow__ called a million times
%timeit objs ** 2     # ~30 ms    
```

<v-clicks>

- Same shape, same operation, same syntax.
- For `int64`, NumPy has a C kernel.
- For `object`, it doesn't. There's no general C function for arbitrary Python objects.
- The relocation contract requires a kernel.
- **Object dtype opts you out of every performance property NumPy offers.**

</v-clicks>

---

# Bridge

<v-clicks>

- So the *operations* live in C.
- But what about the *data*?
- What does an array actually look like?

</v-clicks>

---
layout: section
---

# Idea 2

## Arrays aren't what you think they are

---

# What an ndarray actually is

From `numpy/_core/include/numpy/ndarraytypes.h`, slightly trimmed:

```c {1-10|3}
typedef struct {
    PyObject_HEAD
    char        *data;       // pointer to the actual bytes
    int          nd;         // number of dimensions
    npy_intp    *dimensions; // shape: e.g. [1000, 512, 512, 3]
    npy_intp    *strides;    // bytes to step per axis
    PyArray_Descr *descr;    // dtype info
    int          flags;
    /* ... */
} PyArrayObject;
```

## A small **header** pointing at a flat **buffer**.

---

# A surprising timing

```python {1|3|4}
big = np.random.random((10_000, 10_000))   # 800 MB

%timeit big.T                              # ~40 ns
%timeit big.T.copy()                       # ~810 ms
```

<v-clicks>

- Transposing 800MB in **40 nanoseconds**.
- The same operation with `.copy()` costs about twenty million times more.
- What is `.T` actually doing, then?

</v-clicks>

---

# The picture: buffer

```
buffer in memory:    [1] [2] [3] [4] [5] [6] (each box = 8 bytes, float64)
```

<v-clicks>

- Six contiguous elements.
- That's all the data there is.

</v-clicks>

---

# The picture: header

```
buffer in memory:    [1] [2] [3] [4] [5] [6] (each box = 8 bytes, float64)
                      ^
                      |
          a:  shape=(2, 3)   strides=(24, 8)
```

<v-clicks>

- The header points at the buffer.
- Shape says how the bytes are laid out conceptually.
- Strides say how many bytes to step per axis.

</v-clicks>

---

# The picture: transpose

```
buffer in memory:    [1] [2] [3] [4] [5] [6] (each box = 8 bytes, float64)
                      ^
                      |
          a:    shape=(2, 3)   strides=(24, 8)
          a.T:  shape=(3, 2)   strides=(8, 24)
```

<v-clicks>

- Same buffer.
- Transpose just **swapped two numbers** in the strides field.
- The data didn't move. That's the whole operation.

</v-clicks>

---

# One buffer, two headers

```python{|1-2|3-4|5-6}
>>> a = np.zeros((2, 3))
>>> b = a.T
>>> b[0, 0] = 42
>>> a[0, 0]
42.0
>>> b.base is a
True
```

<v-clicks>

- We never touched `a`. We wrote through `b`.
- There was only ever one buffer, so there was only ever one place to write.
- The header/buffer split isn't trivia. It decides who sees your writes.

</v-clicks>

---

# Verifying the picture

```python{|1-3|4-5|6-7|9-13}
>>> a = np.zeros((2, 3))
>>> a.itemsize
8         # bytes per element
>>> a.shape
(2, 3)    # elements per dim
>>> a.strides
(24, 8)   # bytes to step per dim

>>> b = a.T
>>> b.shape, b.strides
((3, 2), (8, 24))
>>> a.flags['C_CONTIGUOUS'], b.flags['C_CONTIGUOUS']
(True, False)
```

<v-clicks>

- Strides are in **bytes**. `(24, 8)` is "skip a row" then "skip a column" for `float64`.
- `b` is not C-contiguous. But it *is* `F_CONTIGUOUS`: still perfectly regular, just column-major.
- Non-contiguous doesn't mean scrambled. That regularity is exactly what lets BLAS take `b` without a copy.

</v-clicks>

---

# The reshape contract

<v-clicks>

- `reshape` returns a **view** when the requested shape is compatible with the existing memory layout.
- It **copies** when it isn't.
- "Compatible" means: NumPy can produce the new shape by choosing new strides
  over the same buffer, without rearranging any bytes.

  - A contiguous array (C or F) can almost always be reshaped without copying.
  - A non-contiguous array sometimes can. It depends on which axes you touch.

> Heuristic: if you've done a transpose, fancy indexing, or an axis-rearranging operation recently, **assume reshape might copy**.

</v-clicks>

---

# Stop guessing: ask

```python{|1-3|5-7}
>>> flat = images.transpose(0, 3, 1, 2).reshape(1000, -1)
>>> np.shares_memory(flat, images)
False                      # it copied

>>> flat.base is None
False                      # ...but base says nothing useful here
```

<v-clicks>

- `np.shares_memory` is the question you actually mean: *did these end up on the same bytes?*
- `.base` is tempting and misleading. A copy still has a `base`, it just points at the copy.
- On pathological strides `shares_memory` can be slow. `np.may_share_memory` is the cheap conservative answer.

</v-clicks>

---

# Why copies hurt

A copy isn't slow because the CPU is busy.
It's slow because the bytes have to *move*.

<v-clicks>

- A copy reads every byte and writes every byte, so 6.3 GB in means **12.6 GB moved**.
- This machine streams about 40 GB/s, so that is ~300 ms of pure traffic.
- That is the **floor**, and only a streaming copy hits it. `big.T.copy()` had to fault in fresh pages *and* read against the grain of the cache, so it paid 810 ms for traffic worth 40 ms.
- While that happens, the cache fills with data we won't reuse.
- The *next* operation pays again to pull its inputs back in.

**The currency is bandwidth, not bytes.**

</v-clicks>


---

# The villain returns: recall

```python {lines:true}
images = load_images()                   # (1000, 512, 512, 3)
images = images.transpose(0, 3, 1, 2)
flat = images.reshape(1000, -1)          # 6.3 GB copy
```

- Remember this from the start?
- I said something here cost 6.3 gigabytes.
- Let's read it!

---

# The villain returns: diagnose

```python {1-2|4-6|8-10}
images = load_images()                  # shape (1000, 512, 512, 3)
                                        # C-contiguous ✓

images = images.transpose(0, 3, 1, 2)   # shape (1000, 3, 512, 512)
                                        # strides reordered, same buffer
                                        # C-contiguous ✗
                                        
flat = images.reshape(1000, -1)         # needs contiguous layout
                                        # buffer doesn't match → copy
                                        # 6.3 GB allocated and moved
```

<v-clicks>

- Reshape needs to walk the new shape in a regular stride pattern. The transpose broke that.
- The cost wasn't on the line that did the work.
- It was set up three lines earlier.

</v-clicks>

---

# The villain returns

```python {1|2|3}
images = load_images()
images = np.ascontiguousarray(                # explicit copy here
    images.transpose(0, 3, 1, 2))
flat = images.reshape(1000, -1)               # now free
```

<v-clicks>

- The fix doesn't make the copy go away.
- The 6.3 GB still gets moved.
- `.copy()` would work too. `ascontiguousarray` names the thing we actually want.
- What changed: the copy is now on the line that asks for it, instead of hiding inside `reshape`.
- **The model doesn't avoid copies. It makes them visible.**

</v-clicks>

---

# Bridge

<v-clicks>

- We've seen what arrays *are*.
- We've seen what operations *cost*.
- But what about operations between *mismatched shapes*?

</v-clicks>

---
layout: section
---

# Idea 3

## Broadcasting is a contract

---

# The contract

<v-clicks>

- Broadcasting is a deal NumPy offers you.
- You give it two arrays of mismatched shape.
- It runs the operation **as if the smaller one were the size of the larger**, without ever allocating that larger version.
- The only thing that gets allocated is the **result**.

</v-clicks>

---

# The contract in code

```python{1|2|4}
a = np.zeros((1000, 1000))   # 8 MB
b = np.arange(1000)          # 8 KB

result = a + b               # 8 MB (just the result)
```

What didn't happen:

- `b` was **not** tiled to (1000, 1000)
- No 8 MB intermediate was allocated
- The C kernel iterated over `a`'s shape, re-reading the same 8 KB of `b`

---

# How? Stride zero

```python{|1-3|5-6}
>>> b = np.arange(1000)
>>> b.strides
(8,)

>>> np.broadcast_to(b, (1000, 1000)).strides
(0, 8)
```

<v-clicks>

- Step **zero bytes** to move down a row. Never move. Read the same 8 KB a thousand times.
- There is no tiling code and no special case in the kernel. It's the stride machinery from idea 2, handed a 0.
- Broadcasting isn't a separate feature. It's what the header can already express.

</v-clicks>

---

# The rules

1. Align shapes from the **trailing axis**.
2. Each axis pair must be **equal, or one of them must be 1**.
3. Missing axes are treated as 1.

```{1-3|5-7|9-13}
a:           (1000, 1000)
b:                 (1000)    ← prepended as (1, 1000)
result:      (1000, 1000)    ✓

a:           (1000, 1, 5)
b:           (   1, 3, 5)
result:      (1000, 3, 5)    ✓

a:           (1000, 3, 5)
b:           (1000, 4, 5)
─────────────────────────────────────────────────────
ValueError: operands could not be broadcast together
            with shapes (1000,3,5) (1000,4,5)
```

---

# Why the contract is worth keeping

Remember why copies hurt: bandwidth and cache eviction.

- Broadcasting refuses to allocate the tile, so neither cost gets paid.
- Better than warm: the stride-0 axis re-reads 8 KB that never leaves L1.
- Contiguous, predictable strides are also the precondition for handing off to **SIMD** instructions or **BLAS** routines underneath.
- You don't ask for any of this. It's what staying inside the contract buys you.

---

# The trap

Broadcasting prevents *one* specific intermediate.
It does **not** prevent intermediates from chained operations.

```python
result = (a - a.mean(axis=1, keepdims=True)) ** 2
```

What gets allocated:

<v-clicks>

- `a.mean(...)`        → small, shape `(N, 1)`
- `a - a.mean(...)`    → **full size** intermediate
- `(...) ** 2`         → **full size** intermediate

</v-clicks>

<v-clicks>

- The mean broadcast: no tile, contract held.
- But chaining still costs you intermediates.

</v-clicks>

---
layout: section
---

# The closing

---

# The promise, kept

```python
result = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
```

```python {1-3|5-7|9-10|12-13}
a.mean(axis=1, keepdims=True)   # C kernel (idea 1)
                                # shape (N, 1), keepdims preserves
                                # the column for broadcasting
                                 
a - a.mean(...)                 # broadcasts (N, 1) against (N, M)
                                # no tile allocated (idea 3)
                                # full-size intermediate

(...) ** 2                      # C kernel, elementwise
                                # another full-size intermediate
                                 
.sum(axis=1)                    # C kernel, reduction
                                # collapses to shape (N,)
```

<v-clicks>

## You can read this yourself now.

</v-clicks>

---
layout: section
---

# One more thing

## The intermediates we never killed

---

# The cost we accepted

```python
result = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
```

<v-clicks>

- Broadcasting saved the tile.
- But each chained step still wrote a **full-size intermediate** to memory.
- NumPy evaluates one operation at a time. It finishes `a - mean`, stores it, then starts `** 2`.
- For a big `a`, that's two full arrays written out and read straight back. Pure bandwidth (idea 2).

</v-clicks>

---

# numexpr: same expression, one pass

```python {1|3|5}
import numexpr as ne

m = a.mean(axis=1, keepdims=True)         # small, shape (N, 1)

result = ne.evaluate('sum((a - m) ** 2, axis=1)')
```

<v-clicks>

- numexpr compiles the string into one fused loop over the buffers.
- It walks the data **once**, in cache-sized chunks, across threads.
- `a - m` and `** 2` never become full arrays. The intermediates are gone.

```python
# a is (2000, 2000) float64, 32 MB
%timeit ((a - a.mean(1, keepdims=True)) ** 2).sum(1)  # ~23 ms
%timeit ne.evaluate('sum((a - m) ** 2, axis=1)')      # ~5 ms
```

</v-clicks>

---

# The catch

<v-clicks>

- It's not free magic. numexpr supports a **subset** of NumPy: arithmetic, comparisons, a handful of functions and reductions.
- The expression is a **string**, so you give up the syntax checking and tooling that real code gets.
- For a single operation there's nothing to fuse, and if the array already fits in cache there's nothing to win. At (1000, 1000) the same expression is a wash.
- You can get most of the way there without the dependency:

```python
np.subtract(a, m, out=buf)      # ~6 ms, no new allocation
np.multiply(buf, buf, out=buf)
buf.sum(axis=1)
```

- Reach for numexpr when you have a **chain** of elementwise operations over arrays too big for cache. That's exactly where NumPy's intermediates hurt.

</v-clicks>

---

# The takeaway

- An ndarray is a header pointing at a buffer.
- Operations relocate to C.
- Broadcasting holds a contract, and it holds it with a stride of 0.
- The cost is wherever copies happen. Usually not where you wrote it.
- When a chain of operations is the cost, a tool like numexpr can fuse it away.

## The cost isn't where you think it is!

---

# What to do with this

<v-clicks>

- When you next look at NumPy code, yours or someone else's, read it the way we just read these examples.
  - Where are the C kernels?
  - Is this reshape a view or a copy? (`np.shares_memory` will tell you)
  - What is broadcasting allocating, and what isn't it?

- The model isn't useful because it makes you write clever code.
- It's useful because it makes the costs **visible**, and once you can see them, you can decide which ones to pay.

</v-clicks>

---

# References

- [Introduction to Numerical Computing with NumPy | SciPy 2019 | Alex Chabot-Leclerc](https://www.youtube.com/watch?v=ZB7BZMhfPgk). The gentle on-ramp if today went too fast.
- [Advanced NumPy | SciPy Japan 2019 | Juan Nunez-Iglesias](https://www.youtube.com/watch?v=cYugp9IN1-Q). Goes deeper on strides, views, and the C side.
- [Array Programming with NumPy | Harris et al.](https://arxiv.org/abs/2006.10256). The canonical paper.
- [Internal organization of NumPy arrays](https://numpy.org/doc/stable/dev/internals.html). Authoritative on memory layout.
- [Advanced NumPy | Scientific Python Lectures](https://lectures.scientific-python.org/advanced/advanced_numpy/index.html). Strides, ufuncs, and the C API in depth.
- [numexpr documentation](https://numexpr.readthedocs.io/). Fusing chained expressions to skip the intermediates.

---
layout: center
---

# Thank you

Questions?
