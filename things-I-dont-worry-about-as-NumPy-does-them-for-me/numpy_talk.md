---
title: Things I don't worry about as NumPy does them for me
sub_title: A mental model for the things NumPy is quietly doing on your behalf
author: Kai Striega
options:
  incremental_lists: true
---

<!-- jump_to_middle -->

# Why we're here

* I spent years maintaining SciPy, which sits on top of NumPy.
* In that time I read a *lot* of NumPy code written by very smart people.
* Most performance problems weren't because people didn't know NumPy.
They were because people had the **wrong mental model** of what NumPy
was doing underneath.

<!-- end_slide -->

# The villain

```python +line_numbers {1|2|3}
images = load_images()                 # (1000, 512, 512, 3)
images = images.transpose(0, 3, 1, 2)
flat = images.reshape(1000, -1)        # ← 3 GB copy
```

* Three innocent-looking lines.
* One of them just allocated and moved 3 gigabytes.
* **The cost isn't where you think it is.**

<!-- end_slide -->

# The promise

By the end of this talk, you'll look at this expression:

```python
result = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
```

...and know exactly what NumPy just did to memory.

* Three ideas. Not tricks. Lenses.

<!-- end_slide -->

<!-- jump_to_middle -->

Idea 1
===

# The work isn't where you think it is

<!-- end_slide -->

# The familiar comparison

```python {1|4|7}
data = np.arange(1_000_000, dtype=np.float64)

# Pure Python
%timeit  [x ** 2 for x in data]    # ~400 ms

# NumPy
%timeit data ** 2                  # ~2 ms
```

* Yes — NumPy is faster.
* *But that's not the lesson*.

<!-- end_slide -->

# Watch what's actually happening

```python {1-3|5-9|11-15|17-21}
import sys

python_lines_visited = 0

def counter(frame, event, arg):
    global python_lines_visited
    if event == 'line':
        python_lines_visited += 1
    return counter

python_lines_visited = 0
sys.settrace(counter)
result = [x ** 2 for x in data]
sys.settrace(None)
# python_lines_visited: 1,018,399 

python_lines_visited = 0
sys.settrace(counter)
result = data ** 2
sys.settrace(None)
# python_lines_visited:  1
```

<!-- end_slide -->

# Three million versus one

* Python's bytecode interpreter ran more than **a million times** in the first version.
* **Once** in the second.
* NumPy isn't accelerating Python.
* It's *relocating the work* somewhere Python never touches.

<!-- end_slide -->

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

* *This* is the loop that ran. In C. Once. Over the whole array.
* This is a **ufunc** — a vectorised C kernel.

<!-- end_slide -->

# When the relocation breaks

```python {1-2|4-5|7-8}
ints = np.arange(1_000_000, dtype=np.int64)
objs = np.arange(1_000_000, dtype=object)

# INT64_square kernel runs in C
%timeit ints ** 2     # ~2 ms     

# Python's __pow__ called a million times
%timeit objs ** 2     # ~80 ms    
```

* Same shape, same operation, same syntax.
* For `int64`, NumPy has a C kernel.
* For `object`, it doesn't — there's no general C function for arbitrary Python objects.

* The relocation contract requires a kernel.
* **Object dtype opts you out of every performance property NumPy offers.**

<!-- end_slide -->

# Bridge

* So the *operations* live in C.
* But what about the *data*?
* What does an array actually look like?

<!-- end_slide -->

<!-- jump_to_middle -->

Idea 2
===

# Arrays aren't what you think they are

<!-- end_slide -->

# A surprising timing

```python {1|3|4}
big = np.zeros((10_000, 10_000))    # 800 MB

%timeit big.T                       # ~100 ns
%timeit big.T.copy()                # ~400 ms
```

* Transposing 800MB in **100 nanoseconds**.
* The same operation with `.copy()` costs four million times more.
* What is `.T` actually doing, then?

<!-- end_slide -->

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

* A small **header** pointing at a flat **buffer**.

<!-- end_slide -->

# The picture — buffer

```
buffer in memory:    [1] [2] [3] [4] [5] [6]
```

* Six contiguous bytes.
* That's all the data there is.

<!-- end_slide -->

# The picture — header

```
buffer in memory:    [1] [2] [3] [4] [5] [6]
                      ^
                      |
          a:  shape=(2, 3)   strides=(24, 8)
```

* The header points at the buffer.
* Shape says how the bytes are laid out conceptually.
* Strides say how many bytes to step per axis.

<!-- end_slide -->

# The picture — transpose

```
buffer in memory:    [1] [2] [3] [4] [5] [6]
                      ^
                      |
          a:    shape=(2, 3)   strides=(24, 8)
          a.T:  shape=(3, 2)   strides=(8, 24)
```

* Same buffer.
* Transpose just **swapped two numbers** in the strides field.
* The data didn't move. That's the whole operation.

<!-- end_slide -->

# Verifying the picture

```python 
>>> a = np.zeros((2, 3))
>>> a.shape
(2, 3)
>>> a.strides
(24, 8)
>>> a.flags['C_CONTIGUOUS']
True
>>> b = a.T
>>> b.shape
(3, 2)
>>> b.strides
(8, 24)
>>> b.flags['C_CONTIGUOUS']
False
>>> b.flags['F_CONTIGUOUS']
True
```

* Strides are in **bytes** — `(24, 8)` is "skip a row" then "skip a column" for a `float64` array.
* Transpose breaks C-contiguity but creates F-contiguity.
* Non-contiguous doesn't mean scrambled.

<!-- end_slide -->

# The reshape contract

* `reshape` returns a **view** when the requested shape is compatible with the existing memory layout.
* It **copies** when it isn't.
* "Compatible" means: NumPy can produce the new shape by choosing new strides
over the same buffer, without rearranging any bytes.

   - A C-contiguous array can always be reshaped without copying.
   - A non-contiguous array sometimes can — it depends on which axes you touch.

> Heuristic: if you've done a transpose, fancy indexing, or any
> axis-rearranging operation recently, **assume reshape might copy**.
> Check `flags` if you care.

<!-- end_slide -->

# The villain returns — recall

```python +line_numbers
images = load_images()                   # (1000, 512, 512, 3)
images = images.transpose(0, 3, 1, 2)
flat = images.reshape(1000, -1)          # ← 3 GB copy
```

* Remember this from the start?
* I said something here cost 3 gigabytes.
* Let's read it!

<!-- end_slide -->

# The villain returns — diagnose

```python {1-2|4-6|8-10}
images = load_images()                  # shape (1000, 512, 512, 3)
                                        # C-contiguous ✓
                                        
images = images.transpose(0, 3, 1, 2)   # shape (1000, 3, 512, 512)
                                        # strides reordered — same buffer
                                        # C-contiguous ✗
                                        
flat = images.reshape(1000, -1)         # needs contiguous layout
                                        # buffer doesn't match → copy
                                        # 3 GB allocated and moved
```

* The cost wasn't on the line that did the work.
* It was set up three lines earlier.

<!-- end_slide -->

# The villain returns — resolve

```python {1-3|2|3}
images = load_images()
images = images.transpose(0, 3, 1, 2).copy()  # explicit copy here
flat = images.reshape(1000, -1)               # now free
```

(`np.ascontiguousarray(...)` does the same thing.)

* The fix doesn't make the copy go away.
* The 3 GB still gets moved.
* What changed: the copy is now on the line that says `copy`, instead of hiding inside `reshape`.
* **The model doesn't avoid copies. It makes them visible.**

<!-- end_slide -->

# Bridge

* We've seen what arrays *are*.
* We've seen what operations *cost*.
* But what about operations between *mismatched shapes*?

<!-- end_slide -->

<!-- jump_to_middle -->

Idea 3
===

# Broadcasting is a contract

<!-- end_slide -->

# The contract

* Broadcasting is a deal NumPy offers you.
* You give it two arrays of mismatched shape.
* It runs the operation **as if the smaller one were the size of the larger** — without ever allocating that larger version.
* The only thing that gets allocated is the **result**.

<!-- end_slide -->

# The contract in code

```python
a = np.zeros((1000, 1000))   # 8 MB
b = np.arange(1000)          # 8 KB

result = a + b               # 8 MB (just the result)
```

What didn't happen:

* `b` was **not** tiled to (1000, 1000)
* No 8 MB intermediate was allocated
* The C kernel iterated over `a`'s shape, reading `b` modularly

<!-- end_slide -->

# The rules

1. Align shapes from the **trailing axis**.
2. Each axis pair must be **equal, or one of them must be 1**.
3. Missing axes are treated as 1.

``` 
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

<!-- end_slide -->

# Why the contract is worth keeping

* Because broadcasting doesn't materialise, the C kernel can stream through memory in a tight loop.
* That keeps the **cache** warm and lets NumPy hand off to **SIMD** instructions or **BLAS** routines underneath.
* You don't ask for any of this.
* It's the consequence of staying inside the contract.

<!-- end_slide -->

# The trap

* Broadcasting prevents *one* specific intermediate.
* It does **not** prevent intermediates from chained operations.

```python
result = (a - a.mean(axis=1, keepdims=True)) ** 2
```

What gets allocated:

- `a.mean(...)`        → small, shape `(N, 1)`
- `a - a.mean(...)`    → **full size** intermediate
- `(...) ** 2`         → **full size** intermediate (or reused buffer)

* The mean broadcast: no tile, contract held.
* But chaining still costs you intermediates.

> Most ufuncs take an `out=` parameter that writes into a pre-allocated
> buffer. Worth knowing exists; we won't go further.

<!-- end_slide -->

# Bridge

You now have all three lenses.

* The work lives in C.
* The array is a header pointing at a buffer.
* Broadcasting is a contract that saves the tile.

* Let's use them!

<!-- end_slide -->

<!-- jump_to_middle -->

The closing
===

# Reading the promise

<!-- end_slide -->

# The promise, kept

```python
result = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
```

``` {1-13|1-3|5-7|9-10|12-13}
a.mean(axis=1, keepdims=True)   ▏ C kernel (idea 1)
                                ▏ shape (N, 1) — keepdims preserves
                                ▏ the column for broadcasting
                                ▏
a - a.mean(...)                 ▏ broadcasts (N, 1) against (N, M)
                                ▏ no tile allocated (idea 3)
                                ▏ full-size intermediate
(...) ** 2                      ▏ C kernel, elementwise
                                ▏ another full-size intermediate
                                ▏
.sum(axis=1)                    ▏ C kernel, reduction
                                ▏ collapses to shape (N,)
```

## You could do this without me!

<!-- end_slide -->

# The takeaway

1. An ndarray is a header pointing at a buffer.
2. Operations relocate to C, broadcasting holds a contract, and the cost is wherever copies happen — usually not where you wrote them.

## The cost isn't where you think it is!

<!-- end_slide -->


# What to do with this

* When you next look at NumPy code, yours or someone else's, read it the way we just read these examples.
    - Where are the C kernels?
    - Is this reshape a view or a copy?
    - What is broadcasting allocating, and what isn't it?

* The model isn't useful because it makes you write clever code.
* It's useful because it makes the costs **visible**, and once you can see them, you can decide which ones to pay.

<!-- end_slide -->

# References

- 
- 
- 
- 

<!-- end_slide -->

<!-- jump_to_middle -->

# Thank you

Questions?
