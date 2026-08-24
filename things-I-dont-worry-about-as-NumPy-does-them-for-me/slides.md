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

# AI Disclosure

- This talk is based on my own thoughts, experiences and expertise
- I created **most** of the content
- AI assisted by:
    - designing the ASCII diagrams
    - reviewing the code examples
    - writing the benchmarking code, which was reviewed by myself
    - generating git commit messages that are actually useful
- The model used was Claude Opus 4.8

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

<!--
Three innocent-looking lines. Nobody would flag any of these in review.

[click] Transpose to channels-first. Still fine.

[click] Reshape. And *this* is the line where 6.3 gigabytes just got
allocated and moved.

Don't explain it yet, just plant it. We come back to this twice.
**The cost isn't where you think it is.**
-->

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

<!--
Everyone has seen this comparison.

[click] Pure Python: 72 milliseconds.

[click] NumPy: a quarter of a millisecond. Call it 300x.

Yes, NumPy is faster. *But that's not the lesson.* Everybody already knows
NumPy is faster, it's why they're using it. The question nobody asks is
**why**, and the answer is not "because C is fast".
-->

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

<!--
A trace hook. It counts every Python line executed, nothing else.

[click] The list comprehension: a million and one lines.

[click] The same result through NumPy: one.

A million versus one. Python's bytecode interpreter ran a million times in
the first version and **once** in the second.

So NumPy isn't accelerating Python. It's *relocating the work* somewhere
Python never touches. Hold onto that word, relocating. Everything in this
section is a consequence of it.
-->

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

<!--
*This* is the loop that ran. In C. Over the whole array.

One call, one loop, no interpreter anywhere in sight. That's where the
million lines went. They didn't get faster, they stopped existing.
-->

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

<!--
Same shape, same operation, same syntax. The only difference is the dtype.

[click] For int64 NumPy has a C kernel, so we're back to a third of a
millisecond.

[click] For object it doesn't, and there's no general C function for
arbitrary Python objects. So it falls back to calling `__pow__` a million
times. 30 milliseconds, a hundred times slower.

The relocation contract requires a kernel. No kernel, no relocation.
**Object dtype opts you out of every performance property NumPy offers.**
If you spot `dtype=object` in a hot path, you've found the problem.
-->

---

# Bridge

- So the *operations* live in C.
- But what about the *data*?
- What does an array actually look like?

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

<!--
800 megabytes of random data.

[click] Transposing all 800 MB takes forty nanoseconds.

[click] The same operation with `.copy()` costs about 810 milliseconds.
Roughly twenty million times more.

Forty nanoseconds isn't enough time to touch 800 MB. It's barely enough
time to touch anything at all. So what is `.T` actually doing? The next
few slides are the answer.
-->

---

# The picture: buffer

```
buffer in memory:    [1] [2] [3] [4] [5] [6] (each box = 8 bytes, float64)
```

<!--
Six contiguous elements, eight bytes each. That's all the data there is.

Everything else we're about to talk about is bookkeeping sitting on top
of this one flat run of bytes.
-->

---

# The picture: header

```
buffer in memory:    [1] [2] [3] [4] [5] [6] (each box = 8 bytes, float64)
                      ^
                      |
          a:  shape=(2, 3)   strides=(24, 8)
```

<!--
The header points at the buffer.

Shape says how the bytes are laid out conceptually: two rows of three.
Strides say how many bytes to step along each axis: 24 to move down a row,
8 to move across a column.

Note what hasn't changed. The buffer is identical. All we've added is a
description of it.
-->

---

# The picture: transpose

```
buffer in memory:    [1] [2] [3] [4] [5] [6] (each box = 8 bytes, float64)
                      ^
                      |
          a:    shape=(2, 3)   strides=(24, 8)
          a.T:  shape=(3, 2)   strides=(8, 24)
```

<!--
Same buffer. Same six boxes, in the same order.

Transpose **swapped two numbers** in the strides field. 24, 8 became 8, 24.
That's it, that's the entire operation. The data didn't move.

And that is your forty nanoseconds.
-->

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

<!--
Watch what happens when two headers point at one buffer.

[click] `a` is zeros, `b` is its transpose.

[click] We write a 42 through `b`...

[click] ...and read it straight back out of `a`.

We never touched `a`. We wrote through `b`. There was only ever one buffer,
so there was only ever one place for that write to land.

The header/buffer split isn't trivia. It decides who sees your writes. This
is the bug people file against NumPy that turns out not to be a bug.
-->

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

<!--
You don't have to take the diagram on faith. The array will tell you.

[click] `itemsize`: 8 bytes per element.

[click] `shape`: elements per dimension.

[click] `strides`: **bytes** to step per dimension. (24, 8) means "skip a
whole row", then "skip one column", for float64.

[click] And the transpose has both reversed. `b` is not C-contiguous.

But it *is* F_CONTIGUOUS, and that matters. Non-contiguous doesn't mean
scrambled. `b` is still perfectly regular, just column-major. That
regularity is exactly what lets BLAS take `b` without copying it first.
-->

---

# Stop guessing: ask

```python{1|2-3|4-5}
>>> flat = images.transpose(0, 3, 1, 2).reshape(1000, -1)
>>> np.shares_memory(flat, images)
False                      # it copied
>>> np.may_share_memory(flat, images)
False                      # it copied, errs towards True
```

<!--
First, the rule underneath all of this. `reshape` returns a **view** when
the requested shape is compatible with the existing memory layout, and
**copies** when it isn't. Compatible means NumPy can produce the new shape
by picking new strides over the same buffer, without rearranging any bytes.
A contiguous array can almost always be reshaped for free. A non-contiguous
one sometimes can, depending which axes you touch.

Heuristic worth writing down: if you've done a transpose, fancy indexing,
or any axis-rearranging operation recently, **assume reshape might copy**.

[click] So don't guess, ask. `shares_memory` is the question you actually
mean: did these two end up on the same bytes?

[click] `may_share_memory` is the cheap, conservative version. It's
conservative towards *True*, it answers True when it can't be sure. So a
False from it is definitive.

The trap here is `.base`, and it's worth naming out loud. `flat.base is None`
returns False, which reads like "it's a view, nothing was copied". Both
things are true at once: `flat` really is a view, of the intermediate copy
that reshape was forced to make. `.base` answers "do I own my buffer?", not
"do I share bytes with the array you care about". Only `shares_memory` takes
both arrays, which is why it's the one that can answer.
-->

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
- **The currency is bandwidth, not bytes.**

</v-clicks>


---

# The villain returns: recall

```python {lines:true}
images = load_images()                   # (1000, 512, 512, 3)
images = images.transpose(0, 3, 1, 2)
flat = images.reshape(1000, -1)          # 6.3 GB copy
```

<!--
Remember this from the start? I said something here cost 6.3 gigabytes.

We've got the vocabulary now. Let's actually read it.
-->

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

<!--
Loaded C-contiguous. Nothing wrong yet.

[click] The transpose reorders the strides. Same buffer, no data moved,
but it is no longer C-contiguous.

[click] And reshape needs to walk the new shape in a regular stride
pattern. The transpose broke that, so it copies. 6.3 GB allocated and moved.

The cost wasn't on the line that did the work. It was set up three lines
earlier.

The fix is to wrap the transpose in `np.ascontiguousarray`, and then the
reshape is free. But notice what that does *not* do. It doesn't make the
copy go away, the 6.3 GB still moves. `.copy()` would work just as well,
`ascontiguousarray` just names the thing we actually want.

What changed is that the copy now sits on the line that asks for it,
instead of hiding inside `reshape`. **The model doesn't avoid copies. It
makes them visible.** That's the argument of the whole talk in one line.
-->

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
layout: section
---

# One more thing

## The intermediate we never killed

---

# The cost we accepted

Broadcasting prevents *one* specific intermediate. It does **not** prevent the ones chaining creates.

```python
result = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
```

<v-clicks>

- `a.mean(...)` → small, shape `(N, 1)`. The broadcast saved the tile, contract held.
- `a - a.mean(...)` → **full size** intermediate.
- `(...) ** 2` → another **full size** intermediate.
- NumPy evaluates one operation at a time. It finishes `a - mean`, stores it, then starts `** 2`.
- For a big `a`, that's two full arrays written out and read straight back. Pure bandwidth (idea 2).
- Everything in this talk says that costs us two allocations. So let's count them.

</v-clicks>

---

# Count them

NumPy registers its allocations with `tracemalloc`, so we can just look.

```python {1-3|5-7}
tracemalloc.start()
((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
tracemalloc.get_traced_memory()      # a is (2000, 2000), 32 MB

# predicted: 64 MB, two full-size intermediates
# measured:  32 MB
```

<v-clicks>

- One intermediate. Not two.
- Nobody wrote `out=`. Nobody imported anything.
- **NumPy elided it.**

</v-clicks>

---

# Only a temporary can have a refcount of 1

```python
x = a + b + b
#   ^^^^^ this temporary is never given a name,
#         so nothing else holds a reference to it
```

Simplified from `numpy/_core/src/multiarray/temp_elide.c`:

```c {|5|6-9|10|11}
static int
can_elide_temp(PyObject *olhs, PyObject *orhs, int *cannot)
{
    PyArrayObject *alhs = (PyArrayObject *)olhs;
    if (!check_unique_temporary(olhs) ||        // refcount == 1
            !PyArray_CheckExact(olhs) ||        // not a subclass
            !PyArray_ISNUMBER(alhs) ||          // not object dtype
            !PyArray_CHKFLAGS(alhs, NPY_ARRAY_OWNDATA) ||
            !PyArray_ISWRITEABLE(alhs) ||
            PyArray_NBYTES(alhs) < NPY_MIN_ELIDE_BYTES) {
        return 0;
    }
```

<!--
`LOAD_FAST` bumps the refcount of every *named* variable it pushes. So a
refcount of 1 is a reliable signal: this array is nobody else's. And if
nobody else can see it, overwriting it changes no observable behaviour.

[click] That is this line. On 3.13 and earlier `check_unique_temporary` is
just `Py_REFCNT(lhs) == 1` behind a portability shim. Remember it, it is
the line that breaks in twenty slides' time.

[click] The rest is NumPy refusing to be clever. An exact ndarray, not a
subclass whose `__array_finalize__` might notice. A numeric dtype, not
object. Owning its own data, and writeable.

[click] And big enough to be worth it, which is the 256 KiB we come to next.

[click] Fail any one of them and it bails out and allocates, exactly as
before.

So when all of it passes, NumPy rewrites `tmp ** 2` into `tmp **= 2` and
reuses the buffer. CPython plays the same trick to grow strings in place.
-->

---

# The paranoid part

A Cython extension can also call `PyNumber_Add` with a refcount of 1. That array might not be a temporary at all.

<v-clicks>

- So before eliding, NumPy calls `backtrace()` and walks up to 10 stack frames.
- Every frame must sit inside libpython or NumPy itself, until it reaches `_PyEval_EvalFrameDefault`.
- Anything else on the stack and it refuses. It has to know the interpreter called it.
- That stack walk costs about 10 microseconds, which buys a rule:

```python
NPY_MIN_ELIDE_BYTES = 256 * 1024     # below this, don't even check
```

- 128 KiB: two allocations. 256 KiB: one. The cliff is exactly there.

</v-clicks>

---

# When it doesn't fire

```python {1-2|4-5|7-8}
t = a - m                  # t is named, refcount > 1
t ** 2                     # no elision, allocates

np.sqrt(a - m)             # ufunc call, not an operator
                           # no elision, allocates

(a - m) * col              # col is (N, 1), shapes differ
                           # elision refuses to broadcast
```

<!--
Three ways to lose it. Give the temporary a name and you've taken it away,
`t` has a refcount above 1 so there's nothing to elide.

[click] The hooks live in the operators, not in `np.sqrt`. A plain ufunc
call allocates.

[click] And `can_elide_temp` demands matching shapes. Broadcasting saves
you the tile, but it costs you the elision.
-->

---

# And then it stopped

```python
# python 3.13:   32 MB,  8.3 ms
# python 3.14:   64 MB, 22.4 ms      <- same numpy, same expression
```

<v-clicks>

- CPython 3.14 moved the evaluation stack to *stackrefs*.
- Temporaries no longer reliably show a refcount of 1, so the heuristic stopped recognising them.
- Nothing raised. Nothing warned. The expression got 2.7x slower and kept returning the right answer.
- Fixed for correctness ([numpy#28681](https://github.com/numpy/numpy/issues/28681)), and CPython added `PyUnstable_Object_IsUniqueReferencedTemporary` for it. On 3.14.0 I still measure no elision.
- **This is the point of the whole talk.** You cannot notice this without a model of what NumPy was doing for you.

</v-clicks>

---

# The fix, and what it doesn't fix

The refcount test from earlier now lives behind a shim:

```c {|4-5|6-8|9-11}
static int
check_unique_temporary(PyObject *lhs)
{
#if PY_VERSION_HEX == 0x030E00A7 && !defined(PYPY_VERSION)
#error "NumPy is broken on CPython 3.14.0a7, please update to a newer version"
#elif PY_VERSION_HEX >= 0x030E00B1 && !defined(PYPY_VERSION)
    // see https://github.com/python/cpython/issues/133164
    return PyUnstable_Object_IsUniqueReferencedTemporary(lhs);
#else
    // equivalent to Py_REFCNT(lhs) == 1 except on 3.13t
    return PyUnstable_Object_IsUniquelyReferenced(lhs);
#endif
}
```

<!--
This is the same check we looked at earlier, the one guarding `can_elide_temp`.
It used to be a bare `Py_REFCNT(olhs) != 1`. Now it is thirteen lines of
version detection, and every branch is a scar.

[click] For exactly one CPython alpha, 3.14.0a7, there was no way to get the
right answer at all, so NumPy refuses to compile against it. That is what it
looks like when an assumption this deep breaks.

[click] From 3.14 beta 1 onwards, stop guessing from the refcount and ask the
interpreter directly. That is the API CPython added for this.

[click] And everywhere else, the old meaning, refcount equals one. Even that
needs a backport, because on free-threaded 3.13 the plain refcount answer was
never trustworthy either.

Now the honest part. This makes NumPy's answer **correct**, not fast. The
question "is this a temporary" is being asked properly again, and on 3.14 the
answer is usually no, so on 3.14.0 I still measure two full-size intermediates
and 22 milliseconds. The optimisation has not come back. What got fixed is
that NumPy is no longer eliding based on a signal that stopped meaning
anything, which is the bug that mattered.
-->

---

# The takeaway

- An ndarray is a header pointing at a buffer.
- Operations relocate to C.
- Broadcasting holds a contract, and it holds it with a stride of 0.
- The cost is wherever copies happen. Usually not where you wrote it.
- NumPy quietly elides a chained temporary for you, and quietly stops when the interpreter changes underneath it.

## The cost isn't where you think it is!

---

# What to do with this

<v-clicks>

- When you next look at NumPy code, yours or someone else's, read it the way we just read these examples.
  - Where are the C kernels?
  - Is this reshape a view or a copy? (`np.shares_memory` will tell you)
  - What is broadcasting allocating, and what isn't it?
  - How many intermediates is this chain really paying for? (`tracemalloc` will tell you)

- The model isn't useful because it makes you write clever code.
- It's useful because it makes the costs **visible**, and once you can see them, you can decide which ones to pay.

</v-clicks>

---

# If NumPy is your thing, come work with me!

- Endgame is a technology-led consultancy providing expert mathematical and economic advice
- **If it touches the energy sector, we run the numbers to understand it**

<div class="flex justify-center items-start gap-20 mt-12">
  <img :src="'/LogoWithBackground-02.png'" class="h-52 rounded-lg" alt="Endgame Analytics logo" />
  <div class="flex flex-col items-center gap-3">
    <img :src="'/pycon-au-2026-careers-qr.png'" class="h-52 rounded-lg" alt="QR code to the Endgame Analytics careers page" />
    <span class="text-sm opacity-75">Scan for open roles</span>
  </div>
</div>

---


# References

- [Introduction to Numerical Computing with NumPy | SciPy 2019 | Alex Chabot-Leclerc](https://www.youtube.com/watch?v=ZB7BZMhfPgk). The gentle on-ramp if today went too fast.
- [Advanced NumPy | SciPy Japan 2019 | Juan Nunez-Iglesias](https://www.youtube.com/watch?v=cYugp9IN1-Q). Goes deeper on strides, views, and the C side.
- [Array Programming with NumPy | Harris et al.](https://arxiv.org/abs/2006.10256). The canonical paper.
- [Internal organization of NumPy arrays](https://numpy.org/doc/stable/dev/internals.html). Authoritative on memory layout.
- [Advanced NumPy | Scientific Python Lectures](https://lectures.scientific-python.org/advanced/advanced_numpy/index.html). Strides, ufuncs, and the C API in depth.
- [`temp_elide.c`](https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/temp_elide.c). 400 lines, and the comment at the top explains the whole trick.
- [numpy#28681](https://github.com/numpy/numpy/issues/28681) and [cpython#133164](https://github.com/python/cpython/issues/133164). What stackrefs did to elision, and the C API added to fix it.

---
layout: center
---

# Thank you

Questions?
