#!/usr/bin/env python
"""Benchmark and verify every claim made in slides.md.

Run inside the pinned conda environment (see environment.yml) so the
versions match the ones the slide timings were measured against:

    conda env create -f environment.yml
    conda run -n numpy-talk python benchmark_claims.py

Each section maps to a slide. For every claim we print:
  * what the slide says
  * what this machine actually measures
  * a PASS / CHECK marker

Timing numbers in the slides are order-of-magnitude illustrations, so for
those we check we are in the right ballpark rather than demanding an exact
match. Structural claims (shapes, strides, flags, view-vs-copy) are exact
and asserted.
"""

import gc
import sys
import timeit

import numpy as np

try:
    import numexpr as ne
    HAVE_NUMEXPR = True
except ImportError:
    HAVE_NUMEXPR = False


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def line(claim, slide_value, measured, ok):
    mark = "PASS " if ok else "CHECK"
    print(f"  [{mark}] {claim}")
    print(f"          slide : {slide_value}")
    print(f"          actual: {measured}")


def bench(stmt, setup="pass", globals_=None, min_time=0.2):
    """Return best per-call time in seconds using an adaptive loop count."""
    timer = timeit.Timer(stmt, setup=setup, globals=globals_)
    # find a loop count that runs for at least a few ms
    number, _ = timer.autorange()
    # repeat to take the minimum (most stable) reading
    runs = timer.repeat(repeat=5, number=number)
    return min(runs) / number


def fmt_time(seconds):
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def within(measured, low, high):
    return low <= measured <= high


# --------------------------------------------------------------------------
# Idea 1 -- "The familiar comparison"  (slides.md L115-123)
# --------------------------------------------------------------------------

def familiar_comparison():
    header("Idea 1 / The familiar comparison  (L113-129)")
    data = np.arange(1_000_000, dtype=np.float64)

    t_py = bench("[x ** 2 for x in data]", globals_={"data": data})
    t_np = bench("data ** 2", globals_={"data": data})

    line(
        "pure-Python list comprehension on 1e6 floats",
        "~72 ms",
        fmt_time(t_py),
        # generous band: hardware varies a lot
        within(t_py, 0.05, 1.5),
    )
    line(
        "NumPy  data ** 2  on 1e6 floats",
        "~0.25 ms",
        fmt_time(t_np),
        within(t_np, 0.0002, 0.02),
    )
    line(
        "NumPy is faster than pure Python",
        "yes (much)",
        f"{t_py / t_np:.0f}x faster",
        t_np < t_py,
    )


# --------------------------------------------------------------------------
# Idea 1 -- "Watch what's actually happening"  (slides.md L134-156)
# --------------------------------------------------------------------------

def trace_line_counts():
    header("Idea 1 / Watch what's actually happening -- settrace  (L136-158)")
    data = np.arange(1_000_000, dtype=np.float64)

    python_lines_visited = 0

    def counter(frame, event, arg):
        nonlocal python_lines_visited
        if event == "line":
            python_lines_visited += 1
        return counter

    # sys.settrace only traces *newly called* frames, and on CPython 3.12+ a
    # list comprehension is inlined into the current frame -- so we must run
    # each snippet inside its own function call to get it traced.
    def run_listcomp():
        return [x ** 2 for x in data]

    def run_vectorised():
        return data ** 2

    # version 1: list comprehension
    python_lines_visited = 0
    sys.settrace(counter)
    run_listcomp()
    sys.settrace(None)
    py_lines = python_lines_visited

    # version 2: vectorised
    python_lines_visited = 0
    sys.settrace(counter)
    run_vectorised()
    sys.settrace(None)
    np_lines = python_lines_visited

    line(
        "list comp trips the line tracer ~1 per element",
        "~1,018,399",
        f"{py_lines:,}",
        py_lines > 1_000_000,
    )
    line(
        "data ** 2 trips the line tracer",
        "1",
        f"{np_lines}",
        np_lines <= 2,  # exact count is interpreter-version dependent
    )
    print("          note: exact counts depend on the CPython version "
          f"(here {sys.version.split()[0]}).")


# --------------------------------------------------------------------------
# Idea 1 -- "When the relocation breaks"  (slides.md L199-217)
# --------------------------------------------------------------------------

def relocation_breaks():
    header("Idea 1 / When the relocation breaks -- object dtype  (L199-217)")
    ints = np.arange(1_000_000, dtype=np.int64)
    objs = np.arange(1_000_000, dtype=object)

    t_int = bench("ints ** 2", globals_={"ints": ints})
    t_obj = bench("objs ** 2", globals_={"objs": objs})

    line("int64 ** 2 (C kernel)", "~0.28 ms", fmt_time(t_int), within(t_int, 0.0002, 0.02))
    line("object ** 2 (Python __pow__ per element)", "~30 ms",
         fmt_time(t_obj), within(t_obj, 0.01, 0.5))
    line("object dtype is dramatically slower", "yes",
         f"{t_obj / t_int:.0f}x slower", t_obj > t_int * 5)


# --------------------------------------------------------------------------
# Idea 2 -- "A surprising timing"  (slides.md L265-282)
# --------------------------------------------------------------------------

def surprising_timing():
    header("Idea 2 / A surprising timing -- transpose vs copy  (L265-282)")
    # np.random.random, NOT np.zeros. np.zeros comes from calloc, so its pages
    # are lazily mapped to the shared zero page until touched -- timing a copy
    # out of it measures page-fault behaviour as much as data movement.
    big = np.random.random((10_000, 10_000))  # 800 MB

    size_mb = big.nbytes / 1e6
    line("np.random.random((10_000, 10_000)) size", "800 MB",
         f"{size_mb:.0f} MB", abs(size_mb - 800) < 1)

    t_view = bench("big.T", globals_={"big": big})
    t_copy = bench("big.T.copy()", globals_={"big": big}, min_time=0.5)

    line("big.T (view, swaps strides)", "~40 ns",
         fmt_time(t_view), within(t_view, 1e-9, 5e-6))
    line("big.T.copy() (transposing copy of 800 MB)", "~810 ms",
         fmt_time(t_copy), within(t_copy, 0.05, 3.0))
    line("copy costs ~twenty million times more than the view",
         "~20,000,000x", f"{t_copy / t_view:,.0f}x",
         t_copy / t_view > 1_000_000)

    # The slide claims .T.copy() pays far more than streaming bandwidth would
    # predict, for two reasons: fresh pages, and a cache-hostile access
    # pattern. Break the cost apart so the claim is visible, not asserted.
    dst = np.empty_like(big)
    np.copyto(dst, big)  # fault the destination in first
    t_stream = bench("np.copyto(dst, big)",
                     globals_={"np": np, "big": big, "dst": dst}, min_time=0.5)
    t_alloc = bench("big.copy()", globals_={"big": big}, min_time=0.5)

    line("streaming copy into warm pages (the floor)", "~40 ms",
         fmt_time(t_stream), within(t_stream, 0.005, 0.3))
    line("linear copy with fresh allocation (adds page faults)",
         "slower than the floor", fmt_time(t_alloc), t_alloc > t_stream)
    line("transposing copy (adds cache-hostile strides)",
         "slower again", fmt_time(t_copy), t_copy > t_alloc)


# --------------------------------------------------------------------------
# Idea 2 -- "One buffer, two headers"  (slides.md L338-358)
# --------------------------------------------------------------------------

def view_aliasing():
    header("Idea 2 / One buffer, two headers -- views alias  (L338-358)")
    a = np.zeros((2, 3))
    b = a.T
    b[0, 0] = 42

    line("write through b, read through a", "42.0", a[0, 0], a[0, 0] == 42.0)
    line("b.base is a", "True", b.base is a, b.base is a)


# --------------------------------------------------------------------------
# Idea 2 -- "Stop guessing: ask"  (slides.md L406-425)
# --------------------------------------------------------------------------

def shares_memory_diagnostic():
    header("Idea 2 / Stop guessing -- np.shares_memory vs .base  (L406-425)")
    # same structure as the villain, at a size we can actually allocate
    images = np.zeros((10, 8, 8, 3))
    flat = images.transpose(0, 3, 1, 2).reshape(10, -1)

    shares = np.shares_memory(flat, images)
    line("np.shares_memory(flat, images) after the transposed reshape",
         "False -> it copied", shares, shares is False)

    # the slide's point: .base is NOT a copy detector. The copy has a base too,
    # it just points at the intermediate rather than at `images`.
    line("flat.base is None", "False -> base says nothing useful",
         flat.base is None, flat.base is not None)

    # may_share_memory is the cheap conservative version
    line("np.may_share_memory agrees it did not share", "False",
         np.may_share_memory(flat, images),
         np.may_share_memory(flat, images) is False)


# --------------------------------------------------------------------------
# Idea 2 -- "Verifying the picture"  (slides.md L360-386)
# --------------------------------------------------------------------------

def verify_strides():
    header("Idea 2 / Verifying the picture -- shape, strides, flags  (L360-386)")
    a = np.zeros((2, 3))
    b = a.T

    line("a.itemsize", "8", a.itemsize, a.itemsize == 8)
    line("a.shape", "(2, 3)", a.shape, a.shape == (2, 3))
    line("a.strides", "(24, 8)", a.strides, a.strides == (24, 8))
    line("b = a.T ; b.shape", "(3, 2)", b.shape, b.shape == (3, 2))
    line("b.strides", "(8, 24)", b.strides, b.strides == (8, 24))
    line("a.T is a view (shares the buffer)", "yes",
         np.shares_memory(a, b), np.shares_memory(a, b))


# --------------------------------------------------------------------------
# Idea 2 -- "Non-contiguous doesn't mean scrambled"  (slides.md L360-386)
# --------------------------------------------------------------------------

def contiguity_flags():
    header("Idea 2 / Non-contiguous doesn't mean scrambled -- flags  (L360-386)")
    a = np.zeros((2, 3))
    b = a.T

    line("a.flags['C_CONTIGUOUS']", "True",
         a.flags["C_CONTIGUOUS"], a.flags["C_CONTIGUOUS"] is True)
    line("b.flags['C_CONTIGUOUS']", "False",
         b.flags["C_CONTIGUOUS"], b.flags["C_CONTIGUOUS"] is False)
    line("b.flags['F_CONTIGUOUS']", "True",
         b.flags["F_CONTIGUOUS"], b.flags["F_CONTIGUOUS"] is True)


# --------------------------------------------------------------------------
# Idea 2 -- "Why copies hurt" / bandwidth  (slides.md L427-445)
# --------------------------------------------------------------------------

def bandwidth():
    header("Idea 2 / Why copies hurt -- memory bandwidth  (L427-445)")
    # Measure copy bandwidth on a large contiguous buffer. Pre-allocate and
    # pre-touch the destination so we time pure memory traffic, not the
    # allocation + first-touch page faults of a fresh np.copy().
    nbytes = 500_000_000  # 0.5 GB float64
    src = np.ones(nbytes // 8, dtype=np.float64)
    dst = np.empty_like(src)
    np.copyto(dst, src)  # warm up / fault in the destination pages

    t = bench("np.copyto(dst, src)", globals_={"np": np, "src": src, "dst": dst},
              min_time=0.5)
    # a copy both reads src and writes dst, so traffic is 2x the array size
    gb_moved = 2 * src.nbytes / 1e9
    gb_per_s = gb_moved / t

    line("RAM moves data at roughly (read+write traffic)", "~40 GB/s",
         f"{gb_per_s:.1f} GB/s", within(gb_per_s, 20, 100))
    # implied wall-clock to copy 6.3 GB (read 6.3 + write 6.3 = 12.6 GB moved)
    implied = (2 * 6.3) / gb_per_s
    line("implied wall-clock for a 6.3 GB streaming copy",
         "~300 ms", f"{implied * 1000:.0f} ms",
         within(implied, 0.2, 0.45))


# --------------------------------------------------------------------------
# The villain -- transpose then reshape copies  (slides.md L73-89, L447-505)
# --------------------------------------------------------------------------

def villain():
    header("The villain -- transpose then reshape forces a copy  (L447-505)")

    # full-size byte arithmetic (we do NOT allocate 6.3 GB; we compute it)
    shape = (1000, 512, 512, 3)
    nbytes = np.prod(shape) * 8  # float64
    gb = nbytes / 1e9
    line("images (1000, 512, 512, 3) float64 size", "6.3 GB",
         f"{gb:.1f} GB", abs(gb - 6.3) < 0.1)

    # verify the view-vs-copy *behaviour* at a small, safe scale
    small = np.zeros((10, 8, 8, 3))  # tiny stand-in, same structure
    line("fresh array is C-contiguous", "C-contiguous ok",
         small.flags["C_CONTIGUOUS"], small.flags["C_CONTIGUOUS"])

    t = small.transpose(0, 3, 1, 2)
    line("after transpose(0,3,1,2): C-contiguous?", "C-contiguous broken",
         t.flags["C_CONTIGUOUS"], t.flags["C_CONTIGUOUS"] is False)

    flat = t.reshape(t.shape[0], -1)
    shares = np.shares_memory(flat, small)
    line("reshape of the transposed array shares memory?",
         "no -> reshape COPIES", shares, shares is False)

    # control: reshape on a contiguous array is a free view
    flat_c = small.reshape(small.shape[0], -1)
    shares_c = np.shares_memory(flat_c, small)
    line("reshape of the contiguous array shares memory?",
         "yes -> reshape is a VIEW", shares_c, shares_c is True)

    # control: an explicit copy before reshape makes reshape free (a view).
    # The slide uses np.ascontiguousarray because it names the intent.
    t_copied = np.ascontiguousarray(small.transpose(0, 3, 1, 2))
    line("ascontiguousarray gives back a C-contiguous array",
         "C-contiguous ok", t_copied.flags["C_CONTIGUOUS"],
         t_copied.flags["C_CONTIGUOUS"])
    flat2 = t_copied.reshape(t_copied.shape[0], -1)
    shares2 = np.shares_memory(flat2, t_copied)
    line("ascontiguousarray then reshape -> reshape is a view of the copy",
         "yes", shares2, shares2 is True)


# --------------------------------------------------------------------------
# Idea 3 -- "The contract in code"  (slides.md L540-555)
# --------------------------------------------------------------------------

def broadcasting_sizes():
    header("Idea 3 / The contract in code -- sizes  (L540-555)")
    a = np.zeros((1000, 1000))
    b = np.arange(1000)
    result = a + b

    line("a = np.zeros((1000, 1000)) size", "8 MB",
         f"{a.nbytes / 1e6:.0f} MB", abs(a.nbytes / 1e6 - 8) < 0.1)
    line("b = np.arange(1000) size", "8 KB",
         f"{b.nbytes / 1e3:.0f} KB", abs(b.nbytes / 1e3 - 8) < 0.1)
    line("result = a + b size (just the result)", "8 MB",
         f"{result.nbytes / 1e6:.0f} MB", abs(result.nbytes / 1e6 - 8) < 0.1)
    line("b was NOT tiled to (1000, 1000)", "no full intermediate",
         "b.shape stays " + str(b.shape), b.shape == (1000,))


# --------------------------------------------------------------------------
# Idea 3 -- "How? Stride zero"  (slides.md L557-576)
# --------------------------------------------------------------------------

def stride_zero():
    header("Idea 3 / How? Stride zero -- broadcasting is a 0 stride  (L557-576)")
    b = np.arange(1000)
    line("b.strides", "(8,)", b.strides, b.strides == (8,))

    bt = np.broadcast_to(b, (1000, 1000))
    line("np.broadcast_to(b, (1000, 1000)).strides", "(0, 8)",
         bt.strides, bt.strides == (0, 8))
    line("the broadcast view allocated no new buffer", "shares b's memory",
         np.shares_memory(bt, b), np.shares_memory(bt, b))

    # the same 0 stride is what a + b uses under the hood
    a = np.zeros((1000, 1000))
    bcast = np.broadcast_arrays(a, b)[1]
    line("the operand NumPy actually feeds the kernel for a + b",
         "stride 0 on the broadcast axis", bcast.strides,
         bcast.strides[0] == 0)


# --------------------------------------------------------------------------
# Idea 3 -- "The rules"  (slides.md L578-600)
# --------------------------------------------------------------------------

def broadcasting_rules():
    header("Idea 3 / The rules -- shape resolution  (L578-600)")

    r1 = np.broadcast_shapes((1000, 1000), (1000,))
    line("(1000,1000) + (1000,) -> ", "(1000, 1000)", r1, r1 == (1000, 1000))

    r2 = np.broadcast_shapes((1000, 1, 5), (1, 3, 5))
    line("(1000,1,5) + (1,3,5) -> ", "(1000, 3, 5)", r2, r2 == (1000, 3, 5))

    try:
        np.broadcast_shapes((1000, 3, 5), (1000, 4, 5))
        ok, msg = False, "no error raised"
    except ValueError as exc:
        ok, msg = True, "ValueError raised"
    line("(1000,3,5) + (1000,4,5) -> ", "ValueError", msg, ok)


# --------------------------------------------------------------------------
# The trap / closing -- intermediates exist  (slides.md L613-641, L685-700)
# --------------------------------------------------------------------------

def intermediates():
    header("The trap -- chained ops allocate full-size intermediates  (L613-641)")
    a = np.random.rand(1000, 1000)

    mean = a.mean(axis=1, keepdims=True)
    line("a.mean(axis=1, keepdims=True) shape", "(N, 1)",
         mean.shape, mean.shape == (1000, 1))

    diff = a - mean
    line("a - mean is a full-size intermediate", "full size",
         diff.shape, diff.shape == a.shape and diff.nbytes == a.nbytes)

    sq = diff ** 2
    line("(...) ** 2 is another full-size intermediate", "full size",
         sq.shape, sq.shape == a.shape and sq.nbytes == a.nbytes)

    final = sq.sum(axis=1)
    line(".sum(axis=1) collapses to shape (N,)", "(N,)",
         final.shape, final.shape == (1000,))


# --------------------------------------------------------------------------
# One more thing -- numexpr fuses the chain  (slides.md L702-726)
# --------------------------------------------------------------------------

def numexpr_fusion():
    header("One more thing / numexpr -- fused single pass  (L702-726)")
    if not HAVE_NUMEXPR:
        print("  [SKIP ] numexpr not installed")
        return

    # the slide states this size explicitly: (2000, 2000) float64, 32 MB.
    # It has to exceed cache or there is nothing for fusion to win.
    a = np.random.rand(2000, 2000)
    m = a.mean(axis=1, keepdims=True)

    # correctness first: both expressions must agree
    numpy_result = ((a - a.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    ne_result = ne.evaluate("sum((a - m) ** 2, axis=1)")
    agree = np.allclose(numpy_result, ne_result)
    line("numexpr result matches NumPy result", "must agree",
         f"allclose={agree}", agree)

    t_np = bench(
        "((a - a.mean(1, keepdims=True)) ** 2).sum(1)",
        globals_={"a": a},
    )
    t_ne = bench(
        "ne.evaluate('sum((a - m) ** 2, axis=1)')",
        globals_={"a": a, "m": m, "ne": ne},
    )

    print(f"          (array size used here: {a.shape}, "
          f"{a.nbytes / 1e6:.0f} MB)")
    line("NumPy chained expression", "~23 ms", fmt_time(t_np),
         within(t_np, 0.001, 0.1))
    line("numexpr fused expression", "~5 ms", fmt_time(t_ne),
         within(t_ne, 0.0005, 0.1))
    line("numexpr is faster than chained NumPy", "yes",
         f"{t_np / t_ne:.2f}x", t_ne < t_np)

    # "The catch" (L728-747): out= gets most of the win with no dependency.
    buf = np.empty_like(a)

    def inplace():
        np.subtract(a, m, out=buf)
        np.multiply(buf, buf, out=buf)
        return buf.sum(axis=1)

    line("out= / in-place matches the chained result", "must agree",
         f"allclose={np.allclose(numpy_result, inplace())}",
         np.allclose(numpy_result, inplace()))
    t_out = bench("inplace()", globals_={"inplace": inplace})
    line("out= / in-place, no new dependency", "~6 ms", fmt_time(t_out),
         within(t_out, 0.001, 0.05))
    line("out= recovers most of the numexpr win", "close to numexpr",
         f"{t_np / t_out:.2f}x vs numexpr's {t_np / t_ne:.2f}x",
         t_out < t_np)

    # "The catch": when the array fits in cache there is nothing to win.
    small = np.random.rand(1000, 1000)
    sm = small.mean(axis=1, keepdims=True)
    t_np_s = bench("((a - a.mean(1, keepdims=True)) ** 2).sum(1)",
                   globals_={"a": small})
    t_ne_s = bench("ne.evaluate('sum((a - m) ** 2, axis=1)')",
                   globals_={"a": small, "m": sm, "ne": ne})
    line("at (1000, 1000) / 8 MB the two are a wash", "roughly equal",
         f"numexpr {t_np_s / t_ne_s:.2f}x", t_np_s / t_ne_s < 2.0)


# --------------------------------------------------------------------------

def main():
    print("Benchmarking the claims in slides.md")
    print(f"numpy {np.__version__}", end="")
    if HAVE_NUMEXPR:
        print(f" | numexpr {ne.__version__}", end="")
    print(f" | python {sys.version.split()[0]}")

    sections = [
        familiar_comparison,
        trace_line_counts,
        relocation_breaks,
        surprising_timing,
        view_aliasing,
        shares_memory_diagnostic,
        verify_strides,
        contiguity_flags,
        bandwidth,
        villain,
        broadcasting_sizes,
        stride_zero,
        broadcasting_rules,
        intermediates,
        numexpr_fusion,
    ]
    for section in sections:
        try:
            section()
        except Exception as exc:  # keep going; report which claim blew up
            print(f"  [ERROR] {section.__name__}: {exc!r}")
        gc.collect()

    print()
    print("Done. 'PASS' = matches the slide within tolerance; "
          "'CHECK' = worth a look.")


if __name__ == "__main__":
    main()
