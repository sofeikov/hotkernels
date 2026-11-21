<section class="hero">
<p class="eyebrow">NumPy on Apple Silicon and x86-64 Linux</p>
<div class="hero-grid">
<div>
<h1>The slow parts of NumPy, <em>rewritten underneath it</em>.</h1>
<p class="lede">hotkernels replaces the loops inside 23 NumPy operations with hand-written ARM64 kernels. Your code does not change. Every kernel is checked against NumPy's own output, and a call no kernel takes costs about a nanosecond more than before.</p>
<div class="cta">
<a class="btn primary" href="[[ root ]]start/">Get started</a>
<a class="btn" href="[[ root ]]report/">Read how it works</a>
</div>
</div>
<pre class="code"><code><span class="kn">import</span> numpy <span class="k">as</span> np
<span class="kn">import</span> hotkernels
<span></span>
hotkernels.install()
<span></span>
x = np.random.standard_normal(<span class="m">1_000_000</span>)
x[x > <span class="m">0</span>]           <span class="c"># 16x faster</span>
np.clip(x, <span class="m">-1</span>, <span class="m">1</span>)   <span class="c"># 5x faster</span>
np.argmax(x)       <span class="c"># 7x faster</span></code></pre>
</div>
</section>

<div class="stats">
<div><b>40</b><span>kernels for 23 NumPy operations, all of them bound to NumPy</span></div>
<div><b>5x+</b><span>faster than NumPy for 17 of the 22 in the results table, at a million elements</span></div>
<div><b>21 / 23</b><span>return NumPy's result bit for bit; the other two are opt-in</span></div>
<div><b>~1 ns</b><span>added to a call that no kernel takes</span></div>
</div>

## How much faster

Each bar is one NumPy call on an array of 1,048,576 elements, timed as a native function against NumPy's own loop on an Apple M1 Pro. Switch to `clang -O3` to see the same kernels against the plain C loop a compiler makes of the operation: that comparison is the one that says whether writing assembly was worth it.

<figure class="widget">
<div data-widget="speedups"></div>
<figcaption>Speedup of each kernel at 1,048,576 elements. Read from the results table in the repository's README when this site was built. The <a href="[[ root ]]report/results/">results chapter</a> has every size, the confidence intervals, and the timings from Python.</figcaption>
</figure>

## What stands behind the numbers

- **Checked against NumPy, not against a description of it.** Every kernel is compared with NumPy's own output on inputs that include NaN, infinities, signed zeros, ties and denormals. Thirty-eight of the 40 match bit for bit. The two float `cumsum` kernels regroup the additions, which is where their speed comes from; they stay inside a declared error bound and are installed only on request.
- **Memory-safe by test.** Each kernel is run with its buffers pressed against pages the process may not touch. Reading or writing one byte outside a buffer crashes the check.
- **Assembly only where it wins.** A kernel keeps its assembly only if it is measurably faster than the compiled C. Where it is not, the C ships.
- **Nothing breaks when a kernel does not apply.** A strided array, another element type, an argument a kernel does not implement: the call goes to NumPy exactly as written. With hotkernels installed, NumPy's own test suite gives the same outcome for each of its 47,234 tests.

There are limits, and the site does not hide them: figures from one machine, an Apple M1 Pro, with the x86-64 kernels measured separately and written up in [a chapter of their own]([[ root ]]report/x86-64/), and the workloads tracked [over time]([[ root ]]report/history/) on an Apple M4 Max; one or two element types per operation, and a few results that depend on the arguments. They are set out in [what the numbers do not say]([[ root ]]report/results/).

## How it was built, and why you might care

The kernels were written by coding agents, not by hand. An agent edits one assembly file, runs an automated judge, and keeps the edit only if the judge finds it correct and significantly faster. Nobody reviews the assembly line by line, so the judge is the only thing between a subtly wrong kernel and your data, and most of the engineering went into making it trustworthy.

The report walks through all of it from the ground up. It assumes you know NumPy well and have never read a line of assembly.

<div class="cards">
<a class="card" href="[[ root ]]report/numpy/"><h3>Where the time goes</h3><p>A NumPy call is a fixed cost plus a loop. Which loops are slow, and which are already at the limit.</p><span class="more">Chapter 1 →</span></a>
<a class="card" href="[[ root ]]report/machine/"><h3>The machine, in fifteen minutes</h3><p>Vector registers, dependency chains, branch prediction and caches: the four ideas every optimisation here rests on.</p><span class="more">Chapter 2 →</span></a>
<a class="card" href="[[ root ]]report/judge/"><h3>The judge</h3><p>Guard pages, thousands of generated cases per kernel, and a bootstrap test that decides what counts as faster.</p><span class="more">Chapter 3 →</span></a>
<a class="card" href="[[ root ]]report/loop/"><h3>Agents in a loop</h3><p>[[ data.attempts.total ]] logged attempts, [[ data.attempts.reverted ]] of them thrown away. Browse every one.</p><span class="more">Chapter 4 →</span></a>
<a class="card" href="[[ root ]]report/selection/"><h3>The kernels</h3><p>How <code>x[mask]</code> runs without a branch, why float <code>cumsum</code> cannot be both exact and fast, and a histogram that replaces arithmetic with a proof.</p><span class="more">Chapters 5 and 6 →</span></a>
<a class="card" href="[[ root ]]report/delivery/"><h3>Into NumPy</h3><p>Five kinds of hook that route unmodified code to the kernels, including <code>x[mask]</code>, which is not a function at all.</p><span class="more">Chapter 7 →</span></a>
</div>
