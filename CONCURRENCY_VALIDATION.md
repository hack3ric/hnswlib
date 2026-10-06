# Concurrent C++ workload validation

Status: first-milestone functional and performance acceptance checks passed in
the tested configurations. This report covers the
first milestone in [CONCURRENCY_PLAN.md](CONCURRENCY_PLAN.md): concurrent search
and insertion of new labels. Concurrent deletion, replacement, reclamation, and
Python concurrency remain deferred.

## Scope and reference

The baseline is commit `655288845a6dc8d6ba47e8b3f1fc324ad2df985a`, preserved in a
separate worktree. Existing library implementations, bindings, and tests are
unchanged. Production changes are confined to the new
`hnswlib/concurrent_hnsw.h`; CMake additionally registers the new tests and an
explicitly invoked benchmark.

Measurements use an Intel Xeon w3-2423 (six physical cores, twelve logical CPUs),
GCC 16.2.1, and Linux. C++ comparisons use identical
`-std=c++11 -O3 -DNDEBUG -march=native -pthread` flags. Single-thread runs are
pinned to CPU 0 and four-thread runs to CPUs 0–3. No other validation jobs run
alongside the performance measurements. Results describe this environment and
these workloads, rather than a guarantee for every platform or dataset.

## Correctness and compatibility

| Requirement | Evidence |
| --- | --- |
| Publication hides incomplete results | Deterministic pauses during reservation, after initialization, and before publication; queries return only published labels |
| Empty-index and root coordination | Simultaneous first insertions and a reader retaining an old root while a higher root is published |
| No lost reciprocal updates | Two writers forced to compute against the same full adjacency list; one must revalidate after the other commits |
| Safe transitions between read-only and mixed work | A paused unlocked reader delays the first writer; new queries still finish using snapshots |
| Safe sole-builder optimization | A second builder waits while the first prepares outgoing links without snapshot locks |
| Maintenance excludes both reader paths | Mutations fail inside a read-only query callback; queries and insertions fail while quiescent save is paused |
| Duplicate labels and bounded resources | Concurrent duplicate insertion, fixed element capacity, explicit levels, maximum level, and upper-arena exhaustion |
| Failure recovery | Injected failures before and after graph exposure; failed-first-insertion recovery; compaction with live nodes after failed holes; subsequent capacity reuse |
| Callback and stream failures | Scratch/admission RAII after callback exceptions; failed output streams leave the index usable |
| Supported query types | Mixed L2, inner product, filters, epsilon stopping, and multivector document stopping; returned labels and distances checked against input data |
| Stable graph quality | L2 self recall 1.0 in the mixed stress run; inner-product exact-reference recall 0.995; multivector vector-neighbor recall 1.0 |
| Post-run invariants | Counts, label mappings, live states, root level, edge IDs/levels, degree limits, and duplicate/self edges |
| Metric compatibility | Inner-product and multivector serial graphs are byte-for-byte identical to legacy graphs; ordinary query results also match exactly |
| Persistence | Legacy-to-concurrent and concurrent-to-legacy loading; larger load capacity; byte-for-byte serial graph equivalence, including insertion after soft-deleting the root |
| Quiescent behavior | Soft deletion, undelete, updates, save, failed-slot cleanup, and rejection of online resize |

The unmodified baseline passed all 16 registered C++ tests in RelWithDebInfo.
The implementation passed all 18 registered tests; subsequent changes were
checked with both new test executables. The existing `test_updates` executable
also passed in both normal and `update` modes using the unchanged generated
100,000-vector dataset; recall exceeded 0.99 in both modes.

The mixed-workload executable passed:

- GCC and Clang with explicit C++11 compilation.
- TSan, with no suppressions and deadlock detection enabled.
- ASan/UBSan with prefetch enabled and `UBSAN_OPTIONS=halt_on_error=1`.
- Exception-disabled compilation, alongside `no_exceptions_api_test` and the
  serial compatibility fixture.

All 14 existing Python tests passed. The existing recall evaluation produced
0.995 for both baseline and current builds.

The serial compatibility fixture is separate from the mixed TSan executable:
constructing its legacy reference triggers pre-existing legacy lock-order
warnings. The new multivector test uses an exact scan rather than the legacy
brute-force type, whose packed document payload causes an unaligned label load
under UBSan. Neither legacy implementation was changed or its warnings suppressed
to obtain the concurrent-index sanitizer result.

## Existing benchmark framework

The unchanged `tests/python/speedtest.py` ran three times per build/configuration,
alternating baseline/current order. It uses 400,000 16-dimensional vectors, seed
1, `M=16`, construction `ef=60`, 64 insertion threads, query `ef=15`, and `k=1`.
Each process measures three query batches of `5000 * query_threads` queries.
The table gives the median of the per-process query medians and the median build
time across processes.

| Query threads | Baseline build, s | Current build, s | Baseline query batch, s | Current query batch, s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4.8900 | 4.8827 | 0.133974 | 0.132925 |
| 4 | 4.8418 | 4.8544 | 0.132972 | 0.129770 |

Recall was 0.9998–1.0 for the baseline and 1.0 for the current build. These
measurements show no regression in the existing framework. They do not by
themselves establish the new concurrent type's performance: that type is measured
separately below.

## Concurrent-type performance

Generated logs, raw measurements, comparison CSVs, and run metadata are excluded
from Git commits and retained locally under
`tests/benchmarks/concurrency_2026_10_05/`. This report records the measured
summaries and reproduction procedure. The matrix covers 8, 16, 32, and
128 dimensions, serial and parallel construction/querying, 20,000 and 100,000
items, and a search-`ef` sweep. Every configuration receives a warm-up process for
each implementation. Recorded runs alternate order. Each process excludes its
first query pass and reports the median of five measured passes of 10,000 queries.

Build/query settings and recall targets are identical across implementations.
Serial construction is also checked for byte-for-byte graph equivalence. Recall
uses exact neighbors for 100 independent queries, outside the measured region.
Parallel graph construction can vary with scheduling; report that variation
alongside timing uncertainty rather than attributing every difference to the
concurrency protocol.

There are 126 recorded runs: seven baseline/concurrent pairs per 8-dimensional
configuration and three pairs per other configuration. The table shows medians,
with each cell ordered **baseline → concurrent**. Construction uses `M=16` and
`ef=200`; query `ef=50`, except `ef=100` for 128-dimensional vectors.

| Items / dimensions | Build/query threads | Build, s | Query throughput, queries/s | Query p99, µs | Recall |
| --- | --- | --- | --- | --- | --- |
| 20,000 / 8 | 1/1 | 1.468 → 1.417 | 59,914 → 60,071 | 21.2 → 20.9 | 1.000 → 1.000 |
| 20,000 / 8 | 1/4 | 1.520 → 1.448 | 181,170 → 179,459 | 32.9 → 32.7 | 1.000 → 1.000 |
| 20,000 / 8 | 4/4 | 0.425 → 0.437 | 210,498 → 209,447 | 26.5 → 26.6 | 1.000 → 1.000 |
| 20,000 / 16 | 1/1 | 2.454 → 2.278 | 32,801 → 34,375 | 37.7 → 39.0 | 0.995 → 0.995 |
| 20,000 / 16 | 1/4 | 2.443 → 2.349 | 113,023 → 117,295 | 53.3 → 51.0 | 0.995 → 0.995 |
| 20,000 / 16 | 4/4 | 0.655 → 0.633 | 128,932 → 134,631 | 43.8 → 39.8 | 0.995 → 0.995 |
| 20,000 / 32 | 1/1 | 2.937 → 2.811 | 26,758 → 28,262 | 46.1 → 43.9 | 0.918 → 0.918 |
| 20,000 / 32 | 1/4 | 2.998 → 2.864 | 89,723 → 97,844 | 68.1 → 58.7 | 0.918 → 0.918 |
| 20,000 / 32 | 4/4 | 0.803 → 0.781 | 103,808 → 109,717 | 56.2 → 52.5 | 0.918 → 0.920 |
| 20,000 / 128 | 1/1 | 5.085 → 4.670 | 8,378 → 8,863 | 150.5 → 134.6 | 0.723 → 0.723 |
| 20,000 / 128 | 1/4 | 5.056 → 4.727 | 32,930 → 35,622 | 169.5 → 184.1 | 0.723 → 0.723 |
| 20,000 / 128 | 4/4 | 1.325 → 1.226 | 34,855 → 35,720 | 156.1 → 159.1 | 0.717 → 0.716 |
| 100,000 / 32 | 1/1 | 22.151 → 19.995 | 15,444 → 17,939 | 108.7 → 75.6 | 0.810 → 0.810 |
| 100,000 / 32 | 4/4 | 5.531 → 5.349 | 64,536 → 76,373 | 91.5 → 71.2 | 0.803 → 0.805 |

For uncertainty, compute the paired log time ratio for each process pair, then
its mean and Student-t 95% interval (`df = pairs - 1`); transform back to percentage
change. Recall uses paired absolute differences, reported in percentage points.
No outliers were removed. The intervals describe repeated builds/runs on the fixed
datasets and queries, not variation across all possible datasets.

No configuration showed a time increase or recall decrease beyond those
intervals. The most closely investigated case was short vectors; its timing
changes and 95% intervals were:

| Build/query threads, 8 dimensions | Build time change | Query time change | Query p99 change |
| --- | --- | --- | --- |
| 1/1 | −4.03% [−5.91, −2.13] | −0.20% [−2.71, +2.38] | −0.99% [−3.57, +1.66] |
| 1/4 | −2.86% [−10.34, +5.26] | +1.26% [−3.27, +6.02] | +1.34% [−8.07, +11.70] |
| 4/4 | +2.72% [−0.99, +6.57] | −0.54% [−2.80, +1.77] | +0.05% [−5.57, +6.01] |

These small differences meet the user's acceptance of performance within
statistical error. Several larger-vector configurations improve throughput and
build time. Tail-latency intervals can be wide, especially with three pairs;
additional runs would be needed to resolve effects smaller than those intervals.
For example, 128-dimensional queries after serial construction with four query
threads have a p99 change estimate of +3.35%, with an interval of [−24.74%,
+41.92%]. The median table and log-ratio estimates use different summaries of the
same raw runs, so their point estimates need not coincide.

The serial 20,000-item, 32-dimensional search-effort sweep preserves recall at
every measured setting:

| Search ef | Recall, baseline / concurrent | Queries/s, baseline / concurrent |
| ---: | --- | --- |
| 10 | 0.561 / 0.561 | 99,782 / 109,684 |
| 20 | 0.736 / 0.736 | 57,402 / 62,434 |
| 50 | 0.918 / 0.918 | 26,758 / 28,262 |
| 100 | 0.986 / 0.986 | 13,916 / 14,738 |

## Mixed workload and lock acquisition

The following are medians from three runs on 20,000 32-dimensional vectors,
`M=16`, construction `ef=200`, search `ef=50`. Half the index is built before
readers and writers overlap. Final recall is measured after all writers finish.

| Readers / writers | Queries/s | Inserts/s | Query p99, µs | Insert p99, µs | Final recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 1 | 27,023 | 6,002 | 50.8 | 288.5 | 0.918 |
| 3 / 1 | 75,479 | 6,169 | 58.7 | 284.0 | 0.918 |
| 6 / 2 | 116,517 | 9,893 | 70.8 | 396.8 | 0.918 |

Eight-thread runs use CPUs 0–7 and therefore include two SMT siblings on this
six-core host. The local raw mixed runs include variation in throughput and
latency rather than only the medians.

Separate instrumented runs measured adjacency-lock acquisition:

| Readers / writers | Acquisitions | Mean acquisition time, ns | Acquisitions over 1 µs |
| --- | ---: | ---: | ---: |
| 1 / 1 | 2,719,031 | 63.7 | 212 |
| 3 / 1 | 7,109,916 | 106.7 | 6,081 |
| 6 / 2 | 11,061,088 | 112.4 | 5,126 |

Acquisition times include instrumentation overhead and scheduling delays; these
runs are not used for throughput comparison. The increased acquisition time with
more workers shows why traversal and pruning stay outside node locks.

Memory and initialization overhead are described in
[CONCURRENCY.md](CONCURRENCY.md#storage-and-failure-handling). In particular,
aligned records and revision counters consume additional capacity-proportional
memory, and construction/loading temporarily holds both packed and aligned
level-zero records. Search and insertion scratch may allocate.

## Reproduction

```sh
cmake -S . -B build/concurrent-checked -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/concurrent-checked
ctest --test-dir build/concurrent-checked --output-on-failure

g++ -std=c++11 -O3 -DNDEBUG -march=native -pthread -I BASELINE \
  -DHNSWLIB_BENCH_LEGACY_ONLY tests/cpp/concurrent_benchmark.cpp -o baseline-bench
g++ -std=c++11 -O3 -DNDEBUG -march=native -pthread -I . \
  tests/cpp/concurrent_benchmark.cpp -o concurrent-bench

taskset -c 0 ./baseline-bench legacy 20000 32 1 50 1 1
taskset -c 0 ./concurrent-bench concurrent 20000 32 1 50 1 1
taskset -c 0-3 ./concurrent-bench mixed 20000 32 4 50 1
```

Replace `BASELINE` with the unmodified worktree path. Build a separate benchmark
with `-DHNSWLIB_BENCH_LOCK_WAIT` for lock acquisition measurements. Repeating the
commands with identical parameters is necessary to assess timing uncertainty;
single runs are not an acceptance test.
