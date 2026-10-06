# Concurrent C++ search and insertion

`ConcurrentHierarchicalNSW` is the opt-in, fixed-capacity C++ implementation of
the first milestone in [CONCURRENCY_PLAN.md](CONCURRENCY_PLAN.md). Include
`hnswlib/concurrent_hnsw.h`. The existing `HierarchicalNSW` and Python index retain
their existing implementations and concurrency contracts.

```cpp
#include "hnswlib/concurrent_hnsw.h"

hnswlib::L2Space space(128);
hnswlib::ConcurrentIndexOptions options;
options.upper_layer_capacity = 200000; // total upper-layer lists, not bytes
options.max_level = 32;
hnswlib::ConcurrentHierarchicalNSW<float> index(
    &space, 1000000, 16, 200, 100, options);
index.setEf(100);

// Calls from independent threads may overlap:
// index.addPoint(vector, unique_label);
// auto result = index.searchKnn(query, 10);
```

## Contract

- Searches and insertion of distinct new labels may overlap. Searches are weakly
  consistent and return only initialized, published vectors. ANN discovery of a
  particular concurrent insertion is not guaranteed.
- Insertion of an existing label returns an error instead of updating it. The
  integer third argument to `addPoint` is an explicit level, including level zero.
  The boolean third argument retains its meaning as replacement, which is not
  supported by this type. Internal IDs returned by explicit-level insertion may
  change during failed-slot cleanup; use external labels for later operations.
- `searchKnn`, `searchKnnCloserFirst`, filters, stop conditions, and copying data
  by label are supported. Labels returned by stop-condition searches are external
  labels. Distances and labels remain associated with immutable vectors.
- `markDelete`, `unmarkDelete`, `updatePointNoExceptions(data, label)`, save,
  failed-insertion cleanup, and integrity inspection require quiescence. They
  acquire exclusive admission or return an error promptly. They do not wait for
  callers to stop submitting work. Quiescent soft deletion retains the existing
  tombstone semantics; it does not reclaim capacity.
- Capacity cannot be resized. Load a saved index with a larger capacity to create
  a larger instance. Loading and object destruction require external lifetime
  coordination; do not destroy an object while calls are using it.
- `setEf` is atomic. Each query samples its own value.
- Callbacks and custom distance functions must tolerate concurrent invocation.
  Keep the space and its distance parameters alive and immutable. Use a separate
  mutable stop condition for each search. Same-thread reentrant insertion into
  the index from a callback returns an error. Quiescent mutations from a callback also fail
  admission, rather than blocking on the callback's own operation. Do not retain
  callback data pointers after the operation returns.
- `NoExceptions` APIs return status for ordinary operation errors, including
  insertion failure. As in the existing search API, a throwing search callback
  can propagate an exception; RAII still releases its scratch and admission.
  Exception-disabled compilation is supported. It cannot recover from a failing
  allocation that terminates the process under `-fno-exceptions`.

## Storage and failure handling

Vectors, base-layer links, locks, state records, and upper-layer backing storage
have fixed capacity. The default upper-layer list budget is
`capacity / (M - 1) + capacity / 16 + 64`. The explicit budget and maximum level
are checked before consuming a node slot. This budget is not a guarantee that
every possible random or explicit level sequence fits; exhaustion returns an
error rather than allocating more backing storage or silently lowering a level.
Temporary search heaps, scratch buffers, and result containers may allocate.
Label-map buckets are reserved at initialization; individual label-map entries
still allocate. Node lifetime state occupies the reserved fourth byte of the
base-layer header, next to the deletion flag, rather than a separate array on
the distance-computation path. The implementation requires byte-sized atomic
flags. Quiescent serialization clears this runtime state in its output buffer,
preserving the legacy file bytes without changing the live state objects.

The concurrent type aligns vectors to 64 bytes for payloads of at least 64 bytes,
and to 16 bytes for smaller payloads. A reserved ID after maximum level-zero
degree keeps read-only search prefetches inside allocated storage. Save translates
these records to the original packed layout in bounded batches; load reconstructs
the aligned layout. The legacy type's storage and algorithms are unchanged.

For `M=16` and float vectors, record sizes are:

| Dimensions | Legacy bytes/record | Concurrent bytes/record |
| --- | ---: | ---: |
| 8 | 172 | 176 |
| 16 | 204 | 256 |
| 32 | 268 | 320 |
| 128 | 652 | 704 |

There are also eight revision bytes per capacity slot, the reserved upper-layer
arena, and two visited bytes per capacity slot for each allocated query scratch
list. The legacy storage's initial scratch list is retained for quiescent updates.
The default upper arena uses approximately `8.78 * capacity + 4352` bytes at
`M=16`, versus roughly `4.53 * capacity` bytes of occupied upper lists in an
ordinary random build. Actual memory also includes the same node/label mutexes,
level and tower-pointer arrays, and label-map allocations as the legacy type.
Construction/loading temporarily holds both packed and aligned level-zero
records while converting them; subsequent operation does not relocate backing
storage. Save needs about 64 KiB of conversion scratch, or one record if larger.

Each insertion reserves a label and slot, initializes the payload, prepares its
entire outgoing tower, and only then installs reciprocal links. Until all links
are installed, the node can route a traversal but cannot appear in results or be
selected by another builder as a normal neighbor. Publication is a release store
of the live state. Root publication follows it; readers obtain the root with an
acquire load and derive its level from immutable node metadata.

If insertion throws after reserving a slot, that slot is quarantined and its
label reservation is released. Searches exclude it from results but can safely
traverse any exposed links because its initialized payload remains intact.
`getFailedInsertionCount()` reports these slots. Once operations have stopped,
`cleanupFailedInsertions()` removes edges to failed slots, compacts live slots and
towers, remaps internal IDs, and restores capacity. Saving also performs this
cleanup. `getCurrentElementCount()` counts published nodes, including soft-deleted
nodes, and excludes reservations and failures.

EBR is unnecessary in this milestone: no storage visible to an operation is
overwritten, relocated, or reused while shared operations are active. Quiescent
cleanup is the only path that recycles failed insertion storage. Concurrent
reclaimable deletion and immutable-vector replacement remain deferred.

## Synchronization

Admission separates shared operations from quiescent maintenance. It does not
serialize insertions with searches. Label-operation stripes serialize conflicting
insertions; reservation and label-map locks cover only bookkeeping and private
initialization. Graph traversals never retain a node lock while taking another.

Read-only searches register once in the reader phase. Writers, copying data by
label, and searches using snapshots register in the normal admission gate.
Maintenance first closes that gate, then atomically closes an empty reader phase.
If a fast reader registers first, the phase check fails and maintenance releases
the gate without modifying the graph. If maintenance closes the phase first, the
query falls back to the closed gate and returns an error. Closing the phase uses
acquire/release ordering so observing its closure also observes the earlier gate
closure. Successful maintenance reopens both after its writes are complete.

During mixed operations, each adjacency read copies a list under its node mutex.
Writers snapshot a list and revision, compute pruning outside the lock, and commit
only if that revision still matches. Otherwise they retry with current contents.
Concurrent search and insertion invoke distance functions and user callbacks
outside adjacency locks. Quiescent `updatePointNoExceptions` delegates to the
legacy update algorithm and retains its callback behavior; exclusive admission
prevents that operation from overlapping any search or insertion.

When no writers are active, searches register in a read-only phase and read
adjacency directly. The first writer closes this phase atomically and waits for
its already registered readers to leave before mutating the graph. New searches
immediately use locked snapshots, including while that writer is waiting. All
writers can then proceed concurrently. The last writer reopens the read-only
phase. Acquire/release ordering connects completed writes to subsequent unlocked
reads. No writer waits while holding a label or node lock.

A long-running reader can delay the first writer of a new mixed phase. It does
not prevent new searches from completing. This transition is covered by a
deterministic test; callers should account for slow callbacks in write latency.
Callbacks must not wait for another thread's insertion into the same index:
that insertion may be waiting for the callback's current read or preparation
phase to finish.

A sole builder can also read existing adjacency directly while preparing its
private outgoing tower. Registration of another builder closes that opportunity:
the joining builder waits for the private preparation to finish before accessing
the graph. Reciprocal updates run after releasing this preparation guard and use
the usual locking/revision protocol. Once builders overlap, preparation uses
locked snapshots. This preserves parallel construction while avoiding unnecessary
snapshot copies for a single insertion thread; its transition has a deterministic
test as well.

## Persistence

Quiescent save uses the existing binary format after failed-slot compaction.
Existing indexes can be loaded into the concurrent type, and concurrent indexes
can be loaded by `HierarchicalNSW`. The load constructor is:

```cpp
hnswlib::ConcurrentHierarchicalNSW<float> restored(
    &space, std::string("index.bin"), 1000000, options);
```

Loaded towers must fit the configured arena and maximum level. Runtime state,
mutexes, and revisions are reconstructed rather than serialized.

## Validation commands

```sh
cmake -S . -B build/concurrent -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/concurrent
ctest --test-dir build/concurrent --output-on-failure
build/concurrent/concurrent_benchmark legacy 20000 32 1 50
build/concurrent/concurrent_benchmark concurrent 20000 32 1 50
```

`concurrent_search_insert_test` includes deterministic publication, first-node,
lost-update, admission, phase-transition, rollback, and callback tests plus an
overlapping workload that checks every returned distance against the input
vector. Mixed queries cover L2, inner product, filters, epsilon stopping, and
multivector document stopping. It also checks fixed capacities, explicit levels, legacy save/load,
quiescent updates, and post-run graph invariants. Run it with TSan, ASan/UBSan,
and `HNSWLIB_ENABLE_EXCEPTIONS=OFF` as well.

`concurrent_compatibility_test` additionally checks byte-for-byte serial graph
equivalence, including insertion after soft-deleting the root. It is a separate
executable because constructing its legacy reference triggers pre-existing TSan
lock-order warnings in `HierarchicalNSW`. Run the mixed-workload TSan test without
suppressions; do not disable deadlock detection to hide findings in the new code.

`concurrent_benchmark` reports build time, query throughput, query p99 latency,
and recall against exact neighbors. For a preserved baseline, compile the same
source with `HNSWLIB_BENCH_LEGACY_ONLY` and the unmodified checkout's include path.
Use the same compiler flags and repeat comparisons with identical datasets and
parameters. The existing Python speed and recall framework remains part of
regression validation; this C++ benchmark additionally measures the new API.
Its `mixed` mode overlaps writers and readers, reports both query and insertion
p99 latency, and measures final recall against a stable exact reference.
Compile a separate executable with `HNSWLIB_BENCH_LOCK_WAIT` to measure adjacency
lock acquisition counts, mean acquisition time, and acquisitions exceeding one
microsecond. Those times include clock overhead and scheduling delays; use the
uninstrumented executable for throughput comparisons.

Performance acceptance and the broader completion audit are tracked in
[CONCURRENCY_VALIDATION.md](CONCURRENCY_VALIDATION.md), including baseline
comparisons, timing uncertainty, matched-recall results, and mixed workloads.
The deferred roadmap remains in [CONCURRENCY_PLAN.md](CONCURRENCY_PLAN.md).
