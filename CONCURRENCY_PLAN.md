# Concurrent workloads in hnswlib

Status: first C++ search/insert milestone implemented and validated. See
[CONCURRENCY.md](CONCURRENCY.md) for the concrete API and synchronization protocol,
and [CONCURRENCY_VALIDATION.md](CONCURRENCY_VALIDATION.md) for measured evidence,
statistical uncertainty, and platform/workload limits. Later phases remain deferred.

## Objective and agreed scope

The long-term objective is concurrent mixed search, insertion, and deletion on a
fixed-capacity HNSW index, with epoch-based reclamation (EBR) allowing deleted
storage to be reused safely.

The first milestone is narrower:

- Support concurrent **C++ search and insertion of new labels**.
- Support weakly consistent ANN searches during insertion.
- Fix adjacency access, node publication, entry-point coordination, and shared
  metadata races.
- Validate genuinely overlapping workloads.
- Defer concurrent deletion, deleted-slot replacement, reverse-edge bookkeeping,
  and reclamation to a later phase.
- Defer Python bindings.

Fixed capacity at creation or loading is acceptable. Index growth and backing
storage relocation are outside the first milestone.

EBR is not needed in this milestone as long as storage visible to active
operations is never freed, overwritten, relocated, or reused. Adjacency locks
and a publication protocol are still required. Failed insertions may quarantine
exposed storage until quiescent cleanup. Revisit EBR when adding concurrent
reclamation, immutable-vector replacement, or copy-on-write adjacency that
retires old buffers.

Adding concurrency must not cause correctness or performance regressions in
existing supported workloads. In particular, search-only throughput/latency,
insert-only throughput/build time, and recall measured under the current testing
and benchmarking framework must not degrade merely because mixed-workload
concurrency has been added. This is an acceptance requirement, not an optional
optimization. Improvements in mixed-workload performance do not offset regressions
in existing workloads.

Keep changes confined to concurrency-related implementation, tests, benchmarks,
and documentation. Do not rewrite or tune the other index implementations.
Performance differences within measured statistical uncertainty are acceptable;
use repeated comparable measurements to distinguish those from a reproducible
regression.

Existing-label updates overwrite vectors today. The proposed initial restriction
is that updates, soft deletion, undelete, and deleted-slot replacement require
quiescence. Their existing sequential behavior should remain available. The
precise mechanism for rejecting unsupported overlap is an implementation design
decision, not permission to leave a data race in an advertised concurrent API.

## Search and operation semantics

Searches are weakly consistent, not snapshots of the entire graph:

- A search overlapping insertion may or may not discover the inserted item.
- Every returned label and distance must refer to a valid, initialized vector.
- No reader may access partially initialized nodes or race with adjacency writes.
- Successful insertion must complete the defined publication protocol before
  returning. ANN search still does not guarantee discovery of any particular item.
- Conflicting mutations of the same label must be serialized or rejected with a
  defined status. Unrelated insertions should execute concurrently.
- Failure must not expose a searchable half-inserted item or leak capacity
  indefinitely.

When concurrent deletion is implemented, a search beginning after deletion
completes must exclude that deleted incarnation. A later insertion using the same
external label creates a new incarnation. Logical deletion and physical storage
reclamation will have separate completion conditions.

Load, save, resize, clear, integrity inspection, and destruction initially require
quiescence. Concurrent mode rejects resize. Configuration setters must either use
appropriate synchronization or explicitly require quiescence.

## Findings in the existing implementation

The main implementation is in [hnswlib/hnswalg.h](hnswlib/hnswalg.h).

| Area | Current behavior | Required change |
| --- | --- | --- |
| Search | `searchBaseLayerST` and upper-layer traversal read mutable adjacency arrays without locking | Read consistent adjacency snapshots |
| Insertion | `addPointWithLevel` reserves an ID and installs a label mapping before node initialization finishes | Separate reservation, construction, and publication |
| Reciprocal links | Writers mutate neighbor lists under locks that ordinary search does not acquire | Use a common adjacency access protocol |
| Entry point | Entry-point ID and maximum level lack a complete mixed-workload reader protocol | Obtain and publish a consistent pair |
| Metadata | Random generators, deletion flags, configuration, and other shared fields need an audit | Define synchronization for every shared field |
| Updates | `updatePoint` overwrites a vector in place | Initially require quiescence; later publish immutable versions |
| Replacement | Deleted slots can immediately receive new labels and vectors | Defer concurrent replacement until safe retirement and reuse exist |
| Scratch | `VisitedListPool` allocates additional lists on demand | Use RAII leases; define resource limits if bounded scratch is required |

The existing multithreaded replacement test performs insertion, deletion, and
replacement in separate phases. It does not establish mixed-workload safety.

## Reference: DiskANN3

The reference reviewed was Microsoft DiskANN at commit
`bceaf45dc6aa694485553816b448d7b8f55df393`:

- [EBR implementation](https://github.com/microsoft/DiskANN/blob/bceaf45dc6aa694485553816b448d7b8f55df393/diskann-inmem/src/epoch.rs)
- [Adjacency synchronization](https://github.com/microsoft/DiskANN/blob/bceaf45dc6aa694485553816b448d7b8f55df393/diskann-inmem/src/neighbors.rs)
- [In-memory provider and documented limitations](https://github.com/microsoft/DiskANN/blob/bceaf45dc6aa694485553816b448d7b8f55df393/diskann-inmem/src/provider.rs)
- [Concurrent in-memory index RFC](https://github.com/microsoft/DiskANN/blob/bceaf45dc6aa694485553816b448d7b8f55df393/rfcs/01206-inmem2.md)

The useful separation is EBR for vector-slot lifetime and locks for adjacency
access. EBR alone does not make concurrent mutation of ordinary vector or
adjacency memory safe. The provider also documents incomplete insert/delete
protection and failed-insert rollback; hnswlib needs explicit protocols for these
cases rather than copying those limitations.

## First milestone design

### 1. Introduce a storage access boundary

Centralize the following operations inside the implementation:

- Resolve a node and verify that its state permits access.
- Obtain an initialized vector and its label.
- Copy an adjacency list at a given level.
- Commit a validated adjacency update.
- Reserve, publish, or abort an insertion.
- Obtain a consistent entry-point and maximum-level snapshot.

Route construction, ordinary search, filtered search, stop-condition search, and
neighbor-selection heuristics through these accessors. Audit all direct reads of
packed storage, including prefetch paths.

Implementation decision: expose the opt-in `ConcurrentHierarchicalNSW` C++ type,
which privately owns existing HNSW storage and reuses its pruning heuristic and
file format. Composition prevents callers from bypassing the concurrency protocol
through legacy public fields or raw-pointer helpers. Retain `HierarchicalNSW`
unchanged for compatibility and performance comparison. Keep the library
header-only and C++11-compatible.

### 2. Use fixed backing storage and explicit reservations

Preallocate node metadata, vector storage, base-layer adjacency, node locks, and
reservation bookkeeping at creation or loading.

Fixed element capacity does not itself bound upper-layer memory: HNSW levels are
random and explicit-level insertion is supported. Use a preallocated upper-layer
arena with a defined budget and supported level limit. Reserve a complete tower
before making an insertion visible. An exhausted arena must return an error
without corrupting the index or silently changing the requested level.

Keep reserved slots distinct from initialized and published nodes. Search must
not infer readability from the current element counter. Define accounting for
reserved, published, and aborted slots, and preserve documented count semantics.

Use synchronized random-level generation or a suitable per-operation generator;
do not share an unsynchronized random engine.

Fixed capacity means no index growth or backing-storage relocation. It does not
automatically mean zero heap allocations in search heaps, scratch buffers, and
returned results. A completely allocation-free operation path would require
additional bounds and is not an agreed requirement.

### 3. Protect adjacency through short critical sections

For the initial implementation, use per-node `std::mutex` protection:

1. Lock the node.
2. Validate its state and requested level.
3. Copy the adjacency list into operation-local scratch.
4. Unlock.
5. Perform distance calculations and traversal outside the lock.

Apply this at every HNSW layer and in construction traversal as well as search.

Writers compute proposed neighbors outside locks, then validate an adjacency
revision before committing. A changed revision requires recomputation or merging
against current contents; a stale proposal must not erase another insertion's
links.

Write down a single lock-order protocol for label operations, entry-point
metadata, and adjacency changes. Avoid retaining one node lock while acquiring
another. Where a multi-node commit is necessary, acquire a sorted, deduplicated
set of locks. Concurrent searches and insertions must not invoke user filters or
distance callbacks while holding graph locks. Quiescent legacy updates remain
isolated by exclusive admission and retain their existing callback behavior.

Copy-on-write adjacency is a possible later optimization if measured contention
warrants its additional memory and reclamation complexity.

Performance refinement: automatically admit unlocked adjacency reads during
read-only phases. The first writer closes that phase and drains previously
registered readers before mutation. New searches use locked snapshots and can
overlap writers; the last writer reopens the fast path. Test the registration,
drain, and reopening races explicitly. A slow reader may delay the first writer,
so include that transition in latency validation.

### 4. Define insertion publication and rollback

An insertion should:

1. Serialize operations on its external label and determine whether it is new.
2. Reserve storage and bookkeeping needed to complete or roll back construction.
3. Initialize vector, label, level metadata, and empty adjacency lists privately.
4. Enter a controlled construction state if graph traversal needs to access the
   initialized node before insertion completes.
5. Build outgoing and reciprocal links using the common adjacency protocol.
6. Commit result visibility and label lookup consistently.
7. Release construction ownership and return success.

Distinguish private construction from initialized, traversable construction if
the algorithm needs both. Unfinished nodes must not appear in ordinary search
results. No reference may expose a partially initialized vector or tower.

The first node and entry-point promotion require special coordination. Publish a
consistent entry-point/level pair in a short critical section. Do not hold an
index-wide lock throughout every insertion. Explicitly handle simultaneous first
insertions and searches of an empty index.

Reserve all fallible resources before graph exposure wherever possible. For
failure after exposure, specify how reciprocal links are detached and how any
readers that already obtained references finish safely. Without EBR in this
milestone, exposed aborted storage must not be immediately overwritten: either
prove the post-exposure path cannot fail or quarantine it until a quiescent
cleanup. Quarantine must be accounted for and recoverable. This decision is a
correctness prerequisite for the publication implementation.

### 5. Audit all supported query paths and metadata

- Cover `searchKnn`, closer-first results, filters, and stop-condition searches.
- Ensure label conversion cannot access an unfinished node.
- Audit deletion-aware and bare search paths against the declared operation
  restrictions; do not infer broad safety from a counter sampled once.
- Validate handles/IDs, list bounds, and level availability before access and
  before deriving prefetch addresses.
- Use RAII to release visited-list and scratch leases on all exits, including
  callback exceptions and status-return errors.
- Define synchronization for `ef`, counters, metrics, and configuration fields.
- Document thread-safety requirements for custom distance and filter callbacks.
- Prevent unsupported concurrent operations from bypassing the safe path through
  an overload, explicit-level insertion, or a low-level helper.

### 6. Preserve quiescent operations and compatibility

The first milestone should avoid unnecessary changes to the on-disk format.
If the new in-memory representation differs, translate to/from the existing
logical format during quiescent save/load where feasible. Do not serialize locks,
construction states, or process-local pointers.

Load must allocate the full configured capacity and reconstruct runtime metadata
before admitting operations. A larger load-time capacity remains permissible;
online resize does not.

Preserve throwing and `NoExceptions` APIs and exception-disabled builds. Specify
how duplicate-label insertion behaves in concurrent mode: the initial proposal
is a defined rejection requiring quiescent update, rather than silently taking
the current unsafe in-place update path.

Python remains outside this milestone. Existing Python sequential functionality
should not be inadvertently broken by core changes, but concurrent Python usage
will not be advertised until its bindings have been audited.

## First milestone validation

### Deterministic concurrency tests

Use scheduling hooks or barriers to force the following interleavings:

- Search while an insertion is reserved but not initialized.
- Search while an initialized insertion is linked but not result-visible.
- Search while another thread changes the adjacency being expanded.
- Two writers proposing updates to the same adjacency list.
- Simultaneous first insertions and search of an initially empty index.
- Entry-point promotion while a reader obtains its root snapshot.
- Two insertions using the same external label.
- Capacity or upper-layer reservation failure.
- Failure after partial graph exposure, according to the selected rollback
  protocol.
- Callback failure and early status-return paths releasing all resources.
- Rejection or safe exclusion of unsupported mixed mutations.

### Stress tests and invariants

Run searches and new-label insertions at the same time, across thread counts,
small and near-full capacities, overlapping neighborhoods, and different level
distributions. Include explicit-level insertions and filtered/stop-condition
queries.

After quiescence, check:

- All searchable nodes are fully initialized.
- Every edge points to a valid node with the required level.
- Degree limits hold and duplicate/self edges obey existing invariants.
- Labels are unique and map to the intended nodes.
- Counts and reservation bookkeeping agree.
- Failed insertions leave no searchable partial result or permanently lost
  reservation.

Run TSan on overlapping workloads, plus ASan/UBSan and the existing C++ regression
suite. Sanitizer success supplements the synchronization argument; it does not
replace it.

### Performance and recall

Capture a baseline from the unmodified implementation using the current testing
and benchmarking framework. Compare the implementation against that baseline on
both search-only and insert-only workloads, including existing single-threaded
and multithreaded configurations, and run the existing recall evaluations. Keep
hardware, compiler/build flags, datasets, index/query parameters, thread counts,
and seeds comparable. Repeat measurements sufficiently to distinguish normal
run-to-run variation from a reproducible regression.

Preserve or improve existing throughput, latency, build time, and recall. Do not
hide a regression by changing benchmark settings, relaxing correctness checks,
lowering recall targets, or comparing speed at different recall levels. Evaluate
search performance at matched recall as well as with the existing fixed settings.
If synchronization causes a reproducible regression, revise the design or retain
a safe specialized path for the existing workload before declaring completion.
Do not remove synchronization required for mixed-workload correctness. Report
legacy and concurrent configurations separately if both are offered; an unchanged
legacy path alone does not establish the concurrent path's performance.

Also compare the new implementation across mixed read/insert ratios. Measure:

- Query and insertion throughput.
- Tail latency and lock contention.
- Recall at stable graph checkpoints.
- Memory overhead and resource exhaustion behavior.
- Scaling with thread count and contention on popular nodes.

Do not compare recall against a changing ground truth without defining the
measurement interval. Establish measurement variability before evaluating results;
noise tolerance is not a budget for an actual slowdown or recall loss. Any proposed
exception to the no-regression requirement must be explicitly discussed with the
user rather than accepted as an inherent cost of concurrency.

## Deferred phases

These sections preserve the long-term design direction. They are not part of the
first implementation milestone.

### Concurrent existing-label updates

Publish immutable vector versions instead of overwriting payloads in place.
Use EBR to retain previous versions while readers finish, with a bounded version
pool and explicit backpressure. Serialize same-label mutations and repair graph
connections through the synchronized accessors.

Define the update commit point and failure behavior so an error after publication
does not falsely imply that the old value remains unchanged.

### EBR and generation-aware slot references

Introduce an independently tested epoch registry with RAII guards. Every operation
that retains a node or vector reference must pin before resolving it, including
search, insertion traversal, update, data access, and maintenance.

Requirements include:

- A correct registration-versus-epoch-advancement handshake.
- Conservative atomic ordering and a written safety argument.
- Safe nested guards and release on every exit path.
- Retirement ordering that protects readers arriving during a long operation.
- Serialized reclamation passes and explicit epoch overflow handling.
- Bounded participant and retirement resources.

Use slot-and-generation handles for edges, entry points, label mappings, queued
work, and candidates. Validate generation/state atomically before reading a
payload. EBR protects acquired references; generations prevent stale references
from resolving to a different incarnation after reuse. Do not silently wrap a
generation back into an old identity.

Reclamation should run opportunistically and through an explicit maintenance
API. Never wait for a grace period while holding a guard or lock that prevents
that grace period. A stalled reader may delay reclamation indefinitely; return a
retryable resource error rather than overwrite protected storage.

### Reclaimable deletion and graph repair

The deletion API and its relationship to reversible `markDelete`/`unmarkDelete`
remain undecided. The user deferred deletion rather than approving either a new
API or changed existing semantics.

The intended lifecycle is approximately:

```text
FREE -> BUILDING -> LIVE -> DELETING -> RETIRED -> FREE
                    |
                    + <-> SOFT_DELETED  (if reversible deletion is retained)
```

Logical deletion first hides the node from results and prevents new incoming
links. Keep the payload available for routing and repair until it can be detached.
Repair graph connectivity, replace the entry point when necessary, then retire
storage and wait for the EBR grace period before reuse.

HNSW edges are directed. Outgoing neighbors do not enumerate all incoming edges.
The proposed production approach is exact reverse-edge bookkeeping allocated
from bounded outgoing-edge capacity. Individual in-degree is not bounded by
`M`; only the total edge budget bounds reverse storage.

Update forward and reverse relationships together. Process incoming edges in
bounded batches and repair affected sources using surviving neighbors and the
deleted node's neighborhood at the corresponding level, pruned with HNSW's
heuristic. Edge commits must revalidate target state under the synchronization
used by deletion so they cannot recreate an edge after cleanup.

Reverse bookkeeping increases memory and mutation cost. A full adjacency scan is
a simpler alternative but makes deletion work proportional to graph size. Use
memory estimates and workload measurements to settle this tradeoff before
implementation; a full scan can also serve as a test oracle.

Test last-node deletion, entry-point deletion, simultaneous deletion of neighbors,
delete/reinsert of the same label, delayed edge commits, stale handles, repeated
reuse, and long-lived readers. Measure recall over long churn runs: memory safety
alone does not establish graph quality.

### Persistence after reclamation is introduced

Reusable slots and generation-aware handles may require a versioned concurrent
index format with legacy import. Quiescent save must finish pending maintenance,
drain reclamation, and serialize logical nodes and valid edges rather than runtime
state. Load reconstructs free pools, generations, reverse relationships, and EBR
metadata.

Keep live count, soft-deleted count, occupied slots, pending reclamation, and
reusable capacity distinct, with explicit compatibility semantics for existing
count APIs.

### Python integration

After the C++ contract is stable, audit automatic-label allocation, first-node
coordination, direct map/storage access, GIL transitions, callback reentrancy,
insufficient-result behavior, and pickling. No graph lock may be held while
acquiring the GIL for a Python callback.

## Delivery sequence

Git commits may be used to organize changes into coherent, reviewable steps
throughout implementation and validation. Keep each commit focused and record
the relevant validation and any unresolved acceptance requirements.

| Stage | Deliverable | Scope |
| --- | --- | --- |
| 1 | Finalize concurrent-mode API, operation restrictions, lock ordering, resource limits, and baseline measurements | First milestone |
| 2 | Storage accessors, adjacency snapshots, shared-metadata synchronization | First milestone |
| 3 | Insertion reservation, publication, root coordination, and failure cleanup | First milestone |
| 4 | Deterministic tests, mixed search/insert stress tests, sanitizer and compatibility validation | First milestone |
| 5 | Benchmark and tune concurrent C++ search/insertion; document supported operations | First milestone |
| 6 | EBR, generation-aware handles, and immutable-vector updates | Deferred |
| 7 | Deletion API decision, graph repair, retirement, and slot reuse | Deferred |
| 8 | Reclamation-aware persistence and sustained-churn validation | Deferred |
| 9 | Python concurrency support | Deferred |

The first milestone is complete when supported C++ search/insert paths follow a
documented synchronization and publication protocol, unsupported overlap has a
defined safe outcome, deterministic and sanitizer tests pass, quiescent
compatibility is preserved, and the existing framework shows no correctness,
search-only performance, insert-only performance, or recall regressions beyond
established measurement variability. Report baseline and implementation results
alongside mixed-workload measurements; mixed-workload gains alone do not satisfy
these acceptance criteria.
