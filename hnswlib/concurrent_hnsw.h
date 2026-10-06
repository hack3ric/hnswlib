#pragma once

#include "hnswalg.h"
#include <climits>
#include <cstdint>
#include <thread>

#ifndef HNSWLIB_CONCURRENT_TEST_HOOK
#define HNSWLIB_CONCURRENT_TEST_HOOK(event, id) ((void)0)
#endif

namespace hnswlib {

// Capacity is measured in upper-layer adjacency lists, not bytes. Zero selects
// a conservative default; exhaustion is reported rather than growing storage.
struct ConcurrentIndexOptions {
    size_t upper_layer_capacity = 0;
    int max_level = 32;
};

// Opt-in, fixed-capacity search/insert index. Composition deliberately prevents
// access to the legacy index's unsynchronized mutation and raw-pointer APIs.
// Vectors never change while shared operations are admitted. Adjacency snapshots
// hold at most one node lock; user code and distance functions run without it.
template<typename dist_t>
class ConcurrentHierarchicalNSW : public AlgorithmInterface<dist_t> {
    using Graph = HierarchicalNSW<dist_t>;
    using Candidate = std::pair<dist_t, tableint>;
    using Candidates = std::priority_queue<Candidate, std::vector<Candidate>,
                                            typename Graph::CompareByFirst>;
    enum State : unsigned char { Empty, Routing, Live, Failed };

    struct Storage : Graph {
        std::unique_ptr<char[]> arena;
        std::unique_ptr<char[]> records;
        using Graph::Graph;
        ~Storage() {
            // The base destructor owns individual legacy towers, whereas these
            // pointers refer into one arena owned by this derived object.
            if (arena)
                for (size_t i = 0; i < this->cur_element_count; ++i)
                    this->linkLists_[i] = nullptr;
            if (records) this->data_level0_memory_ = nullptr;
        }
    };

    // Shared admission protects against maintenance, NOT against insertion.
    // Maintenance fails promptly if an operation (including a callback) is live.
    class Admission {
        std::atomic<size_t>* gate_;
        std::atomic<size_t>* read_phase_;
        bool exclusive_;
        bool admitted_;
     public:
        // A null gate means the query already holds an unlocked SearchPhase,
        // which maintenance must also exclude. Other shared operations use gate.
        Admission(std::atomic<size_t>* gate, bool exclusive, std::atomic<size_t>* read_phase = nullptr)
            : gate_(nullptr), read_phase_(nullptr), exclusive_(exclusive), admitted_(!gate) {
            if (!gate) return;
            size_t value = gate->load(std::memory_order_relaxed);
            const size_t busy = std::numeric_limits<size_t>::max();
            if (exclusive) {
                value = 0;
                if (gate->compare_exchange_strong(value, busy, std::memory_order_acquire)) {
                    value = 0;
                    if (read_phase && !read_phase->compare_exchange_strong(
                            value, writer_bit_, std::memory_order_acq_rel)) {
                        gate->store(0, std::memory_order_release);
                        return;
                    }
                    gate_ = gate;
                    read_phase_ = read_phase;
                }
            } else {
                while (value < busy - 1) {
                    if (gate->compare_exchange_weak(value, value + 1, std::memory_order_acquire)) {
                        gate_ = gate;
                        break;
                    }
                }
            }
            admitted_ = gate_ != nullptr;
        }
        ~Admission() {
            if (gate_) {
                if (exclusive_) {
                    if (read_phase_) read_phase_->store(0, std::memory_order_release);
                    gate_->store(0, std::memory_order_release);
                }
                else gate_->fetch_sub(1, std::memory_order_release);
            }
        }
        explicit operator bool() const { return admitted_; }
        Admission(const Admission&) = delete;
        Admission& operator=(const Admission&) = delete;
    };

    // A vector-backed scratch pool makes returning a lease non-allocating.
    // One slot is reserved for every outstanding lease before allocating it.
    class VisitedPool {
        mutable std::mutex mutex_;
        mutable std::vector<VisitedList*> available_;
        mutable size_t allocated_ = 0;
        size_t capacity_;
     public:
        explicit VisitedPool(size_t capacity) : capacity_(capacity) {}
        ~VisitedPool() { for (auto p : available_) delete p; }
        VisitedList* acquire() const {
            VisitedList* p;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (available_.empty()) {
                    available_.reserve(allocated_ + 1);
                    p = new VisitedList(static_cast<int>(capacity_));
                    ++allocated_;
                } else {
                    p = available_.back();
                    available_.pop_back();
                }
            }
            // Reset may clear the whole marks array on tag wraparound; this
            // list is already exclusively leased, so do it outside the pool lock.
            p->reset();
            return p;
        }
        void release(VisitedList* p) const {
            std::lock_guard<std::mutex> lock(mutex_);
            available_.push_back(p);
        }
    };
    struct VisitedLease {
        const VisitedPool& pool;
        VisitedList* list;
        explicit VisitedLease(const VisitedPool& p) : pool(p), list(p.acquire()) {}
        ~VisitedLease() { pool.release(list); }
        VisitedLease(const VisitedLease&) = delete;
        VisitedLease& operator=(const VisitedLease&) = delete;
    };

    std::unique_ptr<Storage> graph_;
    ConcurrentIndexOptions options_;
    using StateByte = std::atomic<unsigned char>;
    static_assert(sizeof(StateByte) == 1 && alignof(StateByte) == 1,
                  "Concurrent HNSW requires byte-sized atomic flags");
    // The fourth header byte is unused by the legacy format. Keeping state next
    // to the existing deletion flag avoids a separate cache miss per candidate.
    StateByte& state(size_t id) const {
        return *reinterpret_cast<StateByte*>(reinterpret_cast<unsigned char*>(graph_->get_linklist0(id)) + 3);
    }
    void initializeState(size_t id, State value) {
        new (&state(id)) StateByte(static_cast<unsigned char>(value));
    }

    // These fields are immutable for an admitted operation. Cache them locally
    // so an indirect distance callback does not force repeated metadata loads.
    struct Vectors {
        const char* base;
        size_t stride, offset;
        DISTFUNC<dist_t> function;
        const void* parameter;
        explicit Vectors(const Graph& g)
            : base(g.data_level0_memory_), stride(g.size_data_per_element_), offset(g.offsetData_),
              function(g.fstdistfunc_), parameter(g.dist_func_param_) {}
        const char* data(tableint id) const { return base + id * stride + offset; }
        dist_t distance(const void* query, tableint id) const { return function(query, data(id), parameter); }
        bool eligible(tableint id) const {
            const char* record = base + id * stride;
            return reinterpret_cast<const StateByte*>(record + 3)->load(std::memory_order_acquire) == Live &&
                   !(static_cast<unsigned char>(record[2]) & Graph::DELETE_MARK);
        }
    };
    std::vector<uint64_t> revisions_;  // protected by the corresponding node lock
    mutable std::atomic<size_t> gate_{0};
    struct OperationScope {
        const ConcurrentHierarchicalNSW* index;
        OperationScope* previous;
        static OperationScope*& current() {
            static thread_local OperationScope* head = nullptr;
            return head;
        }
        static bool contains(const ConcurrentHierarchicalNSW* index) {
            for (auto* scope = current(); scope; scope = scope->previous)
                if (scope->index == index) return true;
            return false;
        }
        explicit OperationScope(const ConcurrentHierarchicalNSW* owner)
            : index(owner), previous(current()) { current() = this; }
        ~OperationScope() { current() = previous; }
    };
    // The first writer closes the unlocked-reader phase and drains only readers
    // already in it. New searches use locked snapshots and overlap all writers.
    // The last writer reopens the fast path. No graph-wide lock covers insertion.
    static const size_t writer_bit_ = size_t(1) << (sizeof(size_t) * CHAR_BIT - 1);
    mutable std::atomic<size_t> read_phase_{0};
    std::mutex writers_mutex_;
    size_t writers_ = 0;
    std::atomic<bool> solo_preparation_{false};

    struct SearchPhase {
        std::atomic<size_t>& phase;
        bool unlocked = false;
        explicit SearchPhase(std::atomic<size_t>& p) : phase(p) {
            size_t value = phase.load(std::memory_order_acquire);
            while (!(value & writer_bit_) && value < writer_bit_ - 1) {
                if (phase.compare_exchange_weak(value, value + 1, std::memory_order_acquire)) {
                    unlocked = true;
                    break;
                }
            }
        }
        ~SearchPhase() { if (unlocked) phase.fetch_sub(1, std::memory_order_release); }
    };

    struct WriterPhase {
        ConcurrentHierarchicalNSW& owner;
        explicit WriterPhase(ConcurrentHierarchicalNSW& o) : owner(o) {
            {
                std::lock_guard<std::mutex> lock(owner.writers_mutex_);
                if (++owner.writers_ == 1) owner.read_phase_.fetch_or(writer_bit_, std::memory_order_acq_rel);
            }
            HNSWLIB_CONCURRENT_TEST_HOOK("writer_entered", kInvalidInternalId);
            while (owner.read_phase_.load(std::memory_order_acquire) & ~writer_bit_) std::this_thread::yield();
            while (owner.solo_preparation_.load(std::memory_order_acquire)) std::this_thread::yield();
        }
        ~WriterPhase() {
            std::lock_guard<std::mutex> lock(owner.writers_mutex_);
            if (--owner.writers_ == 0) owner.read_phase_.fetch_and(~writer_bit_, std::memory_order_release);
        }
    };

    struct PreparationPhase {
        ConcurrentHierarchicalNSW& owner;
        bool unlocked;
        explicit PreparationPhase(ConcurrentHierarchicalNSW& o) : owner(o) {
            std::lock_guard<std::mutex> lock(owner.writers_mutex_);
            unlocked = owner.writers_ == 1;
            if (unlocked) owner.solo_preparation_.store(true, std::memory_order_relaxed);
        }
        ~PreparationPhase() {
            if (unlocked) owner.solo_preparation_.store(false, std::memory_order_release);
        }
    };
    std::atomic<tableint> root_{kInvalidInternalId};
    std::atomic<size_t> published_{0};
    std::atomic<size_t> failed_{0};
    std::atomic<size_t> ef_{10};
    std::mutex reservation_mutex_;
    std::mutex bootstrap_mutex_;
    size_t used_layers_ = 0;  // reservation_mutex_, or exclusive admission
    VisitedPool visited_;

    static size_t checkCapacity(size_t n) {
        if (n == 0 || n > static_cast<size_t>(INT_MAX))
            HNSWLIB_THROW_RUNTIME_ERROR("Concurrent capacity must be between 1 and INT_MAX");
        return n;
    }

    static size_t recordStride(size_t links, size_t payload, size_t alignment) {
        const size_t maximum = std::numeric_limits<size_t>::max();
        if (links > maximum - sizeof(tableint) - sizeof(labeltype) - 2 * alignment ||
            payload > maximum - links - sizeof(tableint) - sizeof(labeltype) - 2 * alignment)
            HNSWLIB_THROW_RUNTIME_ERROR("Concurrent record size overflow");
        const size_t offset = (links + sizeof(tableint) + sizeof(labeltype) + alignment - 1) & ~(alignment - 1);
        return (offset + payload + alignment - 1) & ~(alignment - 1);
    }

    static size_t checkConfiguration(SpaceInterface<dist_t>* space, size_t capacity, size_t M) {
        checkCapacity(capacity);
        if (!space || M < 2) HNSWLIB_THROW_RUNTIME_ERROR("Invalid concurrent space or M");
        const size_t links = 2 * std::min(M, size_t(10000)) * sizeof(tableint) + sizeof(linklistsizeint);
        const size_t stride = recordStride(links, space->get_data_size(), 64);
        if (capacity > (std::numeric_limits<size_t>::max() - 63) / stride)
            HNSWLIB_THROW_RUNTIME_ERROR("Concurrent record arena size overflow");
        return capacity;
    }

    void alignRecords() {
        Graph& g = *graph_;
        const size_t alignment = g.data_size_ >= 64 ? 64 : 16;
        const size_t stride = recordStride(g.size_links_level0_, g.data_size_, alignment);
        if (g.max_elements_ > (std::numeric_limits<size_t>::max() - alignment + 1) / stride)
            HNSWLIB_THROW_RUNTIME_ERROR("Concurrent record arena size overflow");
        // One extra ID after the level-zero list is a prefetch sentinel, even
        // at maximum degree. The label and vector do not overlap this slot.
        const size_t label_offset = g.size_links_level0_ + sizeof(tableint);
        const size_t data_offset = (label_offset + sizeof(labeltype) + alignment - 1) & ~(alignment - 1);
        std::unique_ptr<char[]> records(new char[g.max_elements_ * stride + alignment - 1]);
        char* base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(records.get()) + alignment - 1)
                                            & ~(uintptr_t(alignment) - 1));
        for (size_t i = 0; i < g.cur_element_count; ++i) {
            memcpy(base + i * stride, g.get_linklist0(i), g.size_links_level0_);
            memcpy(base + i * stride + label_offset,
                   g.data_level0_memory_ + i * g.size_data_per_element_ + g.label_offset_, sizeof(labeltype));
            memcpy(base + i * stride + data_offset, g.getDataByInternalId(i), g.data_size_);
            auto* list = reinterpret_cast<linklistsizeint*>(base + i * stride);
            auto* links = reinterpret_cast<tableint*>(list + 1);
            links[g.maxM0_] = 0;
            // Unused legacy entries can be stale but must address allocated
            // storage if the prefetch loop reads one past the live edge count.
            for (size_t j = g.getListCount(list); j < g.maxM0_; ++j)
                if (links[j] >= g.max_elements_) links[j] = 0;
        }
        free(g.data_level0_memory_);
        g.data_level0_memory_ = base;
        graph_->records = std::move(records);
        g.size_data_per_element_ = stride;
        g.label_offset_ = label_offset;
        g.offsetData_ = data_offset;
    }

    void initialize() {
        Graph& g = *graph_;
        checkCapacity(g.max_elements_);
        if (g.M_ < 2 || options_.max_level < 0 || options_.max_level > 1024)
            HNSWLIB_THROW_RUNTIME_ERROR("Invalid concurrent M or maximum level");
        if (g.M_ > 10000 || g.maxM_ != g.M_ || g.maxM0_ != 2 * g.M_ ||
            g.offsetLevel0_ != 0 || g.offsetData_ != g.size_links_level0_ ||
            g.label_offset_ != g.offsetData_ + g.data_size_ ||
            g.size_data_per_element_ != g.label_offset_ + sizeof(labeltype))
            HNSWLIB_THROW_RUNTIME_ERROR("Incompatible concurrent index layout or space");
        if (!options_.upper_layer_capacity)
            options_.upper_layer_capacity = g.max_elements_ / (g.M_ - 1) + g.max_elements_ / 16 + 64;
        if (options_.upper_layer_capacity > std::numeric_limits<size_t>::max() / g.size_links_per_element_)
            HNSWLIB_THROW_RUNTIME_ERROR("Upper-layer arena size overflow");
        g.label_lookup_.reserve(g.max_elements_);
        revisions_.resize(g.max_elements_, 0);
        size_t needed = 0;
        for (size_t i = 0; i < g.cur_element_count; ++i) {
            if (g.element_levels_[i] > options_.max_level)
                HNSWLIB_THROW_RUNTIME_ERROR("Loaded tower exceeds concurrent maximum level");
            needed += g.element_levels_[i];
        }
        if (needed > options_.upper_layer_capacity)
            HNSWLIB_THROW_RUNTIME_ERROR("Loaded towers exceed upper-layer capacity");
        if (g.cur_element_count &&
            (g.enterpoint_node_ >= g.cur_element_count || g.maxlevel_ != g.element_levels_[g.enterpoint_node_]))
            HNSWLIB_THROW_RUNTIME_ERROR("Invalid loaded entry point");
        for (size_t i = 0; i < g.cur_element_count; ++i) {
            auto found = g.label_lookup_.find(g.getExternalLabel(i));
            if (found == g.label_lookup_.end() || found->second != i)
                HNSWLIB_THROW_RUNTIME_ERROR("Invalid loaded label mapping");
            for (int layer = 0; layer <= g.element_levels_[i]; ++layer) {
                auto* list = g.get_linklist_at_level(i, layer);
                size_t size = g.getListCount(list);
                if (size > (layer ? g.maxM_ : g.maxM0_))
                    HNSWLIB_THROW_RUNTIME_ERROR("Invalid loaded adjacency size");
                auto* links = reinterpret_cast<tableint*>(list + 1);
                for (size_t j = 0; j < size; ++j)
                    if (links[j] >= g.cur_element_count || links[j] == i || g.element_levels_[links[j]] < layer)
                        HNSWLIB_THROW_RUNTIME_ERROR("Invalid loaded graph edge");
            }
        }
        std::unique_ptr<char[]> arena(new char[options_.upper_layer_capacity * g.size_links_per_element_]);
        for (size_t i = 0; i < g.cur_element_count; ++i) {
            size_t bytes = g.element_levels_[i] * g.size_links_per_element_;
            char* tower = bytes ? arena.get() + used_layers_ * g.size_links_per_element_ : nullptr;
            if (bytes) memcpy(tower, g.linkLists_[i], bytes);
            free(g.linkLists_[i]);
            g.linkLists_[i] = tower;
            used_layers_ += g.element_levels_[i];
        }
        graph_->arena = std::move(arena);
        alignRecords();
        for (size_t i = 0; i < g.cur_element_count; ++i) initializeState(i, Live);
        published_.store(g.cur_element_count, std::memory_order_relaxed);
        if (g.cur_element_count) root_.store(g.enterpoint_node_, std::memory_order_release);
    }

    bool eligible(tableint id) const {
        return state(id).load(std::memory_order_acquire) == Live && !graph_->isMarkedDeleted(id);
    }

    uint64_t neighbors(tableint id, int level, std::vector<tableint>& out) const {
        Graph& g = *graph_;
        HNSWLIB_CONCURRENT_TEST_HOOK("adjacency_lock_begin", id);
        std::lock_guard<std::mutex> lock(g.link_list_locks_[id]);
        HNSWLIB_CONCURRENT_TEST_HOOK("adjacency_lock_acquired", id);
        const auto* list = g.get_linklist_at_level(id, level);
        const tableint* data = reinterpret_cast<const tableint*>(list + 1);
        out.assign(data, data + g.getListCount(const_cast<linklistsizeint*>(list)));
        return revisions_[id];
    }

    void writeNeighbors(tableint id, int level, const std::vector<tableint>& values) {
        auto* list = graph_->get_linklist_at_level(id, level);
        if (!values.empty()) memcpy(list + 1, values.data(), values.size() * sizeof(tableint));
        graph_->setListCount(list, static_cast<unsigned short>(values.size()));
        ++revisions_[id];
    }

    template<bool locking = true>
    tableint descend(const void* query, tableint entry, int bottom) const {
        const Graph& g = *graph_;
        const Vectors vectors(g);
        dist_t best = vectors.distance(query, entry);
        std::vector<tableint> links;
        if (locking) links.reserve(g.maxM0_);
        for (int layer = g.element_levels_[entry]; layer > bottom; --layer) {
            bool changed;
            do {
                changed = false;
                const tableint* adjacent;
                size_t size;
                if (locking) {
                    neighbors(entry, layer, links);
                    adjacent = links.data(); size = links.size();
                } else {
                    auto* list = g.get_linklist_at_level(entry, layer);
                    adjacent = reinterpret_cast<tableint*>(list + 1); size = g.getListCount(list);
                }
                for (size_t i = 0; i < size; ++i) {
                    tableint candidate = adjacent[i];
                    dist_t d = vectors.distance(query, candidate);
                    if (d < best) { best = d; entry = candidate; changed = true; }
                }
            } while (changed);
        }
        return entry;
    }

    static std::vector<Candidate> candidateStorage(size_t capacity) {
        std::vector<Candidate> storage;
        storage.reserve(capacity);
        return storage;
    }

    template<bool locking = true, bool bare = false, bool callbacks = true>
    Candidates searchLayer(const void* query, tableint entry, int layer, size_t ef,
                           BaseFilterFunctor* filter = nullptr,
                           BaseSearchStopCondition<dist_t>* stop = nullptr) const {
        const Graph& g = *graph_;
        const Vectors vectors(g);
        VisitedLease visited(visited_);
        vl_type* marks = visited.list->mass;
        const vl_type tag = visited.list->curV;
        // Reserve a modest initial frontier rather than repeatedly allocating
        // 1, 2, 4, ... entries. Large ef values can still grow on demand.
        const size_t initial = std::min(g.max_elements_, std::min(size_t(256), ef ? ef : g.M_)) + 1;
        Candidates result(typename Graph::CompareByFirst(), candidateStorage(initial));
        Candidates candidates(typename Graph::CompareByFirst(), candidateStorage(initial));
        std::vector<tableint> links;
        if (locking) links.reserve(g.maxM0_);
        const bool all_live = !locking && !failed_.load(std::memory_order_relaxed) && !g.num_deleted_.load();
        dist_t lower = std::numeric_limits<dist_t>::max();
        dist_t first = bare ? g.fstdistfunc_(query, g.getDataByInternalId(entry), g.dist_func_param_)
                            : vectors.distance(query, entry);
        if ((bare || all_live || vectors.eligible(entry)) && (!callbacks || !filter || (*filter)(g.getExternalLabel(entry)))) {
            result.emplace(first, entry);
            lower = first;
            if (callbacks && stop) stop->add_point_to_result(g.getExternalLabel(entry), g.getDataByInternalId(entry), first);
        }
        candidates.emplace(-first, entry);
        marks[entry] = tag;
        while (!candidates.empty()) {
            const auto current = candidates.top();
            if (bare ? -current.first > lower : (callbacks && stop ? stop->should_stop_search(-current.first, lower)
                     : (-current.first > lower && result.size() >= ef))) break;
            candidates.pop();
            const tableint* adjacent;
            size_t size;
            if (locking) {
                neighbors(current.second, layer, links);
                adjacent = links.data(); size = links.size();
            } else {
                auto* list = g.get_linklist_at_level(current.second, layer);
                adjacent = reinterpret_cast<tableint*>(list + 1); size = g.getListCount(list);
            }
            #if defined(USE_SSE) && HNSWLIB_USE_PREFETCH
            if (size) {
                _mm_prefetch(reinterpret_cast<const char*>(marks + adjacent[0]), _MM_HINT_T0);
                const size_t ahead = std::min(size_t(adjacent[0]) + 64, g.max_elements_ - 1);
                _mm_prefetch(reinterpret_cast<const char*>(marks + ahead), _MM_HINT_T0);
                _mm_prefetch(bare ? g.getDataByInternalId(adjacent[0]) : vectors.data(adjacent[0]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(adjacent + 1), _MM_HINT_T0);
            }
#endif
            for (size_t j = 0; j < size; ++j) {
#if defined(USE_SSE) && HNSWLIB_USE_PREFETCH
                // Bare search only visits level zero. All its allocated list
                // entries address allocated slots, including a sentinel after
                // maximum degree. Prefetch never interprets a vector or label
                // as a neighbor, nor dereferences a stale unused edge.
                const tableint next = adjacent[bare ? j + 1 : std::min(j + 1, size - 1)];
                _mm_prefetch(reinterpret_cast<const char*>(marks + next), _MM_HINT_T0);
                _mm_prefetch(bare ? g.getDataByInternalId(next) : vectors.data(next), _MM_HINT_T0);
#endif
                tableint id = adjacent[j];
                if (marks[id] == tag) continue;
                marks[id] = tag;
                dist_t d = bare ? g.fstdistfunc_(query, g.getDataByInternalId(id), g.dist_func_param_)
                                : vectors.distance(query, id);
                if (!(callbacks && stop ? stop->should_consider_candidate(d, lower) : (result.size() < ef || d < lower))) continue;
                candidates.emplace(-d, id);
#if defined(USE_SSE) && HNSWLIB_USE_PREFETCH
                // The next expansion needs adjacency; its payload has already
                // been read to compute the candidate distance.
                _mm_prefetch(reinterpret_cast<const char*>(g.get_linklist_at_level(candidates.top().second, layer)),
                             _MM_HINT_T0);
#endif
                if ((bare || all_live || vectors.eligible(id)) && (!callbacks || !filter || (*filter)(g.getExternalLabel(id)))) {
                    result.emplace(d, id);
                    if (callbacks && stop) stop->add_point_to_result(g.getExternalLabel(id), g.getDataByInternalId(id), d);
                }
                while (callbacks && stop ? stop->should_remove_extra() : result.size() > ef) {
                    const auto removed = result.top();
                    result.pop();
                    if (callbacks && stop) stop->remove_point_from_result(g.getExternalLabel(removed.second),
                                                            g.getDataByInternalId(removed.second), removed.first);
                }
                if (!result.empty()) lower = result.top().first;
            }
        }
        return result;
    }

    // The same diversification heuristic and ordering as getNeighborsByHeuristic2,
    // with immutable vector access and a single allocation for the kept set.
    void prune(Candidates& candidates, size_t limit) const {
        if (candidates.size() < limit) return;
        const Vectors vectors(*graph_);
        std::priority_queue<Candidate> closest(std::less<Candidate>(), candidateStorage(candidates.size()));
        std::vector<Candidate> kept;
        kept.reserve(limit);
        while (!candidates.empty()) {
            closest.emplace(-candidates.top().first, candidates.top().second);
            candidates.pop();
        }
        while (!closest.empty() && kept.size() < limit) {
            const auto candidate = closest.top();
            closest.pop();
            bool good = true;
            for (const auto& neighbor : kept) {
                if (vectors.distance(vectors.data(neighbor.second), candidate.second) < -candidate.first) {
                    good = false;
                    break;
                }
            }
            if (good) kept.push_back(candidate);
        }
        for (const auto& candidate : kept) candidates.emplace(-candidate.first, candidate.second);
    }

    // Optimistic read/compute/validate/commit. Distance callbacks and pruning
    // never run under the node lock, and a concurrent writer cannot be lost.
    void addBacklink(tableint source, tableint target, int level,
                     std::vector<tableint>& old, std::vector<tableint>& proposed) {
        Graph& g = *graph_;
        const Vectors vectors(g);
        const size_t limit = level ? g.maxM_ : g.maxM0_;
        for (;;) {
            uint64_t revision;
            {
                HNSWLIB_CONCURRENT_TEST_HOOK("adjacency_lock_begin", source);
                std::lock_guard<std::mutex> lock(g.link_list_locks_[source]);
                HNSWLIB_CONCURRENT_TEST_HOOK("adjacency_lock_acquired", source);
                auto* list = g.get_linklist_at_level(source, level);
                tableint* links = reinterpret_cast<tableint*>(list + 1);
                size_t size = g.getListCount(list);
                // target is a new, unpublished ID and each selected neighbor
                // is unique. No other insertion can install this same edge.
                if (size < limit) {
                    links[size] = target;
                    g.setListCount(list, static_cast<unsigned short>(size + 1));
                    ++revisions_[source];
                    return;
                }
                old.assign(links, links + size);
                revision = revisions_[source];
            }
            HNSWLIB_CONCURRENT_TEST_HOOK("backlink_snapshot", target);
            proposed.reserve(limit);
            Candidates candidates(typename Graph::CompareByFirst(), candidateStorage(limit + 1));
            candidates.emplace(vectors.distance(vectors.data(target), source), target);
            for (auto id : old)
                candidates.emplace(vectors.distance(vectors.data(id), source), id);
            prune(candidates, limit);
            proposed.clear();
            while (!candidates.empty()) { proposed.push_back(candidates.top().second); candidates.pop(); }
            HNSWLIB_CONCURRENT_TEST_HOOK("adjacency_lock_begin", source);
            std::lock_guard<std::mutex> lock(g.link_list_locks_[source]);
            HNSWLIB_CONCURRENT_TEST_HOOK("adjacency_lock_acquired", source);
            if (revision != revisions_[source]) continue;
            writeNeighbors(source, level, proposed);
            return;
        }
    }

    struct Insertion {
        ConcurrentHierarchicalNSW& owner;
        tableint id;
        labeltype label;
        bool committed = false;
        Insertion(ConcurrentHierarchicalNSW& owner_, tableint id_, labeltype label_)
            : owner(owner_), id(id_), label(label_) {}
        ~Insertion() {
            if (committed) return;
            // Readers may already have copied an edge to this initialized node.
            // Preserve its vector/tower until exclusive cleanup, even on failure.
            owner.state(id).store(Failed, std::memory_order_release);
            owner.failed_.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(owner.graph_->label_lookup_lock);
            owner.graph_->label_lookup_.erase(label);
        }
    };

    template<bool locking>
    std::vector<std::vector<tableint>> prepare(const void* data, tableint id, tableint entry, int level) {
        Graph& g = *graph_;
        const tableint fallback = entry;
        const int top = g.element_levels_[entry];
        if (level < top) entry = descend<locking>(data, entry, level);
        std::vector<std::vector<tableint>> selected_by_level(std::min(level, top) + 1);
        for (int layer = std::min(level, top); layer >= 0; --layer) {
            Candidates candidates = searchLayer<locking, false, false>(data, entry, layer, g.ef_construction_);
            if (g.isMarkedDeleted(fallback)) {
                candidates.emplace(g.fstdistfunc_(data, g.getDataByInternalId(fallback), g.dist_func_param_), fallback);
                if (candidates.size() > g.ef_construction_) candidates.pop();
            }
            prune(candidates, g.M_);
            std::vector<tableint>& selected = selected_by_level[layer];
            selected.reserve(g.M_);
            while (!candidates.empty()) { selected.push_back(candidates.top().second); candidates.pop(); }
            if (selected.empty()) selected.push_back(fallback);
            entry = selected.back();
            // No incoming edge exists yet; no other builder selects a Routing
            // node. Publishing a backlink later publishes this initialized list.
            writeNeighbors(id, layer, selected);
        }
        return selected_by_level;
    }

    StatusOr<tableint> insert(const void* data, labeltype label, int requested_level) {
        Graph& g = *graph_;
        std::unique_lock<std::mutex> label_lock(g.getLabelOpMutex(label));
        tableint id;
        int level;
        {
            std::lock_guard<std::mutex> reserve(reservation_mutex_);
            std::lock_guard<std::mutex> lookup(g.label_lookup_lock);
            if (g.label_lookup_.count(label)) return Status("Label already exists; use a quiescent update");
            if (g.cur_element_count >= g.max_elements_) return Status("Concurrent capacity exhausted; clean up failed insertions if any");
            level = requested_level;
            if (level < 0) {
                std::uniform_real_distribution<double> distribution(0.0, 1.0);
                const double sample = distribution(g.level_generator_);
                // Zero is permitted by uniform_real_distribution. Reject its
                // unbounded tower instead of converting infinity to an int.
                if (sample == 0) return Status("Random level exceeds concurrent maximum level");
                level = static_cast<int>(-log(sample) * g.mult_);
            }
            if (level > options_.max_level) return Status("Insertion exceeds concurrent maximum level");
            if (static_cast<size_t>(level) > options_.upper_layer_capacity - used_layers_)
                return Status("Upper-layer arena capacity exhausted");
            id = static_cast<tableint>(g.cur_element_count.load());
            HNSWLIB_CONCURRENT_TEST_HOOK("reserved", id);
            // Allocate the map entry before consuming storage. It remains hidden
            // by state validation until publication.
            g.label_lookup_.emplace(label, id);
            g.element_levels_[id] = level;
            g.linkLists_[id] = level ? graph_->arena.get() + used_layers_ * g.size_links_per_element_ : nullptr;
            used_layers_ += level;
            memset(g.data_level0_memory_ + id * g.size_data_per_element_, 0, g.size_data_per_element_);
            if (level) memset(g.linkLists_[id], 0, level * g.size_links_per_element_);
            g.setExternalLabel(id, label);
            memcpy(g.getDataByInternalId(id), data, g.data_size_);
            ++g.cur_element_count;
            initializeState(id, Routing);
        }
        Insertion insertion{*this, id, label};
        HNSWLIB_CONCURRENT_TEST_HOOK("initialized", id);
        tableint entry = root_.load(std::memory_order_acquire);
        if (entry == kInvalidInternalId) {
            std::lock_guard<std::mutex> bootstrap(bootstrap_mutex_);
            entry = root_.load(std::memory_order_acquire);
            if (entry == kInvalidInternalId) {
                state(id).store(Live, std::memory_order_release);
                ++published_;
                root_.store(id, std::memory_order_release);
                insertion.committed = true;
                return id;
            }
        }
        const int top = g.element_levels_[entry];
        std::vector<std::vector<tableint>> selected_by_level;
        {
            PreparationPhase phase(*this);
            HNSWLIB_CONCURRENT_TEST_HOOK("prepare", id);
            selected_by_level = phase.unlocked ? prepare<false>(data, id, entry, level)
                                              : prepare<true>(data, id, entry, level);
        }
        // A traversal may enter through an upper-layer backlink and descend all
        // the way to level zero, even if this insertion later fails. Prepare the
        // whole outgoing tower before exposing the first reciprocal link.
        std::vector<tableint> old, proposed;
        for (int layer = std::min(level, top); layer >= 0; --layer) {
            for (auto neighbor : selected_by_level[layer]) addBacklink(neighbor, id, layer, old, proposed);
            HNSWLIB_CONCURRENT_TEST_HOOK("layer_linked", id);
        }
        HNSWLIB_CONCURRENT_TEST_HOOK("before_publish", id);
        state(id).store(Live, std::memory_order_release);
        ++published_;
        tableint previous = root_.load(std::memory_order_acquire);
        while (g.element_levels_[previous] < level &&
               !root_.compare_exchange_weak(previous, id, std::memory_order_acq_rel, std::memory_order_acquire)) {}
        insertion.committed = true;
        return id;
    }

    void syncRoot() {
        graph_->enterpoint_node_ = root_.load(std::memory_order_acquire);
        graph_->maxlevel_ = graph_->enterpoint_node_ == kInvalidInternalId ? -1 : graph_->element_levels_[graph_->enterpoint_node_];
        graph_->ef_ = ef_.load(std::memory_order_relaxed);
    }

    // Quiescent legacy updates see only legacy flags. Recreate the atomic
    // objects even if a distance callback throws.
    struct SavedStates {
        ConcurrentHierarchicalNSW& owner;
        explicit SavedStates(ConcurrentHierarchicalNSW& o) : owner(o) {
            for (size_t i = 0; i < owner.graph_->cur_element_count; ++i) {
                StateByte& flag = owner.state(i);
                flag.~StateByte();
                *reinterpret_cast<unsigned char*>(&flag) = 0;
            }
        }
        ~SavedStates() {
            for (size_t i = 0; i < owner.graph_->cur_element_count; ++i) owner.initializeState(i, Live);
        }
    };

    // Emit the canonical packed format, independent of the aligned in-memory
    // layout. Copy bounded batches; no second full index is needed to save.
    Status save(std::ostream& output) {
        Graph& g = *graph_;
        StreamExceptionsOff guard(output);
        return invokeWithoutStreamThrow([&]() -> Status {
            if (!output) return Status("Cannot save index: output stream is in a failed state");
            const size_t data_offset = g.size_links_level0_;
            const size_t label_offset = data_offset + g.data_size_;
            const size_t stride = label_offset + sizeof(labeltype);
            writeBinaryPOD(output, g.offsetLevel0_);
            writeBinaryPOD(output, g.max_elements_);
            writeBinaryPOD(output, g.cur_element_count);
            writeBinaryPOD(output, stride);
            writeBinaryPOD(output, label_offset);
            writeBinaryPOD(output, data_offset);
            writeBinaryPOD(output, g.maxlevel_);
            writeBinaryPOD(output, g.enterpoint_node_);
            writeBinaryPOD(output, g.maxM_);
            writeBinaryPOD(output, g.maxM0_);
            writeBinaryPOD(output, g.M_);
            writeBinaryPOD(output, g.mult_);
            writeBinaryPOD(output, g.ef_construction_);
            if (!output.good()) return Status("Failed writing index metadata");
            const size_t batch = std::max(size_t(1), size_t(65536) / stride);
            std::vector<char> buffer(batch * stride);
            const size_t count = g.cur_element_count;
            for (size_t begin = 0; begin < count; begin += batch) {
                const size_t length = std::min(batch, count - begin);
                for (size_t j = 0; j < length; ++j) {
                    char* record = buffer.data() + j * stride;
                    memcpy(record, g.get_linklist0(begin + j), data_offset);
                    record[3] = 0;  // Never persist runtime publication state.
                    memcpy(record + data_offset, g.getDataByInternalId(begin + j), g.data_size_);
                    const labeltype label = g.getExternalLabel(begin + j);
                    memcpy(record + label_offset, &label, sizeof(label));
                }
                output.write(buffer.data(), length * stride);
                if (!output.good()) return Status("Failed writing level 0 memory block");
            }
            for (size_t i = 0; i < count; ++i) {
                const unsigned int bytes = g.element_levels_[i] * g.size_links_per_element_;
                writeBinaryPOD(output, bytes);
                if (bytes) output.write(g.linkLists_[i], bytes);
                if (!output.good()) return Status("Failed writing link list elements");
            }
            return OkStatus();
        });
    }

    // Exclusive admission: no operation can retain an old ID or payload pointer.
    // Remap edges first, then compact nodes and towers in allocation order.
    void cleanup() {
        if (!failed_.load(std::memory_order_relaxed)) return;
        Graph& g = *graph_;
        const size_t old_count = g.cur_element_count;
        std::vector<tableint> remap(old_count, kInvalidInternalId);
        tableint count = 0;
        for (size_t i = 0; i < old_count; ++i)
            if (state(i).load(std::memory_order_relaxed) == Live) remap[i] = count++;
        for (size_t i = 0; i < old_count; ++i) {
            if (remap[i] == kInvalidInternalId) continue;
            for (int layer = 0; layer <= g.element_levels_[i]; ++layer) {
                auto* list = g.get_linklist_at_level(i, layer);
                tableint* links = reinterpret_cast<tableint*>(list + 1);
                size_t size = g.getListCount(list), kept = 0;
                for (size_t j = 0; j < size; ++j)
                    if (remap[links[j]] != kInvalidInternalId) links[kept++] = remap[links[j]];
                g.setListCount(list, static_cast<unsigned short>(kept));
            }
        }
        size_t used = 0;
        for (size_t i = 0; i < old_count; ++i) {
            if (remap[i] == kInvalidInternalId) continue;
            tableint dest = remap[i];
            int level = g.element_levels_[i];
            char* tower = level ? graph_->arena.get() + used * g.size_links_per_element_ : nullptr;
            if (level) memmove(tower, g.linkLists_[i], level * g.size_links_per_element_);
            memmove(g.data_level0_memory_ + dest * g.size_data_per_element_,
                    g.data_level0_memory_ + i * g.size_data_per_element_, g.size_data_per_element_);
            g.element_levels_[dest] = level;
            g.linkLists_[dest] = tower;
            initializeState(dest, Live);
            revisions_[dest] = 0;
            used += level;
        }
        for (auto& item : g.label_lookup_) item.second = remap[item.second];
        for (size_t i = count; i < old_count; ++i) initializeState(i, Empty);
        tableint root = root_.load(std::memory_order_relaxed);
        if (root != kInvalidInternalId) root_.store(remap[root], std::memory_order_relaxed);
        g.cur_element_count = count;
        used_layers_ = used;
        failed_.store(0, std::memory_order_relaxed);
        syncRoot();
    }

 public:
    using DistanceLabelPriorityQueue = typename AlgorithmInterface<dist_t>::DistanceLabelPriorityQueue;
    using DistanceLabelVector = typename AlgorithmInterface<dist_t>::DistanceLabelVector;
    using AlgorithmInterface<dist_t>::addPoint;

    ConcurrentHierarchicalNSW(SpaceInterface<dist_t>* space, size_t capacity,
                             size_t M = 16, size_t ef_construction = 200, size_t seed = 100,
                             ConcurrentIndexOptions options = ConcurrentIndexOptions())
        : graph_(new Storage(space, checkConfiguration(space, capacity, M), M, ef_construction, seed)),
          options_(options), visited_(capacity) { initialize(); }

    ConcurrentHierarchicalNSW(SpaceInterface<dist_t>* space, const std::string& location,
                             size_t capacity = 0, ConcurrentIndexOptions options = ConcurrentIndexOptions())
        : graph_(new Storage(space, location, false, capacity)),
          options_(options), visited_(graph_->max_elements_) { initialize(); }

    size_t getMaxElements() const { return graph_->max_elements_; }
    size_t getCurrentElementCount() const { return published_.load(std::memory_order_acquire); }
    size_t getDeletedCount() const { return graph_->num_deleted_.load(std::memory_order_acquire); }
    size_t getFailedInsertionCount() const { return failed_.load(std::memory_order_acquire); }
    void setEf(size_t ef) { ef_.store(ef, std::memory_order_relaxed); }

    Status addPointNoExceptions(const void* data, labeltype label, bool replace_deleted = false) override {
        if (replace_deleted) return Status("Concurrent deleted-slot replacement is not supported");
        auto result = addPointWithLevel(data, label, -1);
        return result.ok() ? OkStatus() : result.status();
    }

    tableint addPoint(const void* data, labeltype label, int level) {
        auto result = addPointWithLevel(data, label, level);
        if (!result.ok()) HNSWLIB_THROW_RUNTIME_ERROR(result.status().message());
        return result.value();
    }

    StatusOr<tableint> addPointWithLevel(const void* data, labeltype label, int level) {
        if (OperationScope::contains(this)) return Status("Insertion from a callback on the same index is not supported");
        Admission operation(&gate_, false);
        if (!operation) return Status("Index maintenance is in progress");
        if (!data || level < -1) return Status("Invalid insertion data or level");
        OperationScope scope(this);
        WriterPhase writer(*this);
#if defined(__EXCEPTIONS) || _HAS_EXCEPTIONS == 1
        try { return insert(data, label, level); }
        catch (...) { return Status("Insertion failed; exposed storage is quarantined until cleanupFailedInsertions"); }
#else
        return insert(data, label, level);
#endif
    }

    StatusOr<DistanceLabelPriorityQueue> searchKnnNoExceptions(
        const void* query, size_t k, BaseFilterFunctor* filter = nullptr) const override {
        SearchPhase phase(read_phase_);
        Admission operation(phase.unlocked ? nullptr : &gate_, false);
        if (!operation) return Status("Index maintenance is in progress");
        if (!query && k) return Status("Null query");
        OperationScope scope(this);
        DistanceLabelPriorityQueue result;
        tableint entry = root_.load(std::memory_order_acquire);
        if (!k || entry == kInvalidInternalId) return result;
        HNSWLIB_CONCURRENT_TEST_HOOK("root_snapshot", entry);
        const size_t ef = std::max(k, ef_.load(std::memory_order_relaxed));
        Candidates candidates;
        if (phase.unlocked) {
            entry = descend<false>(query, entry, 0);
            if (!filter && !failed_.load(std::memory_order_relaxed) && !graph_->num_deleted_.load())
                candidates = searchLayer<false, true, false>(query, entry, 0, ef);
            else
                candidates = searchLayer<false>(query, entry, 0, ef, filter);
        } else {
            entry = descend(query, entry, 0);
            candidates = filter ? searchLayer(query, entry, 0, ef, filter)
                                : searchLayer<true, false, false>(query, entry, 0, ef);
        }
        while (candidates.size() > k) candidates.pop();
        while (!candidates.empty()) {
            const auto item = candidates.top();
            result.emplace(item.first, graph_->getExternalLabel(item.second));
            candidates.pop();
        }
        return result;
    }

    StatusOr<DistanceLabelVector> searchStopConditionClosestNoExceptions(
        const void* query, BaseSearchStopCondition<dist_t>& stop, BaseFilterFunctor* filter = nullptr) const {
        SearchPhase phase(read_phase_);
        Admission operation(phase.unlocked ? nullptr : &gate_, false);
        if (!operation) return Status("Index maintenance is in progress");
        if (!query) return Status("Null query");
        OperationScope scope(this);
        DistanceLabelVector result;
        tableint entry = root_.load(std::memory_order_acquire);
        if (entry == kInvalidInternalId) return result;
        Candidates candidates = phase.unlocked
            ? searchLayer<false>(query, descend<false>(query, entry, 0), 0, 0, filter, &stop)
            : searchLayer(query, descend(query, entry, 0), 0, 0, filter, &stop);
        result.resize(candidates.size());
        size_t i = result.size();
        while (!candidates.empty()) {
            const auto item = candidates.top();
            result[--i] = {item.first, graph_->getExternalLabel(item.second)};
            candidates.pop();
        }
        stop.filter_results(result);
        return result;
    }

    DistanceLabelVector searchStopConditionClosest(
        const void* query, BaseSearchStopCondition<dist_t>& stop, BaseFilterFunctor* filter = nullptr) const {
        auto result = searchStopConditionClosestNoExceptions(query, stop, filter);
        if (!result.ok()) HNSWLIB_THROW_RUNTIME_ERROR(result.status().message());
        return std::move(result).value();
    }

    template<typename data_t>
    StatusOr<std::vector<data_t>> getDataByLabelNoExceptions(labeltype label) const {
        Admission operation(&gate_, false);
        if (!operation) return Status("Index maintenance is in progress");
        std::lock_guard<std::mutex> lookup(graph_->label_lookup_lock);
        auto it = graph_->label_lookup_.find(label);
        if (it == graph_->label_lookup_.end() || !eligible(it->second)) return Status("Label not found");
        const size_t dim = *static_cast<size_t*>(graph_->dist_func_param_);
        std::vector<data_t> result(dim);
        memcpy(result.data(), graph_->getDataByInternalId(it->second), dim * sizeof(data_t));
        return result;
    }

    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        auto result = getDataByLabelNoExceptions<data_t>(label);
        if (!result.ok()) HNSWLIB_THROW_RUNTIME_ERROR(result.status().message());
        return std::move(result).value();
    }

    Status cleanupFailedInsertions() {
        Admission operation(&gate_, true, &read_phase_);
        if (!operation) return Status("Cleanup requires a quiescent index");
        cleanup();
        return OkStatus();
    }

    Status checkIntegrityNoExceptions() {
        Admission operation(&gate_, true, &read_phase_);
        if (!operation) return Status("Integrity inspection requires a quiescent index");
        cleanup();
        Graph& g = *graph_;
        if (g.cur_element_count != published_.load() || g.label_lookup_.size() != published_.load())
            return Status("Inconsistent published-node accounting");
        const tableint root = root_.load(std::memory_order_relaxed);
        if (g.cur_element_count ? root >= g.cur_element_count : root != kInvalidInternalId)
            return Status("Invalid entry point");
        for (size_t i = 0; i < g.cur_element_count; ++i) {
            if (state(i).load() != Live) return Status("Non-live node after cleanup");
            if (g.element_levels_[i] > g.element_levels_[root]) return Status("Entry point is below maximum level");
            auto found = g.label_lookup_.find(g.getExternalLabel(i));
            if (found == g.label_lookup_.end() || found->second != i) return Status("Invalid label mapping");
            for (int layer = 0; layer <= g.element_levels_[i]; ++layer) {
                auto* list = g.get_linklist_at_level(i, layer);
                const size_t size = g.getListCount(list);
                if (size > (layer ? g.maxM_ : g.maxM0_)) return Status("Adjacency capacity exceeded");
                const tableint* links = reinterpret_cast<tableint*>(list + 1);
                for (size_t j = 0; j < size; ++j) {
                    if (links[j] >= g.cur_element_count || links[j] == i || g.element_levels_[links[j]] < layer)
                        return Status("Invalid graph edge");
                    for (size_t k = 0; k < j; ++k)
                        if (links[j] == links[k]) return Status("Duplicate graph edge");
                }
            }
        }
        return OkStatus();
    }

    Status markDeleteNoExceptions(labeltype label) {
        Admission operation(&gate_, true, &read_phase_);
        if (!operation) return Status("Soft deletion requires a quiescent index");
        return graph_->markDeleteNoExceptions(label);
    }
    void markDelete(labeltype label) {
        Status s = markDeleteNoExceptions(label);
        if (!s.ok()) HNSWLIB_THROW_RUNTIME_ERROR(s.message());
    }
    Status unmarkDeleteNoExceptions(labeltype label) {
        Admission operation(&gate_, true, &read_phase_);
        if (!operation) return Status("Undelete requires a quiescent index");
        return graph_->unmarkDeleteNoExceptions(label);
    }
    void unmarkDelete(labeltype label) {
        Status s = unmarkDeleteNoExceptions(label);
        if (!s.ok()) HNSWLIB_THROW_RUNTIME_ERROR(s.message());
    }

    Status updatePointNoExceptions(const void* data, labeltype label) {
        Admission operation(&gate_, true, &read_phase_);
        if (!operation) return Status("Update requires a quiescent index");
        tableint id = graph_->getInternalIdByLabel(label);
        if (id == kInvalidInternalId || !data) return Status("Invalid label or update data");
        cleanup();
        syncRoot();
        id = graph_->getInternalIdByLabel(label);
        SavedStates flags(*this);
        return graph_->updatePoint(data, id, 1.0f);
    }

    Status resizeIndexNoExceptions(size_t) { return Status("Concurrent indexes have fixed capacity"); }
    void resizeIndex(size_t capacity) {
        Status s = resizeIndexNoExceptions(capacity);
        HNSWLIB_THROW_RUNTIME_ERROR(s.message());
    }

    Status saveIndexNoExceptions(std::ostream& output) {
        Admission operation(&gate_, true, &read_phase_);
        if (!operation) return Status("Save requires a quiescent index");
        cleanup();
        syncRoot();
        return save(output);
    }
    Status saveIndexNoExceptions(const std::string& location) override {
        // Obtain admission before opening/truncating the destination.
        Admission operation(&gate_, true, &read_phase_);
        if (!operation) return Status("Save requires a quiescent index");
        cleanup();
        syncRoot();
        std::ofstream output(location, std::ios::binary);
        if (!output.is_open()) return Status("Cannot save index: failed to open output file");
        return save(output);
    }
};

}  // namespace hnswlib
