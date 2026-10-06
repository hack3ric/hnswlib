#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace {
std::function<void(const char*, unsigned)> test_hook;
void hook(const char* event, unsigned id) { if (test_hook) test_hook(event, id); }
}
#define HNSWLIB_CONCURRENT_TEST_HOOK(event, id) hook(event, id)
#include "hnswlib/concurrent_hnsw.h"

namespace {
#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Requirement failed at " << __FILE__ << ':' << __LINE__ << ": " #condition << '\n'; \
    std::abort(); } } while (0)

using Index = hnswlib::ConcurrentHierarchicalNSW<float>;

class Pause {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false;
 public:
    void stop() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return released; });
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        REQUIRE(cv.wait_for(lock, std::chrono::seconds(10), [&] { return entered; }));
    }
    void resume() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
};

void publicationAndAdmission() {
    for (const char* stage : {"reserved", "initialized", "before_publish"}) {
        hnswlib::L2Space space(4);
        Index index(&space, 4);
        float a[4] = {}, b[4] = {1, 2, 3, 4};
        index.addPoint(a, 100, 0);
        Pause pause;
        test_hook = [&](const char* event, unsigned id) {
            if (id == 1 && std::strcmp(event, stage) == 0) pause.stop();
        };
        std::thread writer([&] { index.addPoint(b, 200, 1); });
        pause.wait();
        auto result = index.searchKnn(b, 4);
        REQUIRE(result.size() == 1 && result.top().second == 100);
        REQUIRE(index.getCurrentElementCount() == 1);
        // Reservation still holds the lookup mutex until initialization ends.
        if (std::strcmp(stage, "reserved") != 0)
            REQUIRE(!index.getDataByLabelNoExceptions<float>(200).ok());
        REQUIRE(!index.markDeleteNoExceptions(100).ok());
        REQUIRE(!index.unmarkDeleteNoExceptions(100).ok());
        REQUIRE(!index.updatePointNoExceptions(b, 100).ok());
        REQUIRE(!index.cleanupFailedInsertions().ok());
        REQUIRE(!index.checkIntegrityNoExceptions().ok());
        std::ostringstream output;
        REQUIRE(!index.saveIndexNoExceptions(output).ok());
        REQUIRE(output.str().empty());
        pause.resume();
        writer.join();
        test_hook = {};
        REQUIRE(index.getCurrentElementCount() == 2);
        REQUIRE(index.searchKnn(b, 1).top().second == 200);
        REQUIRE(index.checkIntegrityNoExceptions().ok());
    }
}

void rootPromotionDuringSearch() {
    hnswlib::L2Space space(1);
    Index index(&space, 4);
    float a = 0, b = 1;
    index.addPoint(&a, 100, 0);
    Pause publication, root_snapshot;
    test_hook = [&](const char* event, unsigned id) {
        if (id == 1 && std::strcmp(event, "before_publish") == 0) publication.stop();
        if (id == 0 && std::strcmp(event, "root_snapshot") == 0) root_snapshot.stop();
    };
    std::thread writer([&] { index.addPoint(&b, 200, 3); });
    publication.wait();
    std::thread reader([&] { REQUIRE(index.searchKnn(&b, 1).top().second == 200); });
    root_snapshot.wait();
    publication.resume();
    writer.join();
    // The reader retains the old root, whose level and payload remain valid
    // after a higher root has been published by the overlapping writer.
    root_snapshot.resume();
    reader.join();
    test_hook = {};
    REQUIRE(index.checkIntegrityNoExceptions().ok());
}

void maintenanceExcludesNewReaders() {
    hnswlib::L2Space space(1);
    Index index(&space, 4);
    float value = 1;
    index.addPoint(&value, 100);
    Pause writing;
    struct Buffer : std::stringbuf {
        Pause& pause;
        bool first = true;
        explicit Buffer(Pause& p) : pause(p) {}
        std::streamsize xsputn(const char* data, std::streamsize size) override {
            if (first) { first = false; pause.stop(); }
            return std::stringbuf::xsputn(data, size);
        }
    } buffer(writing);
    std::ostream output(&buffer);
    std::thread saver([&] { REQUIRE(index.saveIndexNoExceptions(output).ok()); });
    writing.wait();
    REQUIRE(!index.searchKnnNoExceptions(&value, 1).ok());
    hnswlib::EpsilonSearchStopCondition<float> stop(1, 1, 4);
    REQUIRE(!index.searchStopConditionClosestNoExceptions(&value, stop).ok());
    REQUIRE(!index.getDataByLabelNoExceptions<float>(100).ok());
    REQUIRE(!index.addPointNoExceptions(&value, 200).ok());
    writing.resume();
    saver.join();
    REQUIRE(index.searchKnn(&value, 1).top().second == 100);
}

void simultaneousFirstInsertions() {
    hnswlib::L2Space space(1);
    Index index(&space, 32);
    Pause pause;
    test_hook = [&](const char* event, unsigned id) {
        if (!id && std::strcmp(event, "initialized") == 0) pause.stop();
    };
    float a = 0, b = 1;
    std::thread writer([&] { index.addPoint(&a, 0, 2); });
    pause.wait();
    REQUIRE(index.searchKnn(&a, 1).empty());
    index.addPoint(&b, 1, 1);
    REQUIRE(index.searchKnn(&b, 1).top().second == 1);
    pause.resume();
    writer.join();
    test_hook = {};
    REQUIRE(index.searchKnn(&a, 1).top().second == 0);
    REQUIRE(index.checkIntegrityNoExceptions().ok());
}

void conflictingBacklinks() {
    hnswlib::L2Space space(4);
    Index index(&space, 16, 2, 20);
    float root[4] = {}, a[4] = {0.01f, 0, 0, 0}, b[4] = {-0.01f, 0, 0, 0};
    index.addPoint(root, 100, 0);
    // Fill the root's level-zero adjacency (2*M), forcing optimistic pruning
    // rather than the simple append path for both new points.
    for (int i = 0; i < 4; ++i) {
        float axis[4] = {};
        axis[i] = 1;
        index.addPoint(axis, 101 + i, 0);
    }
    Pause pause;
    std::atomic<bool> once{false};
    test_hook = [&](const char* event, unsigned id) {
        if (id == 5 && std::strcmp(event, "backlink_snapshot") == 0 && !once.exchange(true)) pause.stop();
    };
    std::thread writer([&] { index.addPoint(a, 105, 0); });
    pause.wait();
    index.addPoint(b, 106, 0);
    pause.resume();
    writer.join();
    test_hook = {};
    REQUIRE(index.searchKnn(root, 1).top().second == 100);
    REQUIRE(index.searchKnn(a, 1).top().second == 105);
    REQUIRE(index.searchKnn(b, 1).top().second == 106);
    REQUIRE(index.checkIntegrityNoExceptions().ok());
}

void readOnlyPhaseTransition() {
    hnswlib::L2Space space(1);
    Index index(&space, 8);
    float a = 0, b = 1;
    index.addPoint(&a, 0);
    Pause reader_pause;
    std::atomic<bool> writer_entered{false}, writer_initialized{false};
    test_hook = [&](const char* event, unsigned) {
        if (std::strcmp(event, "writer_entered") == 0) writer_entered = true;
        if (std::strcmp(event, "initialized") == 0) writer_initialized = true;
    };
    struct Filter : hnswlib::BaseFilterFunctor {
        Pause& pause;
        Index& index;
        float* vector;
        Filter(Pause& p, Index& i, float* v) : pause(p), index(i), vector(v) {}
        bool operator()(hnswlib::labeltype) override {
            // Mutation from a callback must return an error, not wait for this
            // very reader or its label-operation lock.
            REQUIRE(!index.addPointNoExceptions(vector, 100).ok());
            REQUIRE(!index.markDeleteNoExceptions(0).ok());
            REQUIRE(!index.unmarkDeleteNoExceptions(0).ok());
            REQUIRE(!index.updatePointNoExceptions(vector, 0).ok());
            REQUIRE(!index.cleanupFailedInsertions().ok());
            REQUIRE(!index.checkIntegrityNoExceptions().ok());
            std::ostringstream output;
            REQUIRE(!index.saveIndexNoExceptions(output).ok() && output.str().empty());
            pause.stop();
            return true;
        }
    } filter(reader_pause, index, &a);
    std::thread reader([&] { index.searchKnn(&a, 1, &filter); });
    reader_pause.wait();
    std::thread writer([&] { index.addPoint(&b, 1); });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!writer_entered.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    REQUIRE(writer_entered);
    REQUIRE(!writer_initialized);
    // A new search uses snapshots and completes even while the first writer
    // waits for a paused reader of the previous unlocked phase.
    REQUIRE(index.searchKnn(&a, 1).top().second == 0);
    reader_pause.resume();
    reader.join(); writer.join();
    test_hook = {};
    REQUIRE(writer_initialized);
    REQUIRE(index.searchKnn(&b, 1).top().second == 1);
    REQUIRE(index.checkIntegrityNoExceptions().ok());
}

void soleBuilderTransition() {
    hnswlib::L2Space space(1);
    Index index(&space, 8);
    float a = 0, b = 1, c = 2;
    index.addPoint(&a, 0, 0);
    Pause preparation;
    std::atomic<unsigned> entered{0};
    std::atomic<bool> second_initialized{false};
    test_hook = [&](const char* event, unsigned id) {
        if (std::strcmp(event, "writer_entered") == 0) ++entered;
        if (id == 1 && std::strcmp(event, "prepare") == 0) preparation.stop();
        if (id == 2 && std::strcmp(event, "initialized") == 0) second_initialized = true;
    };
    std::thread first([&] { index.addPoint(&b, 1, 0); });
    preparation.wait();
    std::thread second([&] { index.addPoint(&c, 2, 0); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (entered < 2 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    REQUIRE(entered == 2 && !second_initialized);
    REQUIRE(index.searchKnn(&a, 3).size() == 1);
    preparation.resume();
    first.join(); second.join();
    test_hook = {};
    REQUIRE(second_initialized);
    REQUIRE(index.searchKnn(&a, 3).size() == 3);
    REQUIRE(index.checkIntegrityNoExceptions().ok());
}

void duplicateAndCapacity() {
    hnswlib::L2Space space(1);
    hnswlib::ConcurrentIndexOptions options;
    options.upper_layer_capacity = 1;
    options.max_level = 2;
    Index index(&space, 2, 4, 16, 100, options);
    float a = 0, b = 1;
    REQUIRE(!index.addPointWithLevel(&a, 9, 3).ok());
    REQUIRE(index.getCurrentElementCount() == 0);
    REQUIRE(index.addPointWithLevel(&a, 10, 1).ok());
    REQUIRE(!index.addPointWithLevel(&b, 11, 1).ok());
    REQUIRE(index.addPointWithLevel(&b, 11, 0).ok());
    REQUIRE(!index.addPointNoExceptions(&a, 12).ok());
    REQUIRE(!index.addPointNoExceptions(&b, 10).ok());
    REQUIRE(!index.addPointNoExceptions(&b, 10, true).ok());
    REQUIRE(!index.resizeIndexNoExceptions(4).ok());
    REQUIRE(index.getDataByLabel<float>(10)[0] == a);

    Index duplicate(&space, 16);
    std::atomic<unsigned> successes{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back([&] {
        if (duplicate.addPointNoExceptions(&a, 42).ok()) ++successes;
    });
    for (auto& thread : threads) thread.join();
    REQUIRE(successes == 1 && duplicate.getCurrentElementCount() == 1);
    REQUIRE(duplicate.checkIntegrityNoExceptions().ok());
}

#if defined(__EXCEPTIONS) || _HAS_EXCEPTIONS == 1
void failedInsertionsAndCallbacks() {
    hnswlib::L2Space space(1);
    Index index(&space, 4);
    float root = 0, failed = 1, good = 2;
    index.addPoint(&root, 10, 1);
    for (const char* stage : {"initialized", "layer_linked", "before_publish"}) {
        test_hook = [&](const char* event, unsigned) {
            if (std::strcmp(event, stage) == 0) throw std::runtime_error("injected failure");
        };
        REQUIRE(!index.addPointWithLevel(&failed, 11, 1).ok());
        test_hook = {};
        REQUIRE(index.getFailedInsertionCount() == 1);
        REQUIRE(index.getCurrentElementCount() == 1);
        REQUIRE(index.searchKnn(&failed, 3).size() == 1);
        REQUIRE(!index.getDataByLabelNoExceptions<float>(11).ok());
        // Cleanup must wait for readers that retained an exposed failed node.
        Pause pause;
        struct BlockingFilter : hnswlib::BaseFilterFunctor {
            Pause& pause;
            explicit BlockingFilter(Pause& p) : pause(p) {}
            bool operator()(hnswlib::labeltype) override { pause.stop(); return true; }
        } filter(pause);
        std::thread reader([&] { index.searchKnn(&failed, 1, &filter); });
        pause.wait();
        REQUIRE(!index.cleanupFailedInsertions().ok());
        pause.resume();
        reader.join();
        REQUIRE(index.cleanupFailedInsertions().ok());
        REQUIRE(index.getFailedInsertionCount() == 0);
        REQUIRE(index.checkIntegrityNoExceptions().ok());
    }
    index.addPoint(&failed, 11, 1);
    index.addPoint(&good, 12, 1);
    REQUIRE(index.searchKnn(&good, 3).size() == 3);
    struct ThrowFilter : hnswlib::BaseFilterFunctor {
        bool operator()(hnswlib::labeltype) override { throw std::runtime_error("filter failure"); }
    } filter;
    for (int i = 0; i < 10; ++i) {
        bool caught = false;
        try { index.searchKnn(&good, 1, &filter); } catch (const std::runtime_error&) { caught = true; }
        REQUIRE(caught);
    }
    REQUIRE(index.cleanupFailedInsertions().ok());
    REQUIRE(index.searchKnn(&good, 1).top().second == 12);
}

void compactFailedHoles() {
    hnswlib::L2Space space(1);
    Index index(&space, 16);
    for (unsigned i = 0; i < 16; ++i) {
        float value = static_cast<float>(i);
        const bool fail = i % 4 == 1;
        test_hook = [=](const char* event, unsigned) {
            if (fail && std::strcmp(event, "before_publish") == 0) throw std::runtime_error("hole");
        };
        REQUIRE(index.addPointWithLevel(&value, 1000 + i, i % 3).ok() != fail);
        test_hook = {};
    }
    REQUIRE(index.getCurrentElementCount() == 12);
    REQUIRE(index.getFailedInsertionCount() == 4);
    float extra = 20;
    REQUIRE(!index.addPointNoExceptions(&extra, 2000).ok());
    // Save performs the same exclusive cleanup, including moving towers for
    // live nodes following failed slots and remapping the root and labels.
    std::ostringstream saved;
    REQUIRE(index.saveIndexNoExceptions(saved).ok());
    REQUIRE(index.getFailedInsertionCount() == 0);
    REQUIRE(index.checkIntegrityNoExceptions().ok());
    for (unsigned i = 0; i < 16; ++i) {
        if (i % 4 == 1) continue;
        float query = static_cast<float>(i);
        REQUIRE(index.getDataByLabel<float>(1000 + i)[0] == query);
        REQUIRE(index.searchKnn(&query, 1).top().second == 1000 + i);
    }
    for (unsigned i = 0; i < 4; ++i) {
        float value = 20.0f + i;
        REQUIRE(index.addPointWithLevel(&value, 2000 + i, i % 3).ok());
    }
    REQUIRE(index.getCurrentElementCount() == 16);
    REQUIRE(index.checkIntegrityNoExceptions().ok());
}

void failedFirstInsertion() {
    hnswlib::L2Space space(1);
    Index index(&space, 1);
    float value = 1;
    test_hook = [](const char* event, unsigned) {
        if (std::strcmp(event, "initialized") == 0) throw std::runtime_error("first node");
    };
    REQUIRE(!index.addPointNoExceptions(&value, 123).ok());
    test_hook = {};
    REQUIRE(index.searchKnn(&value, 1).empty());
    REQUIRE(index.getCurrentElementCount() == 0 && index.getFailedInsertionCount() == 1);
    REQUIRE(index.cleanupFailedInsertions().ok());
    REQUIRE(index.checkIntegrityNoExceptions().ok());
    REQUIRE(index.addPointNoExceptions(&value, 123).ok());
    REQUIRE(index.searchKnn(&value, 1).top().second == 123);
}
#endif

void persistenceAndQuiescentMutation() {
    hnswlib::L2Space space(2);
    const char* path = "concurrent_search_insert_test.bin";
    float a[2] = {}, b[2] = {2, 2}, c[2] = {4, 4};
    {
        Index index(&space, 8);
        index.addPoint(a, 100);
        index.addPoint(b, 200);
        index.markDelete(100);
        REQUIRE(index.searchKnn(a, 2).size() == 1);
        index.unmarkDelete(100);
        REQUIRE(index.updatePointNoExceptions(c, 200).ok());
        REQUIRE(index.getDataByLabel<float>(200)[0] == 4);
        struct FailedBuffer : std::streambuf {
            std::streamsize xsputn(const char*, std::streamsize) override { return 0; }
            int_type overflow(int_type) override { return traits_type::eof(); }
        } failed_buffer;
        std::ostream failed_output(&failed_buffer);
        REQUIRE(!index.saveIndexNoExceptions(failed_output).ok());
        REQUIRE(index.searchKnn(c, 1).top().second == 200);
        index.saveIndex(path);
    }
    {
        hnswlib::HierarchicalNSW<float> legacy(&space, std::string(path));
        REQUIRE(legacy.getCurrentElementCount() == 2);
        REQUIRE(legacy.searchKnn(c, 1).top().second == 200);
        legacy.addPoint(b, 300);
        legacy.saveIndex(path);
    }
    Index index(&space, std::string(path), 12);
    REQUIRE(index.getMaxElements() == 12);
    REQUIRE(index.getCurrentElementCount() == 3);
    REQUIRE(index.searchKnn(c, 1).top().second == 200);
    float d[2] = {8, 8};
    index.addPoint(d, 400);
    const Index& constant = index;
    REQUIRE(constant.searchKnnCloserFirst(d, 2).front().second == 400);
    hnswlib::EpsilonSearchStopCondition<float> stop(1000, 1, 12);
    auto results = constant.searchStopConditionClosest(d, stop);
    REQUIRE(!results.empty());
    for (auto item : results) REQUIRE(item.second >= 100);  // external labels, not internal IDs
    REQUIRE(index.checkIntegrityNoExceptions().ok());
    std::remove(path);
}


void mixedStress() {
    const size_t n = 2000, dim = 8, writers = 4;
    hnswlib::L2Space space(dim);
    Index index(&space, n, 12, 80);
    std::vector<float> data(n * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> random(0, 1);
    for (auto& x : data) x = random(rng);
    std::atomic<size_t> next{0}, remaining{writers}, queries{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (size_t w = 0; w < writers; ++w) threads.emplace_back([&] {
        while (!start.load()) std::this_thread::yield();
        for (;;) {
            size_t i = next.fetch_add(1);
            if (i >= n) break;
            REQUIRE(index.addPointNoExceptions(data.data() + i * dim, i + 10000).ok());
        }
        --remaining;
    });
    for (size_t r = 0; r < 4; ++r) threads.emplace_back([&, r] {
        while (!start.load()) std::this_thread::yield();
        size_t iteration = r;
        do {
            const float* query = data.data() + (iteration++ % n) * dim;
            struct EvenFilter : hnswlib::BaseFilterFunctor {
                bool operator()(hnswlib::labeltype label) override { return label % 2 == 0; }
            } filter;
            auto results = index.searchKnn(query, 10, r == 3 ? &filter : nullptr);
            if (r == 2 && iteration % 8 == 0) {
                hnswlib::EpsilonSearchStopCondition<float> stop(0.5f, 1, 20);
                for (auto item : index.searchStopConditionClosest(query, stop)) results.push(item);
            }
            while (!results.empty()) {
                auto item = results.top(); results.pop();
                REQUIRE(item.second >= 10000 && item.second < n + 10000);
                if (r == 3) REQUIRE(item.second % 2 == 0);
                float expected = 0;
                for (size_t d = 0; d < dim; ++d) {
                    float delta = query[d] - data[(item.second - 10000) * dim + d];
                    expected += delta * delta;
                }
                REQUIRE(std::abs(expected - item.first) < 1e-4f);
            }
            ++queries;
        } while (remaining.load());
    });
    start.store(true);
    for (auto& thread : threads) thread.join();
    REQUIRE(queries > 0);
    REQUIRE(index.getCurrentElementCount() == n);
    REQUIRE(index.checkIntegrityNoExceptions().ok());
    size_t found = 0;
    for (size_t i = 0; i < n; ++i) found += index.searchKnn(data.data() + i * dim, 1).top().second == i + 10000;
    REQUIRE(found > n * 0.98);
    std::cout << "mixed queries=" << queries << " self recall=" << double(found) / n << '\n';
}

void additionalSpaces() {
    const size_t n = 400, dim = 8;
    hnswlib::InnerProductSpace inner_product(dim);
    hnswlib::MultiVectorL2Space<unsigned> multivector(dim);
    for (bool documents : {false, true}) {
        hnswlib::SpaceInterface<float>* space = documents
            ? static_cast<hnswlib::SpaceInterface<float>*>(&multivector) : &inner_product;
        const size_t stride = space->get_data_size();
        std::vector<char> data(n * stride);
        std::mt19937 rng(71);
        std::uniform_real_distribution<float> random(0, 1);
        for (size_t i = 0; i < n; ++i) {
            for (size_t d = 0; d < dim; ++d) {
                const float value = random(rng);
                memcpy(data.data() + i * stride + d * sizeof(float), &value, sizeof(value));
            }
            if (documents) multivector.set_doc_id(data.data() + i * stride, static_cast<unsigned>(i % 20));
        }
        Index index(space, n, 12, 80);
        index.setEf(n);
        for (size_t i = 0; i < 40; ++i) index.addPoint(data.data() + i * stride, i + 10000);
        std::atomic<bool> done{false};
        std::atomic<size_t> queries{0};
        std::thread reader([&] {
            do {
                const char* query = data.data() + (queries.load() % n) * stride;
                Index::DistanceLabelVector result;
                if (documents) {
                    hnswlib::MultiVectorSearchStopCondition<unsigned, float> stop(multivector, 5, 20);
                    result = index.searchStopConditionClosest(query, stop);
                } else {
                    result = index.searchKnnCloserFirst(query, 10);
                }
                REQUIRE(!result.empty());
                std::unordered_set<unsigned> found_documents;
                for (const auto& item : result) {
                    REQUIRE(item.second >= 10000 && item.second < 10000 + n);
                    const char* point = data.data() + (item.second - 10000) * stride;
                    float expected = documents ? 0.0f : 1.0f;
                    for (size_t d = 0; d < dim; ++d) {
                        float a, b;
                        memcpy(&a, query + d * sizeof(float), sizeof(float));
                        memcpy(&b, point + d * sizeof(float), sizeof(float));
                        expected += documents ? (a - b) * (a - b) : -a * b;
                    }
                    REQUIRE(std::abs(expected - item.first) < 1e-4f);
                    if (documents) found_documents.insert(multivector.get_doc_id(point));
                }
                if (documents) REQUIRE(found_documents.size() == 5);
                ++queries;
            } while (!done.load());
        });
        while (!queries.load()) std::this_thread::yield();
        for (size_t i = 40; i < n; ++i) index.addPoint(data.data() + i * stride, i + 10000);
        done.store(true);
        reader.join();
        REQUIRE(queries > 1);
        REQUIRE(index.checkIntegrityNoExceptions().ok());
        size_t correct = 0;
        for (size_t q = 0; q < 20; ++q) {
            const char* query = data.data() + q * stride;
            // Keep the oracle independent of packed-label storage: the legacy
            // brute-force type performs unaligned label loads for doc payloads.
            Index::DistanceLabelVector truth;
            for (size_t i = 0; i < n; ++i)
                truth.emplace_back(space->get_dist_func()(query, data.data() + i * stride,
                                                         space->get_dist_func_param()), i + 10000);
            std::sort(truth.begin(), truth.end());
            truth.resize(10);
            for (const auto& found : index.searchKnnCloserFirst(query, 10))
                for (const auto& expected : truth) correct += found.second == expected.second;
        }
        std::cout << (documents ? "multivector" : "inner product") << " recall=" << double(correct) / 200 << '\n';
        REQUIRE(correct >= 190);  // ANN discovery, separate from exact distance validation above.
    }
}
}  // namespace

int main() {
    publicationAndAdmission();
    rootPromotionDuringSearch();
    maintenanceExcludesNewReaders();
    simultaneousFirstInsertions();
    conflictingBacklinks();
    readOnlyPhaseTransition();
    soleBuilderTransition();
    duplicateAndCapacity();
#if defined(__EXCEPTIONS) || _HAS_EXCEPTIONS == 1
    failedInsertionsAndCallbacks();
    compactFailedHoles();
    failedFirstInsertion();
#endif
    persistenceAndQuiescentMutation();
    mixedStress();
    additionalSpaces();
    std::cout << "Concurrent search/insert tests passed\n";
}
