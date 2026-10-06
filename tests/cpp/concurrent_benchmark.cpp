// Build with -DHNSWLIB_BENCH_LEGACY_ONLY against an unmodified checkout for the
// baseline. Uses the same uniform-vector, build/query/recall approach as the
// existing Python speed and recall tests, with an exact independent-query oracle.
#ifdef HNSWLIB_BENCH_LOCK_WAIT
void benchmarkLockHook(const char* event, unsigned id);
#define HNSWLIB_CONCURRENT_TEST_HOOK(event, id) benchmarkLockHook(event, id)
#endif
#include "hnswlib/hnswlib.h"
#ifndef HNSWLIB_BENCH_LEGACY_ONLY
#include "hnswlib/concurrent_hnsw.h"
#endif
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>

using Clock = std::chrono::steady_clock;

#ifdef HNSWLIB_BENCH_LOCK_WAIT
// Instrumented runs quantify adjacency-lock acquisition separately from the
// uninstrumented throughput/latency comparison. Aggregate once per worker exit
// so the measurement itself does not add a shared counter to every lock.
std::atomic<uint64_t> lock_count{0}, lock_nanoseconds{0}, slow_lock_count{0};
struct ThreadLockStats {
    uint64_t count = 0, nanoseconds = 0, slow = 0;
    Clock::time_point begin;
    ~ThreadLockStats() {
        lock_count.fetch_add(count, std::memory_order_relaxed);
        lock_nanoseconds.fetch_add(nanoseconds, std::memory_order_relaxed);
        slow_lock_count.fetch_add(slow, std::memory_order_relaxed);
    }
};
void benchmarkLockHook(const char* event, unsigned) {
    static thread_local ThreadLockStats stats;
    if (std::strcmp(event, "adjacency_lock_begin") == 0) stats.begin = Clock::now();
    else if (std::strcmp(event, "adjacency_lock_acquired") == 0) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - stats.begin).count();
        ++stats.count;
        stats.nanoseconds += elapsed;
        stats.slow += elapsed >= 1000;
    }
}
#endif

template<class Function>
void parallel(size_t count, size_t threads, Function fn) {
    std::atomic<size_t> next{0};
    std::vector<std::thread> workers;
    for (size_t t = 0; t < threads; ++t) workers.emplace_back([&] {
        for (;;) {
            size_t begin = next.fetch_add(16, std::memory_order_relaxed);
            if (begin >= count) break;
            for (size_t i = begin; i < std::min(count, begin + 16); ++i) fn(i);
        }
    });
    for (auto& w : workers) w.join();
}

template<class Index>
void benchmark(size_t n, size_t dim, size_t threads, size_t ef, unsigned seed, size_t build_threads) {
    const size_t nq = 10000, k = 10;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> random(0, 1);
    std::vector<float> data(n * dim), queries(nq * dim);
    for (auto& v : data) v = random(generator);
    for (auto& v : queries) v = random(generator);
    hnswlib::L2Space space(dim);
    Index index(&space, n, 16, 200, 100);
    const auto build_start = Clock::now();
    parallel(n, build_threads, [&](size_t i) { index.addPoint(data.data() + i * dim, i); });
    const double build_seconds = std::chrono::duration<double>(Clock::now() - build_start).count();
    index.setEf(ef);
    std::vector<std::vector<std::pair<float, hnswlib::labeltype>>> results(nq);
    std::vector<double> timings;
    std::vector<double> latencies(nq);
    for (size_t repeat = 0; repeat < 6; ++repeat) {
        const auto begin = Clock::now();
        parallel(nq, threads, [&](size_t i) {
            const auto start = Clock::now();
            results[i] = index.searchKnnCloserFirst(queries.data() + i * dim, k);
            latencies[i] = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
        });
        if (repeat) timings.push_back(std::chrono::duration<double>(Clock::now() - begin).count());
    }
    std::sort(timings.begin(), timings.end());
    std::sort(latencies.begin(), latencies.end());
    hnswlib::BruteforceSearch<float> exact(&space, n);
    for (size_t i = 0; i < n; ++i) exact.addPoint(data.data() + i * dim, i);
    size_t correct = 0;
    // Exact recall on 100 independent queries, outside the measured region.
    for (size_t q = 0; q < 100; ++q) {
        auto truth = exact.searchKnn(queries.data() + q * dim, k);
        while (!truth.empty()) {
            for (auto item : results[q]) correct += item.second == truth.top().second;
            truth.pop();
        }
    }
    std::cout << std::setprecision(8) << "n=" << n << " dim=" << dim << " threads=" << threads
              << " build_threads=" << build_threads << " ef=" << ef << " seed=" << seed << " build_s=" << build_seconds
              << " query_s=" << timings[timings.size()/2]
              << " qps=" << nq / timings[timings.size()/2]
              << " p99_us=" << latencies[nq*99/100]
              << " recall=" << double(correct) / (100*k) << '\n';
}

#ifndef HNSWLIB_BENCH_LEGACY_ONLY
void mixedBenchmark(size_t n, size_t dim, size_t threads, size_t ef, unsigned seed) {
    const size_t nq = 1000, k = 10;
    const size_t writers = std::max(size_t(1), threads / 4), readers = threads - writers;
    if (!readers) { std::cerr << "mixed benchmark needs at least two threads\n"; std::exit(1); }
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> random(0, 1);
    std::vector<float> data(n * dim), queries(nq * dim);
    for (auto& v : data) v = random(generator);
    for (auto& v : queries) v = random(generator);
    hnswlib::L2Space space(dim);
    hnswlib::ConcurrentHierarchicalNSW<float> index(&space, n, 16, 200, 100);
    for (size_t i = 0; i < n / 2; ++i) index.addPoint(data.data() + i * dim, i);
    index.setEf(ef);
    std::atomic<size_t> next{n / 2}, remaining{writers};
    std::atomic<bool> start{false};
    std::vector<std::vector<double>> query_latencies(readers), insert_latencies(writers);
    std::vector<std::thread> workers;
    for (size_t t = 0; t < writers; ++t) workers.emplace_back([&, t] {
        auto& latencies = insert_latencies[t];
        latencies.reserve(n / writers);
        while (!start.load()) std::this_thread::yield();
        for (;;) {
            size_t i = next.fetch_add(1);
            if (i >= n) break;
            const auto begin = Clock::now();
            index.addPoint(data.data() + i * dim, i);
            latencies.push_back(std::chrono::duration<double, std::micro>(Clock::now() - begin).count());
        }
        --remaining;
    });
    for (size_t t = 0; t < readers; ++t) workers.emplace_back([&, t] {
        auto& latencies = query_latencies[t];
        latencies.reserve(100000);
        while (!start.load()) std::this_thread::yield();
        size_t q = t;
        do {
            const auto begin = Clock::now();
            auto result = index.searchKnn(queries.data() + (q++ % nq) * dim, k);
            latencies.push_back(std::chrono::duration<double, std::micro>(Clock::now() - begin).count());
            if (result.empty() || result.top().second >= n) std::abort();
        } while (remaining.load());
    });
    const auto begin = Clock::now();
    start.store(true);
    for (auto& w : workers) w.join();
    const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    std::vector<double> qtimes, itimes;
    for (const auto& times : query_latencies) qtimes.insert(qtimes.end(), times.begin(), times.end());
    for (const auto& times : insert_latencies) itimes.insert(itimes.end(), times.begin(), times.end());
    std::sort(qtimes.begin(), qtimes.end());
    std::sort(itimes.begin(), itimes.end());
    hnswlib::BruteforceSearch<float> exact(&space, n);
    for (size_t i = 0; i < n; ++i) exact.addPoint(data.data() + i * dim, i);
    size_t correct = 0;
    for (size_t q = 0; q < 100; ++q) {
        auto truth = exact.searchKnn(queries.data() + q * dim, k);
        auto result = index.searchKnnCloserFirst(queries.data() + q * dim, k);
        while (!truth.empty()) {
            for (auto item : result) correct += item.second == truth.top().second;
            truth.pop();
        }
    }
    if (!index.checkIntegrityNoExceptions().ok()) std::abort();
    std::cout << std::setprecision(8) << "mixed n=" << n << " dim=" << dim << " readers=" << readers
              << " writers=" << writers << " ef=" << ef << " seed=" << seed
              << " seconds=" << seconds << " qps=" << qtimes.size() / seconds
              << " inserts_per_s=" << itimes.size() / seconds
              << " query_p99_us=" << qtimes[qtimes.size()*99/100]
              << " insert_p99_us=" << itimes[itimes.size()*99/100]
              << " final_recall=" << double(correct) / (100*k) << '\n';
#ifdef HNSWLIB_BENCH_LOCK_WAIT
    std::cout << "adjacency_locks=" << lock_count.load()
              << " acquisition_mean_ns=" << double(lock_nanoseconds.load()) / lock_count.load()
              << " acquisitions_over_1us=" << slow_lock_count.load() << '\n';
#endif
}
#endif

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "usage: concurrent_benchmark legacy|concurrent|mixed N dimension threads ef [seed [build_threads]]\n";
        return 1;
    }
    size_t n = std::stoul(argv[2]), dim = std::stoul(argv[3]), threads = std::stoul(argv[4]), ef = std::stoul(argv[5]);
    if (n < 10 || !dim || !threads || !ef) return 1;
    unsigned seed = argc > 6 ? static_cast<unsigned>(std::stoul(argv[6])) : 1;
    size_t build_threads = argc > 7 ? std::stoul(argv[7]) : threads;
    if (!build_threads) return 1;
    if (std::string(argv[1]) == "legacy") {
        benchmark<hnswlib::HierarchicalNSW<float>>(n, dim, threads, ef, seed, build_threads);
#ifndef HNSWLIB_BENCH_LEGACY_ONLY
    } else if (std::string(argv[1]) == "concurrent") {
        benchmark<hnswlib::ConcurrentHierarchicalNSW<float>>(n, dim, threads, ef, seed, build_threads);
    } else if (std::string(argv[1]) == "mixed") {
        mixedBenchmark(n, dim, threads, ef, seed);
#endif
    } else { return 1; }
}
