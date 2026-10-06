#include "hnswlib/concurrent_hnsw.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>

using Index = hnswlib::ConcurrentHierarchicalNSW<float>;

#define REQUIRE(condition) do { if (!(condition)) { \
    std::cerr << "Requirement failed at line " << __LINE__ << ": " #condition << '\n'; \
    std::abort(); } } while (0)

// Kept separate from the mixed-workload TSan executable: constructing the
// legacy reference exercises its pre-existing lock-order inversions even on a
// single thread. No concurrent-index sanitizer findings are suppressed.
void serialGraphCompatibility() {
    const size_t count = 500, dim = 16;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> legacy(&space, count + 20, 8, 60, 100);
    Index concurrent(&space, count + 20, 8, 60, 100);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> random(0, 1);
    std::vector<float> vector(dim);
    for (size_t i = 0; i < count; ++i) {
        for (auto& x : vector) x = random(rng);
        legacy.addPoint(vector.data(), i);
        concurrent.addPoint(vector.data(), i);
    }
    std::ostringstream a, b;
    REQUIRE(legacy.saveIndexNoExceptions(a).ok());
    REQUIRE(concurrent.saveIndexNoExceptions(b).ok());
    REQUIRE(a.str() == b.str());
    const auto root_label = legacy.getExternalLabel(legacy.enterpoint_node_);
    legacy.markDelete(root_label);
    concurrent.markDelete(root_label);
    for (size_t i = count; i < count + 20; ++i) {
        for (auto& x : vector) x = random(rng);
        legacy.addPoint(vector.data(), i);
        concurrent.addPoint(vector.data(), i);
    }
    a.str(""); b.str("");
    REQUIRE(legacy.saveIndexNoExceptions(a).ok());
    REQUIRE(concurrent.saveIndexNoExceptions(b).ok());
    REQUIRE(a.str() == b.str());
}

void metricCompatibility() {
    const size_t count = 400, dim = 8;
    hnswlib::InnerProductSpace inner_product(dim);
    hnswlib::MultiVectorL2Space<unsigned> multivector(dim);
    for (bool documents : {false, true}) {
        hnswlib::SpaceInterface<float>* space = documents
            ? static_cast<hnswlib::SpaceInterface<float>*>(&multivector) : &inner_product;
        hnswlib::HierarchicalNSW<float> legacy(space, count, 12, 80);
        Index concurrent(space, count, 12, 80);
        const size_t stride = space->get_data_size();
        std::vector<char> data(count * stride);
        std::mt19937 rng(71);
        std::uniform_real_distribution<float> random(0, 1);
        for (size_t i = 0; i < count; ++i) {
            char* point = data.data() + i * stride;
            for (size_t d = 0; d < dim; ++d) {
                float value = random(rng);
                memcpy(point + d * sizeof(float), &value, sizeof(value));
            }
            if (documents) multivector.set_doc_id(point, static_cast<unsigned>(i % 20));
            legacy.addPoint(point, i + 10000);
            concurrent.addPoint(point, i + 10000);
        }
        std::ostringstream a, b;
        REQUIRE(legacy.saveIndexNoExceptions(a).ok());
        REQUIRE(concurrent.saveIndexNoExceptions(b).ok());
        REQUIRE(a.str() == b.str());
        legacy.setEf(count);
        concurrent.setEf(count);
        for (size_t q = 0; q < 20; ++q) {
            const void* query = data.data() + q * stride;
            REQUIRE(legacy.searchKnnCloserFirst(query, 10) == concurrent.searchKnnCloserFirst(query, 10));
        }
    }
}

int main() {
    serialGraphCompatibility();
    metricCompatibility();
    std::cout << "Legacy and concurrent serial graphs are byte-for-byte identical\n";
}
