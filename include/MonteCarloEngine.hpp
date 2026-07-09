#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <thread>
#include <vector>

#include "MarketData.hpp"

// Monte Carlo pricer for a European option under Geometric Brownian Motion.
//
// Because a European payoff only depends on the terminal spot price, the
// GBM SDE can be solved in closed form for a single time step:
//   S_T = S0 * exp((r - 0.5*vol^2) * T + vol*sqrt(T) * Z),   Z ~ N(0,1)
// The Monte Carlo estimate of the price is the sample mean of the
// discounted payoff, e^{-rT} * E[payoff(S_T)], which converges to the
// Black-Scholes price as the number of paths grows.
namespace mc {

// Cache-line padded accumulator so that per-thread partial sums never
// share a cache line (avoids false sharing between worker threads).
struct alignas(64) PaddedSum {
    double value = 0.0;
};

// Simulates `numPaths` terminal spot prices and accumulates the discounted
// payoff sum into `sumOut`. Uses a thread-local RNG so each worker has its
// own independent Mersenne Twister stream.
template <typename PayoffT>
inline void simulateChunk(const PayoffT& payoff, const MarketData& mkt,
                           unsigned long numPaths, unsigned int seed,
                           double& sumOut) {
    const double variance = mkt.vol * mkt.vol * mkt.expiry;
    const double rootVariance = std::sqrt(variance);
    const double itoDrift = (mkt.rate - 0.5 * mkt.vol * mkt.vol) * mkt.expiry;
    const double movedSpot = mkt.spot * std::exp(itoDrift);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gaussian(0.0, 1.0);

    double runningSum = 0.0;
    for (unsigned long i = 0; i < numPaths; ++i) {
        const double z = gaussian(rng);
        const double terminalSpot = movedSpot * std::exp(rootVariance * z);
        runningSum += payoff(terminalSpot);
    }

    sumOut = runningSum;
}

// Single-threaded European option pricer (baseline for benchmarking).
template <typename PayoffT>
double priceSingleThreaded(const PayoffT& payoff, const MarketData& mkt,
                            unsigned long numPaths, unsigned int baseSeed = 42) {
    double sum = 0.0;
    simulateChunk(payoff, mkt, numPaths, baseSeed, sum);
    return (sum / static_cast<double>(numPaths)) * std::exp(-mkt.rate * mkt.expiry);
}

// Multithreaded European option pricer. Splits the path count into
// contiguous, roughly equal chunks across `numThreads` worker threads.
// Each thread writes into its own padded accumulator to avoid false
// sharing, and the partial sums are reduced once all threads join.
template <typename PayoffT>
double priceMultiThreaded(const PayoffT& payoff, const MarketData& mkt,
                           unsigned long numPaths, unsigned int numThreads,
                           unsigned int baseSeed = 42) {
    if (numThreads == 0) numThreads = 1;
    if (numThreads > numPaths) numThreads = static_cast<unsigned int>(numPaths);

    std::vector<PaddedSum> partialSums(numThreads);
    std::vector<std::thread> workers;
    workers.reserve(numThreads);

    const unsigned long basePathsPerThread = numPaths / numThreads;
    const unsigned long remainder = numPaths % numThreads;

    for (unsigned int t = 0; t < numThreads; ++t) {
        const unsigned long pathsForThisThread =
            basePathsPerThread + (t < remainder ? 1 : 0);
        // Distinct seed per thread so RNG streams don't overlap.
        const unsigned int threadSeed = baseSeed + t * 7919u + 1u;

        workers.emplace_back([&, pathsForThisThread, threadSeed, t]() {
            simulateChunk(payoff, mkt, pathsForThisThread, threadSeed,
                          partialSums[t].value);
        });
    }

    for (auto& w : workers) w.join();

    double totalSum = 0.0;
    for (const auto& p : partialSums) totalSum += p.value;

    return (totalSum / static_cast<double>(numPaths)) * std::exp(-mkt.rate * mkt.expiry);
}

}  // namespace mc
