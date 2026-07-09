#pragma once

#include <cmath>
#include <random>
#include <vector>

#include "MarketData.hpp"

// Generates full GBM sample paths (spot price at every time step, not just
// the terminal value) purely for visualization purposes. The pricing engine
// itself never needs this -- European payoffs only depend on S_T -- but
// plotting a handful of full paths is useful to sanity-check the simulation.
namespace mc {

// Returns a flat, row-major buffer of shape (numPaths, numSteps + 1):
// paths[p * (numSteps + 1) + s] is the spot price of path p at step s.
// Flat/contiguous storage keeps the whole buffer cache-friendly versus a
// vector-of-vectors.
inline std::vector<double> generatePaths(const MarketData& mkt,
                                          unsigned long numPaths,
                                          unsigned int numSteps,
                                          unsigned int seed = 1234) {
    const double dt = mkt.expiry / static_cast<double>(numSteps);
    const double drift = (mkt.rate - 0.5 * mkt.vol * mkt.vol) * dt;
    const double diffusion = mkt.vol * std::sqrt(dt);

    const std::size_t stride = static_cast<std::size_t>(numSteps) + 1;
    std::vector<double> paths(static_cast<std::size_t>(numPaths) * stride);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gaussian(0.0, 1.0);

    for (unsigned long p = 0; p < numPaths; ++p) {
        double* row = &paths[static_cast<std::size_t>(p) * stride];
        row[0] = mkt.spot;
        for (unsigned int s = 1; s <= numSteps; ++s) {
            const double z = gaussian(rng);
            row[s] = row[s - 1] * std::exp(drift + diffusion * z);
        }
    }

    return paths;
}

}  // namespace mc
