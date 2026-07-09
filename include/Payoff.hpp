#pragma once

#include <algorithm>

// European call/put payoffs as small function objects so the compiler
// can inline them directly into the Monte Carlo hot loop (no vtable).
class CallPayoff {
public:
    explicit CallPayoff(double strike) : strike_(strike) {}

    inline double operator()(double spot) const noexcept {
        return std::max(spot - strike_, 0.0);
    }

private:
    double strike_;
};

class PutPayoff {
public:
    explicit PutPayoff(double strike) : strike_(strike) {}

    inline double operator()(double spot) const noexcept {
        return std::max(strike_ - spot, 0.0);
    }

private:
    double strike_;
};
