#pragma once

// Parameters for a single European option under Geometric Brownian Motion,
// i.e. the Black-Scholes market model:
//   dS_t = r * S_t * dt + Vol * S_t * dW_t
struct MarketData {
    double spot;     // S0: current underlying price
    double strike;   // K: option strike price
    double rate;     // r: risk-free rate (continuously compounded)
    double vol;       // sigma: annualized volatility
    double expiry;   // T: time to expiry in years
};
