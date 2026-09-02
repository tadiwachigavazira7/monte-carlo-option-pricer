#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "MarketData.hpp"
#include "MonteCarloEngine.hpp"
#include "Payoff.hpp"
#include "PathSimulator.hpp"

namespace {

struct Args {
    MarketData mkt{100.0, 100.0, 0.05, 0.20, 1.0};
    unsigned long numPaths = 5'000'000UL;
    unsigned int numThreads = std::thread::hardware_concurrency();
    unsigned int seed = 42;
    bool benchmark = false;
    bool exportData = false;
    std::string outDir = "data";
};

double parseDouble(const char* s) {
    return std::stod(s);
}

Args parseArgs(int argc, char** argv) {
    Args args;

    if (args.numThreads == 0) {
        args.numThreads = 4;
    }

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];

        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                std::exit(1);
            }

            return argv[++i];
        };

        if (flag == "--spot") {
            args.mkt.spot = parseDouble(next());
        } else if (flag == "--strike") {
            args.mkt.strike = parseDouble(next());
        } else if (flag == "--rate") {
            args.mkt.rate = parseDouble(next());
        } else if (flag == "--vol") {
            args.mkt.vol = parseDouble(next());
        } else if (flag == "--expiry") {
            args.mkt.expiry = parseDouble(next());
        } else if (flag == "--paths") {
            args.numPaths = std::stoul(next());
        } else if (flag == "--threads") {
            args.numThreads =
                static_cast<unsigned int>(std::stoul(next()));
        } else if (flag == "--seed") {
            args.seed =
                static_cast<unsigned int>(std::stoul(next()));
        } else if (flag == "--benchmark") {
            args.benchmark = true;
        } else if (flag == "--export") {
            args.exportData = true;
        } else if (flag == "--out") {
            args.outDir = next();
        } else if (flag == "--help") {
            std::cout
                << "Usage: monte_carlo_pricer [options]\n"
                << "  --spot X       underlying spot price (default 100)\n"
                << "  --strike X     strike price (default 100)\n"
                << "  --rate X       risk-free rate (default 0.05)\n"
                << "  --vol X        volatility (default 0.20)\n"
                << "  --expiry X     time to expiry in years (default 1.0)\n"
                << "  --paths N      number of simulated paths (default 5000000)\n"
                << "  --threads N    worker threads (default = hardware concurrency)\n"
                << "  --seed N       RNG base seed (default 42)\n"
                << "  --benchmark    benchmark scaling across CPU thread counts\n"
                << "  --export       write sample_paths.csv and payoffs.csv for Python plots\n"
                << "  --out DIR      output directory for --benchmark/--export (default ./data)\n";

            std::exit(0);
        } else {
            std::cerr
                << "Unknown flag: " << flag
                << " (use --help)\n";

            std::exit(1);
        }
    }

    return args;
}

template <typename PayoffT>
double timedRun(
    bool multiThreaded,
    const PayoffT& payoff,
    const MarketData& mkt,
    unsigned long numPaths,
    unsigned int numThreads,
    unsigned int seed,
    double& priceOut) {

    const auto start = std::chrono::steady_clock::now();

    priceOut = multiThreaded
                   ? mc::priceMultiThreaded(
                         payoff,
                         mkt,
                         numPaths,
                         numThreads,
                         seed)
                   : mc::priceSingleThreaded(
                         payoff,
                         mkt,
                         numPaths,
                         seed);

    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(
               end - start)
        .count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 0) {
        return (values[middle - 1] + values[middle]) / 2.0;
    }

    return values[middle];
}

template <typename PayoffT>
double benchmarkRun(
    bool multiThreaded,
    const PayoffT& payoff,
    const MarketData& mkt,
    unsigned long numPaths,
    unsigned int numThreads,
    unsigned int seed,
    unsigned int repetitions,
    double& priceOut) {

    std::vector<double> timings;
    timings.reserve(repetitions);

    double lastPrice = 0.0;

    for (unsigned int i = 0; i < repetitions; ++i) {
        double price = 0.0;

        const double runtimeMs =
            timedRun(
                multiThreaded,
                payoff,
                mkt,
                numPaths,
                numThreads,
                seed,
                price);

        timings.push_back(runtimeMs);
        lastPrice = price;
    }

    priceOut = lastPrice;

    return median(timings);
}

void runBenchmark(const Args& args) {
    
    const std::vector<unsigned long> pathCounts = {
        100'000UL,
        1'000'000UL,
        5'000'000UL,
        10'000'000UL,
        20'000'000UL
    };

    const unsigned int hardwareThreads =
        std::thread::hardware_concurrency();

    std::vector<unsigned int> threadCounts = {
        1,
        2,
        4
    };
    if (hardwareThreads >= 6) {
        threadCounts.push_back(6);
    }

    if (hardwareThreads >= 8) {
        threadCounts.push_back(8);
    }
    if (hardwareThreads >= 12) {
        threadCounts.push_back(12);
    }

    if (hardwareThreads >= 16) {
        threadCounts.push_back(16);
    }


    constexpr unsigned int repetitions = 5;

    CallPayoff payoff(args.mkt.strike);

    std::ofstream csv(args.outDir + "/benchmark.csv");

    if (!csv) {
        std::cerr
            << "Failed to open benchmark output file: "
            << args.outDir << "/benchmark.csv\n";
        return;
    }

    csv << "num_paths,threads,runtime_ms,speedup,efficiency_pct,"
           "single_price,multi_price,price_difference\n";

    std::cout
        << "\nMonte Carlo Parallel Scaling Benchmark\n"
        << "=======================================\n"
        << "Hardware threads: " << hardwareThreads << "\n"
        << "Repetitions:      " << repetitions << "\n\n";

    std::cout
        << std::left
        << std::setw(12) << "Paths"
        << std::setw(10) << "Threads"
        << std::setw(16) << "Runtime (ms)"
        << std::setw(12) << "Speedup"
        << std::setw(14) << "Efficiency"
        << "Price Diff\n";

    std::cout
        << "-----------------------------------------------------------------\n";

    for (const auto numPaths : pathCounts) {

        /*
         * Establish the serial baseline for this exact workload.
         */
        double singlePrice = 0.0;

        const double singleMs =
            benchmarkRun(
                false,
                payoff,
                args.mkt,
                numPaths,
                1,
                args.seed,
                repetitions,
                singlePrice);

        for (const auto threads : threadCounts) {

            double multiPrice = 0.0;

            const double multiMs =
                benchmarkRun(
                    true,
                    payoff,
                    args.mkt,
                    numPaths,
                    threads,
                    args.seed,
                    repetitions,
                    multiPrice);

            /*
             * Parallel speedup:
             *
             *       T_serial
             * S = -----------
             *       T_parallel
             */
            const double speedup =
                singleMs / multiMs;

            /*
             * Parallel efficiency:
             *
             *             Speedup
             * E = --------------------- * 100
             *          Number Threads
             */
            const double efficiency =
                100.0 * speedup / threads;

            const double priceDifference =
                std::abs(singlePrice - multiPrice);

            std::cout
                << std::left
                << std::setw(12) << numPaths
                << std::setw(10) << threads
                << std::setw(16) << std::fixed
                << std::setprecision(2) << multiMs
                << std::setw(12) << std::setprecision(2)
                << speedup
                << std::setw(14) << std::setprecision(1)
                << efficiency << "%"
                << std::setprecision(8)
                << priceDifference
                << "\n";

            csv
                << numPaths << ","
                << threads << ","
                << multiMs << ","
                << speedup << ","
                << efficiency << ","
                << std::setprecision(10)
                << singlePrice << ","
                << multiPrice << ","
                << priceDifference
                << "\n";
        }

        std::cout << "\n";
    }

    std::cout
        << "Wrote "
        << args.outDir
        << "/benchmark.csv\n";
}

void runExport(const Args& args) {
    // A handful of full paths (with intermediate steps) for the fan chart.
    const unsigned int numStepsForPlot = 100;
    const unsigned long numPathsForPlot = 200;

    auto paths =
        mc::generatePaths(
            args.mkt,
            numPathsForPlot,
            numStepsForPlot,
            args.seed);

    std::ofstream pathsCsv(
        args.outDir + "/sample_paths.csv");

    const std::size_t stride =
        numStepsForPlot + 1;

    for (unsigned long p = 0;
         p < numPathsForPlot;
         ++p) {

        for (std::size_t s = 0;
             s < stride;
             ++s) {

            pathsCsv
                << paths[p * stride + s];

            if (s + 1 < stride) {
                pathsCsv << ",";
            }
        }

        pathsCsv << "\n";
    }

    std::cout
        << "Wrote "
        << args.outDir
        << "/sample_paths.csv\n";

    // Terminal payoffs for a much larger sample,
    // for the payoff histogram.
    const unsigned long numPayoffSamples =
        200'000UL;

    CallPayoff callPayoff(args.mkt.strike);
    PutPayoff putPayoff(args.mkt.strike);

    const double variance =
        args.mkt.vol *
        args.mkt.vol *
        args.mkt.expiry;

    const double rootVariance =
        std::sqrt(variance);

    const double itoDrift =
        (args.mkt.rate -
         0.5 * args.mkt.vol * args.mkt.vol) *
        args.mkt.expiry;

    const double movedSpot =
        args.mkt.spot *
        std::exp(itoDrift);

    std::mt19937_64 rng(args.seed + 999);
    std::normal_distribution<double> gaussian(0.0, 1.0);

    std::ofstream payoffCsv(
        args.outDir + "/payoffs.csv");

    payoffCsv
        << "terminal_spot,call_payoff,put_payoff\n";

    for (unsigned long i = 0;
         i < numPayoffSamples;
         ++i) {

        const double z =
            gaussian(rng);

        const double terminalSpot =
            movedSpot *
            std::exp(rootVariance * z);

        payoffCsv
            << terminalSpot << ","
            << callPayoff(terminalSpot) << ","
            << putPayoff(terminalSpot)
            << "\n";
    }

    std::cout
        << "Wrote "
        << args.outDir
        << "/payoffs.csv\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    if (args.benchmark) {
        runBenchmark(args);
        return 0;
    }

    if (args.exportData) {
        runExport(args);
        return 0;
    }

    CallPayoff callPayoff(args.mkt.strike);
    PutPayoff putPayoff(args.mkt.strike);

    double callPrice = 0.0;
    double putPrice = 0.0;

    const double callMs =
        timedRun(
            true,
            callPayoff,
            args.mkt,
            args.numPaths,
            args.numThreads,
            args.seed,
            callPrice);

    const double putMs =
        timedRun(
            true,
            putPayoff,
            args.mkt,
            args.numPaths,
            args.numThreads,
            args.seed,
            putPrice);

    std::cout
        << std::fixed
        << std::setprecision(6);

    std::cout
        << "Monte Carlo European Option Pricer\n"
        << "  Spot=" << args.mkt.spot
        << " Strike=" << args.mkt.strike
        << " Rate=" << args.mkt.rate
        << " Vol=" << args.mkt.vol
        << " Expiry=" << args.mkt.expiry
        << "\n"
        << "  Paths=" << args.numPaths
        << " Threads=" << args.numThreads
        << "\n\n"
        << "  Call price: " << callPrice
        << "  (" << callMs << " ms)\n"
        << "  Put price:  " << putPrice
        << "  (" << putMs << " ms)\n";

    return 0;
}