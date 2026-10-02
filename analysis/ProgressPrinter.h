#pragma once

#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <memory>
#include <string>
#include <mutex>
#include <sstream>
#include <ctime>
#include <Rtypes.h>

namespace snd::trident {

/**
 * @brief Thread-safe progress reporter for ROOT RDataFrame.
 *
 * Can be attached to any RDataFrame graph via:
 *   df = df.Filter(printer, {"rdfentry_"});
 * or using the Python helper:
 *   df = add_progress_printer(df, total_events=N, every_seconds=30.0);
 *
 * Periodically prints processing speed (kHz/MHz), elapsed time, ETA,
 * and completion percentage without adding overhead or altering event data.
 */
class ProgressPrinter {
public:
    struct State {
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> last_print_count{0};
        std::chrono::steady_clock::time_point start_time{std::chrono::steady_clock::now()};
        std::chrono::steady_clock::time_point last_print_time{std::chrono::steady_clock::now()};
        std::mutex print_mutex;
        std::atomic<bool> summary_printed{false};
    };

    ProgressPrinter(uint64_t every_n = 0,
                    uint64_t total_events = 0,
                    double every_sec = 30.0,
                    const std::string& label = "Progress",
                    bool log_mode = true);

    virtual ~ProgressPrinter() = default;

    // Callable interface for ROOT RDataFrame
    bool operator()(ULong64_t entry = 0) const;

    uint64_t GetProcessed() const;
    uint64_t GetTotal() const;
    void SetTotal(uint64_t total);

    double GetElapsedSeconds() const;
    double GetRate() const;

    void PrintSummary() const;
    void Reset();

    // Formatting utilities
    static std::string FormatNumber(uint64_t val);
    static std::string FormatTime(double sec);
    static std::string FormatRate(double rate);
    static std::string GetCurrentTimestamp();

private:
    void PrintStatus(uint64_t current, std::chrono::steady_clock::time_point now, bool is_done) const;

    uint64_t fEveryN{0};
    uint64_t fTotal{0};
    double fEverySec{30.0};
    std::string fLabel{"Progress"};
    bool fLogMode{true};

    std::shared_ptr<State> fState; //! Transient state shared across all copies / worker threads
};

} // namespace snd::trident
