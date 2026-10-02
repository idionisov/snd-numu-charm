#include "ProgressPrinter.h"
#include <algorithm>

namespace snd::trident {

ProgressPrinter::ProgressPrinter(uint64_t every_n,
                                 uint64_t total_events,
                                 double every_sec,
                                 const std::string& label,
                                 bool log_mode)
    : fEveryN(every_n),
      fTotal(total_events),
      fEverySec(every_sec),
      fLabel(label),
      fLogMode(log_mode),
      fState(std::make_shared<State>())
{
    fState->start_time = std::chrono::steady_clock::now();
    fState->last_print_time = fState->start_time;
}

bool ProgressPrinter::operator()(ULong64_t /*entry*/) const {
    uint64_t current = ++(fState->count);

    // Dynamic clock check frequency to guarantee near-zero overhead
    uint64_t check_interval = 500;
    if (fEveryN > 0 && fEveryN < check_interval) {
        check_interval = fEveryN;
    }

    if ((current % check_interval == 0) || (fTotal > 0 && current == fTotal)) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> dt = now - fState->last_print_time;
        bool time_trigger = (fEverySec > 0.0 && dt.count() >= fEverySec);
        bool count_trigger = (fEveryN > 0 && (current - fState->last_print_count.load()) >= fEveryN);
        bool final_trigger = (fTotal > 0 && current >= fTotal && !fState->summary_printed.load());

        if (time_trigger || count_trigger || final_trigger) {
            std::lock_guard<std::mutex> lock(fState->print_mutex);
            if (final_trigger && fState->summary_printed.load()) {
                final_trigger = false;
            }
            auto now_locked = std::chrono::steady_clock::now();
            std::chrono::duration<double> dt_locked = now_locked - fState->last_print_time;

            if ((fEverySec > 0.0 && dt_locked.count() >= (fEverySec * 0.9)) ||
                (fEveryN > 0 && (current - fState->last_print_count.load()) >= fEveryN) ||
                final_trigger)
            {
                PrintStatus(current, now_locked, final_trigger);
                fState->last_print_time = now_locked;
                fState->last_print_count.store(current);
                if (final_trigger) {
                    fState->summary_printed.store(true);
                }
            }
        }
    }
    return true;
}

uint64_t ProgressPrinter::GetProcessed() const {
    return fState ? fState->count.load() : 0;
}

uint64_t ProgressPrinter::GetTotal() const {
    return fTotal;
}

void ProgressPrinter::SetTotal(uint64_t total) {
    fTotal = total;
}

double ProgressPrinter::GetElapsedSeconds() const {
    if (!fState) return 0.0;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = now - fState->start_time;
    return dt.count();
}

double ProgressPrinter::GetRate() const {
    double sec = GetElapsedSeconds();
    uint64_t n = GetProcessed();
    return (sec > 0.0) ? (static_cast<double>(n) / sec) : 0.0;
}

void ProgressPrinter::PrintSummary() const {
    if (!fState) return;
    std::lock_guard<std::mutex> lock(fState->print_mutex);
    if (fState->summary_printed.exchange(true)) return;
    auto now = std::chrono::steady_clock::now();
    PrintStatus(fState->count.load(), now, true);
}

void ProgressPrinter::Reset() {
    if (!fState) return;
    std::lock_guard<std::mutex> lock(fState->print_mutex);
    fState->count.store(0);
    fState->last_print_count.store(0);
    fState->start_time = std::chrono::steady_clock::now();
    fState->last_print_time = fState->start_time;
    fState->summary_printed.store(false);
}

void ProgressPrinter::PrintStatus(uint64_t current, std::chrono::steady_clock::time_point now, bool is_done) const {
    std::chrono::duration<double> total_elapsed = now - fState->start_time;
    double sec = total_elapsed.count();
    double rate = (sec > 0.0) ? (static_cast<double>(current) / sec) : 0.0;

    std::string rate_str = FormatRate(rate);
    std::string time_str = FormatTime(sec);
    std::string ts = GetCurrentTimestamp();

    std::ostringstream ss;
    ss << "[" << ts << "] [" << fLabel << "] ";
    if (is_done) {
        ss << "[✓ Done] ";
    }

    if (fTotal > 0) {
        double pct = (100.0 * static_cast<double>(current)) / static_cast<double>(fTotal);
        if (pct > 100.0) pct = 100.0;

        double eta_sec = (rate > 0.0 && current < fTotal) ? (static_cast<double>(fTotal - current) / rate) : 0.0;
        std::string eta_str = FormatTime(eta_sec);

        ss << FormatNumber(current) << " / " << FormatNumber(fTotal)
           << " (" << std::fixed << std::setprecision(1) << pct << "%) | "
           << "Elapsed: " << time_str;
        if (!is_done) {
            ss << " | ETA: " << eta_str;
        }
        ss << " | Rate: " << rate_str;
    } else {
        ss << FormatNumber(current) << " events | "
           << "Elapsed: " << time_str << " | "
           << "Rate: " << rate_str;
    }

    std::cout << ss.str() << std::endl;
    std::cout.flush();
}

std::string ProgressPrinter::FormatNumber(uint64_t val) {
    std::string s = std::to_string(val);
    int n = static_cast<int>(s.length()) - 3;
    while (n > 0) {
        s.insert(n, ",");
        n -= 3;
    }
    return s;
}

std::string ProgressPrinter::FormatTime(double sec) {
    int isec = static_cast<int>(sec);
    int hours = isec / 3600;
    int mins = (isec % 3600) / 60;
    int secs = isec % 60;
    char buf[32];
    if (hours > 0) {
        snprintf(buf, sizeof(buf), "%02d:%02d:%02d", hours, mins, secs);
    } else {
        snprintf(buf, sizeof(buf), "%02d:%02d", mins, secs);
    }
    return std::string(buf);
}

std::string ProgressPrinter::FormatRate(double rate) {
    char buf[32];
    if (rate >= 1e6) {
        snprintf(buf, sizeof(buf), "%.2f MHz", rate / 1e6);
    } else if (rate >= 1e3) {
        snprintf(buf, sizeof(buf), "%.2f kHz", rate / 1e3);
    } else {
        snprintf(buf, sizeof(buf), "%.1f ev/s", rate);
    }
    return std::string(buf);
}

std::string ProgressPrinter::GetCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&now_c, &tm_buf);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_buf);
    return std::string(buf);
}

} // namespace snd::trident
