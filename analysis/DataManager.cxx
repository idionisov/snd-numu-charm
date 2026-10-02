#include "DataManager.h"
#include "SNDLHCEventHeader.h"
#include "Scifi.h"
#include "MuFilter.h"
#include "sndGeometryGetter.h"
#include "sndTchainGetter.h"
#include "sndConfiguration.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <glob.h>
#include <iomanip>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string.h>

#include <TKey.h>
#include <TROOT.h>
#include <TObjArray.h>
#include <TDirectory.h>

namespace snd {

    // ROOT macro that generates runtime type information and dictionary inspection methods
    // ROOT I/O and interactive CINT/Cling sessions.
    ClassImp(DataManager);

    static bool ParseRunAndChunk(const std::string& fpath, long long& run, long long& chunk) {
        static const std::regex runRegex("run_(\\d+)");
        static const std::regex chunkRegex("(?:raw|sndsw_raw|digCPP|search)[-_](\\d+)");
        std::smatch runMatch, chunkMatch;
        bool hasRun = std::regex_search(fpath, runMatch, runRegex);
        bool hasChunk = std::regex_search(fpath, chunkMatch, chunkRegex);
        if (hasRun && hasChunk) {
            run = std::stoll(runMatch[1].str());
            chunk = std::stoll(chunkMatch[1].str());
            return true;
        }
        return false;
    }

    static bool ChronologicalFileCompare(const std::string& a, const std::string& b) {
        long long runA = -1, chunkA = -1;
        long long runB = -1, chunkB = -1;
        bool hasA = ParseRunAndChunk(a, runA, chunkA);
        bool hasB = ParseRunAndChunk(b, runB, chunkB);
        if (hasA && hasB) {
            if (runA != runB) return runA < runB;
            if (chunkA != chunkB) return chunkA < chunkB;
            return strverscmp(a.c_str(), b.c_str()) < 0;
        }
        return strverscmp(a.c_str(), b.c_str()) < 0;
    }

    // SFINAE helpers to bridge between different sndsw API versions where argument order of
    // (runNumber, csvPath) vs (csvPath, runNumber) varies across releases.
    template <typename Str>
    auto CallGetTChain(int runNumber, int nFiles, const Str& csvPath, int)
        -> decltype(snd::analysis_tools::GetTChain(runNumber, nFiles, csvPath)) {
        return snd::analysis_tools::GetTChain(runNumber, nFiles, csvPath);
    }

    template <typename Str>
    auto CallGetTChain(int runNumber, int nFiles, const Str& csvPath, long)
        -> decltype(snd::analysis_tools::GetTChain(csvPath, runNumber, nFiles)) {
        return snd::analysis_tools::GetTChain(csvPath, runNumber, nFiles);
    }

    template <typename Str>
    auto CallGetDataBasePath(int runNumber, const Str& csvPath, int)
        -> decltype(snd::analysis_tools::GetDataBasePath(runNumber, csvPath)) {
        return snd::analysis_tools::GetDataBasePath(runNumber, csvPath);
    }

    template <typename Str>
    auto CallGetDataBasePath(int runNumber, const Str& csvPath, long)
        -> decltype(snd::analysis_tools::GetDataBasePath(csvPath, runNumber)) {
        return snd::analysis_tools::GetDataBasePath(csvPath, runNumber);
    }

    template <typename Str>
    auto CallGetGeoPath(int runNumber, const Str& csvPath, int)
        -> decltype(snd::analysis_tools::GetGeoPath(runNumber, csvPath)) {
        return snd::analysis_tools::GetGeoPath(runNumber, csvPath);
    }

    template <typename Str>
    auto CallGetGeoPath(int runNumber, const Str& csvPath, long)
        -> decltype(snd::analysis_tools::GetGeoPath(csvPath, runNumber)) {
        return snd::analysis_tools::GetGeoPath(csvPath, runNumber);
    }

    template <typename Str>
    auto CallGetGeometry(int runNumber, const Str& csvPath, int)
        -> decltype(snd::analysis_tools::GetGeometry(runNumber, csvPath)) {
        return snd::analysis_tools::GetGeometry(runNumber, csvPath);
    }

    template <typename Str>
    auto CallGetGeometry(int runNumber, const Str& csvPath, long)
        -> decltype(snd::analysis_tools::GetGeometry(csvPath, runNumber)) {
        return snd::analysis_tools::GetGeometry(csvPath, runNumber);
    }

    DataManager::DataManager()
        : fChain(nullptr), fOwnChain(true), fEntries(-1), fRunNumber(-1),
          fScifi(nullptr), fMuFilter(nullptr), fGeoInitialized(false) {}

    DataManager::DataManager(
        const std::string& source,
        const std::string& treeName,
        int numThreads
    ) : fChain(nullptr), fOwnChain(true), fEntries(-1), fRunNumber(-1),
        fScifi(nullptr), fMuFilter(nullptr), fGeoInitialized(false) {
        std::vector<std::string> sources{source};
        Initialize(sources, treeName, numThreads);
    }

    DataManager::DataManager(
        const std::vector<std::string>& sources,
        const std::string& treeName,
        int numThreads
    ) : fChain(nullptr), fOwnChain(true), fEntries(-1), fRunNumber(-1),
        fScifi(nullptr), fMuFilter(nullptr), fGeoInitialized(false) {
        Initialize(sources, treeName, numThreads);
    }

    DataManager::DataManager(
        int runNumber,
        int nFiles,
        const std::string& csvDataPath,
        const std::string& treeName,
        int numThreads
    ) : fChain(nullptr), fOwnChain(true), fEntries(-1), fRunNumber(runNumber),
        fScifi(nullptr), fMuFilter(nullptr), fGeoInitialized(false) {
        std::string effectiveTree = treeName.empty() ? "rawConv" : treeName;
        auto chain = CallGetTChain(runNumber, nFiles, csvDataPath, 0);
        if (effectiveTree != "rawConv") {
            chain->SetName(effectiveTree.c_str());
        }
        AdoptChain(chain.release(), true, numThreads);
        fTreeName = effectiveTree;
    }

    DataManager::DataManager(TChain* chain, bool ownChain, int numThreads)
        : fChain(nullptr), fOwnChain(ownChain), fEntries(-1), fRunNumber(-1),
          fScifi(nullptr), fMuFilter(nullptr), fGeoInitialized(false) {
        AdoptChain(chain, ownChain, numThreads);
    }

    DataManager::DataManager(std::unique_ptr<TChain> chain, int numThreads)
        : fChain(nullptr), fOwnChain(true), fEntries(-1), fRunNumber(-1),
          fScifi(nullptr), fMuFilter(nullptr), fGeoInitialized(false) {
        AdoptChain(chain.release(), true, numThreads);
    }

    DataManager::~DataManager() {
        if (fOwnChain && fChain) {
            delete fChain;
            fChain = nullptr;
        }
    }

    DataManager::DataManager(DataManager&& other) noexcept:
        fFiles(std::move(other.fFiles)),
        fTreeName(std::move(other.fTreeName)),
        fBranchNames(std::move(other.fBranchNames)),
        fChain(other.fChain),
        fOwnChain(other.fOwnChain),
        fEntries(other.fEntries),
        fRunNumber(other.fRunNumber),
        fGeoPath(std::move(other.fGeoPath)),
        fScifi(other.fScifi),
        fMuFilter(other.fMuFilter),
        fGeoInitialized(other.fGeoInitialized),
        fConfiguration(std::move(other.fConfiguration)),
        fTimeRangeChecked(other.fTimeRangeChecked),
        fTimeRange(other.fTimeRange),
        fDuration(other.fDuration),
        fTimeRangeFastChecked(other.fTimeRangeFastChecked),
        fTimeRangeFast(other.fTimeRangeFast),
        fDurationFast(other.fDurationFast),
        fFillNumber(other.fFillNumber),
        fLumiFullChecked(other.fLumiFullChecked),
        fLumiFull(other.fLumiFull),
        fLumiFastChecked(other.fLumiFastChecked),
        fLumiFast(other.fLumiFast),
        fLumiExactChecked(other.fLumiExactChecked),
        fLumiExact(other.fLumiExact) {
            other.fChain = nullptr;
            other.fOwnChain = false;
            other.fEntries = -1;
            other.fRunNumber = -1;
            other.fFillNumber = -1;
            other.fScifi = nullptr;
            other.fMuFilter = nullptr;
            other.fGeoInitialized = false;
            other.fTimeRangeChecked = false;
            other.fTimeRange = {0, 0};
            other.fDuration = -1.0;
            other.fTimeRangeFastChecked = false;
            other.fTimeRangeFast = {0, 0};
            other.fDurationFast = -1.0;
            other.fLumiFullChecked = false;
            other.fLumiFull = -1.0;
            other.fLumiFastChecked = false;
            other.fLumiFast = -1.0;
            other.fLumiExactChecked = false;
            other.fLumiExact = -1.0;
    }

    DataManager& DataManager::operator=(DataManager&& other) noexcept {
        if (this != &other) {
            if (fOwnChain && fChain) {
                delete fChain;
            }
            fFiles = std::move(other.fFiles);
            fTreeName = std::move(other.fTreeName);
            fBranchNames = std::move(other.fBranchNames);
            fChain = other.fChain;
            fOwnChain = other.fOwnChain;
            fEntries = other.fEntries;
            fRunNumber = other.fRunNumber;
            fFillNumber = other.fFillNumber;
            fGeoPath = std::move(other.fGeoPath);
            fScifi = other.fScifi;
            fMuFilter = other.fMuFilter;
            fGeoInitialized = other.fGeoInitialized;
            fConfiguration = std::move(other.fConfiguration);
            fTimeRangeChecked = other.fTimeRangeChecked;
            fTimeRange = other.fTimeRange;
            fDuration = other.fDuration;
            fTimeRangeFastChecked = other.fTimeRangeFastChecked;
            fTimeRangeFast = other.fTimeRangeFast;
            fDurationFast = other.fDurationFast;
            fLumiFullChecked = other.fLumiFullChecked;
            fLumiFull = other.fLumiFull;
            fLumiFastChecked = other.fLumiFastChecked;
            fLumiFast = other.fLumiFast;
            fLumiExactChecked = other.fLumiExactChecked;
            fLumiExact = other.fLumiExact;

            other.fChain = nullptr;
            other.fOwnChain = false;
            other.fEntries = -1;
            other.fRunNumber = -1;
            other.fFillNumber = -1;
            other.fScifi = nullptr;
            other.fMuFilter = nullptr;
            other.fGeoInitialized = false;
            other.fTimeRangeChecked = false;
            other.fTimeRange = {0, 0};
            other.fDuration = -1.0;
            other.fTimeRangeFastChecked = false;
            other.fTimeRangeFast = {0, 0};
            other.fDurationFast = -1.0;
            other.fLumiFullChecked = false;
            other.fLumiFull = -1.0;
            other.fLumiFastChecked = false;
            other.fLumiFast = -1.0;
            other.fLumiExactChecked = false;
            other.fLumiExact = -1.0;
        }
        return *this;
    }
    void DataManager::AdoptChain(TChain* chain, bool ownChain, int numThreads) {
        if (numThreads > 1) {
            if (!ROOT::IsImplicitMTEnabled()) {
                ROOT::EnableImplicitMT(numThreads);
            }
        } else if (numThreads == 1) {
            if (ROOT::IsImplicitMTEnabled()) {
                ROOT::DisableImplicitMT();
            }
        }
        if (fOwnChain && fChain) {
            delete fChain;
        }
        fChain = chain;
        fOwnChain = ownChain;
        fFiles.clear();
        fBranchNames.clear();
        fEntries = -1;
        fTimeRangeChecked = false;
        fTimeRangeFastChecked = false;
        fFillNumber = -1;
        fLumiFullChecked = false;
        fLumiFull = -1.0;
        fLumiFastChecked = false;
        fLumiFast = -1.0;
        fLumiExactChecked = false;
        fLumiExact = -1.0;

        if (!fChain) return;

        fTreeName = fChain->GetName();
        TObjArray* fileElements = fChain->GetListOfFiles();
        if (fileElements) {
            TIter next(fileElements);
            TObject* obj = nullptr;
            while ((obj = next())) {
                fFiles.push_back(obj->GetTitle());
            }
        }

        if (!fFiles.empty()) {
            std::sort(fFiles.begin(), fFiles.end(), ChronologicalFileCompare);
            if (fRunNumber <= 0) {
                long long r = -1, c = -1;
                if (ParseRunAndChunk(fFiles.front(), r, c) && r > 0) {
                    fRunNumber = static_cast<int>(r);
                }
            }
            if (fFiles.front().find_first_of("*?[]") == std::string::npos) {
                try {
                    fBranchNames = InspectBranches(fFiles.front(), fTreeName);
                } catch (...) {
                }
            }
        }
    }

    void DataManager::Initialize(
        const std::vector<std::string>& sources,
        const std::string& treeName,
        int numThreads
    ){
        TDirectory::TContext dirContext;
        if (numThreads > 1) {
            if (!ROOT::IsImplicitMTEnabled()) {
                ROOT::EnableImplicitMT(numThreads);
            }
        } else if (numThreads == 1) {
            if (ROOT::IsImplicitMTEnabled()) {
                ROOT::DisableImplicitMT();
            }
        }

        fFiles = ResolveFiles(sources);
        if (fFiles.empty()) {
            std::ostringstream oss;
            oss << "No input ROOT files found matching provided sources:";
            for (const auto& s : sources) oss << " " << s;
            throw std::runtime_error(oss.str());
        }

        if (fRunNumber <= 0) {
            long long r = -1, c = -1;
            if (ParseRunAndChunk(fFiles.front(), r, c) && r > 0) {
                fRunNumber = static_cast<int>(r);
            }
        }

        fTreeName = ResolveTreeName(fFiles.front(), treeName);
        fBranchNames = InspectBranches(fFiles.front(), fTreeName);

        if (fOwnChain && fChain) {
            delete fChain;
        }
        fChain = new TChain(fTreeName.c_str());
        fOwnChain = true;
        for (const auto& fpath : fFiles) {
            fChain->Add(fpath.c_str());
        }
    }

    bool DataManager::HasBranch(const std::string& name) const {
        if (fBranchNames.find(name) != fBranchNames.end()) {
            return true;
        }
        if (fChain) {
            return fChain->GetBranch(name.c_str()) != nullptr;
        }
        return false;
    }

    bool DataManager::HasEventHeader() const {
        if (HasBranch("EventHeader") || HasBranch("EventHeader.")) {
            return true;
        }
        for (const auto& b : fBranchNames) {
            if (b.rfind("EventHeader", 0) == 0) return true;
        }
        return false;
    }

    std::vector<std::string> DataManager::GetBranchNamesList() const {
        std::vector<std::string> result(fBranchNames.begin(), fBranchNames.end());
        std::sort(result.begin(), result.end());
        return result;
    }

    Long64_t DataManager::GetEntries() const {
        if (fEntries < 0 && fChain) {
            fEntries = fChain->GetEntries();
        }
        return fEntries;
    }

    ROOT::RDataFrame DataManager::GetDataFrame() {
        if (!fFiles.empty()) {
            return ROOT::RDataFrame(fTreeName, fFiles);
        }
        if (fChain) {
            return ROOT::RDataFrame(*fChain);
        }
        throw std::runtime_error("Cannot create RDataFrame: No files or chain available!");
    }

    // ROOT::RDF::RNode is a wrapper to represent any node in an RDataFrame computation
    // works identically as ROOT::RDataFrame
    ROOT::RDF::RNode DataManager::GetRNode(ULong64_t rangeLimit) {
        auto df = GetDataFrame();
        if (rangeLimit > 0) {
            return df.Range(rangeLimit);
        }
        return df;
    }

    int DataManager::ReadFillNumberFromTree(const std::string& fpath, const std::string& treeName) {
        TDirectory::TContext dirContext;
        std::unique_ptr<TFile> f(TFile::Open(fpath.c_str(), "READ"));
        if (!f || f->IsZombie()) return -1;

        TTree* tree = dynamic_cast<TTree*>(f->Get(treeName.c_str()));
        if (!tree || tree->GetEntries() <= 0) return -1;

        tree->SetBranchStatus("*", 0);
        tree->SetBranchStatus("EventHeader*", 1);

        const char* bname = tree->GetBranch("EventHeader.") ? "EventHeader." : "EventHeader";
        if (!tree->GetBranch(bname)) return -1;

        SNDLHCEventHeader* header = nullptr;
        tree->SetBranchAddress(bname, &header);

        tree->GetEntry(0);
        int fill = (header && header->GetFillNumber() > 0) ? header->GetFillNumber() : -1;
        return fill;
    }

    int DataManager::ReadRunNumberFromTree(const std::string& fpath, const std::string& treeName) {
        TDirectory::TContext dirContext;
        std::unique_ptr<TFile> f(TFile::Open(fpath.c_str(), "READ"));
        if (!f || f->IsZombie()) return -1;

        TTree* tree = dynamic_cast<TTree*>(f->Get(treeName.c_str()));
        if (!tree || tree->GetEntries() <= 0) return -1;

        tree->SetBranchStatus("*", 0);
        tree->SetBranchStatus("EventHeader*", 1);

        const char* bname = tree->GetBranch("EventHeader.") ? "EventHeader." : "EventHeader";
        if (!tree->GetBranch(bname)) return -1;

        SNDLHCEventHeader* header = nullptr;
        tree->SetBranchAddress(bname, &header);

        tree->GetEntry(0);
        int run = (header && header->GetRunId() > 0)
                  ? static_cast<int>(header->GetRunId())
                  : -1;
        return run;
    }

    double DataManager::CalculateLumiFromTree(TTree* tree, Long64_t start_ts, Long64_t end_ts) {
        if (!tree || tree->GetEntries() <= 0) return 0.0;

        Double_t ts = 0.0;
        Double_t var = 0.0;

        tree->SetBranchStatus("*", 0);
        tree->SetBranchStatus("unix_timestamp", 1);
        tree->SetBranchStatus("var", 1);

        tree->SetBranchAddress("unix_timestamp", &ts);
        tree->SetBranchAddress("var", &var);

        std::vector<double> timestamps;
        std::vector<double> inst_lumi;
        timestamps.reserve(tree->GetEntries());
        inst_lumi.reserve(tree->GetEntries());

        Long64_t nEntries = tree->GetEntries();
        for (Long64_t i = 0; i < nEntries; ++i) {
            tree->GetEntry(i);
            if (start_ts > 0 && ts < static_cast<double>(start_ts)) continue;
            if (end_ts > 0 && ts > static_cast<double>(end_ts)) continue;
            timestamps.push_back(ts);
            inst_lumi.push_back(var);
        }

        tree->ResetBranchAddresses();
        tree->SetBranchStatus("*", 1);

        if (timestamps.size() < 2) return 0.0;

        double sum = 0.0;
        for (size_t i = 1; i < timestamps.size(); ++i) {
            double dt = timestamps[i] - timestamps[i - 1];
            if (dt >= 0.0 && dt < 600.0) {
                sum += dt * inst_lumi[i];
            }
        }

        // Convert nb^-1 to fb^-1: sum / 1e3 / 1e6 = sum / 1e9
        return sum / 1e9;
    }

    double DataManager::FetchLumi(int fillNumber, Long64_t start_ts, Long64_t end_ts, const std::string& lumiDir) {
        if (fillNumber <= 0) return 0.0;
        TDirectory::TContext dirContext;
        char buf[512];
        snprintf(buf, sizeof(buf), "%s/fill_%06d.root", lumiDir.c_str(), fillNumber);
        std::unique_ptr<TFile> f(TFile::Open(buf, "READ"));
        if (!f || f->IsZombie()) return 0.0;
        TTree* tree = dynamic_cast<TTree*>(f->Get("atlas_lumi"));
        if (!tree) return 0.0;
        return CalculateLumiFromTree(tree, start_ts, end_ts);
    }

    static Long64_t ReadTimestampFromTree(const std::string& fpath, const std::string& treeName, Long64_t entryIdx) {
        TDirectory::TContext dirContext;
        // std::unique_ptr to invoke destructor automatically when file goes out of scope
        std::unique_ptr<TFile> f(TFile::Open(fpath.c_str(), "READ"));
        if (!f || f->IsZombie()) return 0;

        TTree* tree = dynamic_cast<TTree*>(f->Get(treeName.c_str()));
        if (!tree || tree->GetEntries() <= 0) return 0;

        // We only need the event header
        tree->SetBranchStatus("*", 0);
        tree->SetBranchStatus("EventHeader*", 1);

        const char* bname = tree->GetBranch("EventHeader.") ? "EventHeader." : "EventHeader";
        if (!tree->GetBranch(bname)) return 0;

        SNDLHCEventHeader* header = nullptr;
        tree->SetBranchAddress(bname, &header);

        Long64_t localEntry = (entryIdx < 0) ? (tree->GetEntries() - 1) : entryIdx;
        tree->GetEntry(localEntry);

        Long64_t ts = (header && header->GetUTCtimestamp() > 0) ? header->GetUTCtimestamp() : 0;
        return ts;
    }

    // Time & Duration Assessment via Full TChain Traversal
    // Exact: Evaluates actual event entries (entry 0 and entry N-1) across the full TChain
    std::pair<Long64_t, Long64_t> DataManager::GetTimeRange(bool cached) {
        if (cached && fTimeRangeChecked) {
            return fTimeRange;
        }

        fTimeRange = {0, 0};
        fDuration = -1.0;
        fTimeRangeChecked = true;

        if (!fChain || GetEntries() <= 0 || !HasEventHeader()) {
            return fTimeRange;
        }

        fChain->LoadTree(0);
        fChain->SetBranchStatus("*", 0);
        fChain->SetBranchStatus("EventHeader*", 1);

        const char* bname = fChain->GetBranch("EventHeader.") ? "EventHeader." : "EventHeader";
        SNDLHCEventHeader* header = nullptr;
        fChain->SetBranchAddress(bname, &header);

        // First event entry across full TChain
        fChain->GetEntry(0);
        Long64_t tStart = (header && header->GetUTCtimestamp() > 0) ? header->GetUTCtimestamp() : 0;

        // Final event entry across full TChain
        Long64_t lastEntry = fChain->GetEntries() - 1;
        fChain->GetEntry(lastEntry);
        Long64_t tEnd = (header && header->GetUTCtimestamp() > 0) ? header->GetUTCtimestamp() : 0;

        // Always restore chain to clean state
        fChain->ResetBranchAddresses();
        fChain->SetBranchStatus("*", 1);

        if (tStart > 0 && tEnd > 0) {
            fTimeRange = {tStart, tEnd};
            fDuration = static_cast<double>(tEnd - tStart);
            if (fDuration < 0.0) {
                fDuration = std::abs(fDuration);
            }
        }

        return fTimeRange;
    }

    double DataManager::GetDuration(bool cached) {
        auto tr = GetTimeRange(cached);
        if (tr.first <= 0 || tr.second <= 0) {
            return -1.0;
        }
        return fDuration;
    }

    std::string DataManager::GetDurationStr(bool cached) {
        double dur = GetDuration(cached);
        if (dur < 0.0) {
            return "N/A";
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        if (dur < 60.0) {
            oss << dur << " s";
        } else if (dur < 3600.0) {
            oss << dur << " s (" << (dur / 60.0) << " min)";
        } else {
            oss << std::setprecision(1) << dur << " s ("
                << std::setprecision(2) << (dur / 3600.0) << " h)";
        }
        return oss.str();
    }

    // Fast Boundary Assessment
    // Fast: Evaluates boundary files using chronological run/chunk & natural version ordering
    std::pair<Long64_t, Long64_t> DataManager::GetTimeRangeFast(bool cached) {
        if (cached && fTimeRangeFastChecked) {
            return fTimeRangeFast;
        }

        fTimeRangeFast = {0, 0};
        fDurationFast = -1.0;
        fTimeRangeFastChecked = true;

        if (!HasEventHeader() || fFiles.empty()) {
            return fTimeRangeFast;
        }

        Long64_t tStart = 0;
        Long64_t tEnd = 0;

        if (fFiles.size() == 1) {
            TDirectory::TContext dirContext;
            std::unique_ptr<TFile> f(TFile::Open(fFiles.front().c_str(), "READ"));
            if (f && !f->IsZombie()) {
                TTree* tree = dynamic_cast<TTree*>(f->Get(fTreeName.c_str()));
                if (tree && tree->GetEntries() > 0) {
                    tree->SetBranchStatus("*", 0);
                    tree->SetBranchStatus("EventHeader*", 1);
                    const char* bname = tree->GetBranch("EventHeader.") ? "EventHeader." : "EventHeader";
                    if (tree->GetBranch(bname)) {
                        SNDLHCEventHeader* header = nullptr;
                        tree->SetBranchAddress(bname, &header);

                        tree->GetEntry(0);
                        if (header && header->GetUTCtimestamp() > 0) {
                            tStart = header->GetUTCtimestamp();
                        }

                        tree->GetEntry(tree->GetEntries() - 1);
                        if (header && header->GetUTCtimestamp() > 0) {
                            tEnd = header->GetUTCtimestamp();
                        }
                    }
                }
            }
        } else {
            // Multi-file: search first non-empty file forward
            for (const auto& fpath : fFiles) {
                Long64_t ts = ReadTimestampFromTree(fpath, fTreeName, 0);
                if (ts > 0) {
                    tStart = ts;
                    break;
                }
            }

            // Search last non-empty file backward
            if (tStart > 0) {
                for (auto it = fFiles.rbegin(); it != fFiles.rend(); ++it) {
                    Long64_t ts = ReadTimestampFromTree(*it, fTreeName, -1);
                    if (ts > 0) {
                        tEnd = ts;
                        break;
                    }
                }
            }
        }

        if (tStart > 0 && tEnd > 0) {
            fTimeRangeFast = {tStart, tEnd};
            fDurationFast = static_cast<double>(tEnd - tStart);
            if (fDurationFast < 0.0) {
                fDurationFast = std::abs(fDurationFast);
            }
        }

        return fTimeRangeFast;
    }

    double DataManager::GetDurationFast(bool cached) {
        auto tr = GetTimeRangeFast(cached);
        if (tr.first <= 0 || tr.second <= 0) {
            return -1.0;
        }
        return fDurationFast;
    }

    std::string DataManager::GetDurationFastStr(bool cached) {
        double dur = GetDurationFast(cached);
        if (dur < 0.0) {
            return "N/A";
        }

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);
        if (dur < 60.0) {
            oss << dur << " s";
        } else if (dur < 3600.0) {
            oss << dur << " s (" << (dur / 60.0) << " min)";
        } else {
            oss << std::setprecision(1) << dur << " s ("
                << std::setprecision(2) << (dur / 3600.0) << " h)";
        }
        return oss.str();
    }

    std::string DataManager::Summary() const {
        std::ostringstream oss;
        oss << "============================================================\n";
        oss << "Snd DataManager Dataset Summary\n";
        oss << "============================================================\n";
        oss << "Tree Name     : " << fTreeName << "\n";
        oss << "Dataset Type  : " << (IsMC() ? "Monte Carlo (MC)" : "Collision Data (Real)") << "\n";
        if (GetRunNumber() > 0) {
            oss << "Run Number    : " << GetRunNumber() << "\n";
        }
        if (GetFillNumber() > 0) {
            oss << "Fill Number   : " << GetFillNumber() << "\n";
        }
        oss << "File Count    : " << fFiles.size() << "\n";
        oss << "Total Entries : " << GetEntries() << "\n";

        if (const_cast<DataManager*>(this)->GetDurationFast() >= 0.0) {
            oss << "Duration      : " << const_cast<DataManager*>(this)->GetDurationFastStr() << "\n";
        }
        if (GetFillNumber() > 0) {
            double lumiFast = const_cast<DataManager*>(this)->GetLumiFast();
            if (lumiFast > 0.0) {
                oss << "Luminosity    : " << const_cast<DataManager*>(this)->GetLumiFastStr() << "\n";
            }
        }

        oss << "Has MCTrack   : " << (HasBranch("MCTrack") ? "True" : "False") << "\n";

        std::string geo = GetGeoPath();
        if (!geo.empty()) {
            oss << "Geofile       : " << geo << "\n";
        }
        oss << "Geometry      : " << (HasGeometry() ? "Initialized (Scifi, MuFilter)" : "Not Initialized") << "\n";

        oss << "First File    : " << (fFiles.empty() ? "None" : fFiles.front()) << "\n";
        oss << "============================================================";
        return oss.str();
    }

    void DataManager::Print() const {
        std::cout << Summary() << std::endl;
    }

    int DataManager::GetRunNumber() const {
        // Tier 1: already cached
        if (fRunNumber > 0) {
            return fRunNumber;
        }

        if (fFiles.empty()) {
            return -1;
        }

        // Tier 2: fast filename parse (requires both run_NNNNNN and a chunk marker)
        {
            long long r = -1, c = -1;
            if (ParseRunAndChunk(fFiles.front(), r, c) && r > 0) {
                const_cast<DataManager*>(this)->fRunNumber = static_cast<int>(r);
                return fRunNumber;
            }
        }

        // Tier 3: EventHeader probe – start from the middle file and spiral outward.
        // Using the middle rather than the first/last is intentional: when a fill
        // spans multiple runs, the chronologically central files most reliably
        // belong to the dominant (longest) run of interest.
        if (!HasEventHeader()) {
            return -1;
        }

        const std::size_t n = fFiles.size();
        const std::ptrdiff_t mid = static_cast<std::ptrdiff_t>(n / 2);

        // Visit indices in order: mid, mid-1, mid+1, mid-2, mid+2, ...
        for (std::ptrdiff_t delta = 0; delta < static_cast<std::ptrdiff_t>(n); ++delta) {
            for (int sign : {0, -1, 1}) {
                if (sign == 0 && delta != 0) continue;   // only try mid+0 once (delta==0)
                if (sign != 0 && delta == 0) continue;   // skip ±0 when delta==0
                std::ptrdiff_t idx = mid + sign * delta;
                if (idx < 0 || idx >= static_cast<std::ptrdiff_t>(n)) continue;
                int run = ReadRunNumberFromTree(fFiles[static_cast<std::size_t>(idx)], fTreeName);
                if (run > 0) {
                    const_cast<DataManager*>(this)->fRunNumber = run;
                    return fRunNumber;
                }
            }
        }

        return -1;
    }

    int DataManager::GetFillNumber() const {
        if (fFillNumber > 0) {
            return fFillNumber;
        }

        if (!HasEventHeader()) {
            return -1;
        }

        if (!fFiles.empty()) {
            for (const auto& fpath : fFiles) {
                int fill = ReadFillNumberFromTree(fpath, fTreeName);
                if (fill > 0) {
                    const_cast<DataManager*>(this)->fFillNumber = fill;
                    return fFillNumber;
                }
            }
        }

        if (fChain && fChain->GetEntries() > 0) {
            fChain->LoadTree(0);
            fChain->SetBranchStatus("*", 0);
            fChain->SetBranchStatus("EventHeader*", 1);
            const char* bname = fChain->GetBranch("EventHeader.") ? "EventHeader." : "EventHeader";
            if (fChain->GetBranch(bname)) {
                SNDLHCEventHeader* header = nullptr;
                fChain->SetBranchAddress(bname, &header);
                fChain->GetEntry(0);
                if (header && header->GetFillNumber() > 0) {
                    const_cast<DataManager*>(this)->fFillNumber = header->GetFillNumber();
                }
                fChain->ResetBranchAddresses();
                fChain->SetBranchStatus("*", 1);
                if (fFillNumber > 0) {
                    return fFillNumber;
                }
            }
        }

        return -1;
    }

    std::string DataManager::GetLumiPath(const std::string& lumiDir) const {
        if (!fFiles.empty()) {
            TDirectory::TContext dirContext;
            std::unique_ptr<TFile> f(TFile::Open(fFiles.front().c_str(), "READ"));
            if (f && !f->IsZombie()) {
                if (f->GetListOfKeys() && f->GetListOfKeys()->Contains("atlas_lumi")) {
                    return fFiles.front();
                }
            }
        }

        int fill = GetFillNumber();
        if (fill <= 0) {
            return "";
        }

        char buf[512];
        snprintf(buf, sizeof(buf), "%s/fill_%06d.root", lumiDir.c_str(), fill);
        return std::string(buf);
    }

    double DataManager::GetLumi(Long64_t start_ts, Long64_t end_ts, const std::string& lumiDir, bool cached) {
        bool isFullFill = (start_ts <= 0 && end_ts <= 0);
        if (isFullFill && cached && fLumiFullChecked) {
            return fLumiFull;
        }

        std::string lumiPath = GetLumiPath(lumiDir);
        if (lumiPath.empty()) {
            if (isFullFill) {
                fLumiFull = 0.0;
                fLumiFullChecked = true;
            }
            return 0.0;
        }

        TDirectory::TContext dirContext;
        std::unique_ptr<TFile> f(TFile::Open(lumiPath.c_str(), "READ"));
        if (!f || f->IsZombie()) {
            if (isFullFill) {
                fLumiFull = 0.0;
                fLumiFullChecked = true;
            }
            return 0.0;
        }

        TTree* tree = dynamic_cast<TTree*>(f->Get("atlas_lumi"));
        if (!tree) {
            if (isFullFill) {
                fLumiFull = 0.0;
                fLumiFullChecked = true;
            }
            return 0.0;
        }

        double lumi = CalculateLumiFromTree(tree, start_ts, end_ts);
        if (isFullFill) {
            fLumiFull = lumi;
            fLumiFullChecked = true;
        }
        return lumi;
    }

    double DataManager::GetLumiFast(const std::string& lumiDir, bool cached) {
        if (cached && fLumiFastChecked) {
            return fLumiFast;
        }

        fLumiFastChecked = true;
        fLumiFast = 0.0;

        auto tr = GetTimeRangeFast(cached);
        if (tr.first <= 0 || tr.second <= 0) {
            return 0.0;
        }

        fLumiFast = GetLumi(tr.first, tr.second, lumiDir, false);
        return fLumiFast;
    }

    double DataManager::GetLumiExact(const std::string& lumiDir, bool cached) {
        if (cached && fLumiExactChecked) {
            return fLumiExact;
        }

        fLumiExactChecked = true;
        fLumiExact = 0.0;

        auto tr = GetTimeRange(cached);
        if (tr.first <= 0 || tr.second <= 0) {
            return 0.0;
        }

        fLumiExact = GetLumi(tr.first, tr.second, lumiDir, false);
        return fLumiExact;
    }

    std::string DataManager::GetLumiStr(Long64_t start_ts, Long64_t end_ts, const std::string& lumiDir, bool cached) {
        double lumi = GetLumi(start_ts, end_ts, lumiDir, cached);
        if (lumi <= 0.0) return "0.0 fb^-1";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << lumi << " fb^-1";
        return oss.str();
    }

    std::string DataManager::GetLumiFastStr(const std::string& lumiDir, bool cached) {
        double lumi = GetLumiFast(lumiDir, cached);
        if (lumi <= 0.0) return "0.0 fb^-1";
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << lumi << " fb^-1";
        return oss.str();
    }

    std::string DataManager::GetDataBasePath(const std::string& csvFilePath) const {
        int run = GetRunNumber();
        if (run > 0) {
            try {
                return CallGetDataBasePath(run, csvFilePath, 0);
            } catch (...) {
                return "";
            }
        }
        return "";
    }

    std::string DataManager::GetGeoPath(const std::string& csvFilePath) const {
        if (!fGeoPath.empty()) {
            return fGeoPath;
        }
        int run = GetRunNumber();
        if (run > 0) {
            try {
                return CallGetGeoPath(run, csvFilePath, 0);
            } catch (...) {
                return "";
            }
        }
        return "";
    }

    std::pair<Scifi*, MuFilter*> DataManager::InitGeometry(const std::string& geoPath, const std::string& csvFilePath) {
        TDirectory::TContext dirContext;
        if (!geoPath.empty()) {
            fGeoPath = geoPath;
        } else if (fGeoPath.empty()) {
            fGeoPath = GetGeoPath(csvFilePath);
        }

        if (fGeoPath.empty()) {
            throw std::runtime_error("DataManager::InitGeometry: Geofile path is empty and could not be determined automatically (run number: "
                + std::to_string(GetRunNumber()) + "). Please specify geoPath explicitly.");
        }

        auto geom = snd::analysis_tools::GetGeometry(fGeoPath);
        fScifi = geom.first;
        fMuFilter = geom.second;
        fGeoInitialized = (fScifi != nullptr && fMuFilter != nullptr);
        return geom;
    }

    bool DataManager::HasGeometry() const {
        return fGeoInitialized && (fScifi != nullptr || fMuFilter != nullptr);
    }

    Scifi* DataManager::GetScifi(bool autoInit) {
        if (!HasGeometry() && autoInit) {
            InitGeometry();
        }
        return fScifi;
    }

    MuFilter* DataManager::GetMuFilter(bool autoInit) {
        if (!HasGeometry() && autoInit) {
            InitGeometry();
        }
        return fMuFilter;
    }

    std::pair<Scifi*, MuFilter*> DataManager::GetGeometry(bool autoInit) {
        return std::make_pair(GetScifi(autoInit), GetMuFilter(autoInit));
    }

    snd::Configuration* DataManager::GetConfiguration(bool autoInit) {
        if (fConfiguration) {
            return fConfiguration.get();
        }
        if (!HasGeometry() && autoInit) {
            try {
                InitGeometry();
            } catch (...) {
                return nullptr;
            }
        }
        if (!HasGeometry()) {
            return nullptr;
        }
        int run = GetRunNumber();
        if (run <= 0) {
            return nullptr;
        }
        try {
            auto opt = snd::Configuration::GetOption(run);
            fConfiguration = std::make_unique<snd::Configuration>(opt, fScifi, fMuFilter);
            return fConfiguration.get();
        } catch (...) {
            return nullptr;
        }
    }

    DataManager DataManager::FromRun(int runNumber,
                                      int nFiles,
                                      const std::string& csvDataPath,
                                      const std::string& treeName,
                                      int numThreads) {
        return DataManager(runNumber, nFiles, csvDataPath, treeName, numThreads);
    }

    std::string DataManager::FetchDataBasePath(int runNumber, const std::string& csvFilePath) {
        return CallGetDataBasePath(runNumber, csvFilePath, 0);
    }

    std::string DataManager::FetchGeoPath(int runNumber, const std::string& csvFilePath) {
        return CallGetGeoPath(runNumber, csvFilePath, 0);
    }

    std::unique_ptr<TChain> DataManager::FetchTChain(int runNumber, int nFiles, const std::string& csvFilePath) {
        return CallGetTChain(runNumber, nFiles, csvFilePath, 0);
    }

    std::unique_ptr<TChain> DataManager::FetchTChain(const std::string& fileName) {
        return snd::analysis_tools::GetTChain(fileName);
    }

    std::pair<Scifi*, MuFilter*> DataManager::FetchGeometry(const std::string& geoPath) {
        TDirectory::TContext dirContext;
        return snd::analysis_tools::GetGeometry(geoPath);
    }

    std::pair<Scifi*, MuFilter*> DataManager::FetchGeometry(int runNumber, const std::string& csvFilePath) {
        TDirectory::TContext dirContext;
        return CallGetGeometry(runNumber, csvFilePath, 0);
    }

    std::vector<std::string> DataManager::ResolveFiles(const std::string& source) {
        if (source.empty()) return {};

        // 1. Exact regular file
        if (std::filesystem::is_regular_file(source)) {
            return {source};
        }

        // 2. Exact directory with *.root files
        if (std::filesystem::is_directory(source)) {
            std::vector<std::string> matches;
            for (const auto& entry : std::filesystem::directory_iterator(source)) {
                if (entry.is_regular_file() && entry.path().extension() == ".root") {
                    matches.push_back(entry.path().string());
                }
            }
            std::sort(matches.begin(), matches.end(), ChronologicalFileCompare);
            return matches;
        }

        // 3. Glob matching
        glob_t globResult;
        int globRet = glob(source.c_str(), GLOB_TILDE, nullptr, &globResult);
        if (globRet == 0 && globResult.gl_pathc > 0) {
            std::vector<std::string> matches;
            for (size_t i = 0; i < globResult.gl_pathc; ++i) {
                std::string p(globResult.gl_pathv[i]);
                if (p.size() >= 5 && p.substr(p.size() - 5) == ".root") {
                    matches.push_back(p);
                }
            }
            globfree(&globResult);
            if (!matches.empty()) {
                std::sort(matches.begin(), matches.end(), ChronologicalFileCompare);
                return matches;
            }
        } else if (globRet == 0) {
            globfree(&globResult);
        }

        // 4. Regex matching within parent directory
        std::filesystem::path p(source);
        std::string dirPart = p.parent_path().string();
        std::string filePart = p.filename().string();
        if (dirPart.empty()) dirPart = ".";

        std::vector<std::string> candDirs;
        if (dirPart.find_first_of("*?[]") != std::string::npos) {
            glob_t dirGlob;
            if (glob(dirPart.c_str(), GLOB_TILDE, nullptr, &dirGlob) == 0) {
                for (size_t i = 0; i < dirGlob.gl_pathc; ++i) {
                    if (std::filesystem::is_directory(dirGlob.gl_pathv[i])) {
                        candDirs.push_back(dirGlob.gl_pathv[i]);
                    }
                }
                globfree(&dirGlob);
            }
        } else if (std::filesystem::is_directory(dirPart)) {
            candDirs.push_back(dirPart);
        }

        std::vector<std::string> rxMatches;
        try {
            std::regex rx(filePart.empty() ? source : filePart);
            for (const auto& d : candDirs) {
                std::error_code ec;
                for (const auto& entry : std::filesystem::directory_iterator(d, ec)) {
                    if (ec) break;
                    if (entry.is_regular_file()) {
                        std::string fname = entry.path().filename().string();
                        if (fname.size() >= 5 && fname.substr(fname.size() - 5) == ".root") {
                            if (std::regex_search(fname, rx)) {
                                rxMatches.push_back(entry.path().string());
                            }
                        }
                    }
                }
            }
        } catch (const std::regex_error&) {
            // invalid regex, ignore
        }

        std::sort(rxMatches.begin(), rxMatches.end(), ChronologicalFileCompare);
        return rxMatches;
    }

    std::vector<std::string> DataManager::ResolveFiles(const std::vector<std::string>& sources) {
        std::vector<std::string> allFiles;
        for (const auto& src : sources) {
            auto resolved = ResolveFiles(src);
            allFiles.insert(allFiles.end(), resolved.begin(), resolved.end());
        }
        std::sort(allFiles.begin(), allFiles.end(), ChronologicalFileCompare);
        allFiles.erase(std::unique(allFiles.begin(), allFiles.end()), allFiles.end());
        return allFiles;
    }

    std::string DataManager::ResolveTreeName(const std::string& firstFile, const std::string& requestedTree) {
        TDirectory::TContext dirContext;
        std::unique_ptr<TFile> f(TFile::Open(firstFile.c_str(), "READ"));
        if (!f || f->IsZombie()) {
            throw std::runtime_error("Could not open ROOT file: " + firstFile);
        }

        if (!requestedTree.empty()) {
            TObject* obj = f->Get(requestedTree.c_str());
            if (obj && obj->InheritsFrom(TTree::Class())) {
                return requestedTree;
            }
            throw std::runtime_error("Requested tree '" + requestedTree + "' not found in " + firstFile);
        }

        for (const auto& candidate : {"rawConv", "cbmsim"}) {
            TObject* obj = f->Get(candidate);
            if (obj && obj->InheritsFrom(TTree::Class())) {
                return candidate;
            }
        }

        std::ostringstream oss;
        oss << "Neither 'rawConv' nor 'cbmsim' tree found in " << firstFile << ". Available keys: ";
        TIter next(f->GetListOfKeys());
        while (TKey* key = dynamic_cast<TKey*>(next())) {
            oss << key->GetName() << " (" << key->GetClassName() << ") ";
        }
        throw std::runtime_error(oss.str());
    }

    std::unordered_set<std::string> DataManager::InspectBranches(const std::string& firstFile, const std::string& treeName) {
        TDirectory::TContext dirContext;
        std::unordered_set<std::string> branches;
        std::unique_ptr<TFile> f(TFile::Open(firstFile.c_str(), "READ"));
        if (!f || f->IsZombie()) return branches;

        TTree* tree = dynamic_cast<TTree*>(f->Get(treeName.c_str()));
        if (!tree) return branches;

        TIter next(tree->GetListOfBranches());
        while (TObject* obj = next()) {
            branches.insert(obj->GetName());
        }
        return branches;
    }

    void DataManager::CopyMetadata(const std::string& sourceFile,
                                const std::string& targetFile,
                                const std::vector<std::string>& excludeTrees) {
        TDirectory::TContext dirContext;
        std::unique_ptr<TFile> fIn(TFile::Open(sourceFile.c_str(), "READ"));
        if (!fIn || fIn->IsZombie()) return;

        std::vector<std::pair<std::string, TObject*>> metadata;
        TIter next(fIn->GetListOfKeys());
        while (TKey* key = dynamic_cast<TKey*>(next())) {
            std::string kname = key->GetName();
            std::string kclass = key->GetClassName();
            if (kclass != "TTree") {
                bool excluded = false;
                for (const auto& excl : excludeTrees) {
                    if (kname == excl) { excluded = true; break; }
                }
                if (!excluded) {
                    TObject* obj = fIn->Get(kname.c_str());
                    if (obj) {
                        metadata.push_back({kname, obj->Clone()});
                    }
                }
            }
        }

        if (!metadata.empty()) {
            std::unique_ptr<TFile> fOut(TFile::Open(targetFile.c_str(), "UPDATE"));
            if (fOut && !fOut->IsZombie()) {
                fOut->cd();
                for (auto& [name, obj] : metadata) {
                    obj->Write(name.c_str(), TObject::kSingleKey | TObject::kOverwrite);
                    delete obj;
                }
            } else {
                for (auto& [name, obj] : metadata) {
                    delete obj;
                }
            }
        }
    }

} // namespace snd
