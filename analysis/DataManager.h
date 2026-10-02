#ifndef SND_DATA_MANAGER_H
#define SND_DATA_MANAGER_H

#include <string>
#include <vector>
#include <unordered_set>
#include <utility>
#include <memory>
#include <iostream>

#include <Rtypes.h>
#include <TChain.h>
#include <TFile.h>
#include <ROOT/RDataFrame.hxx>

class Scifi;
class MuFilter;

namespace snd {

    struct Configuration;

    class DataManager {
    public:
        DataManager();
        explicit DataManager(const std::string& source,
                            const std::string& treeName = "",
                            int numThreads = 0);
        explicit DataManager(const std::vector<std::string>& sources,
                            const std::string& treeName = "",
                            int numThreads = 0);

        // Run-based construction (using snd::analysis_tools::GetTChain & GetDataBasePath)
        explicit DataManager(int runNumber,
                            int nFiles = -1,
                            const std::string& csvDataPath = "",
                            const std::string& treeName = "",
                            int numThreads = 0);

        // Adopt existing TChain (with optional ownership)
        explicit DataManager(TChain* chain, bool ownChain = true, int numThreads = 0);
        explicit DataManager(std::unique_ptr<TChain> chain, int numThreads = 0);

        virtual ~DataManager();

        // Move semantics (disallow copy to protect TChain)
        DataManager(const DataManager&) = delete;
        DataManager& operator=(const DataManager&) = delete;
        DataManager(DataManager&& other) noexcept;
        DataManager& operator=(DataManager&& other) noexcept;

        // Core Accessors
        const std::string& GetTreeName() const { return fTreeName; }
        const std::vector<std::string>& GetFiles() const { return fFiles; }
        size_t GetNumFiles() const { return fFiles.size(); }
        const std::unordered_set<std::string>& GetBranchNames() const { return fBranchNames; }
        std::vector<std::string> GetBranchNamesList() const;

        bool HasBranch(const std::string& name) const;
        bool HasEventHeader() const;
        bool IsMC() const { return HasBranch("MCTrack"); }

        // Run Number & Database Access
        int GetRunNumber() const;
        void SetRunNumber(int runNumber) { fRunNumber = runNumber; }
        std::string GetDataBasePath(const std::string& csvFilePath = "") const;

        // Geofile & Geometry Access (adopting snd::analysis_tools)
        std::string GetGeoPath(const std::string& csvFilePath = "") const;
        void SetGeoPath(const std::string& geoPath) { fGeoPath = geoPath; }
        std::pair<Scifi*, MuFilter*> InitGeometry(const std::string& geoPath = "", const std::string& csvFilePath = "");
        bool HasGeometry() const;
        Scifi* GetScifi(bool autoInit = true);
        MuFilter* GetMuFilter(bool autoInit = true);
        std::pair<Scifi*, MuFilter*> GetGeometry(bool autoInit = true);
        snd::Configuration* GetConfiguration(bool autoInit = true);

        // Entries & Chain
        Long64_t GetEntries() const;
        TChain* GetChain() { return fChain; }
        const TChain* GetChain() const { return fChain; }
        void AdoptChain(TChain* chain, bool ownChain = true, int numThreads = 0);

        // RDataFrame
        ROOT::RDataFrame GetDataFrame();
        ROOT::RDataFrame GetRDF() { return GetDataFrame(); }
        ROOT::RDataFrame rdf() { return GetDataFrame(); }
        ROOT::RDF::RNode GetRNode(ULong64_t rangeLimit = 0);

        // Time & Duration Assessment via Full TChain Traversal
        // Exact: Evaluates actual event entries (entry 0 and entry N-1) across the full TChain
        std::pair<Long64_t, Long64_t> GetTimeRange(bool cached = true);
        double GetDuration(bool cached = true);
        std::string GetDurationStr(bool cached = true);

        // Fast Boundary Assessment
        // Fast: Evaluates boundary files using chronological run/chunk & natural version ordering
        std::pair<Long64_t, Long64_t> GetTimeRangeFast(bool cached = true);
        double GetDurationFast(bool cached = true);
        std::string GetDurationFastStr(bool cached = true);

        // Fill & Luminosity Assessment (ATLAS luminosity in fb^-1)
        int GetFillNumber() const;
        void SetFillNumber(int fillNumber) { fFillNumber = fillNumber; }
        std::string GetLumiPath(const std::string& lumiDir = "/eos/experiment/sndlhc/atlas_lumi") const;

        // Returns integrated luminosity in fb^-1.
        // If start_ts and end_ts are <= 0, returns the full integrated luminosity of the fill
        // without event timestamp extraction (lightning fast). If timestamps are provided,
        // integrates over [start_ts, end_ts].
        double GetLumi(Long64_t start_ts = -1, Long64_t end_ts = -1, const std::string& lumiDir = "/eos/experiment/sndlhc/atlas_lumi", bool cached = true);

        // Uses fast boundary assessment (GetTimeRangeFast) to determine data duration
        // and returns the integrated luminosity over that interval in fb^-1.
        double GetLumiFast(const std::string& lumiDir = "/eos/experiment/sndlhc/atlas_lumi", bool cached = true);

        // Uses exact TChain traversal (GetTimeRange) to determine data duration
        // and returns the integrated luminosity over that interval in fb^-1.
        double GetLumiExact(const std::string& lumiDir = "/eos/experiment/sndlhc/atlas_lumi", bool cached = true);

        std::string GetLumiStr(Long64_t start_ts = -1, Long64_t end_ts = -1, const std::string& lumiDir = "/eos/experiment/sndlhc/atlas_lumi", bool cached = true);
        std::string GetLumiFastStr(const std::string& lumiDir = "/eos/experiment/sndlhc/atlas_lumi", bool cached = true);

        // Formatted reporting
        std::string Summary() const;
        void Print() const;

        // Static Utilities (Data & Geometry discovery)
        static std::vector<std::string> ResolveFiles(const std::string& source);
        static std::vector<std::string> ResolveFiles(const std::vector<std::string>& sources);
        static std::string ResolveTreeName(const std::string& firstFile, const std::string& requestedTree = "");
        static std::unordered_set<std::string> InspectBranches(const std::string& firstFile, const std::string& treeName);
        static void CopyMetadata(const std::string& sourceFile,
                                const std::string& targetFile,
                                const std::vector<std::string>& excludeTrees = {"cbmsim", "rawConv"});

        // Static helpers adopting sndsw analysis tools
        static DataManager FromRun(int runNumber,
                                  int nFiles = -1,
                                  const std::string& csvDataPath = "",
                                  const std::string& treeName = "",
                                  int numThreads = 0);
        static std::string FetchDataBasePath(int runNumber, const std::string& csvFilePath = "");
        static std::string FetchGeoPath(int runNumber, const std::string& csvFilePath = "");
        static std::unique_ptr<TChain> FetchTChain(int runNumber, int nFiles = -1, const std::string& csvFilePath = "");
        static std::unique_ptr<TChain> FetchTChain(const std::string& fileName);
        static std::pair<Scifi*, MuFilter*> FetchGeometry(const std::string& geoPath);
        static std::pair<Scifi*, MuFilter*> FetchGeometry(int runNumber, const std::string& csvFilePath = "");

        // Static luminosity helpers
        static int ReadFillNumberFromTree(const std::string& fpath, const std::string& treeName);
        static double CalculateLumiFromTree(TTree* tree, Long64_t start_ts = -1, Long64_t end_ts = -1);
        static double FetchLumi(int fillNumber, Long64_t start_ts = -1, Long64_t end_ts = -1, const std::string& lumiDir = "/eos/experiment/sndlhc/atlas_lumi");

        // Static run-number helper: reads EventHeader.GetRunId() from entry 0 of a single file
        static int ReadRunNumberFromTree(const std::string& fpath, const std::string& treeName);

    private:
        void Initialize(const std::vector<std::string>& sources,
                        const std::string& treeName,
                        int numThreads);

        std::vector<std::string> fFiles;
        std::string fTreeName;
        std::unordered_set<std::string> fBranchNames;
        TChain* fChain{nullptr}; //! Transient ROOT TChain pointer
        bool fOwnChain{true};

        mutable Long64_t fEntries{-1};
        mutable int fRunNumber{-1};
        mutable int fFillNumber{-1};

        // Geometry & Configuration state
        std::string fGeoPath;
        Scifi* fScifi{nullptr}; //! Transient pointer
        MuFilter* fMuFilter{nullptr}; //! Transient pointer
        bool fGeoInitialized{false};
        std::unique_ptr<snd::Configuration> fConfiguration; //! Cached configuration

        // Full TChain cache
        bool fTimeRangeChecked{false};
        std::pair<Long64_t, Long64_t> fTimeRange{0, 0};
        double fDuration{-1.0};

        // Fast boundary cache
        bool fTimeRangeFastChecked{false};
        std::pair<Long64_t, Long64_t> fTimeRangeFast{0, 0};
        double fDurationFast{-1.0};

        // Lumi cache
        bool fLumiFullChecked{false};
        double fLumiFull{-1.0};
        bool fLumiFastChecked{false};
        double fLumiFast{-1.0};
        bool fLumiExactChecked{false};
        double fLumiExact{-1.0};

        ClassDef(DataManager, 1);
    };

} // namespace snd

#endif // SND_DATA_MANAGER_H
