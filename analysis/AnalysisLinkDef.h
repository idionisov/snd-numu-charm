#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ namespace snd::trident;
#pragma link C++ defined_in namespace snd::trident;

#pragma link C++ struct snd::trident::PreselectionMetrics+;
#pragma link C++ class snd::trident::PreselectionProcessor+;
#pragma link C++ class snd::trident::ProgressPrinter+;

#pragma link C++ namespace snd;
#pragma link C++ enum snd::InteractionType;
#pragma link C++ enum snd::RegionType;
#pragma link C++ enum snd::CharmHadronType;
#pragma link C++ struct snd::NeutrinoTruthConfig+;
#pragma link C++ struct snd::NeutrinoTruthInfo+;
#pragma link C++ class snd::NeutrinoTruthProcessor+;
#pragma link C++ struct snd::MuonNeutrinoTruthInfo+;
#pragma link C++ class snd::MuonNeutrinoTruthProcessor+;
#pragma link C++ class snd::MuonNeutrinoTruthWithDSProcessor+;
#pragma link C++ class snd::DataManager+;
#pragma link C++ class snd::AvgScifiFiducialCut+;
#pragma link C++ namespace snd::analysis_cuts;
#pragma link C++ class snd::analysis_cuts::AvgScifiFiducialCut+;

#endif
