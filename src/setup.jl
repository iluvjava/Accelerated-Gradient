# The code will set up your Julia Environment. 
using Pkg
Pkg.instantiate()
Pkg.add(
    [
        "LuxurySparse", 
        "Plots", 
        "LaTeXStrings", 
        "Arpack", 
        "UnicodePlots", 
        "Zygote", 
        "Plotly", 
        "PGFPlotsX", 
        "MLDatasets", 
        "Statistics", 
        "ProgressMeter"
    ])


