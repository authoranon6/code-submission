# FOMO-UDE: Forecasting and Modeling Tumor Dynamics using Universal Differential Equations

## Abstract
Forecasting tumor growth from sparse and heterogeneous clinical measurements is essential for treatment planning but remains a persistent challenge in oncology. Existing mechanistic models rely on fixed assumptions, while neural networks typically require large datasets and offer limited interpretability. We investigate adaptive tumor growth models based on Universal Differential Equations (UDEs), which integrate mechanistic structure with data-driven learning to handle limited, irregular clinical measurements. Evaluated on multi-species tumor datasets under multiple forecasting scenarios, the UDE model outperforms Gompertz ODE, GRU, LSTM, and Neural ODE baselines (achieving up to 91% RMSE reduction), while symbolic regression via SINDy recovers interpretable governing equations, enabling clinically transparent predictions.

## Running the code
To reproduce the environment, clone the repository and navigate to the folder:
```
git clone https://github.com/authoranon6/code-submission.git
cd code-submission
```
Then, instantiate the environment:
```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```
This installs the exact package versions specified in Project.toml and Manifest.toml. 
You can now run the scripts directly from the terminal:
```
julia --project=. <filename>.jl
```
Or from inside the Julia REPL:
```
julia --project=.
julia> include("<filename>.jl")
```
For instructions on installing Julia, see: https://julialang.org/install/

## Repository Structure
```
.
├── UDE.jl
├── NODE.jl
├── Gompertz.jl
├── GRU.jl
├── LSTM.jl
├── SymbolicRecovery.jl
├── Forecasting.jl
├── data/
│   ├── mouse_data.txt
│   └── preprocessed_tumor_data.csv
├── plots/
│   ├── human/
│     ├── 90-10/ ...
│     ├── 80-20/ ...
│     └── 70-30/ ...
│   └── mouse/
│     ├── 90-10/ ...
│     ├── 80-20/ ...
│     └── 70-30/ ...
├── Manifest.toml
├── Project.toml
└── README.md
```

**Dataset Sources:**

Vaghi, C., Rodallec, A., Fanciullino, R., Ciccolini, J., Mochel, J. P., Mastri, M., and Poignard, C. Population modeling of tumor growth curves and the reduced gompertz model improve prediction of the age of experimental tumors. PLoS Computational Biology, 16(2), 2020. doi: https://doi.org/10.1371/journal.pcbi.1007178.

Laleh, N. G., Loeffler, C. M. L., Grajek, J., Staňková, K., and Pearson, A. T. Classical mathematical models for prediction of response to chemotherapy and immunotherapy. PLoS Computational Biology, 18(2), 2022. doi: https://doi.org/10.1371/journal.pcbi.1009822.
