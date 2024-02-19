# API Reference


## Power spectral densities

### Models and functions

```@autodocs
Modules = [Pioran]
Pages = ["psd.jl"]
Private = false
Order   = [:type, :function]
```

### Helper functions
```@autodocs
Modules = [Pioran]
Pages = ["psd.jl"]
Public = false
Order   = [:function]
```
## Covariance functions

### Models and functions

```@autodocs
Modules = [Pioran]
Pages = ["acvf.jl","Exp.jl","SHO.jl","Celerite.jl"]
Private = false
Order   = [:type, :function]
```

### Helper functions
```@autodocs
Modules = [Pioran]
Pages = ["acvf.jl","Exp.jl","SHO.jl","Celerite.jl"]
Public = false
Order   = [:function]
```

## Gaussian processes

```@autodocs
Modules = [Pioran]
Pages = ["scalable_GP.jl"]
Order   = [:type, :function]
```

## Solvers

### Celerite solver

```@autodocs
Modules = [Pioran]
Pages = ["celerite_solver.jl"]
Order   = [:type, :function]
```

### Direct solver

```@autodocs
Modules = [Pioran]
Pages = ["direct_solver.jl"]
Order   = [:type, :function]
```
## Plotting

### Diagnostics
```@autodocs
Modules = [Pioran]
Pages = ["plots_diagnostics.jl"]
Private = false
Order   = [:function]
```
### Individual plotting functions
```@autodocs
Modules = [Pioran]
Pages = ["plots_diagnostics.jl"]
Public = false
Order   = [:function]
```

## Utilities

```@autodocs
Modules = [Pioran]
Pages = ["utilities.jl"]
Order   = [:type, :function]
```