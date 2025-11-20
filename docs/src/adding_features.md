# Modelling features in the power spectrum

While this package is designed to model broadband power spectral densities that are power-law-like, we can add narrow features using basis functions.

This feature is currently experimental and may not work properly. Use it at your own risk!

## Defining the model and simulating form the model

we defined a power spectral model as sum of two QPOs and a bending power-law.
```@example psd_features
using Plots
using Pioran
using Random
using Distributions
cont = SingleBendingPowerLaw(0.5,1., 3.6)
q1 =  QPO(7.3,0.1,2)
q2 = QPO(9e-4,2,11)
𝓟 = cont + q1 + q2
f = 10 .^range(-3,1,1000)
p1 = Plots.plot(f,𝓟.(f),xlabel="Frequency (day^-1)",ylabel="Power Spectral Density",legend=true,framestyle = :box,xscale=:log10,yscale=:log10,lw=2,ylim=(1e-5,1e3),label="total")
p1 = Plots.plot!(f,q1.(f),label="QPO1",linestyle=:dash)
Plots.plot!(p1,f,q2.(f),label="QPO2",linestyle=:dot)
Plots.plot!(f,cont.(f),label="cont",linestyle=:dash)
p1
```

We now simulate time series from this process. To do so we will use the GP method by approximating the bending power-law with basis functions.

```@example psd_features
t = range(0.,1e3,step=0.1)
yerr = 0.3*ones(length(t));
using QuadGK
```
In this example power spectrum is normalised by its integral (so its integral is the same as the model after approximation) to make proper comparison with the model.
```@example psd_features
f_min,f_max = 1/t[end],1/2/diff(t)[1]
norm = quadgk(x->𝓟(x),f_min,f_max,rtol=1e-14)[1]/2

𝓡 = approx(𝓟, f_min, f_max, 35, norm, basis_function="DRWCelerite")
GP = ScalableGP(0.0, 𝓡)
GPc = GP(t,yerr.^2)
```

We can sample realisations from this process:
```@example psd_features
rng = MersenneTwister(12)
y = [rand(rng,GPc) for i in 1:100]
Plots.plot(t,y[1:3],xlabel="Time",ylabel="Value",legend=false,framestyle = :box,ms=3)
```
As it is not very informative to look at the time series, let's compute the periodogram using `Tonari.jl`.

```@example psd_features
using Tonari
x_GP = mapreduce(permutedims,vcat,y)
fP,I = periodogram(t,x_GP',apply_end_matching=false)
```

```@example psd_features
Δt = t[2]-t[1]
noise_level = 2Δt*mean(yerr.^2)

Plots.plot(fP,I,yscale=:log10,xscale=:log10,xlabel="Frequency (Hz)",ylabel="Power",label="Periodogram",framestyle=:box)
Plots.plot!(f,𝓟.(f),label="Model",linewidth=2)
hline!([noise_level],label="Noise level",linewidth=2,linestyle=:dash,ylim=(noise_level/10,1e3),)
```

We can even compare the periodogram to the one of simulations using Timmer and König. See the documentation of `Tonari.jl` for more details:

```@example psd_features
T, Δt = 504.2, 0.132
simu = Simulation(𝓟, 1e3, 0.1, 20, 20)
rng = MersenneTwister(42)
N = 100
t_TK, x_TK, yerr_TK = sample(rng, simu, N, error_size = 0.25)
fP_TK,I_TK = periodogram(t_TK,x_TK,apply_end_matching=false)
```

The periodograms agree with the input power spectrum.
```@example psd_features
Δt = t[2]-t[1]
noise_level = 2Δt*mean(yerr.^2)
noise_level_TK = 2Δt*mean(yerr_TK.^2)

Plots.plot(fP,I,yscale=:log10,xscale=:log10,xlabel="Frequency (Hz)",ylabel="Power",framestyle=:box,label="Periodogram GP",alpha=1,lw=.5)
Plots.plot!(fP_TK,I_TK,label="Periodogram TK",alpha=0.2)
Plots.plot!(f,𝓟.(f),label="Model",linewidth=2)
hline!([noise_level],label="Noise level",linewidth=2,linestyle=:dash,ylim=(noise_level/4,2e2),)
```