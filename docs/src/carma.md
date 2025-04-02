# Continuous autoregressive moving average (CARMA)

Continuous autoregressive moving average (CARMA) processes are a generalization of ARMA processes to continuous time
[2014ApJ...788...33K](@cite) introduced CARMA processes based on the seminal work of [jonesackerson](@cite), [belcher1994](@cite) and [JONES1981651](@cite).

CARMA processes can be modelled with `Pioran.jl`, as these can be written as a celerite process [2017AJ....154..220F](@citep).

!!! warning
    The implementation of the CARMA process is still experimental and should be used with caution and tested thoroughly.

The CARMA process can be modelled with the [`CARMA`](@ref) type. The quadratic factors can be converted to roots with the [`quad2roots`](@ref) function. The roots can be converted back to quadratic factors with the [`roots2coeffs`](@ref) function. The CARMA process can be used as a kernel in the `ScalableGP` type.

!!! note
    I still have not found good priors for the parameters of the CARMA process. The priors used in the examples are not well tested and should be used with caution. Additionally, I have found difficult to assess the convergence of the MCMC chains. One might need a sampler that can handle the multimodal posterior distribution of the parameters.

Here are some examples of experimentals models for the CARMA process with `Turing.jl`.

```julia
@model function inference_model(y, t, σ, p::Int64, q::Int64)

    # Priors quadratic factors
    qa ~ filldist(FlatPos(0.),p)
    qb ~ filldist(FlatPos(0.),q)

    # Define the CARMA model
    rα = Pioran.quad2roots(qa)
    rβ = Pioran.quad2roots(qb)

    if !all(-f_max .< real.(rα) .< -f_min) || !all(-f_max .< imag.(rα) .< f_max)
        Turing.@addlogprob! -Inf
        return nothing
    end

    if !all(-f_max .< real.(rβ) .< -f_min) || !all(-f_max .< imag.(rβ) .<f_max)
        Turing.@addlogprob! -Inf
        return nothing
    end

    β = Pioran.roots2coeffs(rβ)

    # Prior distribution for the parameters
    variance ~ LogNormal(μₙ, σₙ)
    ν ~ Gamma(2, 0.5)
    μ ~ Normal(x̄, 5 * sqrt(va))
    c ~ LogUniform(1e-6, minimum(y) * 0.99)

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2

    # Make the flux Gaussian
    y = log.(y .- c)

    𝓒 = CARMA(p, q, rα, β, variance)

    # Build the GP
    f = ScalableGP(μ, 𝓒)

    y ~ f(t, σ²)
end
```

```julia
@model function inference_model(y, t, σ, p::Int64, q::Int64)

    # Define the LogNormal distribution for a_1
    a1_dist = Uniform(0.0, f_max^2)
    a2_dist = LogUniform(2 * f_min, 2 * f_max)
    a_3 = LogUniform(f_min, f_max)

    qa = Vector(undef, p)
    qb = Vector(undef, q)

      if p % 2 == 0  # all roots are complex conjugates
          # we first fill the quadratic coefficients with pair indices
          for i in 2:2:p
              qa[i] ~ a2_dist
          end
          # then we fill the quadratic coefficients with odd indices
          for i in 1:2:p-1
              qa[i] ~ a1_dist + qa[i+1]^2 / 4
          end

      else
          qa[end] ~ a_3

          for i in 2:2:p-1
              qa[i] ~ a2_dist
          end
          # then we fill the quadratic coefficients with odd indices
          for i in 1:2:p-2
              qa[i] ~ a1_dist + qa[i+1]^2 / 4
          end

      end

      if q % 2 == 0  # all roots are complex conjugates
          # we first fill the quadratic coefficients with pair indices
          for i in 2:2:q
              qb[i] ~ a2_dist
          end
          # then we fill the quadratic coefficients with odd indices
          for i in 1:2:q-1
              qb[i] ~ a1_dist + qb[i+1]^2 / 4
          end

        else
            qb[end] ~ a_3

            for i in 2:2:q-1
                qb[i] ~ a2_dist
            end
            # then we fill the quadratic coefficients with odd indices
            for i in 1:2:q-2
                qb[i] ~ a1_dist + qb[i+1]^2 / 4
            end

        end
    variance ~ LogNormal(μₙ, σₙ)
    ν ~ Gamma(2, 0.5)
    μ ~ Normal(x̄, 5 * sqrt(va))
    c ~ LogUniform(1e-6, minimum(y) * 0.99)

    # Rescale the measurement variance
    σ² = ν .* σ .^ 2 ./ (y .- c) .^ 2

    # Make the flux Gaussian
    y = log.(y .- c)

    # Define the CARMA model
    # convert the quadratic coefficients to roots
    rα = Pioran.quad2roots(qa)
    rβ = Pioran.quad2roots(qb)

    # check that the roots are in the right range
    if !all(-f_max .< real.(rα) .< -f_min) || !all(-f_max .< imag.(rα) .< f_max)
        Turing.@addlogprob! -Inf
        return nothing
    end

    if !all(-f_max .< real.(rβ) .< -f_min) || !all(-f_max .< imag.(rβ) .< f_max)
        Turing.@addlogprob! -Inf
        return nothing
    end

    # # check that the roots are in the right order
    # if p % 2 == 0
    #     permα = sortperm((imag.(rα[1:2:p])), rev=true)
    # else
    #     permα = sortperm((imag.(rα[1:2:p-1])), rev=true)
    # end
    # if permα != range(1, length(permα))
    #     Turing.@addlogprob! -Inf
    #     return nothing
    # end

    # if q % 2 == 0
    #     permβ = sortperm((imag.(rβ[1:2:q])), rev=true)
    # else
    #     permβ = sortperm((imag.(rβ[1:2:q-1])), rev=true)
    # end
    # if permβ != range(1, length(permβ))
    #     Turing.@addlogprob! -Inf
    #     return nothing
    # end

    β = Pioran.roots2coeffs(rβ)
    𝓒 = CARMA(p, q, rα, β, variance)

    # Build the GP
    f = ScalableGP(μ, 𝓒)

    y ~ f(t, σ²)
    return nothing
end
```