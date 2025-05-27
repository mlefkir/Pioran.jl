using Pioran: SHO, Exp, Celerite

"""
     convert_feature(psd_feature)

Convert a PSD feature to a Celerite covariance function
Only QPO is implemented

# Arguments
- `psd_feature::PowerSpectralDensity`: the PSD feature

# Return
- `covariance::SemiSeparable`: the covariance function
"""
function convert_feature(psd_feature::PowerSpectralDensity)
    if psd_feature isa QPO
        Δ = sqrt(4 * psd_feature.Q^2 - 1)

        a = psd_feature.S₀ * psd_feature.f₀ * psd_feature.Q * 2π
        b = a / Δ
        c = 2π * psd_feature.f₀ / psd_feature.Q / 2
        d = 2π * psd_feature.f₀ * Δ / 2 / psd_feature.Q
        return [a, b, c, d]
    else
        error("Feature $(typeof(psd_feature)) not implemented")
    end
end

"""
     get_covariance_from_psd(psd_features)

Get the covariance function from the PSD features
"""
function get_covariance_from_psd(psd_features)
    if psd_features isa Vector{<:PowerSpectralDensity}
        cov = convert_feature(psd_features[1])
        for i in 2:length(psd_features)
            cov = hcat(cov, convert_feature(psd_features[i]))
        end
        return cov
    else
        return convert_feature(psd_features)
    end
end


"""
     get_normalised_psd(psd_model::PowerSpectralDensity, spectral_points::AbstractVector{<:Real})

Get the PSD normalised at the lowest frequency
"""
function get_normalised_psd(psd_model::PowerSpectralDensity, spectral_points::AbstractVector{<:Real})
    psd_zero = psd_model(spectral_points[1])
    psd_normalised = psd_model(spectral_points) / psd_zero
    return psd_normalised
end

"""
     build_approx(J, f0, fM; basis_function="SHO")

Prepare the approximation of a PSD

# Arguments
- `J::Integer`: the number of basis functions
- `f0::Real`: the lowest frequency
- `fM::Real`: the highest frequency
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"

# Return
- `spectral_points::Vector{Real}`: the spectral points
- `spectral_matrix::Matrix{Real}`: the spectral matrix
"""
function build_approx(J::Int64, f0::Real, fM::Real; basis_function::String = "SHO")
    spectral_points = zeros(J)
    spectral_matrix = zeros(J, J)
    return init_psd_decomp!(spectral_points, spectral_matrix, J, f0, fM, basis_function = basis_function)
end

function init_psd_decomp!(spectral_points::AbstractVector{<:Real}, spectral_matrix::AbstractMatrix{<:Real}, J::Int64, f0::Real, fM::Real; basis_function::String = "SHO")
    # create the spectral_points
    for j in 0:(J - 1)
        spectral_points[j + 1] = f0 * (fM / f0)^(j / (J - 1))
    end

    # fill the spectral matrix
    if basis_function == "SHO"
        for j in 1:J
            for k in 1:J
                spectral_matrix[j, k] = 1 / (1 + (spectral_points[j] / spectral_points[k])^4)
            end
        end
    elseif basis_function == "DRWCelerite"
        for j in 1:J
            for k in 1:J
                spectral_matrix[j, k] = 1 / (1 + (spectral_points[j] / spectral_points[k])^6)
            end
        end
    else
        error("Basis function" * basis_function * "not implemented")
    end
    return spectral_points, spectral_matrix
end

"""
     psd_decomp(psd_normalised, spectral_matrix)

Get amplitudes of the basis functions by solving the linear system of the approximation
"""
function psd_decomp(psd_normalised::AbstractVector{<:Real}, spectral_matrix::AbstractMatrix{<:Real})
    amplitudes = spectral_matrix \ psd_normalised
    return amplitudes
end

"""
     get_approx_coefficients(psd_model, f0, fM; n_components=20, basis_function="SHO")

Get the coefficients of the approximated PSD

# Arguments
- `psd_model::PowerSpectralDensity`: model of the PSD
- `f0::Real`: the lowest frequency
- `fM::Real`: the highest frequency
- `n_components::Integer=20`: the number of basis functions to use
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"

# Return
- `amplitudes::Vector{Real}`: the amplitudes of the basis functions
"""
function get_approx_coefficients(psd_model::PowerSpectralDensity, f0::Real, fM::Real; n_components::Int64 = 20, basis_function::String = "SHO")
    spectral_points, spectral_matrix = build_approx(n_components, f0, fM, basis_function = basis_function)

    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)
    return amplitudes
end

"""
     approximated_psd(f, psd_model, f0, fM; n_components=20, norm=1.0, basis_function="SHO")

Return the approximated PSD. This is essentially to check that the model and the approximation are consistent.

# Arguments
- `f::AbstractVector{<:Real}`: the frequencies at which to evaluate the PSD
- `psd_model::PowerSpectralDensity`: model of the PSD
- `f0::Real`: the lowest frequency
- `fM::Real`: the highest frequency
- `n_components::Integer=20`: the number of basis functions to use
- `norm::Real=1.0`: normalisation of the PSD
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"
- `individual::Bool=false`: return the individual components
"""
function approximated_psd(f, psd_model::PowerSpectralDensity, f0::Real, fM::Real; n_components::Int64 = 20, norm::Real = 1.0, basis_function::String = "SHO", individual = false)
    spectral_points, spectral_matrix = build_approx(n_components, f0, fM, basis_function = basis_function)
    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)

    if individual
        psd = zeros(length(f), n_components)
        if basis_function == "SHO"
            for i in 1:n_components
                psd[:, i] = amplitudes[i] * norm ./ (1 .+ (f ./ spectral_points[i]) .^ 4)
            end
        elseif basis_function == "DRWCelerite"
            for i in 1:n_components
                psd[:, i] = amplitudes[i] * norm ./ (1 .+ (f ./ spectral_points[i]) .^ 6)
            end
        else
            error("Basis function" * basis_function * "not implemented")
        end
    else
        psd = zeros(length(f))
        if basis_function == "SHO"
            for i in 1:n_components
                psd += amplitudes[i] * norm ./ (1 .+ (f ./ spectral_points[i]) .^ 4)
            end
        elseif basis_function == "DRWCelerite"
            for i in 1:n_components
                psd += amplitudes[i] * norm ./ (1 .+ (f ./ spectral_points[i]) .^ 6)
            end
        else
            error("Basis function" * basis_function * "not implemented")
        end
    end
    return psd
end

"""
    approx(psd_model, f_min, f_max, n_components=20, norm=1.0,S_low::Real=20., S_high::Real=20. ; is_integrated_power::Bool = true, basis_function="SHO")

Approximate the PSD with a sum of basis functions to form a covariance function. The PSD model is approximated between `f0=f_min/S_low` and `fM=f_min*S_high`. By default it is normalised by its integral from `f_min` to `f_max` but it can also be normalised by its integral from 0 to infinity using the `variance` argument.

# Arguments
- `psd_model::PowerSpectralDensity`: model of the PSD
- `f_min::Real`: the minimum frequency in the time series
- `f_max::Real`: the maximum frequency in the time series
- `n_components::Integer=20`: the number of basis functions to use
- `norm::Real=1.0`: normalisation of the PSD.
- `S_low::Real=20.0`: scaling factor for the lowest frequency in the approximation.
- `S_high::Real=20.0`: scaling factor for the highest frequency in the approximation.
- `is_integrated_power::Bool = true`: if the norm corresponds to integral of the PSD between `f_min` and `f_max`, if not it is the variance of the process, integral of the PSD from 0 to +inf.
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"

# Return
- `covariance::SumOfSemiSeparable`: the covariance function

# Example
```julia
using Pioran
𝓟 = SingleBendingPowerLaw(1.0, 1.0, 2.0)
𝓡 = approx(𝓟, 1e-4, 1e-1, 30, 2.31,basis_function="SHO")
```

"""
function approx(psd_model::PowerSpectralDensity, f_min::Real, f_max::Real, n_components::Int64 = 20, norm::Real = 1.0, S_low::Real = 20.0, S_high::Real = 20.0; is_integrated_power::Bool = true, basis_function::String = "SHO")

    f0 = f_min / S_low
    fM = f_max * S_high
    spectral_points, spectral_matrix = build_approx(n_components, f0, fM, basis_function = basis_function)

    psd_normalised = get_normalised_psd(psd_model, spectral_points)
    amplitudes = psd_decomp(psd_normalised, spectral_matrix)

    integ = get_norm_psd(amplitudes, spectral_points, f_min, f_max, basis_function, is_integrated_power)
    # normalise the amplitudes
    amplitudes *= norm / integ

    # express the covariance function of the approximation
    if basis_function == "SHO"

        a = amplitudes .* spectral_points * π / √2 # π / √2 was removed as it was also in the expression of var but it is now restored as we do not use the variance anymore
        c = √2 * π .* spectral_points

        covariance = SumOfCelerite(a, a, c, c)

    elseif basis_function == "DRWCelerite"

        # these are the coefficients of the celerite part of the DRWCelerite
        a = amplitudes .* spectral_points * π / 3
        b = √3 * a
        c = π * spectral_points
        d = √3 * c

        # the coefficents of the DRW part are: a, 0, 2c and 0
        aa = [a; a]
        bb = [b; zeros(n_components)]
        cc = [c; 2 * c]
        dd = [d; zeros(n_components)]
        covariance = SumOfCelerite(aa, bb, cc, dd)

    else
        error("Basis function" * basis_function * "not implemented")
    end

    return covariance
end

@doc raw"""
    integral_sho(a, c, x)
Computes the integral of the SHO basis function of amplitude a and width c for a given x.

This integral is obtained using Equation: 4.2.7.1.3 from the "Handbook of Mathematical Formulas and Integrals" 2009

```math
    \int \dfrac{a\, {d}x}{(x/c)^4+1} =\dfrac{ac}{4\sqrt2} \left[\ln{\left(\dfrac{x^2+cx\sqrt2+c^2}{x^2-cx\sqrt2+c^2}\right)}+2\arctan{\left(\dfrac{cx\sqrt2}{c^2-x^2}\right)}\right]
```
"""
function integral_sho(a, c, x)
    norm = @. c .* a / (4√2)
    poly = @. (x^2 + √2c * x + c^2) / (x^2 - √2c * x + c^2)
    return sum(@. norm * (log.(poly) + 2atan.(c * √2 * x, (c^2 - x^2))))
end

@doc raw"""
    integral_drwcelerite(a, c, x)
Computes the integral of the DRWCelerite basis function of amplitude `a` and width `c` for a given `x`.

The DRWCelerite basis function is defined as:

```math
    \int \dfrac{a\, {d}x}{(x/c)^6+1} =\dfrac{ac}{3} \left[ \arctan{(x/c)} +\dfrac{\sqrt3}{4}\ln{\left(\dfrac{x^2+xc\sqrt3+c^2}{x^2-xc\sqrt3+c^2}\right)}+\dfrac{1}{2}\arctan{\left(\dfrac{x^2-c^2}{xc}\right)}\right]
```

"""
function integral_drwcelerite(a, c, x)
    norm = a .* c / 3
    drw = atan.(x ./ c)
    poly = @. (x^2 + √3c * x + c^2) / (x^2 - √3c * x + c^2)
    celerite = @. 0.5 * atan(x^2 - c^2, c * x) + √3 / 4 * log(poly)
    return sum(norm .* (drw + celerite))
end

@doc raw"""
    integrate_basis_function(a,c,x₁,x₂,basis_function)

Computes the integral of the basis function between x₁ and x₂ for a given amplitude a and width c.
"""
function integrate_basis_function(a, c, x₁, x₂, basis_function)
    if basis_function == "SHO"
        return integral_sho(a, c, x₂) - integral_sho(a, c, x₁)
    elseif basis_function == "DRWCelerite"
        return integral_drwcelerite(a, c, x₂) - integral_drwcelerite(a, c, x₁)
    else
        error("Unknown basis function: $basis_function, use 'SHO' or 'DRWCelerite'")
    end
end

@doc raw"""
    get_norm_psd(amplitudes,spectral_points,f_min,f_max,basis_function,is_integrated_power)

Get the normalisation of the sum of basis functions.

# Arguments
- `amplitudes`: amplitude of the basis function
- `spectral_points`: spectral points of the basis function
- `f_min::Real`: the minimum frequency in the time series
- `f_max::Real`: the maximum frequency in the time series
- `basis_function::String="SHO"`: the basis function to use, either "SHO" or "DRWCelerite"
- `is_integrated_power::Bool=true`: if the norm corresponds to integral of the PSD between `f_min` and `f_max` or if it is the integral from 0 to infinity.

"""
function get_norm_psd(amplitudes, spectral_points, f_min, f_max, basis_function, is_integrated_power)
    if is_integrated_power
        # normalise by the integral from f_min to f_max
        integ = integrate_basis_function(amplitudes, spectral_points, f_min, f_max, basis_function)
    else
        # normalise by the total integral of the PSD from 0 to +infty
        if basis_function == "SHO"
            integ = sum(amplitudes .* spectral_points) * π / √2 # removed as it is also in the expression of a
        elseif basis_function == "DRWCelerite"
            integ = sum(amplitudes .* spectral_points) * 2π / 3
        end
    end
    return integ
end
