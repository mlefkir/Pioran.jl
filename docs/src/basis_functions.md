# About the basis functions

The basis functions $\psi_4$ and $\psi_6$ and their Fourier transforms have several properties.

```math
\psi_4(f) = \dfrac{1}{1+f^4} ~~~~~~~~ \psi_6(f) = \dfrac{1}{1+f^6}
```

```math
\begin{align}\begin{split}
\phi_4(\tau) &= \dfrac{\pi}{\sqrt2} \exp\left(-\pi\sqrt{2}|\tau|\right) \left(\cos\left(\pi\sqrt{2}|\tau|\right)+\sin\left(\pi\sqrt{2}|\tau|\right)\right)\\
\phi_6(\tau) &=\dfrac{\pi}{3}\left[\exp{\left(-2\pi |\tau|\right)}+\exp{\left(-\pi|\tau|\right)}\left(\cos\left(\pi\sqrt{3}|\tau|\right)+\sqrt{3}\sin\left(\pi\sqrt{3}|\tau|\right)\right)\right]\end{split}
\end{align}
```

## Approximation
When a model is approximated using a basis function, the `celerite` coefficients are:
### SHO: $\psi_4$

```math
\begin{align*}
a &= A_j f_j \pi/\sqrt2\\
b &=A_j f_j \pi / \sqrt2\\
c &= \pi f_j \sqrt2\\
d &= \pi f_j \sqrt2\\
\end{align*}
```

$A_j$ and $f_j$ are respectively the amplitudes and characteristic frequencies of the basis functions.

### DRWCelerite: $\psi_6$

For the celerite part of the basis function:
```math
\begin{align*}
a &= A_j f_j \pi/3\\
b &=A_j f_j \pi / \sqrt{3}\\
c &= \pi f_j\\
d &= \pi \sqrt{3} f_j\\
\end{align*}
```
For the DRW part:
```math
\begin{align*}
a &= A_j f_j \pi/3\\
b &= 0\\
c &= 2\pi f_j\\
d &= 0
\end{align*}
```

## Integral of the basis functions

We can also obtain the integral of the basis functions in the Fourier domain from $f_\mathrm{min}$ and $f_\mathrm{max}$. This value is used as a normalisation of the covariance function.

For $\psi_4$ we have:

```math
    \int \dfrac{a\, {d}x}{(x/c)^4+1} =\dfrac{ac}{4\sqrt2} \left[\ln{\left(\dfrac{x^2+cx\sqrt2+c^2}{x^2-cx\sqrt2+c^2}\right)}+2\arctan{\left(\dfrac{cx\sqrt2}{c^2-x^2}\right)}\right]
```

For $\psi_6$ we have:

```math
    \int \dfrac{a\, {d}x}{(x/c)^6+1} =\dfrac{ac}{3} \left[ \arctan{(x/c)} +\dfrac{\sqrt3}{4}\ln{\left(\dfrac{x^2+xc\sqrt3+c^2}{x^2-xc\sqrt3+c^2}\right)}+\dfrac{1}{2}\arctan{\left(\dfrac{x^2-c^2}{xc}\right)}\right]
```