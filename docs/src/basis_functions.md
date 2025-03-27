# About the basis functions

The basis functions $\psi_4$ and $\psi_6$ and their Fourier transforms have several properties.

When a model is approximated using $\psi_6$, the `celerite` coefficients are:

```math
\begin{align}
a &= a_j f_j \pi/3\\
b &= 0\\
c &= 2\pi f_j\\
d &= 0
\end{align}
```
```math
\begin{align}
a &= a_j f_j \pi/3\\
b &=a_j f_j \pi / \sqrt{3}\\
c &= \pi f_j\\
d &= \pi \sqrt{3} f_j\\
\end{align}
```

The basis function at zero lag is given by $2\pi/3 a_j f_j$.

## Integrals

We can also obtain the integral of the basis functions in the Fourier domain from $f_\mathrm{min}$ and $f_\mathrm{max}$.

For $\psi_4$ we have:

```math
    \int \dfrac{a\, {d}x}{(x/c)^4+1} =\dfrac{ac}{4\sqrt2} \left[\ln{\left(\dfrac{x^2+cx\sqrt2+c^2}{x^2-cx\sqrt2+c^2}\right)}+2\arctan{\left(\dfrac{cx\sqrt2}{c^2-x^2}\right)}\right]
```

For $\psi_6$ we have:

```math
    \int \dfrac{a\, {d}x}{(x/c)^6+1} =\dfrac{ac}{3} \left[ \arctan{(x/c)} +\dfrac{\sqrt3}{4}\ln{\left(\dfrac{x^2+xc\sqrt3+c^2}{x^2-xc\sqrt3+c^2}\right)}+\dfrac{1}{2}\arctan{\left(\dfrac{x^2-c^2}{xc}\right)}\right]
```