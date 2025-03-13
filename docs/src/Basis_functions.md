# About the basis functions

The basis functions $\psi_4$ and $\psi_6$ and their Fourier transforms have several properties.

When a model is approximated using $\psi_6$, the `celerite` coefficients are:

```math
\begin{align}
a &= a_j f_j \pi/3
b &= 0
c &= 2\pi f_j
d &= 0

a &= a_j f_j \pi/3
b &=a_j f_j \pi / \sqrt{3}
c &= \pi f_j
d &= \pi \sqrt{3} f_j
\end{align}
```

The basis function at zero lag is given by $2\pi/3 a_j f_j$.

We can also obtain the integral of the basis functions in the Fourier domain from $f_\mathrm{min}$ and $f_\mathrm{max}$.