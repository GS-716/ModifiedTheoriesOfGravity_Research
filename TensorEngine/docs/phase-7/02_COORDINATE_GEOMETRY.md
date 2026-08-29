# Geometría coordenada

`CoordinateGeometry.build(ansatz)` traduce únicamente las expresiones escalares
de la IR a SymPy y calcula, con la convención de fase 0:

\[
\Gamma^\rho{}_{\mu\nu}
=\frac12g^{\rho\sigma}
(\partial_\mu g_{\sigma\nu}+\partial_\nu g_{\sigma\mu}
-\partial_\sigma g_{\mu\nu}),
\]

\[
R^\rho{}_{\sigma\mu\nu}
=\partial_\mu\Gamma^\rho{}_{\nu\sigma}
-\partial_\nu\Gamma^\rho{}_{\mu\sigma}
+\Gamma^\rho{}_{\mu\lambda}\Gamma^\lambda{}_{\nu\sigma}
-\Gamma^\rho{}_{\nu\lambda}\Gamma^\lambda{}_{\mu\sigma}.
\]

El resultado conserva métrica e inversa, determinante, Christoffel, Riemann con
el primer índice alto y todos los índices bajos, Ricci, escalar de Ricci y
Einstein. También implementa gradiente, Hessiano y laplaciano de un escalar, y
la derivada covariante de un `ComponentTensor` de rango arbitrario.

SymPy simplifica componentes exactas; no redefine identidades tensoriales
abstractas ni reemplaza a la IR.
