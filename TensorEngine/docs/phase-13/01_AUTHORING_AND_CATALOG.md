# Autoría y catálogo

`ModelBuilder` expone dos invariantes escalares de alto nivel:

\[
R=g^{ac}g^{bd}R_{abcd},\qquad
X=g^{ab}u_a u_b,\quad u_a=\nabla_a\phi.
\]

También permite construir funciones escalares con cualquier aridad declarada.
Estas ayudas solo generan IR: no introducen reglas nuevas ni dependen de SymPy o
xAct.

El catálogo incorporado contiene cinco pruebas complementarias:

| Clave | Lagrangiano |
|---|---|
| `einstein_hilbert` | \(R\) |
| `canonical_scalar` | \(R-X/2-V(\phi)\) |
| `nonminimal_scalar_tensor` | \(F(\phi)R-Z(\phi)X/2-V(\phi)\) |
| `k_essence` | \(R+K(\phi,X)\) |
| `quadratic_ricci_scalar` | \(R+\alpha R^2\) |

El catálogo es una biblioteca de ejemplos, no una lista cerrada: cualquier
`ModelSpec` válido puede entrar en una campaña.
