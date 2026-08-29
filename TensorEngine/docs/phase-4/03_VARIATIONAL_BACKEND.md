# Fase 4.3 — Regla de cadena y backend

`TensorBackend` incorpora métodos para:

- calcular `M_ab`, `P^abcd`, `J^a` y `F_phi`;
- variar directamente una expresión algebraica;
- reconstruir `delta L` desde sus momentos;
- variar `g_ab`, `sqrt(-g)` y restituir `delta u_a=nabla_a delta phi`.
- construir la variación independiente `delta R_abcd` antes de Palatini.

La variación cruda se conserva en la forma

\[
\delta L=M_{ab}\delta g^{ab}
+P^{abcd}\delta R_{abcd}
+F_\phi\delta\phi
+J^a\delta u_a.
\]

`VariationalContext` obtiene desde `ModelSpec` los nombres geométricos, el
espacio de índices y los escalares constantes. Esto permite que el mismo motor
funcione con nombres personalizados sin codificarlos en las reglas.

El backend `structural-python` declara las capacidades `elementary_variation`,
`lagrangian_momenta` y `riemann_projection`.

## Papel de xAct

Mathematica/xAct aún no participa en la ejecución de esta fase. El formato de
los momentos y las variaciones ya está preparado para enviarse a un backend
xAct. Su primera función necesaria será canonizar y verificar las expresiones
que surjan al expandir `delta R_abcd` mediante Palatini y al integrar por partes.
