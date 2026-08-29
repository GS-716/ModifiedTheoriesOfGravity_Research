# Fase 5.2 — Ecuaciones de Euler-Lagrange

Después de restituir la dependencia geométrica e integrar por partes, el motor
usa la forma universal compatible con las convenciones de fase 0:

\[
E_{ab}
=M_{ab}
-P_{(a}{}^{cde}R_{b)cde}
-2\nabla^c\nabla^dP_{acdb}
-\frac12g_{ab}L.
\]

La simetrización en `a,b` se aplica explícitamente; `E_ab` se almacena siempre
en forma manifiestamente simétrica.

Para el campo escalar,

\[
E_\phi=F_\phi-\nabla_aJ^a.
\]

`EulerLagrangeResult` conserva:

- `metric_euler`;
- `scalar_euler`;
- las tres formas del potencial de frontera;
- la variación completa dentro de `sqrt(-g)`;
- la variación de la densidad `sqrt(-g)L`.

## Casos de referencia

Las pruebas verifican:

- `L=R`, que produce la forma del tensor de Einstein;
- `L=-2 Lambda`, que produce `Lambda g_ab`;
- el término cinético escalar canónico;
- un potencial `V(phi)`;
- desaparición de `nabla nabla P` para Einstein–Hilbert.
