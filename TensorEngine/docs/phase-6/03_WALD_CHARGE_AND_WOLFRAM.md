# Fase 6.3 — Potencial de carga de Wald y validación externa

Para

\[
P^{abcd}=\frac{\partial L}{\partial R_{abcd}},
\]

el potencial antisimétrico adoptado es

\[
Q_\xi^{ab}
=-2P^{abcd}\nabla_c\xi_d
+4\xi_d\nabla_cP^{abcd}.
\]

La proyección antisimétrica en `a,b` se aplica explícitamente. Para
Einstein–Hilbert, `nabla P=0` y

\[
Q_\xi^{ab}
=-\nabla^a\xi^b+\nabla^b\xi^a,
\]

que es el potencial de Komar con la normalización del proyecto. Un sector que
depende sólo de `phi` y `nabla phi` contribuye a `J_xi^a` y a las restricciones,
pero no añade un término propio a `Q_xi^{ab}` dentro del alcance actual.

## Ambigüedades controladas

`Theta`, `J` y `Q` admiten las ambigüedades estándar por términos exactos y por
la adición de una divergencia al lagrangiano. TensorEngine almacena la
representante producida directamente por el potencial de fase 5 y la fórmula
anterior. No identifica representantes distintos sin una regla declarada.

## Wolfram/xAct

La operación `verify_phase6` ejecuta nueve referencias independientes:

1. variación de Lie de la métrica inversa;
2. variación de Lie del escalar;
3. reducción Einstein–Hilbert a Komar;
4. antisimetría de `Q_EH`;
5. descomposición fuera de capa de `J_EH`;
6. identidad de Noether y Bianchi contraída;
7. antisimetría de la fórmula general con un `P` de tipo Riemann;
8. descomposición para el acoplamiento no minimal `f(phi)R`, que fija el signo
   del término `4 xi_d nabla_c P^{abcd}`;
9. ausencia de carga adicional en el sector escalar de primer orden.

La ejecución local del 28 de agosto de 2026 con Wolfram Engine 15.0.0,
xTensor 1.3.0 y xTras 1.4.2 aprobó las nueve comprobaciones, sin fallos ni
resultados indeterminados.

El informe se regenera con:

```powershell
python scripts/validate_phase6_wolfram.py
```
