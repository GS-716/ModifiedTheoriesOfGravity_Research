# Matriz de verificación integral

La fase 8 reúne en una sola política los controles que antes estaban dispersos
entre módulos y pruebas. `RunVerifier` consume el `ModelSpec`, los cuatro
momentos y el resultado Euler–Lagrange; opcionalmente incorpora variación cruda,
Noether–Wald, componentes y evidencia Wolfram.

## Controles obligatorios

- validez del modelo y ausencia de símbolos no declarados;
- firmas de índices de `M_ab`, `P^abcd`, `J^a`, `F_phi`, `E_ab`, `E_phi` y
  potenciales de frontera;
- simetría de `M_ab` y `E_ab`;
- cuatro simetrías de Riemann para `P^abcd`;
- reconstrucción determinista de momentos y objetos Euler–Lagrange;
- `Theta=Theta_g+Theta_phi` y factorización por `sqrt(-g)`;
- integración por partes del sector escalar;
- idempotencia de la canonización.

Cuando están disponibles también se comprueban reconstrucción de la variación
cruda, corriente e identidad de Noether, antisimetría de la carga de Wald y
reproyección completa de las ecuaciones por componentes.
