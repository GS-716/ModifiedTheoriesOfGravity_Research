# Controles genéricos y límites

Para cada cálculo soportado, xAct recibe y canoniza ocho residuales:

1. simetría de `M_ab`;
2. antisimetría del primer par de `P^{abcd}`;
3. antisimetría del segundo par;
4. intercambio de pares de `P^{abcd}`;
5. simetría de `E_ab`;
6. suma de los sectores de frontera;
7. factorización de la variación de densidad;
8. antisimetría de `Q_xi^{ab}` cuando se calcula Noether–Wald.

El decodificador es genérico para el dominio `L(g,R,phi,nabla phi)`, pero esta
fase no presenta la identidad diferencial completa de Noether como control
genérico xAct. Su reducción exige una política específica de conmutación de
derivadas y Bianchi. Se conserva como `undetermined` cuando el backend
estructural no puede decidirla, y las fórmulas universales continúan cubiertas
por la validación especializada de fase 6.

Esta separación evita convertir una incapacidad de simplificación en un fallo
matemático o, peor aún, en un cero supuesto.
