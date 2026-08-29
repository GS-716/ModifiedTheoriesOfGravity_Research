# Proyección tensorial y validación Wolfram

## Evaluador

`SympyComponentBackend` proyecta recursivamente una expresión IR. Resuelve:

- índices libres y contracciones de Einstein por nombre y varianza;
- métrica covariante, contravariante y mixta;
- Riemann con combinaciones arbitrarias de índices altos y bajos;
- el gradiente escalar `u_a`;
- derivadas covariantes, incluidas divergencias contraídas;
- tensores adicionales registrados explícitamente por el usuario.

`evaluate()` retorna `ComponentEvaluation`, serializable en la IR. Para trabajo
numérico o manipulación escalar posterior, `evaluate_sympy()` conserva las
expresiones nativas de SymPy. `evaluate_field_equations()` proyecta `E_ab` y
`E_phi` y selecciona las componentes métricas independientes `a<=b` sin borrar
ecuaciones nulas, de modo que la trazabilidad no dependa del ansatz.

## Validación independiente

La operación `verify_phase7` del puente construye FLRW directamente como un
`CTensor` de xCoba. No consume los resultados de SymPy. Comprueba ocho grupos:

1. inversa métrica;
2. `Gamma^0_ii`;
3. `Gamma^i_0i=Gamma^i_i0`;
4. componentes de Ricci;
5. escalar de Ricci;
6. componentes de Einstein;
7. anulación de componentes fuera de la diagonal;
8. `Box(phi)` para un escalar homogéneo.

El informe reproducible se genera con
`python scripts/validate_phase7_wolfram.py` y se guarda por defecto en
`outputs/phase7_wolfram_validation.json`.
