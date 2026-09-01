# Gramática de fuentes lagrangianas

Una `LagrangianSourceSpec` describe el modelo mediante una expresión textual y
declaraciones separadas. Los nombres predefinidos son:

- `R`: \(g^{ac}g^{bd}R_{abcd}\);
- `X`: \(g^{ab}\nabla_a\phi\nabla_b\phi\);
- `phi`: campo escalar.
- `RicciUU`: \(R_{ab}u^au^b\), expandido a métricas, Riemann y gradientes.

Los alias provienen de `DEFAULT_INVARIANTS`, un registro extensible, no de una
lista de teorías. También se admite `contract(Riemann(...), metric(...),
gradient(...), ...)`, con índices como cadenas literales. Véase la
[guía de invariantes](../frontend-invariants.md) para la gramática completa.

Se admiten parámetros declarados, funciones escalares declaradas y los
operadores `+`, `-`, `*`, `/`, `**`, paréntesis y signos unarios. Las constantes
deben ser enteras; `1/2` genera una fracción exacta. Para potencias se usa `**`,
nunca `^`.

Ejemplo:

```json
{
  "schema_version": "1.0",
  "name": "scalar_tensor",
  "expression": "F(phi)*R - Z(phi)*X/2 - V(phi)",
  "normalization": "1/kappa",
  "dimension": {"value": 4},
  "parameters": [{"name": "kappa"}],
  "functions": [
    {"name": "F", "arity": 1},
    {"name": "Z", "arity": 1},
    {"name": "V", "arity": 1}
  ]
}
```
