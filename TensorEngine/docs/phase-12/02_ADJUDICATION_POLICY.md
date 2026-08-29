# Política de adjudicación

Una comprobación interna `undetermined` puede pasar a `passed` únicamente si se
cumplen simultáneamente estas condiciones:

1. la evidencia procede de `verify_model`;
2. el nombre y los fingerprints de modelo y cálculo ya fueron validados;
3. la comprobación xAct declara explícitamente la clave interna en
   `adjudicates`;
4. toda evidencia declarada para esa clave tiene estado `passed`.

La política nunca modifica una comprobación `failed`. Tampoco adjudica reportes
fijos de fases anteriores, resultados sin fingerprints, coincidencias de nombre
implícitas ni evidencia parcial. Si existen pruebas contradictorias, el control
interno conserva `undetermined`.

En la referencia de fase 12 se adjudican exactamente
`noether_current_decomposition` y `diffeomorphism_noether_identity`.
