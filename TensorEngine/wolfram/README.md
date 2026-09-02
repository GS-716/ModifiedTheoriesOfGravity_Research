# Backend Wolfram/xAct

`TensorEngineBridge.wl` recibe solicitudes JSON desde Python mediante
`wolframscript`. `ping` reporta por separado Wolfram Engine, xTensor, xPert y
xTras. `verify_phase5` ejecuta 19 identidades de referencia y devuelve estados
estructurados con cualquier residual no nulo.

No se usa una API web: Python inicia un kernel local, intercambia archivos JSON
temporales y conserva la representación intermedia como contrato común. Las
operaciones IR-xAct se habilitarán de manera incremental y deberán devolver una
respuesta estructurada, nunca texto Mathematica sin validar. La validación de
fase 5 ya aplica el mapa explícito
`R_TE^a_bcd = -R_xAct_cd b^a` antes de comparar con xPert.

`verify_phase6` comprueba variaciones de Lie, corriente de Noether, Bianchi
contraída y potencial de Iyer–Wald. Su resultado usa el mismo contrato tipado y
los mismos estados `passed`, `failed` y `undetermined`.

`verify_phase7` construye el ansatz FLRW como `CTensor` y usa xCoba para
comprobar conexión, curvatura, Einstein y el laplaciano de un escalar homogéneo.

`verify_model` recibe residuales IR de una corrida concreta, los traduce mediante
un decodificador enumerado y ejecuta once controles con tres estrategias:
canonización algebraica, proyección de Bianchi de Riemann y reducción
diferencial. Esta última ordena derivadas covariantes y aplica la Bianchi
contraída mediante xTras. La respuesta repite los fingerprints del modelo y del
cálculo; Python comprueba el eco antes de aceptar la evidencia.

Los identificadores de la IR se codifican en símbolos internos de Wolfram sin
evaluar texto. Esto mantiene nombres válidos de Python como `alpha_1` y evita
que `_` se interprete como patrón. El decodificador valida cada nodo y devuelve
`undetermined` con un motivo concreto cuando el transporte no es seguro. La
ausencia de transporte o de reducción nunca produce un `passed`.

Cuando el transporte falla, cada check conserva un `transport_diagnostic`
JSON-seguro con código, categoría, ruta dentro del residual, tipo de nodo,
símbolo y el fragmento IR concreto. El residual textual permanece disponible
para consumidores anteriores. TensorEngine integra estos datos en
`verification.json`, `results.json` y en el resumen LaTeX/PDF.

Cada comprobación puede declarar `adjudicates`: una lista explícita de controles
internos que esa evidencia puede resolver. El puente no realiza la adjudicación;
solo devuelve la prueba. Python aplica la política una vez validados los
fingerprints.
