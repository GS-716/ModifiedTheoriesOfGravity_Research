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

Cada comprobación puede declarar `adjudicates`: una lista explícita de controles
internos que esa evidencia puede resolver. El puente no realiza la adjudicación;
solo devuelve la prueba. Python aplica la política una vez validados los
fingerprints.
