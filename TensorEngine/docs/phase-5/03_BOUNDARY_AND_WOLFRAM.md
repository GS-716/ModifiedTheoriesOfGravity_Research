# Fase 5.3 — Frontera y puente Wolfram/xAct

El potencial escalar es

\[
\Theta_\phi^a=J^a\delta\phi.
\]

Para la métrica inversa, el potencial se almacena como

\[
\Theta_g^a
=-2P^{abcd}g_{bm}g_{cn}\nabla_d\delta g^{mn}
+2g_{bm}g_{cn}(\nabla_dP^{abcd})\delta g^{mn}.
\]

El resultado total satisface la estructura

\[
\delta(\sqrt{-g}L)
=\sqrt{-g}\left(
E_{ab}\delta g^{ab}
+E_\phi\delta\phi
+\nabla_a\Theta^a
\right),
\qquad
\Theta^a=\Theta_g^a+\Theta_\phi^a.
\]

## Wolfram/xAct

`WolframXActBridge` implementa el transporte local:

```text
Python IR -> request.json -> wolframscript -> xAct -> response.json -> Python IR
```

No requiere una API web. `TensorEngineBridge.wl` implementa:

- `ping`, que separa la versión real de Wolfram Engine de las versiones de
  xTensor, xPert y xTras;
- `verify_phase5`, que construye referencias independientes, canoniza sus
  residuales y devuelve `passed`, `failed` o `undetermined` para cada identidad.

`WolframXActBridge.validate_phase5()` convierte la respuesta en
`WolframPhase5Report`. El informe conserva versiones, convenciones, residuales
textuales de xAct y adaptadores al contrato común `VerificationRecord`.

La ejecución reproducible desde la raíz del proyecto es:

```powershell
python scripts/validate_phase5_wolfram.py
```

El resultado se escribe en `outputs/phase5_wolfram_validation.json`. La prueba
de integración automatizada es deliberadamente opt-in para que la suite básica
no dependa de una licencia local:

```powershell
$env:TENSOR_ENGINE_RUN_WOLFRAM_TESTS="1"
python -m pytest tests/test_wolfram_bridge.py
```

## Mapa de convenciones

xAct almacena la curvatura con el par de índices del conmutador primero. La
comparación con xPert fija explícitamente el mapa usado por el puente:

\[
R_{\mathrm{TE}}{}^a{}_{bcd}
=-R_{\mathrm{xAct}\,cd\,b}{}^a.
\]

Este mapa se aplica sólo en la frontera con xAct; no modifica las convenciones
normativas internas de TensorEngine.

## Validación ejecutada

El 28 de agosto de 2026 se ejecutó localmente con Wolfram Engine 15.0.0,
xTensor 1.3.0, xPert 1.0.6 y xTras 1.4.2. Las 19 comprobaciones finalizaron
como `passed`, sin fallos ni resultados indeterminados. Cubren simetrías y
Bianchi, el momento de Einstein–Hilbert, los términos algebraico y diferencial
de curvatura, `E_ab`, ambos potenciales de frontera, integración por partes
escalar y las identidades de Palatini comparadas con xPert.

La traducción genérica completa IR-xAct-IR permanece como una ampliación del
backend externo; esta validación usa referencias cerradas y auditables de fase 5.
