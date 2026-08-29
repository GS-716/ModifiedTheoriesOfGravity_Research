# Criterios de aceptación de fase 12

La fase queda cerrada cuando:

1. la solicitud genérica contiene once controles y estrategias enumeradas;
2. Wolfram Engine/xAct reduce los once residuales de referencia a cero;
3. las dos identidades diferenciales respaldan explícitamente sus claves
   internas;
4. Python rechaza evidencia de otro cálculo y exige unanimidad para adjudicar;
5. `VerificationReport` y `RunManifest` conservan la cadena de adjudicación;
6. la corrida integral de referencia termina en `success`, sin fallos ni
   indeterminados;
7. la regresión Python y las pruebas vivas Wolfram/xAct terminan sin fallos.

Los residuales no demostrados en modelos futuros seguirán siendo
`undetermined`; la fase no introduce reglas de simplificación optimistas.
