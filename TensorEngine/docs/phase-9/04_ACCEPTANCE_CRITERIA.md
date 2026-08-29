# Criterios de aceptación de la fase 9

La fase se considera cerrada cuando:

1. toda corrida puede serializarse y reconstruirse sin pérdida de IR;
2. el ID es estable frente al reloj y cambia frente al contenido matemático;
3. JSON, informe LaTeX y manifiesto se escriben de forma determinista y atómica;
4. los hashes y tamaños del manifiesto coinciden con los archivos reales;
5. el resultado satisface el contrato declarativo de la etapa `export`;
6. estados `partial` y `failed` se preservan y nunca se promocionan a éxito;
7. la ruta calculada permanece bajo la raíz de salida;
8. la suite de regresión completa continúa aprobada;
9. una corrida escalar–tensor de referencia genera todos los artefactos.
