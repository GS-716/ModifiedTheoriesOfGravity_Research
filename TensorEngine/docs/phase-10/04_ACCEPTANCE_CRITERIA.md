# Criterios de aceptación de la fase 10

La fase se considera cerrada cuando:

1. una sola llamada ejecuta todas las etapas obligatorias en orden;
2. cada etapa satisface en ejecución su contrato declarativo;
3. la normalización global de la acción afecta los cálculos posteriores;
4. Noether, componentes y exportación pueden activarse independientemente;
5. los estados parciales y fallidos mantienen su semántica;
6. notebooks y software pueden observar eventos sin contener lógica del motor;
7. modelos y ansatz tienen entrada/salida JSON reconstruible y atómica;
8. la CLI valida, ejecuta, exporta y devuelve códigos de salida estables;
9. una corrida escalar-tensor completa genera un bundle íntegro;
10. la suite acumulada no presenta regresiones.
