# Configuración, estados y eventos

`EngineOptions` controla las ramas Noether, componentes y exportación. El
ansatz y la raíz de salida se proporcionan por corrida, evitando estado global
oculto.

`EngineRun` expone:

- estado final derivado del informe de verificación;
- paquete matemático reconstruible;
- resultados contractuales ordenados por etapa;
- etapas opcionales omitidas;
- bundle exportado, si existe;
- resumen JSON pequeño para interfaces de usuario;
- política `acceptable(strict=False)`.

El modo normal acepta `success` y `partial`, pero nunca `failed`. El modo
estricto solo acepta `success`; no modifica ni elimina residuales.

Un callback recibe eventos `started`, `completed` y `failed`. Esto permite que
notebooks, aplicaciones o servicios muestren progreso sin insertar lógica
matemática fuera del motor.
