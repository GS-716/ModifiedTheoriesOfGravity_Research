# Contrato de campaña

`CampaignSpec` contiene una clave estable por entrada y el `ModelSpec` completo.
Exige claves y nombres de modelo únicos. `CampaignRunner` crea una instancia
aislada del motor para cada teoría y aplica las mismas `EngineOptions` y el mismo
tipo de validación externa a todas.

Cada `CampaignRecord` conserva:

- estado, conteos de verificación y duración;
- `run_id` y ruta del bundle cuando la corrida existe;
- diagnóstico cuando una teoría no pudo completarse.

Una excepción de un modelo no interrumpe los siguientes. El reporte global es
`failed` si alguna entrada falla, `partial` si ninguna falla pero alguna queda
parcial, y `success` solo si todas son exitosas. La política estricta no cambia
los resultados, únicamente la aceptación del reporte.
