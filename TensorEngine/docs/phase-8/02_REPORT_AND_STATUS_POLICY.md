# Informe y política de estados

`VerificationReport` es inmutable, serializable y contiene versión de esquema,
modelo, backend, comprobaciones, fuentes externas y resumen.

El estado agregado se determina sin excepciones implícitas:

- `success`: todas las comprobaciones disponibles son `passed`;
- `partial`: no hay fallos, pero existe al menos un `undetermined`;
- `failed`: existe al menos una comprobación `failed`.

Todo resultado `failed` o `undetermined` conserva un residual. Una identidad que
requiere Bianchi multitémino, conmutación de derivadas o un backend más potente
permanece `undetermined`; nunca se sustituye por cero ni cuenta como aprobada.

Las claves son estables y únicas. Esto permite comparar informes entre modelos,
versiones y backends sin analizar texto de consola.
