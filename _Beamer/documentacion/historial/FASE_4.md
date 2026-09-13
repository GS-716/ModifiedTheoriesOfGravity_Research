# Fase 4: revelado acumulativo de las deducciones

**Estado: retirada por decisión del usuario.** Se eliminaron las 44 pausas
al iniciar la fase 5. El registro siguiente es histórico; los mapas y las
tablas de las fases anteriores se conservan. El PDF actual es estático.

La fase 3 queda cerrada con sus ocho tablas y el registro `FASE_3.md`.
Respaldo previo a la fase 4: `backups/fase3-final-20260912-105031/`,
con el PDF, los fuentes y el registro de la fase 3.

Se introducen 44 pausas entre bloques completos de razonamiento en
22 diapositivas. Cada una tiene tres estados acumulativos: lo mostrado
permanece visible y en su posición durante los siguientes avances.
No se insertan pausas dentro de ecuaciones ni se altera su numeración.

## Versiones

- `main.pdf`: 193 páginas físicas para exponer las 149 diapositivas.
  Cada avance del visor revela el siguiente bloque en las diapositivas
  seleccionadas. El pie conserva el número de diapositiva sobre 149.
- `main-lectura.pdf`: 149 páginas, una por diapositiva, con todo visible.
  Se genera a partir del mismo contenido mediante `fase4/lectura.tex`.

## Cobertura

| Diapositiva | Tema | Páginas físicas en exposición |
| --- | --- | --- |
| 21 | Resultado del segundo cálculo de Lie | 21-23 |
| 23 | Comparación con el cálculo escalar | 25-27 |
| 28 | Contracción de curvatura en EH | 32-34 |
| 40 | Palatini | 46-48 |
| 41 | Sustitución de delta Gamma | 49-51 |
| 43 | Integraciones por partes | 53-55 |
| 44 | Separación de volumen y frontera | 56-58 |
| 60 | Corriente explícita de Noether | 74-76 |
| 61 | Divergencia del potencial de Noether | 77-79 |
| 84 | Identidad algebraica generalizada | 102-104 |
| 85 | Variación completa con escalar | 105-107 |
| 90 | Bianchi con escalar | 112-114 |
| 94 | Cancelaciones del sector escalar en Noether | 118-120 |
| 109 | Variación del sector cinético | 135-137 |
| 110 | Integración por partes del término escalar | 138-140 |
| 117 | Momento de curvatura de Q beta | 147-149 |
| 119 | Grupo métrico proporcional a 3 | 151-153 |
| 120 | Grupo métrico con signo negativo | 154-156 |
| 125 | Momento del gradiente y dependencia explícita | 161-163 |
| 128 | Contracción de curvatura de Q beta | 166-168 |
| 131 | Ecuaciones asociadas a Q beta | 171-173 |
| 146 | Comprobación final de la condición LL | 188-190 |

Los mapas progresivos sin flechas y las tablas se mantienen sin cambios.
No se adelanta la fase 5 ni se añade contenido teórico.

## Verificación

- Al retirar las 44 instrucciones de pausa, el fuente coincide con el
  respaldo anterior, salvo espacios en blanco. Las 62 etiquetas y los
  fuentes de las fases 2 y 3 permanecen intactos.
- Revisión visual de los 66 estados de las diapositivas seleccionadas.
- Comparación de cada estado completo y cada página de lectura con la
  fase 3: 148 imágenes exactamente iguales; en la diapositiva 41 hay
  únicamente 49 píxeles de diferencia de rasterización en el mismo
  superíndice a de P, inspeccionado con ampliación. No cambian el símbolo,
  la expresión ni la distribución.
- Verificación de que ningún avance modifica o borra contenido ya visible.
- Ambas compilaciones terminan sin avisos de desborde ni referencias pendientes.

Los controles reproducibles y las imágenes están en `.review/fase4/`.
Las compilaciones de exposición y lectura utilizan directorios separados
para conservar sus referencias y numeraciones independientes.
