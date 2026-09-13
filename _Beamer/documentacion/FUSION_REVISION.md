# Fusión y revisión del Beamer

Versión final: 13 de septiembre de 2026. Respaldo iniciado el 12 de septiembre.

## Carpeta activa y procedencia

`_Beamer` es la carpeta activa unificada. La comparación mostró que `Beamer_holografia_generalizada_SISTEMA` conservaba el desarrollo anterior y añadía ocho frames de holografía generalizada, sus mapas y navegación para ocho secciones. Sus tablas, cierre y holografía LL coincidían con `_Beamer`.

Se integraron sus cambios en `_Beamer`, se revisó la nueva teoría y se regeneraron los índices. Ambos estados de entrada están conservados en `backups/pre-fusion-20260912/`: `base/` contiene la versión de 154 páginas y `sistema/` la de 165. Los auxiliares del sistema anterior se conservan solo en el respaldo, no como fuentes activas.

## Compactación y lectura

- Holografía LL: de 21 páginas a 10, incluidas apertura y balance. Se conservan los rótulos H1–H15. Se agrupan la construcción mecánica, el caso de momento fijo y campos, la separación geométrica y las dos integraciones por partes. Se usa explícitamente la densidad con subíndices `vol` y `sup` en lugar de B y S.
- Holografía generalizada: 10 frames. Se amplían los pasos justificativos y los controles dentro del límite pedido.
- Estrategia y alcance se integran en un solo frame. Bianchi y Noether se incorporan al resumen variacional. Todas las secciones conservan o reciben un resumen explícito de resultados; los resúmenes escalar y de contraste se restituyen. No se retira ninguna ecuación numerada anterior.
- Las ocho secciones tienen mapa completo de apertura. Una fila de casillas muestra los mismos hitos durante el desarrollo: verde y marca de verificación para lo completado, azul para el siguiente bloque, gris para lo pendiente. Las otras seis secciones conservan además su mapa completo al cierre, antes del resumen. En holografía, mapa de apertura y resumen se incluyen en el límite de 10 frames. Para ello se reúnen la homogeneidad y la identidad LL, y las hipótesis y la deducción del defecto generalizado.
- Los frames antiguos que necesitaban espacio para esa fila recibieron ajustes locales de separación entre párrafos, renglones y ecuaciones, sin reducir sus tamaños de letra.
- Resultado: 151 páginas frente a 165 de SISTEMA. Son 138 páginas de exposición y 13 de índice detallado. Los índices están al final y se accede a ellos desde «Contenido».

## Revisión de mapas, resúmenes y notación

Se detectaron dos mapas de apertura omitidos por la macro de compactación: HLL y holografía generalizada. También se recuperaron de la compilación verificada tres frames de controles de Q_beta ausentes de la fuente activa, y se corrigió el mapa generalizado de cinco pasos frente a un total declarado de seis. El mapa LL menciona Bianchi explícitamente; el generalizado incluye terceros gradientes, homogeneidad, defecto, superficie efectiva, Q_beta y límites variacionales.

Los ocho resúmenes están en las páginas 27, 55, 59, 69, 90, 117, 127 y 136. Los mapas de apertura se encuentran en 6, 28, 56, 60, 70, 91, 118 y 128. La página 137 conserva una síntesis global adicional.

El título ahora sitúa el trabajo en teorías de gravedad modificada. Se distingue L (escalar), script L (densidad) y ell (escala de longitud fija de EQT). Se reserva theta para la coordenada angular y phi para el campo escalar. El generador del primer ejemplo de Lie pasa de X a xi; X sigue siendo el invariante cinético en los capítulos escalares (su uso auxiliar local en las demostraciones de simetría está definido allí). El operador holográfico usa una H caligráfica legible. La página 5 explica las convenciones y que los momentos definidos respecto de métricas covariante y contravariante no son simplemente la misma derivada con índices bajados. Estos cambios son notacionales, no modificaciones de la acción.

La copia `pre-mapas-resumenes-20260913/` conserva el estado de trabajo anterior a esta revisión. La carpeta SISTEMA duplicada se retira solo tras verificar por SHA-256 todos sus archivos contra el respaldo previo.

## Revisión de la generalización

La referencia LL es Padmanabhan y Kothawala, §2.7, [arXiv:1302.2151](https://arxiv.org/abs/1302.2151). La generalización es una deducción a partir de la variación del Beamer, no un resultado que se atribuya a ese artículo.

1. **Volumen completo.** Fuera de LL, el volumen incluye el término con la conexión y la divergencia de P. Omitirlo no reproduce la acción original.
2. **Orden diferencial.** Si P depende de la curvatura, su divergencia puede introducir terceros gradientes métricos en esa separación. Se explicita el operador de Euler con un término adicional. Para Q_beta, lineal en curvatura y con P dependiente de g y u, el volumen tiene primeros gradientes métricos y hasta segundos gradientes escalares.
3. **Hipótesis de homogeneidad.** Se exige un peso definido bajo un escalamiento métrico constante, manteniendo el escalar y sus derivadas coordenadas fijos. Una suma con pesos distintos se analiza sector a sector.
4. **Defecto.** La identidad se deriva directamente de la variación completa. Que la divergencia de P sea cero basta para anular el defecto, pero que el defecto sea cero no implica que la teoría sea LL.
5. **Superficie efectiva.** Su definición es algebraicamente válida para peso distinto de cero. La fórmula actúa sobre el volumen original. Si se cambia también el volumen para conservar la densidad original, la identidad de la pareja nueva sigue conteniendo el defecto. Se retiró la afirmación demasiado fuerte de que esta redefinición recuperaba por sí sola la estructura LL.
6. **Q_beta.** Se conserva y deriva el peso D/2−2, que no coincide con D/2−m pese a que m=1. La contracción y el signo de la corrección en D=3 se verifican explícitamente. En D=4 no se puede dividir por el peso.
7. **Ansatz.** Se añade un control directo de la divergencia contraída y de la corrección superficial sobre la métrica estática. Muestra por qué anular el defecto en una geometría no establece una identidad LL de la teoría.
8. **Principio variacional.** La variación de la superficie no tiene por qué coincidir sola con la corriente de frontera completa: falta la corriente del volumen. Si la variación escalar no se fija, debe conservarse su momento en la frontera. No se presenta como resuelto un problema general de contraterminos o condiciones de contorno.

## Puntos de simplificación para una revisión posterior

Se conservaron las deducciones indexadas de Palatini, las simetrías de P y las variaciones de Q_beta. Una reducción adicional importante requeriría elegir entre lectura detallada y exposición oral. Los candidatos más claros son integrar controles EH/cosmológico en sus tablas comparativas, y reunir algunas páginas de resultado con la última página de su derivación. No se aplicaron esas supresiones porque las tablas repiten resultados, pero no sustituyen los pasos de comprobación.

## Comprobaciones reproducibles

La versión del repositorio incorpora `herramientas/generar_indice.py` y `herramientas/verificar_pdf.py`, con instrucciones en el README del Beamer. Los respaldos y las imágenes de revisión se conservan localmente y no se publican. La lectura integral resulta ordenada como material de estudio detallado; no se presenta como una charla breve.

En `.review/fusion/` quedan la compilación aislada, los renders, `audit.py` y `check_math.py`. El auditado de enlaces comprueba el regreso a Contenido en todas las páginas y la cobertura completa de los frames por los índices. Conserva las 77 ecuaciones numeradas de la versión SISTEMA y sus números, y añade G1–G10.

Resultado de la auditoría: 1.682 enlaces internos resueltos, ocho mapas de apertura, ocho resúmenes, 10 páginas por capítulo holográfico y ninguna advertencia de desborde ni referencia indefinida. Se inspeccionaron visualmente las 151 páginas renderizadas, incluidos los índices; los ajustes finales de notación se volvieron a compilar y revisar en sus páginas afectadas.

Las pruebas auxiliares verifican la contracción de P_beta en dimensiones 3–6, la divergencia covariante sobre el ansatz estático y la identidad de Euler de tercer gradiente con un ejemplo homogéneo. Son controles adicionales, no sustitutos de las deducciones incluidas.
