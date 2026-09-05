# ModifiedTheoriesOfGravity_Research

Repositorio de investigación sobre teorías modificadas de la gravedad y sus
acoplamientos con un campo escalar. Reúne el material teórico, las presentaciones
y una herramienta de cálculo reproducible: **TensorEngine**.

El proyecto permite formular un lagrangiano covariante, obtener sus momentos y
ecuaciones variacionales, proyectar los resultados sobre una geometría elegida
y documentar el cálculo. Uno de sus contextos de aplicación es el estudio de los
modelos EQT del Draft 4, incluyendo geometrías circulares en tres dimensiones.

## Justificación y objetivos

Comparar lagrangianos mediante derivaciones independientes exige repetir muchas
operaciones y dificulta detectar diferencias de índices, signos o convenciones.
TensorEngine concentra esas operaciones en una representación tensorial común
para que cada caso pueda estudiarse bajo el mismo contrato matemático.

Los objetivos del proyecto son:

- Explorar cómo cambian los momentos, las ecuaciones de campo y los términos de
  frontera al modificar el lagrangiano.
- Conservar la derivación abstracta y distinguirla de la proyección coordenada
  y de la evaluación de perfiles particulares.
- Contrastar identidades con Wolfram Engine/xAct cuando exista soporte, dejando
  constancia de los resultados no validados.
- Producir resultados trazables en Python, JSON y LaTeX/PDF que sirvan para
  comparar modelos y preparar material de investigación.

## Modelo teórico general

El dominio de trabajo es una acción local y covariante

$$
S[g,\phi]=\kappa\int_{\mathcal M}d^D x\,\sqrt{-g}\,
L(g^{ab},R_{abcd},\phi,u_a),
\qquad u_a=\nabla_a\phi.
$$

Se considera una métrica lorentziana no degenerada, su conexión de Levi-Civita y
un campo escalar real. La normalización global $\kappa$ es configurable. El
lagrangiano puede contener contracciones y funciones de la curvatura y del
gradiente escalar, sin derivadas de la curvatura ni segundas derivadas del campo
como argumentos de entrada.

| Grupo de teorías dentro de este dominio | Ejemplos de lagrangianos |
|---|---|
| Gravedad de Einstein con término cosmológico | $R-2\Lambda$ |
| Extensiones métricas de tipo $f(R)$ representables en la IR | $R+\alpha R^2$ |
| Gravedad con invariantes cuadráticos de curvatura | $R+\alpha R_{ab}R^{ab}$, $R+\alpha R_{abcd}R^{abcd}$ |
| Sectores escalares y acoplamientos no mínimos de primer gradiente | $F(\phi)R+K(\phi,X)-V(\phi)$ |
| Acoplamientos algebraicos entre curvatura y gradiente | $R_{ab}\nabla^a\phi\nabla^b\phi$, $RX$ |
| Truncamientos EQT compatibles con estos argumentos | Casos del Draft 4 expresables mediante los invariantes anteriores |

Aquí $X=g^{ab}\nabla_a\phi\nabla_b\phi$, **sin factor $-1/2$ incorporado**.
La pertenencia a este dominio no garantiza que todas las expresiones puedan
reducirse a componentes explícitas con los backends disponibles.

Quedan fuera del contrato actual las teorías con conexión independiente,
torsión o no metricidad, términos no locales, múltiples campos dinámicos y
lagrangianos que dependan explícitamente de $\nabla R$ o
$\nabla_a\nabla_b\phi$. Por ello, el motor no cubre automáticamente todas las
teorías escalares-tensoriales ni todos los modelos de gravedad modificada.
Tampoco presupone que las ecuaciones obtenidas sean de segundo orden.

## Flujo de investigación

~~~mermaid
flowchart TD
    A["Definir el lagrangiano, parámetros y dimensión"] --> B["TensorEngine: compilar a la IR tensorial"]
    B --> C["Derivar momentos, variación y ecuaciones"]
    C --> D["Resultados tensoriales abstractos"]
    D --> E["Proyectar con el ansatz geométrico"]
    E --> F["Especializar perfiles y funciones, si se solicita"]
    C --> G["Verificar identidades en Python y, opcionalmente, xAct"]
    D --> H["Conservar resultados y generar reportes"]
    E --> H
    F --> H
    G --> H
    H --> I["Analizar, comparar y documentar los modelos"]
~~~

El ansatz entra después de la derivación covariante. En el Draft 4 genérico se
mantienen $f(r)$ arbitraria y $\phi=\Phi(r,\varphi)$ estacionaria. El perfil
$\phi=p\varphi$ se impone únicamente mediante una especialización explícita.
FLRW constituye otra geometría de referencia y conserva su campo homogéneo
$\phi(t)$.

## Organización del repositorio

| Carpeta | Contenido |
|---|---|
| [TensorEngine](TensorEngine/README.md) | Motor tensorial, backends, documentación y pruebas |
| [TensorEngine/notebooks](TensorEngine/notebooks/README.md) | Interfaz de trabajo y ejemplos editables para ejecutar modelos |
| [Papers](Papers/) | Bibliografía y documentos de referencia, incluido el Draft 4 |
| [Beamer](Beamer/) | Fuentes LaTeX y material de presentación de la investigación |

Para empezar, consulta la [guía del motor](TensorEngine/README.md) y abre el
[notebook de pruebas](TensorEngine/notebooks/01_quickstart_tensor_engine.ipynb).
La salida distingue resultados calculados, simbólicos, no disponibles y
verificaciones indeterminadas; generar un PDF no equivale a validar toda la
teoría ni a demostrar que un perfil sea una solución.

## Repositorio remoto

La dirección actual es
[GS-716/ModifiedTheoriesOfGravity_Research](https://github.com/GS-716/ModifiedTheoriesOfGravity_Research).

Para una nueva copia:

~~~powershell
git clone https://github.com/GS-716/ModifiedTheoriesOfGravity_Research.git
~~~

Si ya tienes una copia local, conserva su carpeta y actualiza el remoto desde ella:

~~~powershell
git remote set-url origin https://github.com/GS-716/ModifiedTheoriesOfGravity_Research.git
git remote -v
~~~

El nombre del repositorio no obliga a renombrar el paquete Python `tensor_engine`
ni la carpeta local de trabajo.
