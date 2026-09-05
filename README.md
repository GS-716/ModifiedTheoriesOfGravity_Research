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
    H --> J{"¿Resolver las ecuaciones de campo?"}
    J -- "No" --> I["Analizar, comparar y documentar los modelos"]
    J -- "Sí" --> K["FieldEquationsSolver: reducir, clasificar y resolver formalmente"]
    K --> I
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
| [FieldEquationsSolver](FieldEquationsSolver/README.md) | Reducción, clasificación y resolución formal opcional de las ecuaciones de campo |
| [ResearchWorkflow](ResearchWorkflow/README.md) | Notebook integrado y salidas producidas durante el trabajo interactivo |
| [Papers](Papers/) | Bibliografía y documentos de referencia, incluido el Draft 4 |
| [Beamer](Beamer/) | Fuentes LaTeX y material de presentación de la investigación |

Para empezar, consulta la [guía del motor](TensorEngine/README.md) y abre el
[flujo de investigación](ResearchWorkflow/01_modified_gravity_workflow.ipynb).
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

## Guía de instalación y ejecución en una PC nueva

Esta guía describe una instalación reproducible de **ModifiedTheoriesOfGravity_Research**
y de **TensorEngine** en Windows. Las versiones indicadas son las comprobadas en
la instalación de desarrollo actual; conviene mantenerlas fijas mientras se
reproduce un cálculo publicado.

### Versiones de referencia

| Componente | Versión comprobada | Obligatorio | Uso |
|---|---:|:---:|---|
| Windows x64 | Entorno actual de Windows x64 | Sí | Sistema operativo de referencia |
| Git | 2.51.2.windows.1 | Sí | Clonar y actualizar el repositorio |
| Python | 3.12.6 | Sí | Frontend, IR, backends y reportes |
| pip | 25.0 | Sí | Instalar el paquete y sus dependencias |
| SymPy | 1.14.0 | Sí | Álgebra simbólica y operaciones tensoriales |
| pytest | 9.1.1 | Para pruebas | Suite automatizada |
| Wolfram Engine | 15.0.0 | Opcional, recomendado | Validación externa |
| wolframscript | 1.14.0 | Si se usa Wolfram | Ejecutar Wolfram Engine desde PowerShell/Python |
| xAct/xTensor | 1.3.0 | Si se usa Wolfram | Validación tensorial en Wolfram |
| xAct/xPert | 1.0.6 | Opcional | Perturbaciones, si el cálculo las requiere |
| xAct/xTras | 1.4.2 | Opcional | Herramientas adicionales de simplificación |
| xAct/xCoba | No instalado en la referencia | Opcional | Componentes en Wolfram; no es requisito del backend Python |
| Strawberry Perl | 5.42.2 | Para xAct en Windows | Intérprete usado por algunas herramientas de xAct |
| MiKTeX | 26.5 | Para PDF | Compilar los reportes LaTeX |

Las versiones futuras pueden funcionar, pero deben tratarse como una nueva
configuración y verificarse con las pruebas antes de usarla para comparar
resultados. Las huellas matemáticas y los reportes deben conservarse junto con
la versión del entorno.

### 1. Instalar las herramientas del sistema

Instala Git, Python 3.12.6 x64, Strawberry Perl 5.42.2 x64 y MiKTeX 26.5.
Durante la instalación de Python habilita el lanzador `py` y, si es posible,
la opción de añadir Python al `PATH`. En MiKTeX habilita la instalación
automática de paquetes bajo demanda.

Comprueba las versiones desde PowerShell:

~~~powershell
git --version
py -3.12 --version
perl --version
pdflatex --version
~~~

Si `perl` o `pdflatex` no se reconocen, añade sus directorios `bin` al `PATH`.
En la instalación de referencia son, respectivamente, `C:\Strawberry\perl\bin`
y `D:\MiKTeX\miktex\bin\x64`.

### 2. Clonar el repositorio

~~~powershell
git clone https://github.com/GS-716/ModifiedTheoriesOfGravity_Research.git
cd ModifiedTheoriesOfGravity_Research
~~~

Si ya existe una copia local del repositorio renombrado:

~~~powershell
git remote set-url origin https://github.com/GS-716/ModifiedTheoriesOfGravity_Research.git
git fetch origin
git switch main
git pull --ff-only origin main
~~~

### 3. Crear el entorno Python del proyecto

TensorEngine y FieldEquationsSolver se instalan como paquetes editables para que
el notebook, las pruebas y los scripts usen el código local del repositorio.

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip==25.0
python -m pip install -e ".\TensorEngine[dev]"
python -m pip install -e ".\FieldEquationsSolver[dev]"
~~~

Si PowerShell bloquea la activación del entorno para el usuario actual:

~~~powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
~~~

Verifica el entorno instalado:

~~~powershell
python --version
python -m pip --version
python -c "import sympy; print(sympy.__version__)"
python -c "import tensor_engine; print(tensor_engine.__file__)"
python -c "import field_equations_solver; print(field_equations_solver.__file__)"
~~~

Jupyter es opcional para ejecutar los notebooks. Si se desea utilizarlo:

~~~powershell
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name tensor-engine --display-name "Python 3.12 (TensorEngine)"
jupyter lab
~~~

El motor también puede ejecutarse desde scripts Python, por lo que Jupyter no
es necesario para los cálculos ni para las pruebas.

### 4. Instalar y configurar Wolfram Engine

Descarga e instala **Wolfram Engine 15.0.0 para Windows x64** desde Wolfram
Research y activa la licencia correspondiente. Comprueba que el ejecutable
esté disponible:

~~~powershell
wolframscript -version
~~~

Si `wolframscript` no está en el `PATH`, usa la ruta completa del ejecutable o
añade al `PATH` el directorio de Wolfram Engine que contiene `wolframscript.exe`.
La instalación de referencia responde como `WolframScript 1.14.0` y utiliza
Wolfram Engine `15.0.0`.

### 5. Instalar xAct y comprobarlo

Descarga xAct desde [xact.es](https://xact.es/download/install) y copia la
carpeta `xAct` dentro de la carpeta `AddOns\\Applications` de Wolfram Engine.
En la instalación de referencia la ruta es:

~~~text
C:\Program Files\Wolfram Research\Wolfram Engine\15.0\AddOns\Applications\xAct
~~~

Comprueba que xTensor pueda cargarse. En PowerShell, conserva las comillas
simples exteriores para que los backticks de Wolfram lleguen intactos:

~~~powershell
wolframscript -code 'Needs["xAct`xTensor`"]; Print["xAct cargado correctamente"]' -local
~~~

TensorEngine utiliza el puente Python para ejecutar una comprobación más
completa y registrar las versiones detectadas:

~~~powershell
python -c "from tensor_engine import WolframXActBridge; print(WolframXActBridge().ping())"
~~~

La respuesta de referencia indica `status: success`, Wolfram `15.0.0`,
`xAct_available: True`, xTensor `1.3.0`, xPert `1.0.6` y xTras `1.4.2`.
Si xAct no está disponible, TensorEngine conserva los resultados simbólicos y
marca la validación como no disponible; no convierte esa ausencia en una
identidad aprobada.

### 6. Ejecutar un cálculo de prueba

Con el entorno virtual activo y desde la raíz del repositorio, ejecuta primero
las pruebas de ambos paquetes:

~~~powershell
Push-Location TensorEngine
python -m pytest -q
Pop-Location
Push-Location FieldEquationsSolver
python -m pytest -q
Pop-Location
~~~

Después abre el notebook:

~~~powershell
jupyter lab ResearchWorkflow/01_modified_gravity_workflow.ipynb
~~~

La interfaz permite declarar el lagrangiano, sus parámetros y la dimensión,
seleccionar un ansatz como `draft4_circular_ansatz()` o FLRW y solicitar una
especialización posterior de `f(r)` y `phi`. El flujo conserva separadas la
derivación abstracta, la proyección, la especialización, la validación xAct y
la exportación.

Un uso mínimo desde Python tiene esta forma:

~~~python
from tensor_engine import (
    DimensionSpec,
    LagrangianSourceSpec,
    TensorEngine,
    draft4_circular_ansatz,
)

source = LagrangianSourceSpec(
    name="einstein_hilbert",
    expression="R",
    dimension=DimensionSpec(3),
)
run = TensorEngine().run(
    source.compile(),
    ansatz=draft4_circular_ansatz(),
    output_root="outputs",
)
print(run.summary_data())
~~~

Las corridas del notebook se guardan bajo `ResearchWorkflow/outputs/`. Allí se
encuentran `results.json`, el manifiesto, `presentation.json`, el `.tex` y,
cuando MiKTeX está disponible, el PDF. `run.abstract` conserva las expresiones
covariantes y `run.projected` las expresiones proyectadas; una limitación de
componentes no impide conservar la forma abstracta.

### 7. Lista de comprobación para otra PC

Antes de confiar en una corrida, confirma lo siguiente:

- `python --version` devuelve 3.12.6 y el entorno virtual está activo.
- SymPy devuelve 1.14.0 y `python -m pytest -q` termina correctamente.
- `wolframscript -version` devuelve 1.14.0 y Engine responde como 15.0.0.
- `Needs["xAct`xTensor`"]` se ejecuta sin error.
- El puente Python reporta `xAct_available: True` si se espera validación xAct.
- `pdflatex --version` está disponible si se necesita el PDF.
- El notebook usa el kernel del entorno `tensor-engine`.
- La corrida registra versiones, estados de validación y limitaciones en sus
  artefactos; no se deben comparar únicamente las expresiones del PDF.

Si una actualización cambia el resultado, conserva el reporte anterior,
registra las nuevas versiones y ejecuta la suite completa antes de aceptar la
nueva configuración como equivalente.
