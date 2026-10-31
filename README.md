# Marketing Mix Model con PyMC-Marketing

[![CI](https://github.com/hysmarcos/marketing-mix-model/actions/workflows/ci.yml/badge.svg)](https://github.com/hysmarcos/marketing-mix-model/actions/workflows/ci.yml)

> In English: A Bayesian Marketing Mix Model built with PyMC-Marketing on Meta's Robyn demo dataset (208 weeks, 5 paid channels). It estimates the sales each channel drives, passes strict convergence checks and predicts 12 held-out weeks with a 2.9% error. A static page with an in-browser budget simulator shows that the same weekly budget, reallocated, drives about 35% more media sales.

## La pregunta

¿Cómo reparto el presupuesto de marketing del próximo trimestre entre los canales?

La respuesta corta: pasar inversión de vía pública, TV y búsqueda a prensa y Facebook. Con el mismo presupuesto semanal, las ventas que generan los medios suben un 35% según el modelo.

Página con los resultados y el simulador: https://hysmarcos.github.io/marketing-mix-model/

Los datos son simulados por Robyn, la herramienta de MMM de Meta. No son datos de una empresa real.

[![Simulador de presupuesto con el reparto óptimo aplicado](docs/img/simulador.png)](https://hysmarcos.github.io/marketing-mix-model/)

## Resultados

Un Marketing Mix Model (MMM) estima cuántas ventas genera cada canal por cada peso invertido. El retorno por canal es la cantidad de ventas que vuelve por cada peso. A la derecha de la línea en 1, el canal recupera lo invertido.

![Retorno por canal con su rango del 90%](docs/img/roi.png)

Vía pública es el único canal que no recupera lo invertido (0,5 por cada peso). Facebook tiene el retorno más alto (12,3), pero con un rango amplio, porque tiene poca inversión en los datos.

| Canal | Inversión semanal actual | Inversión semanal óptima |
|---|---:|---:|
| TV | $ 9.474 | $ 6.326 |
| Vía pública | $ 9.649 | $ 0 |
| Prensa | $ 5.938 | $ 13.753 |
| Facebook | $ 3.510 | $ 12.195 |
| Búsqueda | $ 12.542 | $ 8.838 |
| Total | $ 41.112 | $ 41.112 |

El plan actual es el promedio de las últimas 13 semanas del dataset. Para el reparto óptimo, el algoritmo SLSQP de scipy busca el reparto con más ventas de medios en el simulador, con el mismo total semanal. Cada canal puede recibir hasta 1,5 veces su inversión semanal más alta en los datos.

![Aporte mensual de cada canal a las ventas](docs/img/aportes.png)

## Comparación con Meridian

Meridian es el MMM de Google. Lo entrené con los mismos datos, los mismos controles y las mismas 12 semanas de prueba, con sus priors por defecto.

![Retorno por canal según PyMC-Marketing y Meridian](docs/img/meridian.png)

| Control | PyMC-Marketing | Meridian |
|---|---:|---:|
| R-hat máximo | 1,002 | 1,008 |
| Error en las 12 semanas de prueba (MAPE) | 2,9% | 3,0% |

Los dos modelos predicen igual de bien las semanas de prueba. Coinciden en dos canales:

1. TV rinde alrededor de 5 pesos por cada peso invertido.
2. Vía pública no recupera lo invertido.

En prensa, Facebook y búsqueda, Meridian da retornos más bajos. Los rangos de los dos modelos se superponen, así que los datos no alcanzan para decidir cuál tiene razón. La diferencia viene de los priors. El prior de Meridian sobre el retorno empuja los valores hacia 1,2, y esos tres canales tienen poca inversión en los datos.

Conclusión para el negocio: bajar la inversión en vía pública es una decisión firme, porque los dos modelos la apoyan. El tamaño de la mejora al pasar plata a prensa, Facebook y búsqueda es incierto. Un experimento de incrementalidad en esos canales resolvería la duda.

## Cómo funciona el modelo

El modelo es bayesiano. No da un solo número por canal. Da un rango de valores posibles, cada uno con su probabilidad. La página muestra el rango donde cae el 90% de esos valores.

El modelo tiene estas partes:

1. Inversión semanal de 5 canales pagos: TV, vía pública, prensa, Facebook y búsqueda.
2. Adstock geométrico. El adstock modela que el efecto de un aviso sigue hasta 8 semanas después.
3. Saturación logística. La saturación modela que cada peso extra en un canal rinde menos que el anterior.
4. Controles: newsletter, ventas de la competencia, dos eventos especiales, feriados de Alemania y tendencia. El demo de Robyn usa feriados de Alemania.
5. Estacionalidad anual con 2 pares de ondas de Fourier (ondas que repiten un ciclo por año).

Un prior es lo que el modelo supone antes de ver los datos. El modelo usa los priors por defecto de PyMC-Marketing 1.3.0:

| Parámetro | Prior |
|---|---|
| Adstock (`adstock_alpha`) | Beta(1, 3) |
| Saturación, eficiencia (`saturation_lam`) | Gamma(3, 1) |
| Saturación, efecto máximo (`saturation_beta`) | HalfNormal(2) |
| Intercepto | Normal(0, 2) |
| Controles (`gamma_control`) | Normal(0, 2) |
| Estacionalidad (`gamma_fourier`) | Laplace(0, 1) |
| Ruido (`y_sigma`) | HalfNormal(2) |

PyMC-Marketing escala las ventas y la inversión antes de entrenar. La newsletter y las ventas de la competencia se escalan en `src/mmm/data.py`, porque PyMC-Marketing no escala los controles.

### Controles de calidad

Las últimas 12 semanas quedaron fuera del entrenamiento. El modelo las predijo sin verlas. `uv run mmm fit` se detiene y no guarda nada si falla uno de los tres primeros controles.

| Control | Resultado | Condición |
|---|---:|---|
| Divergencias del sampler | 0 | debe ser 0 |
| R-hat máximo | 1,002 | menor que 1,01 |
| ESS mínimo | 1.049 | mayor que 400 |
| Error en las 12 semanas de prueba (MAPE) | 2,9% | solo informativo |

El sampler es el motor que calcula el modelo. El R-hat mide si sus 4 cadenas llegaron al mismo resultado. El ESS es la cantidad efectiva de muestras independientes. El MAPE es el error porcentual promedio.

### El simulador

El simulador de la página corre en el navegador, sin servidor. Usa 200 muestras del modelo exportadas en `site/data/resultados.json`. Supone la misma inversión todas las semanas del trimestre. Arranca con la inversión real de las últimas 7 semanas, porque el efecto de esa inversión sigue unas semanas más (adstock). Un test compara este cálculo con la función de adstock de PyMC-Marketing.

El cálculo existe dos veces: en Python (`src/mmm/export.py`) y en JavaScript (`site/js/simulator.js`). Un test compara los dos resultados en 4 casos y exige una diferencia relativa menor que 1e-6.

### Calibración con experimentos

Un experimento de incrementalidad (lift test) mide cuántas ventas extra genera un cambio de inversión en un canal. Si hay experimentos, el modelo los usa para ajustar sus curvas de saturación.

Para calibrar el modelo, seguí estos pasos:

1. Copiá `data/lift_tests_plantilla.csv` y agregá una fila por experimento.
2. Entrená el modelo con `uv run mmm fit --lift-tests <tu archivo>.csv`.

Todas las columnas van en pesos por semana:

| Columna | Qué es |
|---|---|
| `channel` | Canal del experimento, por ejemplo `facebook_S` |
| `x` | Inversión semanal antes del experimento |
| `delta_x` | Cambio de la inversión semanal durante el experimento |
| `delta_y` | Ventas extra por semana que midió el experimento |
| `sigma` | Error estándar de `delta_y` |

Si una fila tiene un error, `mmm fit` se detiene antes de entrenar y nombra la fila y la columna.

El modelo publicado no está calibrado, porque no hay experimentos reales para estos datos. El demo de Robyn trae experimentos de ejemplo, pero sus montos no coinciden con el dataset. Por ejemplo, el experimento de Facebook indica $ 421.000 de inversión en semanas donde el dataset registra $ 14.049. Calibrar con esos valores empeoraría el modelo.

## Cómo ejecutar el proyecto

Necesitás `uv` y Node 24. Desde la carpeta del repositorio, instalá las dependencias:

```bash
uv sync
```

Entrená el modelo. Tarda unos minutos y guarda el resultado en `artifacts/`:

```bash
uv run mmm fit
```

Exportá los resultados para la página, los casos de test y las imágenes del README:

```bash
uv run mmm export
```

Corré los tests de Python y de JavaScript:

```bash
uv run pytest
```

```bash
node --test "site/js/*.test.js"
```

Entrená Meridian y exportá su comparación. Meridian corre en su propio entorno, porque pide otra versión de ArviZ. Necesita Python 3.13 o anterior:

```bash
uv run --script scripts/run_meridian.py
```

Después, ejecutá de nuevo `uv run mmm export` para generar el gráfico de comparación del README.

Abrí la página en `http://localhost:8000`:

```bash
uv run python -m http.server -d site 8000
```

Los notebooks de `notebooks/` muestran la exploración de los datos, el modelo y los resultados. GitHub los muestra con los resultados guardados.

## Estructura

```
src/mmm/      carga de datos, modelo, controles de calidad, exportación y gráficos
notebooks/    exploración, modelo y resultados
site/         página estática y simulador en JavaScript
tests/        tests de Python y casos compartidos con JavaScript
scripts/      comparación con Meridian y captura del simulador
data/raw/     dataset de demo de Robyn
```

## Fuente de los datos y licencia

El dataset es el demo `dt_simulated_weekly` de [Robyn](https://github.com/facebookexperimental/Robyn), con licencia MIT. `data/raw/FUENTE.md` registra el commit de origen.

El código de este repositorio tiene licencia MIT.
