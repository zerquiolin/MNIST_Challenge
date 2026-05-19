# Clasificador de Dígitos MNIST

Proyecto de clasificación de dígitos MNIST orientado a producción, construido con PyTorch y expuesto mediante una API de inferencia con FastAPI.

El proyecto incluye una solución completa del reto basada en notebooks, módulos reutilizables de Python, generación de artefactos del modelo, validación de API, pruebas automatizadas y soporte para Docker. También está estructurado para documentación bilingüe mediante notebooks y archivos README en inglés y español.

## Idiomas de la Documentación

- Documentación en inglés: `README.md`
- Documentación en español: `README_es.md`
- Notebook del reto en inglés: `notebooks/challenge.ipynb`
- Notebook del reto en español: `notebooks/challenge_es.ipynb`

## Contexto del Proyecto

Este proyecto resuelve la kata senior de machine learning para clasificar imágenes de dígitos escritos a mano mediante una API en Python. El reto solicita un modelo capaz de recibir una imagen de un dígito escrito a mano y retornar su clase predicha a través de una interfaz HTTP.

La solución es una implementación de red neuronal en PyTorch orientada a producción, con clara separación de responsabilidades, buenas prácticas de código limpio y componentes modulares para carga de datos, preprocesamiento, definición del modelo, entrenamiento, evaluación, inferencia, validación de API y despliegue. El pipeline completo de entrenamiento y evaluación está documentado en `notebooks/challenge.ipynb` y en su versión en español, `notebooks/challenge_es.ipynb`. Estos notebooks incluyen los principios de diseño, decisiones de implementación, justificaciones, proceso de entrenamiento, comportamiento del aprendizaje, fortalezas, limitaciones y posibles mejoras futuras.

El proyecto también incluye una aplicación FastAPI que expone el modelo entrenado mediante HTTP. El endpoint `/predict` recibe una cadena de imagen codificada en base64, la valida, la preprocesa y retorna el dígito predicho junto con una puntuación de confianza. Aceptar imágenes en base64 puede ser riesgoso si no se controla adecuadamente, por lo que la API incluye salvaguardas contra payloads malformados, tipos de imagen no soportados, entradas demasiado grandes, archivos corruptos, dimensiones de imagen inseguras y codificaciones inválidas.

Para apoyar la mantenibilidad y la transparencia, el repositorio incluye pruebas unitarias y end-to-end que cubren validación, preprocesamiento, comportamiento de inferencia y endpoints de la API. El Makefile proporciona una interfaz de comandos consistente para configuración, generación del modelo, ejecución de notebooks, pruebas, ejecución local de la API y flujos de trabajo con Docker. El mismo proyecto puede ejecutarse localmente con Conda o empaquetarse en Docker para un uso orientado a despliegue.

## Estructura del Proyecto

```text
.
├── app/                         # Aplicación FastAPI, rutas, esquemas, validación, inferencia
├── artifacts/                   # Artefactos generados del modelo, incluyendo mnist_classifier.pt
├── images/                      # Imágenes de muestra de dígitos escritos a mano para validación API/e2e
├── notebooks/                   # Notebooks de solución del reto y trabajo exploratorio
│   ├── challenge.ipynb          # Solución completa del reto en inglés
│   ├── challenge_es.ipynb       # Solución completa del reto en español
│   └── experiments.ipynb        # Notebook exploratorio de apoyo
├── src/                         # Carga de datos, modelo, entrenamiento, preprocesamiento, utilidades
├── tests/                       # Pruebas unitarias y end-to-end
├── Dockerfile                   # Definición de la imagen del contenedor
├── docker-compose.yml           # Definición del servicio de Docker Compose
├── environment.yml              # Definición del entorno Conda
├── Makefile                     # Comandos principales locales, de pruebas, modelo y Docker
├── README.md                    # Documentación en inglés
├── README_es.md                 # Documentación en español
└── requirements.txt             # Dependencias de pip usadas por Docker
```

## Solución del Reto

La solución completa del reto en inglés está disponible en:

```text
notebooks/challenge.ipynb
```

La solución completa del reto en español está disponible en:

```text
notebooks/challenge_es.ipynb
```

La solución en el notebook recorre el flujo de trabajo completo de ML:

- Carga del dataset MNIST.
- Aumento de datos y preprocesamiento.
- Arquitectura CNN implementada en PyTorch.
- Bucle de entrenamiento con validación.
- Evaluación del modelo.
- Exportación del artefacto del modelo.
- Comportamiento de inferencia listo para API.

El trabajo exploratorio adicional está disponible en:

```text
notebooks/experiments.ipynb
```

## Configuración Local

### Prerrequisitos

- Conda.
- Python 3.10 mediante el entorno Conda definido en `environment.yml`.

Crear o actualizar el entorno Conda local:

```bash
make setup
```

Activarlo:

```bash
conda activate mnist-env
```

Si tu shell, editor o entorno de ejecución de notebooks no puede resolver imports desde `app/` o `src/`, ejecuta esto desde la raíz del proyecto:

```bash
export PYTHONPATH=./
```

Esto suele ocurrir en configuraciones basadas principalmente en terminal o en editores configurados manualmente, como Vim o entornos de VS Code personalizados. IDEs como PyCharm suelen detectar automáticamente la raíz del proyecto.

## Ejecución de Notebooks

Inicia Jupyter desde la raíz del repositorio con el entorno Conda activo:

```bash
jupyter notebook
```

Notebooks principales:

```text
notebooks/challenge.ipynb
notebooks/challenge_es.ipynb
```

También puedes ejecutar el notebook del reto configurado desde la línea de comandos:

```bash
make train
```

`make train` actualmente ejecuta la ruta del notebook en inglés configurada en el `Makefile`.

## Generación del Artefacto del Modelo

La API y la imagen Docker esperan el artefacto del modelo entrenado en:

```text
artifacts/mnist_classifier.pt
```

Genera el modelo con:

```bash
make model
```

Por defecto, esto entrena durante `22` épocas. Puedes sobrescribir el número de épocas con `MODEL_EPOCHS`:

```bash
make model MODEL_EPOCHS=5
```

La imagen Docker no entrena el modelo durante la construcción. Genera el artefacto antes de construir o ejecutar el contenedor.

La ruta más rápida es `make model`, que usa el módulo ligero de generación del modelo. También puedes generar o refinar el modelo mediante experimentación en `notebooks/challenge.ipynb` o `notebooks/challenge_es.ipynb`, y luego exportar los pesos resultantes a la misma ruta de artefacto.

### Utilidades para el Artefacto del Modelo

El comando `make model` exporta automáticamente el artefacto del modelo entrenado, por lo que no se necesita un paso manual de guardado en el flujo estándar.

Para código personalizado de entrenamiento o experimentación, `src/utils/model_loader.py` proporciona pequeñas funciones auxiliares para guardar y cargar pesos de modelos PyTorch:

```python
import torch

from src.models.mnist_cnn import MNISTCNN
from src.utils.model_loader import load_model, save_model

model = MNISTCNN()

# Guardar pesos en artifacts/MNISTCNN.pt
save_model(model, "artifacts")

# Cargar los pesos de vuelta en la misma arquitectura
model = load_model(MNISTCNN, "artifacts/MNISTCNN.pt")
```

## Ejecución Local de la API

Inicia el servicio FastAPI:

```bash
make api
```

Verifica el endpoint raíz:

```bash
curl http://localhost:8000/
```

Verifica la disponibilidad del modelo:

```bash
curl http://localhost:8000/status
```

Respuesta esperada cuando el artefacto del modelo está presente:

```json
{"Status":"Ok"}
```

La documentación interactiva de la API está disponible en:

```text
http://localhost:8000/docs
```

## Ejecución con Docker

Docker requiere que Docker Desktop o el daemon de Docker estén en ejecución.

Antes de construir la imagen, asegúrate de que el artefacto del modelo exista:

```bash
make model
```

Construir la imagen:

```bash
make docker-build
```

Ejecutar el contenedor:

```bash
make docker-run
```

La API estará disponible en:

```text
http://localhost:8000
```

También puedes usar Docker Compose:

```bash
make docker-up
```

Detener el servicio de Compose:

```bash
make docker-down
```

La imagen Docker incluye un healthcheck que llama a:

```text
http://127.0.0.1:8000/status
```

El contenedor se considera saludable solo cuando el endpoint retorna:

```json
{"Status":"Ok"}
```

## Pruebas y Verificaciones de Calidad

Ejecutar la suite de pruebas:

```bash
make test
```

Ejecutar verificaciones de linting y formato:

```bash
make lint
```

Ejecutar verificación estática de tipos:

```bash
make typecheck
```

La cobertura de pruebas incluye:

- Pruebas unitarias para preprocesamiento, validación y comportamiento de inferencia.
- Pruebas end-to-end para endpoints de la API.
- Verificaciones de Ruff para linting y formato.
- Verificaciones de Mypy para tipado estático.

## Uso de la API

### Endpoints

- `GET /`: retorna un mensaje básico de bienvenida de la API.
- `GET /status`: retorna la disponibilidad del artefacto del modelo.
- `POST /predict`: predice el dígito a partir de una imagen codificada en base64.

### Solicitud de Predicción

```json
{
  "image_base64": "..."
}
```

### Respuesta de Predicción

```json
{
  "prediction": 5,
  "confidence": 0.99
}
```

Para pruebas interactivas de solicitudes, usa:

```text
http://localhost:8000/docs
```

## Solución de Problemas

### Modelo Faltante

Si `/status` no retorna `{"Status":"Ok"}`, genera el artefacto del modelo:

```bash
make model
```

Luego confirma que este archivo exista:

```text
artifacts/mnist_classifier.pt
```

### Errores de Importación

Si Python no puede importar módulos desde `app/` o `src/`, ejecuta esto desde la raíz del proyecto:

```bash
export PYTHONPATH=./
```

### Daemon de Docker No Disponible

Si los comandos de Docker fallan con un error de conexión al daemon, inicia Docker Desktop o tu daemon de Docker, y luego ejecuta el comando nuevamente.

### Error de Estado en Docker

Si el contenedor inicia pero `/status` retorna un error, el artefacto del modelo falta dentro de la imagen o `MODEL_PATH` apunta a una ubicación incorrecta.

Regenera el modelo y reconstruye la imagen:

```bash
make model
make docker-build
```
