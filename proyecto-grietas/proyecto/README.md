# Detector de riesgo estructural por fotos de grietas

Clasificador de imagenes (Teachable Machine / MobileNetV2) con 4 clases: `sana`, `superficial`,
`peligrosa`, `no_aplica`, mas un analisis complementario con Canny + Hough.

## Estructura
- `app.py`: interfaz web (Gradio): clasificacion + trazado de lineas Canny/Hough en la misma pantalla. Incluye umbral asimetrico para `peligrosa` y aviso de baja confianza.
- `lineas.py`: funcion Canny + Hough que usa la app.
- `evaluar_prueba.py`: evalua el modelo con `prueba/` (matriz de confusion, errores, analisis de umbral).
- `notebooks/canny_hough.py`: deteccion de bordes y lineas sobre una foto.
- `model/`: `keras_model.h5` y `labels.txt`.
- `data/`: fuentes y criterios de etiquetado (las imagenes no se suben).
- `results/`: figuras y tablas para el informe.

## Uso
    pip install -r requirements.txt
    python app.py
    python evaluar_prueba.py --prueba prueba
    python notebooks/canny_hough.py foto.jpg --canny-low 70 --canny-high 180 --hough-threshold 40 --min-length 90 --max-gap 25

## Entrenamiento
Teachable Machine, 10 epocas, lote 16, tasa de aprendizaje 0.001.
Validacion interna de TM (~15% de las imagenes): ~93% (ver tabla de experimentos).

## Resultados
<Completa con la salida de evaluar_prueba.py: accuracy, recall de peligrosa, matriz de confusion.>

## Limitaciones
- Una foto 2D no permite medir profundidad ni saber si el elemento es estructural.
- Los datasets base no etiquetan severidad; esas etiquetas son propias.
- La validacion de Teachable Machine puede estar inflada por imagenes casi identicas; por eso se evalua con `prueba/`.
- El resultado depende del encuadre (ver prueba de recorte central vs imagen completa).
- Herramienta de apoyo: no sustituye la evaluacion de un ingeniero civil.
