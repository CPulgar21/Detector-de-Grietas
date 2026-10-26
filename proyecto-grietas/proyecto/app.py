import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
from tf_keras.models import load_model
from lineas import analizar_lineas

# ---------------- Configuracion (ajusta con los resultados de evaluar_prueba.py) ----------------
RUTA_MODELO = "model/keras_model.h5"
RUTA_ETIQUETAS = "model/labels.txt"
MODO = "fit"                 # "fit" = recorte central | "resize" = imagen completa deformada
UMBRAL_PELIGROSA = 0.25      # si P(peligrosa) >= este valor se avisa aunque no sea la clase ganadora
UMBRAL_CONFIANZA = 0.60      # por debajo, el resultado se marca como "no concluyente"
UMBRAL_NO_APLICA = 0.50
# ------------------------------------------------------------------------------------------------

model = load_model(RUTA_MODELO, compile=False)
with open(RUTA_ETIQUETAS, "r", encoding="utf-8") as f:
    class_names = [l.strip().split(" ", 1)[1] for l in f if l.strip()]

DISCLAIMER = (
    "<b>Herramienta de apoyo, no un diagnóstico.</b> Una foto no permite medir la profundidad "
    "de una grieta ni saber si el elemento es estructural. Ante cualquier duda, consulta a un "
    "ingeniero civil o arquitecto."
)


def preparar(imagen, modo=MODO):
    imagen = imagen.convert("RGB")            # evita fallos con PNG/RGBA o escala de grises
    if modo == "fit":
        imagen = ImageOps.fit(imagen, (224, 224), Image.Resampling.LANCZOS)
    else:
        imagen = imagen.resize((224, 224), Image.Resampling.LANCZOS)
    arr = (np.asarray(imagen).astype(np.float32) / 127.5) - 1
    return arr[None]


def mensaje(probs):
    """Devuelve (nivel, titulo, detalle). nivel: ok | warn | alert | info"""
    p = dict(zip(class_names, probs))
    top = max(p, key=p.get)
    if top == "no_aplica" and p[top] >= UMBRAL_NO_APLICA:
        return ("info", "No parece una pared",
                "La imagen no parece una superficie constructiva. Prueba con otra foto, más cercana a la grieta.")
    if p.get("peligrosa", 0) >= UMBRAL_PELIGROSA:
        return ("alert", "Posible daño importante",
                f"Probabilidad de grieta peligrosa: {100 * p['peligrosa']:.0f}%. Se recomienda revisión profesional.")
    if p[top] < UMBRAL_CONFIANZA:
        return ("info", "Resultado no concluyente",
                "El modelo no está seguro. Repite la foto con mejor luz y de frente a la pared.")
    if top == "sana":
        return ("ok", "Sin grietas visibles", "No se observan grietas en esta foto.")
    if top == "superficial":
        return ("warn", "Posible grieta superficial", "Vigila si crece o cambia con el tiempo.")
    return ("info", top, "")


COLORES = {"ok": "#C1E1A7", "warn": "#EFBC75", "alert": "#E76F51", "info": "#148D8D"}
TEXTO_SOBRE = {"ok": "#0E2C40", "warn": "#0E2C40", "alert": "#0E2C40", "info": "#FFFFFF"}


def tarjeta(nivel, titulo, detalle):
    return (f'<div class="resultado" style="--c:{COLORES[nivel]};--t:{TEXTO_SOBRE[nivel]}">'
            f'<div class="res-titulo">{titulo}</div><div class="res-detalle">{detalle}</div></div>')


def evaluar_grieta(imagen, canny_low, canny_high, hough_thr, min_len, vista):
    if imagen is None:
        return {}, tarjeta("info", "Sube una foto para comenzar", ""), None, None, ""
    modo = "fit" if vista.startswith("Recorte") else "resize"
    probs = model.predict(preparar(imagen, modo), verbose=0)[0]
    resultados = {n: float(p) for n, p in zip(class_names, probs)}
    bordes, lineas, resumen = analizar_lineas(imagen, canny_low, canny_high, hough_thr, min_len)
    return resultados, tarjeta(*mensaje(probs)), lineas, bordes, resumen


CSS = """
:root {--navy:#0E2C40; --petrol:#1A4A5A; --teal:#148D8D; --mint:#C1E1A7; --amber:#EFBC75;}
.gradio-container {max-width: 1080px !important; margin: 0 auto !important; padding: 20px 16px !important;}
.gradio-container .gap, .gradio-container .form {gap: 12px !important;}
.cabecera {padding: 8px 0 4px 0; border-bottom: 2px solid var(--amber); margin-bottom: 4px;}
.cabecera h1 {margin: 0; font-size: clamp(1.5rem, 4vw, 2.2rem); font-weight: 700; color: var(--mint);}
.cabecera p {margin: 6px 0 10px 0; color: #9FC3C3; font-size: clamp(.92rem, 2.4vw, 1.05rem);}
.resultado {background: var(--c); color: var(--t); border-radius: 14px; padding: 20px 22px;}
.res-titulo {font-size: clamp(1.25rem, 3.4vw, 1.6rem); font-weight: 700; line-height: 1.2;}
.res-detalle {margin-top: 4px; font-size: 1rem; opacity: .88;}
.consejo {color: #9FC3C3; font-size: .92rem; line-height: 1.5;}
.aviso {border-top: 1px solid #148D8D66; padding-top: 14px; color: #9FC3C3; font-size: .9rem;}
.aviso b {color: var(--amber);}
footer {display: none !important;}
@media (max-width: 640px) {.gradio-container {padding: 12px 10px !important;} .resultado {padding: 16px 16px;}}
"""

_p = dict(body_background_fill="#0E2C40", background_fill_primary="#0E2C40",
          background_fill_secondary="#1A4A5A", block_background_fill="#1A4A5A",
          block_border_color="#148D8D55", block_border_width="1px", block_shadow="none",
          block_label_background_fill="#1A4A5A", block_label_text_color="#C1E1A7",
          block_title_text_color="#C1E1A7", body_text_color="#EAF4EC",
          body_text_color_subdued="#9FC3C3", border_color_primary="#148D8D66",
          input_background_fill="#0E2C40", input_border_color="#148D8D66",
          button_primary_background_fill="#148D8D", button_primary_background_fill_hover="#1AA3A3",
          button_primary_text_color="#FFFFFF", button_primary_border_color="#148D8D",
          button_secondary_background_fill="#1A4A5A", button_secondary_text_color="#EAF4EC",
          slider_color="#148D8D", panel_background_fill="#1A4A5A")
TEMA = gr.themes.Base(primary_hue="teal", neutral_hue="slate", radius_size="lg",
                      font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
                      ).set(**_p, **{k + "_dark": v for k, v in _p.items()})

with gr.Blocks(title="Detector de Riesgo Estructural") as interfaz:
    gr.HTML('<div class="cabecera"><h1>Detector de Riesgo Estructural</h1>'
            '<p>Sube una foto de un muro o columna. El modelo la clasifica y Canny + Hough '
            'traza las líneas detectadas.</p></div>')
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            entrada = gr.Image(type="pil", show_label=False, height=400)
            boton = gr.Button("Evaluar foto", variant="primary", size="lg")
            gr.HTML('<div class="consejo">Foto de frente a la pared, con buena luz y sin sombras sobre la grieta. '
                    'La grieta debe ocupar buena parte del encuadre.</div>')
            with gr.Accordion("Ajustes avanzados", open=False):
                vista = gr.Radio(["Recorte central", "Imagen completa"], value="Recorte central",
                                 label="Parte de la foto que ve el modelo",
                                 info="En fotos verticales con la grieta cerca de los bordes, prueba 'Imagen completa'.")
                canny_low = gr.Slider(10, 200, value=70, step=5, label="Canny: umbral bajo")
                canny_high = gr.Slider(50, 400, value=180, step=5, label="Canny: umbral alto")
                hough_thr = gr.Slider(10, 150, value=40, step=5, label="Hough: votos mínimos")
                min_len = gr.Slider(20, 300, value=90, step=10, label="Hough: largo mínimo (px)")
        with gr.Column(scale=1, min_width=320):
            salida_msg = gr.HTML(tarjeta("info", "Sube una foto para comenzar", "El resultado aparecerá aquí."))
            salida_label = gr.Label(num_top_classes=4, show_label=False)
            with gr.Tabs():
                with gr.Tab("Líneas detectadas"):
                    salida_lineas = gr.Image(show_label=False)
                with gr.Tab("Bordes (Canny)"):
                    salida_bordes = gr.Image(show_label=False)
            salida_resumen = gr.Markdown()
    gr.HTML(f'<div class="aviso">⚠️ {DISCLAIMER}</div>')
    boton.click(evaluar_grieta, [entrada, canny_low, canny_high, hough_thr, min_len, vista],
                [salida_label, salida_msg, salida_lineas, salida_bordes, salida_resumen])

if __name__ == "__main__":
    interfaz.launch(theme=TEMA, css=CSS)
