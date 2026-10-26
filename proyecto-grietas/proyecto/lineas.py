
import cv2
import numpy as np
from PIL import Image


def _orientacion(ang):
    if ang < 20 or ang > 160:
        return "horizontal"
    if 70 <= ang <= 110:
        return "vertical"
    return "diagonal"


def analizar_lineas(imagen_pil, canny_low=70, canny_high=180, hough_threshold=40,
                    min_length=90, max_gap=25, lado_max=900):
    """imagen_pil -> (bordes_pil, hough_pil, resumen_markdown)"""
    rgb = np.asarray(imagen_pil.convert("RGB"))
    h, w = rgb.shape[:2]
    if max(h, w) > lado_max:                      # los parametros de Hough dependen del tamano
        esc = lado_max / max(h, w)
        rgb = cv2.resize(rgb, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)

    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(suave, int(canny_low), int(canny_high))
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=int(hough_threshold),
                             minLineLength=int(min_length), maxLineGap=int(max_gap))

    salida = rgb.copy()
    angs, largos = [], []
    if lineas is not None:
        for x1, y1, x2, y2 in lineas[:, 0]:
            cv2.line(salida, (x1, y1), (x2, y2), (255, 0, 0), 2)
            angs.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180)
            largos.append(np.hypot(x2 - x1, y2 - y1))

    resumen = [f"**Bordes (Canny):** {100 * (bordes > 0).mean():.1f}% de la imagen",
               f"**Segmentos de línea (Hough):** {len(angs)}"]
    if angs:
        angs, largos = np.array(angs), np.array(largos)
        acum = {}
        for a, l in zip(angs, largos):
            k = _orientacion(a)
            acum[k] = acum.get(k, 0) + l
        total = sum(acum.values())
        resumen.append("**Orientación (por longitud):** " +
                       ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in sorted(acum.items())))
    else:
        resumen.append("No se detectaron líneas con estos parámetros (prueba bajar el umbral de Hough "
                       "o el largo mínimo).")
    resumen.append("\n_Las líneas rojas también pueden aparecer en texturas, poros o juntas: "
                   "no son una medida de gravedad, solo ayudan a ver por dónde corre la grieta._")
    return Image.fromarray(bordes), Image.fromarray(salida), "\n\n".join(resumen)
