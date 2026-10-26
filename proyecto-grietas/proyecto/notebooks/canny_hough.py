"""
Canny + Hough sobre una foto de grieta.

Uso:
    python canny_hough.py foto.jpg
    python canny_hough.py foto.jpg --canny-low 40 --canny-high 120 --hough-threshold 60

"""

import argparse
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def procesar(ruta, canny_low, canny_high, hough_threshold, min_length, max_gap, ancho_max):
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No pude abrir la imagen: {ruta}")

    h, w = img.shape[:2]
    if max(h, w) > ancho_max:
        esc = ancho_max / max(h, w)
        img = cv2.resize(img, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 0)          
    bordes = cv2.Canny(suave, canny_low, canny_high)    
    lineas = cv2.HoughLinesP(
        bordes, rho=1, theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_length, maxLineGap=max_gap,
    )

    resultado = img.copy()
    angulos, largos = [], []
    if lineas is not None:
        for x1, y1, x2, y2 in lineas[:, 0]:
            cv2.line(resultado, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angulos.append(ang)
            largos.append(np.hypot(x2 - x1, y2 - y1))
    return img, gris, bordes, resultado, np.array(angulos), np.array(largos)


def clasificar_orientacion(ang):
    """Etiqueta simple de orientacion (0 = horizontal, 90 = vertical)."""
    if ang < 20 or ang > 160:
        return "horizontal"
    if 70 <= ang <= 110:
        return "vertical"
    return "diagonal"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("imagen")
    p.add_argument("--canny-low", type=int, default=50)
    p.add_argument("--canny-high", type=int, default=150)
    p.add_argument("--hough-threshold", type=int, default=50)
    p.add_argument("--min-length", type=int, default=60)
    p.add_argument("--max-gap", type=int, default=15)
    p.add_argument("--ancho-max", type=int, default=900)
    p.add_argument("--salida", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = p.parse_args()

    os.makedirs(a.salida, exist_ok=True)
    nombre = os.path.splitext(os.path.basename(a.imagen))[0]

    img, gris, bordes, res, ang, lar = procesar(
        a.imagen, a.canny_low, a.canny_high, a.hough_threshold,
        a.min_length, a.max_gap, a.ancho_max)

    
    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("1. Original")
    ax[1].imshow(gris, cmap="gray");                    ax[1].set_title("2. Escala de grises")
    ax[2].imshow(bordes, cmap="gray");                  ax[2].set_title(f"3. Canny ({a.canny_low}/{a.canny_high})")
    ax[3].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)); ax[3].set_title(f"4. Hough ({len(ang)} segmentos)")
    for x in ax:
        x.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(a.salida, f"{nombre}_pasos.png"), dpi=130)
    plt.close()
    cv2.imwrite(os.path.join(a.salida, f"{nombre}_hough.png"), res)

    lineas_txt = [
        f"Imagen: {a.imagen}",
        f"Parametros: Canny {a.canny_low}/{a.canny_high} | Hough umbral={a.hough_threshold}, "
        f"largo minimo={a.min_length}px, hueco maximo={a.max_gap}px",
        f"Pixeles de borde (Canny): {int((bordes > 0).sum())} "
        f"({100 * (bordes > 0).mean():.1f}% de la imagen)",
        f"Segmentos Hough detectados: {len(ang)}",
    ]
    if len(ang):
        ponderado = np.average(ang, weights=lar)
        conteo = {}
        for g, l in zip(ang, lar):
            k = clasificar_orientacion(g)
            conteo[k] = conteo.get(k, 0) + l
        total = sum(conteo.values())
        lineas_txt += [
            f"Largo total de segmentos: {lar.sum():.0f}px | mas largo: {lar.max():.0f}px",
            f"Angulo medio ponderado por largo: {ponderado:.1f} grados (0=horizontal, 90=vertical)",
            "Orientacion (por longitud acumulada): "
            + ", ".join(f"{k} {100 * v / total:.0f}%" for k, v in sorted(conteo.items())),
        ]
    texto = "\n".join(lineas_txt)
    with open(os.path.join(a.salida, f"{nombre}_resumen.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto)


if __name__ == "__main__":
    main()
