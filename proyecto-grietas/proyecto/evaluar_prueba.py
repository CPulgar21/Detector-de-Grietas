"""
Uso:
    python evaluar_prueba.py                      # recorte central (igual que app.py)
    python evaluar_prueba.py --modo resize        # imagen completa deformada a 224x224
    python evaluar_prueba.py --prueba otra_carpeta

Genera en results/:
    matriz_confusion_<modo>.png, errores_<modo>.csv, predicciones_<modo>.csv, resumen_<modo>.txt
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from tf_keras.models import load_model

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def preparar(ruta, modo):
    img = Image.open(ruta).convert("RGB")           
    if modo == "fit":
        img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)   
    else:
        img = img.resize((224, 224), Image.Resampling.LANCZOS)          
    return (np.asarray(img).astype(np.float32) / 127.5) - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="model/keras_model.h5")
    p.add_argument("--etiquetas", default="model/labels.txt")
    p.add_argument("--prueba", default="prueba")
    p.add_argument("--modo", choices=["fit", "resize"], default="fit")
    p.add_argument("--salida", default="results")
    a = p.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    clases = [l.strip().split(" ", 1)[1] for l in open(a.etiquetas, encoding="utf-8") if l.strip()]
    modelo = load_model(a.modelo, compile=False)

    filas = []
    for ci, clase in enumerate(clases):
        carpeta = os.path.join(a.prueba, clase)
        if not os.path.isdir(carpeta):
            print(f"[aviso] no existe la carpeta {carpeta}, la salto")
            continue
        for nombre in sorted(os.listdir(carpeta)):
            if not nombre.lower().endswith(EXT):
                continue
            ruta = os.path.join(carpeta, nombre)
            try:
                x = preparar(ruta, a.modo)
            except Exception as e:
                print(f"[aviso] no pude leer {ruta}: {e}")
                continue
            prob = modelo.predict(x[None], verbose=0)[0]
            filas.append((ruta, ci, prob))

    if not filas:
        raise SystemExit("No encontre imagenes. Revisa la estructura de la carpeta prueba/.")

    n = len(clases)
    y_real = np.array([f[1] for f in filas])
    probs = np.array([f[2] for f in filas])
    y_pred = probs.argmax(1)

    mc = np.zeros((n, n), dtype=int)
    for r, q in zip(y_real, y_pred):
        mc[r, q] += 1

   
    out = [f"Modo de preprocesamiento: {a.modo}", f"Imagenes evaluadas: {len(filas)}",
           f"Accuracy global: {100 * (y_real == y_pred).mean():.1f}%", ""]
    out.append(f"{'clase':<14}{'n':>5}{'recall':>9}{'precision':>11}")
    for i, c in enumerate(clases):
        tot = mc[i].sum()
        rec = mc[i, i] / tot if tot else float("nan")
        pred_i = mc[:, i].sum()
        pre = mc[i, i] / pred_i if pred_i else float("nan")
        out.append(f"{c:<14}{tot:>5}{100 * rec:>8.1f}%{100 * pre:>10.1f}%")

    
    if "peligrosa" in clases and "sana" in clases and "superficial" in clases:
        ip, isn, isu = clases.index("peligrosa"), clases.index("sana"), clases.index("superficial")
        tot_p = mc[ip].sum()
        if tot_p:
            out += ["", f"PELIGROSAS subestimadas: {mc[ip, isn] + mc[ip, isu]} de {tot_p} "
                        f"(como sana: {mc[ip, isn]}, como superficial: {mc[ip, isu]})"]

        
        out += ["", "Umbral asimetrico: se marca 'peligrosa' si P(peligrosa) >= umbral",
                f"{'umbral':>7}{'recall peligrosa':>18}{'falsas alarmas':>16}{'% de no-peligrosas alarmadas':>32}"]
        es_pel = y_real == ip
        for t in [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]:
            alarma = probs[:, ip] >= t
            rec = alarma[es_pel].mean() if es_pel.any() else float("nan")
            fa = int(alarma[~es_pel].sum())
            pct = 100 * alarma[~es_pel].mean() if (~es_pel).any() else float("nan")
            out.append(f"{t:>7.2f}{100 * rec:>17.1f}%{fa:>16}{pct:>31.1f}%")

    texto = "\n".join(out)
    print(texto)
    with open(os.path.join(a.salida, f"resumen_{a.modo}.txt"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")

    
    with open(os.path.join(a.salida, f"predicciones_{a.modo}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "real", "predicho"] + [f"p_{c}" for c in clases])
        for (ruta, r, pr), q in zip(filas, y_pred):
            w.writerow([ruta, clases[r], clases[q]] + [f"{v:.4f}" for v in pr])
    with open(os.path.join(a.salida, f"errores_{a.modo}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "real", "predicho", "confianza_predicha"])
        for (ruta, r, pr), q in zip(filas, y_pred):
            if r != q:
                w.writerow([ruta, clases[r], clases[q], f"{pr[q]:.3f}"])

    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(mc, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(clases, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(clases)
    ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion (prueba, modo {a.modo})")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, mc[i, j], ha="center", va="center",
                    color="white" if mc[i, j] > mc.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(a.salida, f"matriz_confusion_{a.modo}.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
