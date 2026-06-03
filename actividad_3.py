import cv2
import numpy as np
import time
from skimage.filters import threshold_multiotsu

video_path = r"C:\Users\rafae\OneDrive\Escritorio\Vision\Tarea2\video1\desalineado_3.mp4"

kernel_sizes = [47, 9, 13, 21]
filtros = ["PROMEDIO", "GAUSSIANO"]
sigmas = [3, 7]

frame_number = 200
NUM_EJECUCIONES = 10

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()
cap.release()

if not ret:
    print("no hay frame")
    exit()

original = frame.copy()
h, w = frame.shape[:2]

# ROI
x1 = int(w * 0.50)
x2 = int(w * 0.76)
y1 = 0
y2 = int(h * 0.70)

roi = frame[y1:y2, x1:x2]
roi_original = original[y1:y2, x1:x2]

# multi otsu, 3 colores
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
thresholds = threshold_multiotsu(gray, classes=3)
regions = np.digitize(gray, bins=thresholds)

mask_white = (regions == 2).astype(np.uint8) * 255

# trata de limpiar el verde, por brillo
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask_green = cv2.inRange(hsv, (35,50,50), (85,255,255))

mask_codigo = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_green))

kernel = np.ones((5,5), np.uint8)
mask_codigo = cv2.morphologyEx(mask_codigo, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(mask_codigo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# procesamiento
resultados = {}

for k in kernel_sizes:
    resultados[k] = {}

    for filtro in filtros:

        sigma_list = sigmas if filtro == "GAUSSIANO" else [None]

        resultados[k][filtro] = {}

        for s in sigma_list:

            tiempos = []
            mse_val = 0

            for i in range(NUM_EJECUCIONES):

                frame_proc = frame.copy()
                roi_proc = frame_proc[y1:y2, x1:x2]

                inicio = time.perf_counter()

                if filtro == "PROMEDIO":
                    blur = cv2.blur(roi_proc, (k,k))

                elif filtro == "GAUSSIANO":
                    blur = cv2.GaussianBlur(roi_proc, (k,k), s)

                mask_3ch = cv2.merge([mask_codigo]*3)
                roi_proc = np.where(mask_3ch == 255, blur, roi_proc)

                fin = time.perf_counter()
                tiempos.append(fin - inicio)

                if i == 0:
                    mse_val = mse(roi_original, roi_proc)

            resultados[k][filtro][s] = {
                "mse": mse_val,
                "time": sum(tiempos)/len(tiempos)
            }

#  base mse
base_mse = resultados[47]["PROMEDIO"][None]["mse"]

for k in kernel_sizes:

    print(f"Kernel {k} x {k}")
    for filtro in filtros:

        sigma_list = sigmas if filtro == "GAUSSIANO" else [None]

        for s in sigma_list:

            mse_val = resultados[k][filtro][s]["mse"]
            tiempo = resultados[k][filtro][s]["time"]

            porcentaje = (mse_val / base_mse) * 100 if base_mse != 0 else 0

            if filtro == "GAUSSIANO":
                print(f"\n{filtro} (sigma={s})")
            else:
                print(f"\n{filtro}")

            print("MSE:", round(mse_val,2))
            print("Porcentaje:", round(porcentaje,4), "%")
            print("Tiempo promedio:", f"{tiempo:.6f}", "seg")

#cv2.imshow("ROI", roi)
#cv2.imshow("MultiOtsu Regiones", (regions*120).astype(np.uint8))
cv2.imshow("Mascara Final", mask_codigo)

cv2.waitKey(0)
cv2.destroyAllWindows()


for k in kernel_sizes:

    for filtro in filtros:

        sigma_list = sigmas if filtro == "GAUSSIANO" else [None]

        for s in sigma_list:

            frame_show = frame.copy()
            roi_show = frame_show[y1:y2, x1:x2]

            for c in contours:
                if cv2.contourArea(c) > 500:

                    x, y, wc, hc = cv2.boundingRect(c)
                    sub = roi_show[y:y+hc, x:x+wc]

                    if filtro == "PROMEDIO":
                        blur = cv2.blur(sub, (k,k))

                    elif filtro == "GAUSSIANO":
                        blur = cv2.GaussianBlur(sub, (k,k), s)

                    roi_show[y:y+hc, x:x+wc] = blur

            nombre = f"{filtro}_k{k}"
            if s is not None:
                nombre += f"_s{s}"

            #cv2.imwrite(nombre + ".png", frame_show)

            cv2.imshow(nombre, frame_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()