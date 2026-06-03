import cv2
import numpy as np
import time

video = r"C:\Users\rafae\OneDrive\Escritorio\Vision\Tarea2\video1\faltantes_3.mp4"

kernels = [9, 13, 21]
base_kernel = 47
sigmas = [3, 7]

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

cap = cv2.VideoCapture(video)

# Capturar un frame
frame_number = 130
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

if not ret:
    print("Error al leer el frame")
    cap.release()
    exit()

h, w = frame.shape[:2]

# ROI
x1 = int(w * 0.50)
x2 = int(w * 0.76)
y1 = 0
y2 = int(h * 0.70)

# Mask
mask = np.ones((h, w), dtype=np.float32)
mask[y1:y2, x1:x2] = 0
mask = cv2.GaussianBlur(mask, (101,101), 50)
mask = cv2.merge([mask, mask, mask])

# base a comparar RECUERDA QUE DE AQUI SE BASA EL 100%
t0 = time.perf_counter()
base_blur = cv2.blur(frame, (base_kernel, base_kernel))
t1 = time.perf_counter()

base_result = (frame*(1-mask) + base_blur*mask).astype(np.uint8)

mse_base = mse(frame, base_result)
t_base = (t1 - t0)

print("BASE 47x47")
print("MSE:", round(mse_base,2))
print("Porcentaje: 100%")
print("Tiempo:", round(t_base,6), "seg")

# ciclo por kernel
for k in kernels:
    print("Kernel", k, "x", k)

    # PROMEDIO
    t0 = time.perf_counter()
    blur = cv2.blur(frame, (k,k))
    t1 = time.perf_counter()

    resultado = (frame*(1-mask) + blur*mask).astype(np.uint8)
    mse_avg = mse(base_result, resultado)
    t_avg = (t1 - t0)

    print("\nPROMEDIO")
    print("MSE:", round(mse_avg,2))
    print("Porcentaje:", round((mse_avg/mse_base)*100,4), "%")
    print("Tiempo:", round(t_avg,6), "seg")

    cv2.imshow(f"Promedio {k}", resultado)

    # GAUSSIANO
    for s in sigmas:
        t0 = time.perf_counter()
        gauss = cv2.GaussianBlur(frame, (k,k), s)
        t1 = time.perf_counter()

        resultado_g = (frame*(1-mask) + gauss*mask).astype(np.uint8)
        mse_gauss = mse(base_result, resultado_g)
        t_gauss = (t1 - t0)

        print(f"\nGAUSSIANO sigma={s}")
        print("MSE:", round(mse_gauss,2))
        print("Porcentaje:", round((mse_gauss/mse_base)*100,4), "%")
        print("Tiempo:", round(t_gauss,6), "seg")

        cv2.imshow(f"Gauss {k} sigma{s}", resultado_g)

# Mostrar base y original
cv2.imshow("Base 47x47", base_result)
#cv2.imshow("Original", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()