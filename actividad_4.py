import cv2
import numpy as np
import time

video = r"C:\Users\rafae\OneDrive\Escritorio\Vision\Tarea2\video1\faltantes_3.mp4"
gt_path = r"C:\Users\rafae\OneDrive\Escritorio\Vision\Tarea2\gt_faltanes_130.png"

kernels = [3, 5]

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

# GT
gt = cv2.imread(gt_path)
if gt is None:
    print("Error cargando el ground truth")
    exit()

cap = cv2.VideoCapture(video)

h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# ROI
x1 = int(w * 0.50)
x2 = int(w * 0.76)
y1 = int(h * 0)
y2 = int(h * 0.70)


gt = cv2.resize(gt, (w, h))
cap.set(cv2.CAP_PROP_POS_FRAMES, 130)
ret, frame = cap.read()

if not ret:
    print("error frame")
    exit()

for k in kernels:

    fondo_blur = cv2.GaussianBlur(frame, (47, 47), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # SOBEL
    t0 = time.perf_counter()
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    sobel = cv2.convertScaleAbs(sobel)
    t1 = time.perf_counter()

    # LAPLACIANO
    t2 = time.perf_counter()
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=k)
    lap = cv2.convertScaleAbs(lap)
    t3 = time.perf_counter()

    # CANNY
    t4 = time.perf_counter()
    canny = cv2.Canny(gray, 100, 200)
    t5 = time.perf_counter()

    # PREWITT
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    t6 = time.perf_counter()
    prewitt = cv2.filter2D(gray, -1, kernelx)
    t7 = time.perf_counter()

    # ROBERTS
    kernelr = np.array([[1, 0], [0, -1]])
    t8 = time.perf_counter()
    roberts = cv2.filter2D(gray, -1, kernelr)
    t9 = time.perf_counter()

    # GRADIENTE
    t10 = time.perf_counter()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    grad = cv2.magnitude(gx, gy)
    grad = cv2.convertScaleAbs(grad)
    t11 = time.perf_counter()

    def combinar(edge):
        edge_roi = cv2.cvtColor(edge[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        result = fondo_blur.copy()
        result[y1:y2, x1:x2] = edge_roi
        return result

    res_s = combinar(sobel)
    res_l = combinar(lap)
    res_c = combinar(canny)
    res_p = combinar(prewitt)
    res_r = combinar(roberts)
    res_g = combinar(grad)

    print("Kernel:", k)
    

    # MSE vs gt
    print("CANNY:     MSE:", round(mse(gt, res_c), 2), " Tiempo:", round(t5 - t4, 4))
    print("GRADIENTE: MSE:", round(mse(gt, res_g), 2), " Tiempo:", round(t11 - t10, 4))
    print("LAPLACIANO:MSE:", round(mse(gt, res_l), 2), " Tiempo:", round(t3 - t2, 4))
    print("SOBEL:     MSE:", round(mse(gt, res_s), 2), " Tiempo:", round(t1 - t0, 4))
    print("PREWITT:   MSE:", round(mse(gt, res_p), 2), " Tiempo:", round(t7 - t6, 4))
    print("ROBERTS:   MSE:", round(mse(gt, res_r), 2), " Tiempo:", round(t9 - t8, 4))

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Original", frame)
    cv2.imshow("GT Manual", gt)
    cv2.imshow("Canny", res_c)
    cv2.imshow("Gradiente", res_g)
    cv2.imshow("Laplaciano", res_l)
    cv2.imshow("Sobel", res_s)
    cv2.imshow("Prewitt", res_p)
    cv2.imshow("Roberts", res_r)

    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()