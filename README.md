# Spatial Filtering for Background Blurring, Barcode Anonymization and Edge Detection

This project explores the application of spatial filtering techniques in Computer Vision using real-world video sequences captured with an Intel RealSense D415 camera.

The work focuses on two major image processing tasks:

1. Low-pass filtering for image smoothing and information anonymization.
2. High-pass filtering for edge detection and object boundary extraction.

Several filtering techniques were evaluated quantitatively and qualitatively using Mean Squared Error (MSE) and processing time measurements.

## Project Objectives

* Analyze the behavior of low-pass filters for image smoothing.
* Compare Average and Gaussian filtering techniques.
* Apply selective blurring to backgrounds and barcodes.
* Evaluate edge detection algorithms.
* Measure the trade-off between image quality and computational cost.
* Compare quantitative and qualitative performance metrics.

## Experimental Setup

### Hardware

* Intel RealSense D415 Camera

### Software

* Python
* OpenCV
* NumPy
* pyrealsense2

### Capture Configuration

* Resolution: 640 × 480
* Frame Rate: 30 FPS
* Camera Height: 65 cm

## Experimental Scenarios

Videos were recorded on a conveyor belt under multiple conditions:

### Scenario 1

Aligned cans.

### Scenario 2

Misaligned cans.

### Scenario 3

Missing can detection.

### Scenario 4

Barcode anonymization.

## Part I: Low-Pass Spatial Filtering

Low-pass filters reduce high-frequency information, producing smoother images.

The following filters were evaluated:

### Average Filter

Computes the average intensity inside a kernel window.

Advantages:

* Strong smoothing effect.
* Effective for anonymization.
* Simple implementation.

### Gaussian Filter

Applies weighted smoothing using a Gaussian distribution.

Advantages:

* Better detail preservation.
* Adjustable smoothing through sigma (σ).
* More natural blur appearance.

## Window Sizes Evaluated

* 9 × 9
* 13 × 13
* 21 × 21
* 47 × 47 (reference blur level)

The 47×47 filter was used as the maximum blur reference.

## Performance Metric

### Mean Squared Error (MSE)

The similarity between filtered images and the reference blur was measured using:

```text
MSE = (1/MN) Σ[I(i,j)-K(i,j)]²
```

Lower MSE values indicate greater similarity to the reference blur.

## Key Findings

### Best Blur Performance

The Average Filter with a 21×21 kernel produced the closest approximation to the maximum blur reference.

Examples:

| Scenario        | MSE    |
| --------------- | ------ |
| Aligned Cans    | 107.77 |
| Misaligned Cans | 115.36 |
| Missing Can     | 116.97 |

### Fastest Processing

The Gaussian Filter with:

```text
σ = 3
Kernel = 9×9
```

consistently achieved the lowest execution times.

Benefits:

* Fast computation.
* Good visual quality.
* Efficient OpenCV implementation.

## Part II: Barcode Anonymization

To simulate privacy-preserving image processing, barcode regions were automatically segmented and blurred.

### Segmentation Method

Multi-Otsu thresholding with three classes was used to isolate barcode regions.

Pipeline:

```text
Image
   ↓
Multi-Otsu Segmentation
   ↓
ROI Extraction
   ↓
Low-Pass Filtering
   ↓
Barcode Anonymization
```

### Results

The experiments demonstrated that:

* Larger kernels produce stronger anonymization.
* Gaussian filtering preserves more visual structure.
* Average filtering maximizes information removal.
* Surface curvature and reflections increase processing difficulty.

## Part III: Edge Detection

High-pass filters were applied to extract object boundaries.

The following algorithms were compared:

### Roberts

* Fastest method.
* Thin edge generation.
* Lowest MSE values.

### Sobel

* Strong gradient detection.
* More detailed edge structures.

### Prewitt

* Similar behavior to Sobel.
* Slightly lower computational complexity.

### Laplacian

* Sensitive to noise.
* Detects rapid intensity changes.

### Canny

* Best visual quality.
* Multi-stage edge detection.
* Superior qualitative results.

## Evaluation Procedure

Images were compared against manually generated Ground Truth masks.

Two kernel configurations were evaluated:

### Kernel Size 3

* Preserves fine details.
* Produces thinner edges.

### Kernel Size 5

* Produces thicker edges.
* Introduces more error and noise.

## Edge Detection Results

### Best Quantitative Performance

Roberts consistently achieved the lowest MSE values across all scenarios.

Advantages:

* Simpler edge representation.
* Reduced false positives.
* Fast execution.

### Best Qualitative Performance

Canny generated the most visually accurate edges.

Advantages:

* Better contour continuity.
* Higher robustness.
* More realistic object boundaries.

## Comparative Summary

| Task              | Best Method            |
| ----------------- | ---------------------- |
| Maximum Blur      | Average Filter (21×21) |
| Fastest Blur      | Gaussian (σ=3, 9×9)    |
| Lowest Edge MSE   | Roberts                |
| Best Visual Edges | Canny                  |

## Technologies

* Python
* OpenCV
* NumPy
* Intel RealSense SDK
* Multi-Otsu Thresholding

## Applications

This project demonstrates techniques applicable to:

* Privacy-preserving vision systems
* Industrial inspection
* Conveyor belt monitoring
* Automated barcode anonymization
* Object boundary extraction
* Machine vision preprocessing
* Robotics perception systems

## Learning Outcomes

The project provides practical experience in:

* Spatial filtering
* Image smoothing
* Noise reduction
* Thresholding techniques
* ROI-based processing
* Edge detection algorithms
* Performance evaluation using MSE
* Real-time computer vision

## Author

Universidad de Guanajuato – Computer Vision Course

Rafael Alejandro Frías Cortez
