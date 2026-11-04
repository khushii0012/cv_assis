# CCA 3 – Computer Vision Practical Assignment (Units 3, 4, 5)
**Khushi Potfode | PRN 1272251002 | SY MCA, Division D**

## Contents
- `CV_Assignment_Khushi_Potfode.ipynb` – notebook with all four tasks
- `pipeline.py` – the same code as one script
- `input/` – the two original photographs (`IMG_first.jpg`, `IMG_second.jpg`)
- `output/` – result figures (fig1–fig15) and `results.json` (all metrics)
- `requirements.txt`

## Dataset
Photos are in `input/`. The video (`video.mp4`, ~10 s, 60 FPS, 476×848) is **not** committed to Git — place your own video next to `pipeline.py`. Rename the photos' folder paths as needed (script expects `IMG_first.jpg`, `IMG_second.jpg`, `video.mp4` in the working directory).

## Run
```
pip install -r requirements.txt
python pipeline.py        # creates out/ with figures and results.json
```
The YOLO11n weights (`yolo11n.pt`) download automatically on first run. In Colab, upload the three input files to `/content/` and run the notebook.

## Tasks
1. Task 1 – noise, Gaussian/median filtering, PSNR/SSIM, threshold, morphology, Canny, watershed
2. Task 2 – HOG, SIFT keypoints, ratio-test matching, 75% resize and 30° rotation tests
3. Task 3 – Haar cascade vs YOLO11n on photos and video frames
4. Task 4 – frame differencing, MOG2, Lucas–Kanade optical flow
