"""A single RViz Image panel for synchronized depth comparison."""
import cv2
import numpy as np


def comparison_panel(images, metrics, stamp, rate):
    width, height = 400, 300
    canvas = np.full((height + 110, width * 3, 3), 24, dtype=np.uint8)
    titles = [('Stonefish GT | 0.3-10 m', (255, 90, 90)),
              ('LAS2-S | 0.3-10 m', (80, 255, 100)),
              ('Absolute error | 0-2 m', (240, 240, 240))]
    for index, (image, (title, color)) in enumerate(zip(images, titles)):
        x = index * width
        canvas[50:50+height, x:x+width] = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas, title, (x+12, 30), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 1, cv2.LINE_AA)
    text = (f"MAE {metrics['mae_m']:.3f} m | RMSE {metrics['rmse_m']:.3f} m | "
            f"Valid {metrics['valid_fraction']:.1%} | matched {rate:.1f} Hz")
    cv2.putText(canvas, text, (12, height+75), cv2.FONT_HERSHEY_SIMPLEX, .6, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'Acquisition: {stamp/1e9:.6f} s | exact timestamp match | black = invalid',
                (12, height+98), cv2.FONT_HERSHEY_SIMPLEX, .5, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def waiting_panel(counts, last_error, age):
    canvas = np.full((410, 1200, 3), 24, dtype=np.uint8)
    lines = ['DEPTH COMPARISON: waiting for a valid synchronized pair',
             f"Received: GT {counts['gt']} | LAS2-S {counts['estimate']} | valid pairs {counts['pairs']}",
             'Topics: native Stonefish depth + /stereo/depth',
             'Both images must have the same acquisition timestamp.']
    if age is not None:
        lines.append(f'Last valid pair: {age:.1f} s ago (wall time)')
    if last_error:
        lines.append(last_error[:130])
    for index, text in enumerate(lines):
        cv2.putText(canvas, text, (20, 60+index*45), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 190, 80), 1, cv2.LINE_AA)
    return canvas
