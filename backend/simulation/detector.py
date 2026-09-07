import cv2
import numpy as np


class BeaconDetector:

    def detect(self, frame):
        """
        Detect beacon in frame using adaptive thresholding.
        
        Returns:
            Dictionary with x, y, confidence or None if not detected
        """
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Use adaptive threshold to handle varying brightness
        # This works better with brightness randomization
        _, threshold = cv2.threshold(
            gray,
            0,  # Threshold value (ignored with OTSU)
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find largest bright contour
        largest = max(
            contours,
            key=cv2.contourArea
        )

        area = cv2.contourArea(largest)

        # Lower minimum area threshold to catch smaller beacons
        if area < 10:
            return None

        moments = cv2.moments(largest)

        if moments["m00"] == 0:
            return None

        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]

        # Calculate confidence based on area and circularity
        # Larger, more circular = higher confidence
        perimeter = cv2.arcLength(largest, True)
        if perimeter == 0:
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Confidence combines area and shape
        area_confidence = min(1.0, area / 500)  # Normalized by larger area
        shape_confidence = circularity  # 1.0 = perfect circle
        confidence = (area_confidence + shape_confidence) / 2

        return {
            "x": x,
            "y": y,
            "confidence": confidence
        }
