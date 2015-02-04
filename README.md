OCR on google glass using tess-two library
Google glass frame resolution: 1920x1080, we only recognize the region inside the bounding box

Usage:
Long Press -- Exit
Single Tap -- Zoom in
Double Tap -- Zoom out
Fling      -- Move the bounding box up and down

Misc:
Three ways of converting YUV420sp to rgba format:
1, Manually decoding (the first width X height bytes encodes the gray information)
2, Using BitFactory, 1st convert to JPG, then to Bitmap, very slow
3, Using opencv4android library, 1st to mat then to bitmap

we found the performance does not have significant improvement when using color image instead of gray image
