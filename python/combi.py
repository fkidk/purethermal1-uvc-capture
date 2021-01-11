#!/usr/bin/env python
# -*- coding: utf-8 -*-

from uvctypes import *
import cv2
import numpy as np
try:
  from queue import Queue
except ImportError:
  from Queue import Queue
import platform

WIDTH  = 640
HEIGHT = 480

BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktoc(val):
  return (val - 27315) / 100.0

def ctok(val):
  return int(val * 100.0 + 27315)

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
  val = ktoc(val_k)
  cv2.putText(img, "{0:.1f}C".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def display_avg(img, val_k, color):
  val = ktoc(val_k)
  cv2.putText(img, f'{val:.1f}C', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

def color_area(img, thermal_img, threshold=30):
  _threshold = ctok(threshold)
  _, thresholded_img = cv2.threshold(thermal_img, _threshold, 255, cv2.THRESH_BINARY)
  th_img = np.zeros(shape=(HEIGHT, WIDTH, 3))
  th_img[:,:,2] = thresholded_img
  th_img = th_img.astype(np.uint8)
  return cv2.addWeighted(img, 0.5, th_img, 0.5, 0.0)

def get_average(thermal_img, threshold=30):
  _threshold = ctok(threshold)
  values = thermal_img
  values[values < _threshold] = 0  
  masked = np.ma.masked_equal(values, 0)
  avg = np.average(masked)  
  return avg

def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      cam = cv2.VideoCapture(0)
      cam.set(3, WIDTH)
      cam.set(4, HEIGHT)

      try:
        while True:
          data_raw = q.get(True, 500)
          _, img = cam.read()
          if data_raw is None:
            break
          data_raw = cv2.resize(data_raw[:,:], (WIDTH, HEIGHT))
          minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data_raw)
          img = color_area(img, data_raw, threshold=32)
          avg_tmp = get_average(data_raw, threshold=32)
#          display_temperature(img, minVal, minLoc, (255, 0, 0))
          display_temperature(img, maxVal, maxLoc, (0, 255, 255))
          display_avg(img, avg_tmp, (255, 0, 0))
          cv2.imshow('Lepton Radiometry', img)
          if cv2.waitKey(1) == 27:
            break
        cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()
