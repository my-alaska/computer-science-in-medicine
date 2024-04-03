# AGH UST Medical Informatics 03.2021
# Lab 2 : DICOM
import numpy as np
import pydicom
from tkinter import *
from PIL import Image, ImageTk


class MainWindow:
    ds = pydicom.dcmread("data/head.dcm")
    data = ds.pixel_array

    def __init__(self, main):
        # print patient name
        self.start_x = None
        self.start_y = None

        print("Width  :", self.ds.WindowWidth)
        print("Center :", self.ds.WindowCenter)

        print(self.ds.PatientName)

        # todo: from ds get windowWidth and windowCenter
        self.winWidth, self.winCenter = self.ds.WindowWidth, self.ds.WindowCenter

        # prepare canvas
        self.canvas = Canvas(main, width=512, height=512)
        self.canvas.grid(row=0, column=0)
        self.canvas.bind("<Button-1>", self.init_window)
        self.canvas.bind("<B1-Motion>", self.update_window)
        self.canvas.bind("<Button-2>", self.reset_window)
        self.canvas.bind("<Button-3>", self.init_measurement)
        self.canvas.bind("<B3-Motion>", self.update_measurement)
        self.canvas.bind("<ButtonRelease-3>", self.finish_measurement)

        # load image
        # todo: apply transform
        self.array = self.transform_data(self.data, self.winWidth, self.winCenter)

        self.array = self.data
        self.image = Image.fromarray(self.array)

        self.spacing_x, self.spacing_y = self.ds.PixelSpacing

        # ANTIALIAS is deprecated. Use LANCZOS instead
        self.image = self.image.resize((512, 512), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(image=self.image, master=root)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.img)

    def transform_data(self, data, window_width, window_center):
        # todo: transform data (apply window width and center)

        window_min = window_center - window_width / 2
        window_max = window_center + window_width / 2

        transformed_data = np.clip(data, window_min, window_max)
        transformed_data = (
            (transformed_data - window_min) / (window_max - window_min)
        ) * 255

        return transformed_data.astype(np.uint8)

    def init_window(self, event):
        # todo: save mouse position
        self.start_x, self.start_y = event.x, event.y
        print("x: " + str(event.x) + " y: " + str(event.y))

    def update_window(self, event):
        # todo: modify window width and center

        delta_x = event.x - self.start_x
        delta_y = event.y - self.start_y

        self.array2 = self.transform_data(
            self.data, self.winWidth + delta_x, self.winCenter + delta_y
        )
        self.image2 = Image.fromarray(self.array2)
        self.image2 = self.image2.resize((512, 512), Image.LANCZOS)
        self.img2 = ImageTk.PhotoImage(image=self.image2, master=root)
        self.canvas.itemconfig(self.image_on_canvas, image=self.img2)

    def reset_window(self, event):
        self.array = self.data
        self.image = Image.fromarray(self.array)

        # ANTIALIAS is deprecated. Use LANCZOS instead
        self.image = self.image.resize((512, 512), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(image=self.image, master=root)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=NW, image=self.img)

    def init_measurement(self, event):
        # todo: save mouse position
        # todo: create line

        self.start_measurement_x, self.start_measurement_y = event.x, event.y
        self.measurement_line = self.canvas.create_line(
            self.start_measurement_x,
            self.start_measurement_y,
            event.x,
            event.y,
            fill="red",
        )

    def update_measurement(self, event):
        # todo: update line
        # hint: self.canvas.coords(...)
        self.canvas.coords(
            self.measurement_line,
            self.start_measurement_x,
            self.start_measurement_y,
            event.x,
            event.y,
        )

    def finish_measurement(self, event):
        # todo: print measured length in mm
        print("x: " + str(event.x) + " y: " + str(event.y))
        length_pixels = (
            ((event.x - self.start_measurement_x) / self.spacing_x) ** 2
            + ((event.y - self.start_measurement_y) / self.spacing_y) ** 2
        ) ** 0.5
        print("Measured length: {:.2f} mm".format(length_pixels))


# ----------------------------------------------------------------------

root = Tk()
MainWindow(root)
root.mainloop()
