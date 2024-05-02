import vtk

# --- Load DICOM data ---
dir = "mr_brainixA"
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dir)
reader.Update()

# --- Volume Mapper ---
volume_mapper = vtk.vtkSmartVolumeMapper()
volume_mapper.SetInputConnection(reader.GetOutputPort())

# --- Color Transfer Function ---
color_transfer_function = vtk.vtkColorTransferFunction()
color_transfer_function.AddRGBPoint(0, 1.0, 1.0, 1.0)  # White
color_transfer_function.AddRGBPoint(255, 0.0, 0.0, 1.0)  # Blue

# --- Opacity Transfer Function ---
opacity_transfer_function = vtk.vtkPiecewiseFunction()

# --- Volume Property ---
volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_transfer_function)
volume_property.SetScalarOpacity(opacity_transfer_function)

# --- Volume Actor ---
volume_actor = vtk.vtkVolume()
volume_actor.SetMapper(volume_mapper)
volume_actor.SetProperty(volume_property)

# --- Renderer ---
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.0, 0.0, 0.0)
renderer.AddVolume(volume_actor)

# --- Window ---
render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Volume Rendering")
render_window.SetSize(800, 600)
render_window.AddRenderer(renderer)

# --- Interactor ---
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# --- Slider to Change Transfer Function Point ---
tf_slider_rep = vtk.vtkSliderRepresentation2D()
tf_slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
tf_slider_rep.GetPoint1Coordinate().SetValue(0.1, 0.1)
tf_slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
tf_slider_rep.GetPoint2Coordinate().SetValue(0.4, 0.1)
tf_slider_rep.SetMinimumValue(0.0)
tf_slider_rep.SetMaximumValue(1.0)
tf_slider_rep.SetValue(0.5)
tf_slider_rep.SetTitleText("Opacity")

tf_slider_widget = vtk.vtkSliderWidget()
tf_slider_widget.SetInteractor(interactor)
tf_slider_widget.SetRepresentation(tf_slider_rep)
tf_slider_widget.SetAnimationModeToAnimate()
tf_slider_widget.EnabledOn()
tf_slider_widget.AddObserver(
    "InteractionEvent",
    lambda obj, event: opacity_transfer_function.AddPoint(
        127, obj.GetRepresentation().GetValue()
    ),
)

# --- Run ---
interactor.Initialize()
render_window.Render()
interactor.Start()
