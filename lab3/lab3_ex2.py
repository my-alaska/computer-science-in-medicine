import vtk

# Load DICOM data
dir = 'mr_brainixA'
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dir)
reader.Update()

# Create vtkContourFilter for generating iso-surfaces
iso_value = 100
contourFilter = vtk.vtkContourFilter()
contourFilter.SetInputConnection(reader.GetOutputPort())
contourFilter.SetValue(0, iso_value)
contourFilter.Update()

# Define color transfer function
color_transfer_function = vtk.vtkColorTransferFunction()
color_transfer_function.AddRGBPoint(0, 1.0, 1.0, 1.0)  # White
color_transfer_function.AddRGBPoint(255, 1.0, 0.5, 0.0)  # Orange

# Mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(contourFilter.GetOutputPort())
mapper.SetLookupTable(color_transfer_function)

# vtkActor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# Renderer
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# Window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(renderer)
renWin.SetSize(800, 600)

# Interactor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# Slider to change iso value
iso_slider_rep = vtk.vtkSliderRepresentation2D()
iso_slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
iso_slider_rep.GetPoint1Coordinate().SetValue(.7, .1)
iso_slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
iso_slider_rep.GetPoint2Coordinate().SetValue(.9, .1)
iso_slider_rep.SetMinimumValue(0)
iso_slider_rep.SetMaximumValue(255)
iso_slider_rep.SetValue(iso_value)
iso_slider_rep.SetTitleText("ISO Value")

def update_iso_value(value, contourFilter):
    contourFilter.SetValue(0, value)
    renWin.Render()

iso_slider_widget = vtk.vtkSliderWidget()
iso_slider_widget.SetInteractor(iren)
iso_slider_widget.SetRepresentation(iso_slider_rep)
iso_slider_widget.SetAnimationModeToAnimate()
iso_slider_widget.EnabledOn()
iso_slider_widget.AddObserver('InteractionEvent', lambda obj, event: update_iso_value(obj.GetRepresentation().GetValue(), contourFilter))

iren.Initialize()
iren.Start()
