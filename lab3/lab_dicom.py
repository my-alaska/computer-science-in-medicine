import vtk

# --- source: read data
dir = "mr_brainixA"
reader = vtk.vtkDICOMImageReader()
reader.SetDirectoryName(dir)
reader.Update()
imageData = reader.GetOutput()

print(imageData.GetScalarRange())
print(imageData.GetDimensions())
print(imageData)

nFrames = imageData.GetDimensions()[2]
winWidth = 750
winCenter = 100

# --- filter: apply winWidth and winCenter
shiftScaleFilter = vtk.vtkImageShiftScale()
shiftScaleFilter.SetOutputScalarTypeToUnsignedChar()  # output type
shiftScaleFilter.SetInputConnection(reader.GetOutputPort())  # input connection
shiftScaleFilter.SetShift(-winCenter + 0.5 * winWidth)
shiftScaleFilter.SetScale(255.0 / winWidth)
shiftScaleFilter.SetClampOverflow(True)

# --- actor: imageActor (displays 2D images)
actor = vtk.vtkImageActor()
actor.SetDisplayExtent(
    0, 255, 0, 255, 0, nFrames - 1
)  # set region to display (xStart, xEnd, yStart, yEnd, zStart, zEnd)
actor.GetMapper().SetInputConnection(
    shiftScaleFilter.GetOutputPort()
)  # input connection

# --- renderer
ren1 = vtk.vtkRenderer()
ren1.AddActor(actor)

# --- window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren1)
renWin.SetSize(800, 600)

# --- interactor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)


# --- slider to change frame: callback class, sliderRepresentation, slider
class FrameCallback(object):
    def __init__(self, actor, renWin):
        self.renWin = renWin
        self.actor = actor

    def __call__(self, caller, ev):
        value = caller.GetSliderRepresentation().GetValue()
        actor.SetDisplayExtent(0, 255, 0, 255, int(value), int(value))
        self.renWin.Render()


sliderRep = vtk.vtkSliderRepresentation2D()
sliderRep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint1Coordinate().SetValue(0.7, 0.1)
sliderRep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
sliderRep.GetPoint2Coordinate().SetValue(0.9, 0.1)
sliderRep.SetMinimumValue(0)
sliderRep.SetMaximumValue(nFrames - 1)
sliderRep.SetValue(1)
sliderRep.SetTitleText("frame")

slider = vtk.vtkSliderWidget()
slider.SetInteractor(iren)
slider.SetRepresentation(sliderRep)
slider.SetAnimationModeToAnimate()
slider.EnabledOn()
slider.AddObserver("InteractionEvent", FrameCallback(actor, renWin))

# --- run
style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)
iren.Initialize()
iren.Start()
