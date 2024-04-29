import vtk

# source
cube = vtk.vtkCubeSource()

# filter
filter = vtk.vtkTransformPolyDataFilter()
filter.SetInputConnection(cube.GetOutputPort())
transform = vtk.vtkTransform()
transform.RotateX(30)
transform.RotateY(-45)
filter.SetTransform(transform)

# mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(filter.GetOutputPort())

# actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.0, 0.5, 1.0)

# renderer
renderer = vtk.vtkRenderer()
renderer.SetBackground(0.0, 0.0, 0.0)
renderer.AddActor(actor)

# window
render_window = vtk.vtkRenderWindow()
render_window.SetWindowName("Cube")
render_window.SetSize(800, 600)
render_window.AddRenderer(renderer)

# interactor
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# run
interactor.Initialize()
render_window.Render()
interactor.Start()