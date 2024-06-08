import os
import SimpleITK as sitk
from matplotlib import pyplot as plt


def save_combined_central_slice(fixed, moving, transform, file_name_prefix):
    global iteration_number
    central_indexes = [int(i / 2) for i in fixed.GetSize()]

    moving_transformed = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelIDValue()
    )
    # extract the central slice in xy, xz, yz and alpha blend them
    combined = [
        fixed[:, :, central_indexes[2]] + -moving_transformed[:, :, central_indexes[2]],
        fixed[:, central_indexes[1], :] + -moving_transformed[:, central_indexes[1], :],
        fixed[central_indexes[0], :, :] + -moving_transformed[central_indexes[0], :, :],
    ]

    # resample the alpha blended images to be isotropic and rescale intensity
    # values so that they are in [0,255], this satisfies the requirements
    # of the jpg format
    print(f"iteration : {iteration_number} - metric value : {reg.GetMetricValue()}")
    combined_isotropic = []
    for img in combined:
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        min_spacing = min(original_spacing)
        new_spacing = [min_spacing, min_spacing]
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / min_spacing))),
            int(round(original_size[1] * (original_spacing[1] / min_spacing))),
        ]
        resampled_img = sitk.Resample(
            img,
            new_size,
            sitk.Transform(),
            sitk.sitkLinear,
            img.GetOrigin(),
            new_spacing,
            img.GetDirection(),
            0.0,
            img.GetPixelIDValue(),
        )
        combined_isotropic.append(
            sitk.Cast(sitk.RescaleIntensity(resampled_img), sitk.sitkUInt8)
        )
    # tile the three images into one large image and save using the given file
    # name prefix and the iteration number
    sitk.WriteImage(
        sitk.Tile(combined_isotropic, (1, 3)),
        file_name_prefix + format(iteration_number, "03d") + ".jpg",
    )
    iteration_number += 1


os.makedirs("output", exist_ok=True)

# read the images
fixed_image = sitk.ReadImage("data/nativeFixed.mhd", sitk.sitkFloat32)
moving_image = sitk.ReadImage("data/tetniceMoving1.mhd", sitk.sitkFloat32)


plt.imshow(sitk.GetArrayFromImage(fixed_image[100]))
plt.plot()
plt.close()

transform = sitk.CenteredTransformInitializer(
    fixed_image,
    moving_image,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.MOMENTS,
)

# multi-resolution rigid registration using Mutual Information
reg_0 = sitk.ImageRegistrationMethod()
reg_0.SetMetricAsMeanSquares()
reg_0.SetMetricSamplingStrategy(reg_0.REGULAR)
reg_0.SetMetricSamplingPercentage(1)
reg_0.SetInterpolator(sitk.sitkLinear)
reg_0.SetOptimizerAsGradientDescent(
    learningRate=1.0,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=10,
)

reg_1 = sitk.ImageRegistrationMethod()
reg_1.SetMetricAsMattesMutualInformation()
reg_1.SetMetricSamplingStrategy(reg_1.REGULAR)
reg_1.SetMetricSamplingPercentage(1)
reg_1.SetInterpolator(sitk.sitkLinear)
reg_1.SetOptimizerAsGradientDescent(
    learningRate=0.5,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=20,
)

reg = reg_0

reg.SetInitialTransform(transform)

# add iteration callback, save central slice in xy, xz, yz planes
global iteration_number
iteration_number = 0
reg.AddCommand(
    sitk.sitkIterationEvent,
    lambda: save_combined_central_slice(
        fixed_image, moving_image, transform, "output/iteration"
    ),
)
print("Initial metric: ", reg.MetricEvaluate(fixed_image, moving_image))
final_transform = reg.Execute(fixed_image, moving_image)

print(
    "Optimizer's stopping condition, {0}".format(
        reg.GetOptimizerStopConditionDescription()
    )
)
print("Metric value after  registration: ", reg.GetMetricValue())

sitk.WriteTransform(final_transform, "output/ct2mrT1.tfm")
