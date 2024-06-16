import copy
import os
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt


def save_combined_central_slice(fixed, moving, transform, file_name_prefix):
    global iteration_number
    central_indices = [int(i / 2) for i in fixed.GetSize()]

    moving_transformed = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelIDValue()
    )

    # extract the central slice in xy, xz, yz and alpha blend them
    combined = [
        fixed[:, :, central_indices[2]] + -moving_transformed[:, :, central_indices[2]],
        fixed[:, central_indices[1], :] + -moving_transformed[:, central_indices[1], :],
        fixed[central_indices[0], :, :] + -moving_transformed[central_indices[0], :, :],
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
        file_name_prefix + format(iteration_number, "03d") + ".png",
    )
    iteration_number += 1


def get_checkerboard(fixed, moving, c_size):
    # c_size - checkerboard square size
    central_indices = [int(i / 2) for i in fixed.GetSize()]

    fixed = sitk.Cast(fixed[:, :, central_indices[2]], sitk.sitkUInt8)
    moving = sitk.Cast(moving[:, :, central_indices[2]], sitk.sitkUInt8)

    size_x = fixed.GetWidth()
    size_y = fixed.GetHeight()

    cboard = sitk.Image(size_x, size_y, sitk.sitkUInt8)

    for y in range(0, size_y, c_size):
        for x in range(0, size_x, c_size):
            if (x // c_size + y // c_size) % 2 == 0:
                cboard[x : x + c_size, y : y + c_size] = fixed[
                    x : x + c_size, y : y + c_size
                ]
            else:
                cboard[x : x + c_size, y : y + c_size] = moving[
                    x : x + c_size, y : y + c_size
                ]

    return cboard


def plot_3d(image, title, file_name):
    # Przekształcenie obrazu do tablicy numpy
    image_array = sitk.GetArrayFromImage(image)

    # Uzyskanie indeksów niezerowych pikseli
    non_zero_indices = np.where(image_array > 0)

    if non_zero_indices[0].shape[0] == 0:
        return

    # Określenie granic obrazu
    x_min, x_max = non_zero_indices[0].min(), non_zero_indices[0].max()
    y_min, y_max = non_zero_indices[1].min(), non_zero_indices[1].max()
    z_min, z_max = non_zero_indices[2].min(), non_zero_indices[2].max()

    # Utworzenie figur
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Wyświetlenie obrazu 3D
    ax.scatter(
        non_zero_indices[0],
        non_zero_indices[1],
        non_zero_indices[2],
        c=image_array[non_zero_indices],
        cmap="gray",
        marker=".",
    )

    # Ustawienie tytułu i etykiet osi
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Ustawienie granic wykresu na podstawie obrazu
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    plt.savefig(file_name)
    plt.close()


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

reg_2 = sitk.ImageRegistrationMethod()
reg_2.SetMetricAsMattesMutualInformation()
reg_2.SetMetricSamplingStrategy(reg_2.REGULAR)
reg_2.SetMetricSamplingPercentage(1)
reg_2.SetInterpolator(sitk.sitkLinear)
reg_2.SetOptimizerAsGradientDescent(
    learningRate=0.2,
    numberOfIterations=100,
    convergenceMinimumValue=1e-6,
    convergenceWindowSize=5,
)

reg_3 = sitk.ImageRegistrationMethod()
reg_3.SetMetricAsMattesMutualInformation()
reg_3.SetMetricSamplingStrategy(reg_3.REGULAR)
reg_3.SetMetricSamplingPercentage(1)
reg_3.SetInterpolator(sitk.sitkBSpline)
reg_3.SetOptimizerAsRegularStepGradientDescent(
    learningRate=0.1,
    minStep=0.01,
    numberOfIterations=100,
    gradientMagnitudeTolerance=1e-5,
    maximumStepSizeInPhysicalUnits=0.0,
)


if __name__ == "__main__":
    # read the images
    fixed = sitk.ReadImage("data/nativeFixed.mhd", sitk.sitkFloat32)
    moving = sitk.ReadImage("data/tetniceMoving1.mhd", sitk.sitkFloat32)

    regs = [reg_0, reg_1, reg_2, reg_3]

    for i, reg in enumerate(regs):
        os.makedirs(f"output{i}", exist_ok=True)
        fixed_image = copy.deepcopy(fixed)
        moving_image = copy.deepcopy(moving)

        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.MOMENTS,
        )

        reg.SetInitialTransform(transform)

        # add iteration callback, save central slice in xy, xz, yz planes
        global iteration_number
        iteration_number = 0
        reg.AddCommand(
            sitk.sitkIterationEvent,
            lambda: save_combined_central_slice(
                fixed_image, moving_image, transform, f"output{i}/iteration"
            ),
        )
        print("Initial metric: ", reg.MetricEvaluate(fixed_image, moving_image))
        final_transform = reg.Execute(fixed_image, moving_image)

        moving_image_transformed = sitk.Resample(
            moving_image,
            fixed_image,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving_image.GetPixelIDValue(),
        )

        checkerboard_image = get_checkerboard(fixed_image, moving_image_transformed, 10)

        # Wyświetlenie obrazu szachownicy
        plt.imshow(sitk.GetArrayFromImage(checkerboard_image), cmap="gray")
        plt.axis("off")
        plt.savefig(f"output{i}/checkerboard")
        plt.close()

        plot_3d(fixed_image, "fixed image 3d", f"output{i}/3d_fixed")
        plot_3d(moving_image_transformed, "fixed image 3d", f"output{i}/3d_moving")

        print(
            f"Optimizer's stopping condition, {reg.GetOptimizerStopConditionDescription()}"
        )
        print(f"Metric value after  registration: {reg.GetMetricValue()}\n")

        sitk.WriteTransform(final_transform, f"output{i}/ct2mrT1.tfm")
