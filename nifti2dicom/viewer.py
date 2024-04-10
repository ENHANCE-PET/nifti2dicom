import matplotlib
matplotlib.use('Qt5Agg')  # Use an interactive backend

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import numpy as np
import SimpleITK as sitk


def view_medical_images_pair(dicom_path, nifti_path):
    """
    View DICOM and NIfTI medical images side by side with interactive sliders.

    Args:
        dicom_path (str): The path to the DICOM series directory.
        nifti_path (str): The path to the NIfTI file.
    """
    def load_image(path, image_type='dicom'):
        """
        Loads medical image from the specified path.

        Args:
            path (str): Path to the image file or directory.
            image_type (str): Type of the image, either 'dicom' or 'nifti'.

        Returns:
            numpy.array: Loaded image array.
            tuple: Physical spacing between pixels.
        """
        if image_type == 'dicom':
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(path)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
        elif image_type == 'nifti':
            image = sitk.ReadImage(path)
            image = sitk.Flip(image, [False, True, False])  # Flip the image to make it left to right
        else:
            raise ValueError("Unsupported image type. Use 'dicom' or 'nifti'.")

        image_array = sitk.GetArrayFromImage(image)

        spacing = image.GetSpacing()  # Physical spacing between pixels
        return image_array, spacing

    def create_view(ax, image, slice_idx, aspect_ratio, view='sagittal', image_type='DICOM'):
        """
        Creates a view of the medical image slice on the specified axis.

        Args:
            ax (matplotlib.axes.Axes): Matplotlib axis to plot the image on.
            image (numpy.array): Image array.
            slice_idx (int): Index of the slice to display.
            aspect_ratio (float): Aspect ratio of the image.
            view (str): Type of view, one of 'axial', 'coronal', or 'sagittal'.
            image_type (str): Type of the image, either 'DICOM' or 'NIfTI'.
        """
        if view == 'axial':
            img_data = image[slice_idx, :, :]
        elif view == 'coronal':
            img_data = image[:, :, slice_idx]
        else:  # Default to sagittal
            img_data = image[:, slice_idx, :]

        ax.imshow(img_data, cmap='gray', aspect=aspect_ratio, origin='lower')
        # Highlighting the image type in the title
        title = f'{view.capitalize()} View ({image_type})'
        ax.set_title(title, color='cyan' if image_type == 'DICOM' else 'magenta', fontsize=12, fontweight='bold')
        ax.axis('off')

    # Load DICOM and NIfTI images
    dicom_image, dicom_spacing = load_image(dicom_path, 'dicom')
    nifti_image, nifti_spacing = load_image(nifti_path, 'nifti')

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor('#353535')  # Background color of the figure
    plt.subplots_adjust(left=0.25, bottom=0.25)

    aspect_ratio_dicom = dicom_spacing[2] / dicom_spacing[0]
    aspect_ratio_nifti = nifti_spacing[2] / nifti_spacing[0]

    # Initial placeholder images for DICOM and NIfTI
    dicom_img = axes[0].imshow(np.zeros((dicom_image.shape[0], dicom_image.shape[2])), cmap='gray')
    nifti_img = axes[1].imshow(np.zeros((nifti_image.shape[0], nifti_image.shape[2])), cmap='gray')

    for ax in axes:
        ax.set_facecolor('black')  # Axes background color

    def update(val):
        slice_num = int(slider.val)
        view = radio.value_selected
        create_view(axes[0], dicom_image, slice_num, aspect_ratio_dicom, view, 'DICOM')
        create_view(axes[1], nifti_image, slice_num, aspect_ratio_nifti, view, 'NIfTI')
        fig.canvas.draw_idle()

    # Unified Slider for both DICOM and NIfTI images
    max_slices = max(dicom_image.shape[1], nifti_image.shape[1]) - 1
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03], facecolor='lightblue')
    slider = Slider(ax_slider, 'Slice', 0, max_slices, valinit=0, valfmt='%0.0f')
    slider.on_changed(update)

    # Radio Buttons for view selection
    rax = plt.axes([0.05, 0.5, 0.15, 0.15], facecolor='#353535')
    radio = RadioButtons(rax, ('axial', 'sagittal', 'coronal'), activecolor='lightgreen')
    radio.on_clicked(update)

    for label in radio.labels:
        label.set_color('white')  # Set color of radio button labels
    for circle in radio.circles:  # Adjust size and color of radio buttons
        circle.set_radius(0.05)
        circle.set_edgecolor('white')

    update(0)  # Initialize with the first view

    plt.show()




