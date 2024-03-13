![nifti2dicom-logo](/Nifti2dicom-logo.png)


## Nifti2Dicom 🧠💽

[![PyPI version](https://badge.fury.io/py/nifti2dicom.svg)](https://pypi.org/project/nifti2dicom/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://www.gnu.org/licenses/MIT)


Hello there, brave soul! 🌟 Welcome to **Nifti2Dicom** - a project born out of sheer determination, despair, pain and probably a smidge too much caffeine. ☕️ Ever felt like converting NIfTI to DICOM was akin to summoning a minor demon from the pandora's box? 😈 So did we. Which is why we created this snazzy tool to prevent any more unplanned infernal conferences.


## Magic Powers (Features) 🌟

🌌 Dimensional Doorways - Step into our magical portal! Whether you're jumping into a 3D realm or a more mysterious 4D time-warp, we've got you covered. Convert both 3D and 4D nifti images to DICOM. So, if you're clutching a 4D motion-corrected series in nifti, don't fret. We're your dimensional travel agency!
🎨 The Colorful Canvas of Segmentations - Ever dreamt of painting the universe with multilabel nifti segmentations? Well, maybe not. But hey, we can convert those vibrant dreams into 3D DICOM for you. Just hand over your brush, or in this case, your label to region mapping (tutorial brewing in our cauldron), and watch the masterpiece unfold!

## Prerequisites 📋

- **OS**: Universal - because we don't discriminate. 🌍
- **Python**: Version 3.9 is required because even we have our limits. 🐍

## Installation 🔧

We highly recommend using a different realm (virtual environment) for installing nifti2dicom (or any other Python package, really).

### Linux
```bash
python3 -m venv nifti2dicom
source nifti2dicom/bin/activate
```

### Windows
```bash
python -m venv nifti2dicom
nifti2dicom\Scripts\activate.bat  
```

And now, wave a magic wand... just kidding. Do this:

```bash
pip install nifti2dicom
```

## Usage 🚀

Using the mighty **Nifti2Dicom** is (thankfully) less complicated than its origin story:


1. Open your command line or terminal. 💻
2. Enter the following command, replacing the placeholders with your actual paths and desired series description:

### Converting 3d/4d images 
#### For 3d
```bash
   nifti2dicom \
       -d <dicom_dir>               # Directory containing reference DICOM series
       -n <nifti_path>              # Path to the NIFTI file to be converted
       -o <output_dir>              # Directory where the converted DICOM files will be saved
       -desc "<series_description>" # Description for the DICOM series
       -t img                       # Specifies the type of conversion (image in this case)
```
#### For 4d
```bash
   nifti2dicom \
       -d <dicom_dir>               # Directory containing reference DICOM series
       -n <nifti_path>              # Path to the NIFTI file to be converted
       -o <output_dir>              # Directory where the converted DICOM files will be saved
       -desc "<series_description>" # Description for the DICOM series
       -t img                       # Specifies the type of conversion (image in this case)
       -v <sms | ux>                # Specifies the vendor, either "sms" or "ux"
```
Ignore the vendor tag, if you are working on 3d images. It is just relevant for 4D. The logic is the same for 3D.

 ### Converting segmentations single/multilabel segmentations
```bash
    nifti2dicom \
       -d <dicom_dir>               # Directory containing reference DICOM series
       -n <nifti_path>              # Path to the NIFTI file to be converted
       -o <output_dir>              # Directory where the converted DICOM files will be saved
       -desc "<series_description>" # Description for the DICOM series
       -t seg                       # Specifies the type of conversion (segmentation in this case)
       -j <path_to_json>            # Path to the JSON file containing the organ index
```
For converting nifti segmentations to DICOM, you always need to a pass a json file, which contains the mapping of the labels to the region names. The sample .json file can be found [here](/labels_region.json).

### Example:

#### Segmentation conversion
  
```bash
nifti2dicom \
    -d ./refDICOM               # Reference directory with DICOM series
    -n ./brainSegmentation.nii  # Path to the NIFTI segmentation file
    -o ./convertedSegDICOM      # Output directory for the converted segmentation DICOM
    -desc "Brain Segmentation"  # Description for the DICOM series
    -t seg                      # Type of conversion: segmentation
    -j ./organ_index.json       # Path to the JSON file with organ index
```   

#### Image conversion
##### 3d conversion
```bash
nifti2dicom \
    -d ./refDICOM               # Reference directory with DICOM series
    -n ./brainMRI.nii           # Path to the NIFTI image file
    -o ./convertedImgDICOM      # Output directory for the converted image DICOM
    -desc "Fancy Brain Scan"    # Description for the DICOM series
    -t img                      # Type of conversion: image
```
                               
##### 4d conversion
```bash
nifti2dicom \
    -d ./refDICOM               # Reference directory with DICOM series
    -n ./brainMRI.nii           # Path to the NIFTI image file
    -o ./convertedImgDICOM      # Output directory for the converted image DICOM
    -desc "Fancy Brain Scan"    # Description for the DICOM series
    -t img                      # Type of conversion: image
    -v sms                      # Vendor: Siemens (you can replace this with the \
                                # appropriate vendor name, as of now support for united imaging \
                                # and siemens is provided. You can choose one of the two (sms or ux), the default value is ux)
```

 Still confused? You can always type ```nifti2dicom -h``` for help!

## Issues & Feedback 🐛🗣

If you stumble upon any pesky bugs, or have suggestions to prevent other unforeseen exorcisms, [Open an issue](https://github.com/LalithShiyam/nifti2dicom/issues). Also, if you ever come up with a way to bring peace between NIfTI and DICOM, we're all ears (and eyes 👀)!

## License 📜

This project is licensed under the MIT License. Check out the `LICENSE` file to see the fine print.

## Acknowledgments 👏

- To coffee, our eternal ally. ☕️
- The patience of everyone who ever sat near a developer (me) while they mumbled about DICOM headers.
- The spirit animal of this project: A platypus, because just like this software, it's unique, unexpected, and gets the job done (to a reasonable extent - we are managing expectations here)!


## 🎩🔮 A Gentle Wizardly Reminder 🔮🎩

Dear adventurous user, while Nifti2Dicom is sprinkled with a generous dose of magic and wizardry, it's essential to remember that no spell is perfect. Just like the age-old "turn a frog into a prince" trick, sometimes things don't pan out (ask any fairy tale princess). For example our segmentation conversion was tested on slicer 3D with QuantitativeReporting plugin, and we are not sure if it will work on everything.

If you ever find yourself uttering "It doesn't work!" take a deep breath, consider the vastness of the cosmos, and remember — our tool isn't the answer to every cosmic conundrum. It's not a magic bullet (or wand) that'll work wonders in every scenario. But fret not, intrepid one! Reach out, and together, let's see if we can make a tad more magic happen.
