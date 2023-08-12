
## Nifti2Dicom 🧠💽

Hello there, brave soul! 🌟 Welcome to **Nifti2Dicom** - a project born out of sheer determination, despair, and probably a smidge too much caffeine. ☕️ Ever felt like converting NIfTI to DICOM was akin to summoning a minor demon? 😈 So did we. Which is why we created this snazzy tool to prevent any more unplanned infernal conferences.


## Prerequisites 📋

- **OS**: Universal - because we don't discriminate. 🌍
- **Python**: Version 3.9 required, because even we have our limits. 🐍

## Installation 🔧

Wave a magic wand... just kidding. Do this:

```bash
pip install nifti2dicom
```

## Usage 🚀

Using the mighty **Nifti2Dicom** is (thankfully) less complicated than its origin story:

1. Open your command line or terminal. 💻
2. Enter the following command, replacing the placeholders with your actual paths and desired series description:
   
   ```bash
   nifti2dicom <reference_dir> <nifti_path> <output_dir> "<series_description>"
   ```

   **Arguments:**
   - `reference_dir`: Path to the directory containing the reference DICOM series.
   - `nifti_path`: Path to the NIfTI file you wish to convert.
   - `output_dir`: Path to the directory where you'd like the converted DICOM files to reside.
   - `series_description`: A description to be added to the DICOM header. Wrap it in quotes if it contains spaces!

   Example:

   ```bash
   nifti2dicom ./refDICOM ./brainMRI.nii ./convertedDICOM "Fancy Brain Scan"
   ```

## Issues & Feedback 🐛🗣

If you stumble upon any pesky bugs, or have suggestions to prevent other unforeseen exorcisms, [Open an issue](https://github.com/LalithShiyam/nifti2dicom/issues). Also, if you ever come up with a way to bring peace between NIfTI and DICOM, we're all ears (and eyes 👀)!

## License 📜

This project is licensed under the MIT License. Check out the `LICENSE` file to see the fine print.

## Acknowledgments 👏

- To coffee, our eternal ally. ☕️
- The patience of everyone who ever sat near a developer (me) while they mumbled about DICOM headers.
- The spirit animal of this project: A platypus, because just like this software, it's unique, unexpected, and gets the job done!

---

Alright, the emojis should add some fun and flair to your README! Remember to update the `logo_path_here.png` with the actual path if you do have a logo for your project.
