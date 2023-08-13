![nifti2dicom-logo](/Nifti2dicom-logo.png)


## Nifti2Dicom 🧠💽

Hello there, brave soul! 🌟 Welcome to **Nifti2Dicom** - a project born out of sheer determination, despair, pain and probably a smidge too much caffeine. ☕️ Ever felt like converting NIfTI to DICOM was akin to summoning a minor demon from the pandora's box? 😈 So did we. Which is why we created this snazzy tool to prevent any more unplanned infernal conferences.

## Magic Powers (Features) 🌟

🌌 Dimensional Doorways - Step into our magical portal! Whether you're jumping into a 3D realm or a more mysterious 4D time-warp, we've got you covered. Convert both 3D and 4D nifti images to DICOM. So, if you're clutching a 4D motion-corrected series in nifti, don't fret. We're your dimensional travel agency!
🎨 The Colorful Canvas of Segmentations - Ever dreamt of painting the universe with multilabel nifti segmentations? Well, maybe not. But hey, we can convert those vibrant dreams into 3D DICOM for you. Just hand over your brush, or in this case, your label to region mapping (tutorial brewing in our cauldron), and watch the masterpiece unfold!

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
- The spirit animal of this project: A platypus, because just like this software, it's unique, unexpected, and gets the job done (to a reasonable extent - we are managing expectations here)!


## 🎩🔮 A Gentle Wizardly Reminder 🔮🎩

Dear adventurous user, while Nifti2Dicom is sprinkled with a generous dose of magic and wizardry, it's essential to remember that no spell is perfect. Just like the age-old "turn a frog into a prince" trick, sometimes things don't pan out (ask any fairy tale princess).

If you ever find yourself uttering "It doesn't work!" take a deep breath, consider the vastness of the cosmos, and remember — our tool isn't the answer to every cosmic conundrum. It's not a magic bullet (or wand) that'll work wonders in every scenario. But fret not, intrepid one! Reach out, and together, let's see if we can make a tad more magic happen.
