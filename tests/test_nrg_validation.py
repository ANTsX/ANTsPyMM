
import unittest
from antspymm import validate_nrg_file_format  # Replace 'your_module' with the actual name of your Python module

ntfn='/Users/ntustison/Data/Stone/LIMBIC/NRG/ANTsLIMBIC/sub08C105120Yr/ses-1/rsfMRI_RL/000/ANTsLIMBIC_sub08C105120Yr_ses-1_rsfMRI_RL_000.nii.gz'
ntfngood='/Users/ntustison/Data/Stone/LIMBIC/NRG/ANTsLIMBIC/sub08C105120Yr/ses_1/rsfMRI_RL/000/ANTsLIMBIC-sub08C105120Yr-ses_1-rsfMRI_RL-000.nii.gz'

ntfngood2='/Users/ntustison/Data/Stone/LIMBIC/NRG/ANTsLIMBIC////sub08C105120Yr///ses_1/rsfMRI_RL/000////ANTsLIMBIC-sub08C105120Yr-ses_1-rsfMRI_RL-000.nii.gz'

class TestValidateNrgDetailed(unittest.TestCase):
    
    def test_valid_path(self):
        path = '/correct/structure/path/StudyName/SubjectID/20240101/Modality/001/StudyName-SubjectID-20240101-Modality-001.nii.gz'
        separator = '-'
        result, message = validate_nrg_file_format(path, separator)
        self.assertTrue(result, "Valid path should return True")

    def test_invalid_extension(self):
        path = '/incorrect/extension/path/StudyName-SubjectID-20240101-Modality-001/StudyName-SubjectID-20240101-Modality-001.txt'
        separator = '-'
        result, message = validate_nrg_file_format(path, separator)
        self.assertFalse(result, "Path with invalid extension should return False")

    def test_incomplete_path(self):
        path = '/incomplete/path/StudyName-SubjectID'
        separator = '-'
        result, message = validate_nrg_file_format(path, separator)
        self.assertFalse(result, "Incomplete path should return False")

    def test_mismatch_directory_structure(self):
        path = '/mismatch/structure/StudyName-SubjectID-20240101-Modality-001/AnotherName-SubjectID-20240101-Modality-002.nii.gz'
        separator = '-'
        result, message = validate_nrg_file_format(path, separator)
        self.assertFalse(result, "Path with mismatched directory and filename structure should return False")

    def test_nick_00(self):
        separator = '-'
        result, message = validate_nrg_file_format(ntfn, separator)
        self.assertFalse(result, "Path with mismatched directory and filename structure should return False")

    def test_nick_01(self):
        separator = '_'
        result, message = validate_nrg_file_format(ntfn, separator)
        self.assertFalse(result, "Path with mismatched directory and filename structure should return False")

    def test_nick_02(self):
        separator = '-'
        result, message = validate_nrg_file_format(ntfngood, separator)
        self.assertTrue(result, "nrg-etic! superior nrg format construction")

    def test_nick_03(self):
        separator = '-'
        result, message = validate_nrg_file_format(ntfngood2, separator)
        self.assertTrue(result, "nrg-etic! superior nrg format construction but weird slashes")

# Run the test suite
if __name__ == '__main__':
    unittest.main()
