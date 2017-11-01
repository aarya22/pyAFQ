# coding=utf-8
import os
import json
import boto3
import pandas as pd
import dask.dataframe as ddf
import glob
import os.path as op
import nibabel as nib
import dipy.core.gradients as dpg

afq_home = op.join(op.expanduser('~'), 'AFQ_data')
BUNDLES = ["ATR", "CGC", "CST", "HCC", "IFO", "ILF", "SLF", "ARC", "UNC"]

@profile
def fetch_hcp(subjects):
    """
    Fetch HCP diffusion data and arrange it in a manner that resembles the
    BIDS [1]_ specification.

    Parameters
    ----------
    subjects : list
       Each item is an integer, identifying one of the HCP subjects

    Returns
    -------
    dict with remote and local names of these files.

    Notes
    -----
    To use this function, you need to have a file '~/.aws/credentials', that
    includes a section:

    [hcp]
    AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
    AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX

    The keys are credentials that you can get from HCP (see https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)  # noqa

    Local filenames are changed to match our expected conventions.

    .. [1] Gorgolewski et al. (2016). The brain imaging data structure,
           a format for organizing and describing outputs of neuroimaging
           experiments. Scientific Data, 3::160044. DOI: 10.1038/sdata.2016.44.
    """
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')
    base_dir = op.join(afq_home, "HCP")
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    data_files = {}
    for subject in subjects:
        # We make a single session folder per subject for this case, because
        # AFQ api expects session structure:
        sub_dir = op.join(base_dir, 'sub-%s' % subject)
        sess_dir = op.join(sub_dir, "sess-01")
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
            os.mkdir(sess_dir)
            os.mkdir(os.path.join(sess_dir, 'dwi'))
            os.mkdir(os.path.join(sess_dir, 'anat'))
        data_files[op.join(sess_dir, 'dwi', 'sub-%s_dwi.bval' % subject)] =\
            'HCP/%s/T1w/Diffusion/bvals' % subject
        data_files[op.join(sess_dir, 'dwi', 'sub-%s_dwi.bvec' % subject)] =\
            'HCP/%s/T1w/Diffusion/bvecs' % subject
        data_files[op.join(sess_dir, 'dwi', 'sub-%s_dwi.nii.gz' % subject)] =\
            'HCP/%s/T1w/Diffusion/data.nii.gz' % subject
        data_files[op.join(sess_dir, 'anat', 'sub-%s_T1w.nii.gz' % subject)] =\
            'HCP/%s/T1w/T1w_acpc_dc.nii.gz' % subject
        data_files[op.join(sess_dir, 'anat',
                           'sub-%s_aparc+aseg.nii.gz' % subject)] =\
            'HCP/%s/T1w/aparc+aseg.nii.gz' % subject

    for k in data_files.keys():
        if not op.exists(k):
            bucket.download_file(data_files[k], k)
    # Create the BIDS dataset description file text
    dataset_description = {
         "BIDSVersion": "1.0.0",
         "Name": "HCP",
         "Acknowledgements": """Data were provided by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.""",  # noqa
         "Subjects": subjects}

    with open(op.join(base_dir, 'dataset_description.json'), 'w') as outfile:
        json.dump(dataset_description, outfile)

    return data_files

@profile
class AFQ(object):
    """
    This is file folder structure that AFQ requires in your study folder::

        ├── sub01
        │   ├── sess01
        │   │   ├── anat
        │   │   │   ├── sub-01_sess-01_aparc+aseg.nii.gz
        │   │   │   └── sub-01_sess-01_T1.nii.gz
        │   │   └── dwi
        │   │       ├── sub-01_sess-01_dwi.bvals
        │   │       ├── sub-01_sess-01_dwi.bvecs
        │   │       └── sub-01_sess-01_dwi.nii.gz
        │   └── sess02
        │       ├── anat
        │       │   ├── sub-01_sess-02_aparc+aseg.nii.gz
        │       │   └── sub-01_sess-02_T1w.nii.gz
        │       └── dwi
        │           ├── sub-01_sess-02_dwi.bvals
        │           ├── sub-01_sess-02_dwi.bvecs
        │           └── sub-01_sess-02_dwi.nii.gz
        └── sub02
            ├── sess01
            │   ├── anat
            │       ├── sub-02_sess-01_aparc+aseg.nii.gz
            │   │   └── sub-02_sess-01_T1w.nii.gz
            │   └── dwi
            │       ├── sub-02_sess-01_dwi.bvals
            │       ├── sub-02_sess-01_dwi.bvecs
            │       └── sub-02_sess-01_dwi.nii.gz
            └── sess02
                ├── anat
                │   ├── sub-02_sess-02_aparc+aseg.nii.gz
                │   └── sub-02_sess-02_T1w.nii.gz
                └── dwi
                    ├── sub-02_sess-02_dwi.bvals
                    ├── sub-02_sess-02_dwi.bvecs
                    └── sub-02_sess-02_dwi.nii.gz

    For now, it is up to users to arrange this file folder structure in their
    data, with preprocessed data, but in the future, this structure will be
    automatically generated from BIDS-compliant preprocessed data [1]_.

    Notes
    -----
    The structure of the file-system required here resembles that specified
    by BIDS [1]_. In the future, this will be organized according to the
    BIDS derivatives specification, as we require preprocessed, rather than
    raw data.

    .. [1] Gorgolewski et al. (2016). The brain imaging data structure,
           a format for organizing and describing outputs of neuroimaging
           experiments. Scientific Data, 3::160044. DOI: 10.1038/sdata.2016.44.

    """
    @profile
    def __init__(self, raw_path=None, preproc_path=None,
                 sub_prefix="sub", dwi_folder="dwi",
                 dwi_file="*dwi", anat_folder="anat",
                 anat_file="*T1w*", seg_file='*aparc+aseg*',
                 b0_threshold=0, odf_model="DTI", directions="det",
                 bundle_list=BUNDLES, dask_it=False,
                 force_recompute=False,
                 wm_labels=[251, 252, 253, 254, 255, 41, 2]):
        """

        b0_threshold : int, optional
            The value of b under which it is considered to be b0. Default: 0.

        odf_model : string, optional
            Which model to use for determining directions in tractography
            {"DTI", "DKI", "CSD"}. Default: "DTI"

        directions : string, optional
            How to select directions for tracking (deterministic or
            probablistic) {"det", "prob"}. Default: "det".

        dask_it : bool, optional
            Whether to use a dask DataFrame object

        force_recompute : bool, optional
            Whether to ignore previous results, and recompute all, or not.

        wm_labels : list, optional
            A list of the labels of the white matter in the segmentation file
            used. Default: the white matter values for the segmentation
            provided with the HCP data: [251, 252, 253, 254, 255, 41, 2].
        """
        self.directions = directions
        self.odf_model = odf_model
        self.raw_path = raw_path
        self.bundle_list = bundle_list
        self.force_recompute = force_recompute
        self.wm_labels = wm_labels

        self.preproc_path = preproc_path
        if self.preproc_path is None:
            if self.raw_path is None:
                e_s = "must provide either preproc_path or raw_path (or both)"
                raise ValueError(e_s)
            # This creates the preproc_path such that everything else works:
            self.preproc_path = do_preprocessing(self.raw_path)
        # This is the place in which each subject's full data lives
        self.subject_dirs = glob.glob(op.join(preproc_path,
                                              '%s*' % sub_prefix))
        self.subjects = [op.split(p)[-1] for p in self.subject_dirs]
        sub_list = []
        sess_list = []
        dwi_file_list = []
        bvec_file_list = []
        bval_file_list = []
        anat_file_list = []
        seg_file_list = []
        for subject, sub_dir in zip(self.subjects, self.subject_dirs):
            sessions = glob.glob(op.join(sub_dir, '*'))
            for sess in sessions:
                dwi_file_list.append(glob.glob(op.join(sub_dir,
                                                       ('%s/%s/%s.nii.gz' %
                                                        (sess, dwi_folder,
                                                         dwi_file))))[0])

                bvec_file_list.append(glob.glob(op.join(sub_dir,
                                                        ('%s/%s/%s.bvec*' %
                                                         (sess, dwi_folder,
                                                          dwi_file))))[0])

                bval_file_list.append(glob.glob(op.join(sub_dir,
                                                        ('%s/%s/%s.bval*' %
                                                         (sess, dwi_folder,
                                                          dwi_file))))[0])

                anat_file_list.append(glob.glob(op.join(sub_dir,
                                                        ('%s/%s/%s.nii.gz' %
                                                         (sess,
                                                          anat_folder,
                                                          anat_file))))[0])

                seg_file_list.append(glob.glob(op.join(sub_dir,
                                                       ('%s/%s/%s.nii.gz' %
                                                        (sess,
                                                         anat_folder,
                                                         seg_file))))[0])

                sub_list.append(subject)
                sess_list.append(sess)

        self.data_frame = pd.DataFrame(dict(subject=sub_list,
                                            dwi_file=dwi_file_list,
                                            bvec_file=bvec_file_list,
                                            bval_file=bval_file_list,
                                            anat_file=anat_file_list,
                                            seg_file=seg_file_list,
                                            sess=sess_list))
        if dask_it:
            self.data_frame = ddf.from_pandas(self.data_frame,
                                              npartitions=len(sub_list))
        self.set_gtab(b0_threshold)
        self.set_dwi_affine()

    def set_gtab(self, b0_threshold):
        self.data_frame['gtab'] = self.data_frame.apply(
            lambda x: dpg.gradient_table(x['bval_file'], x['bvec_file'],
                                         b0_threshold=b0_threshold),
            axis=1)

    def get_gtab(self):
        return self.data_frame['gtab']

    gtab = property(get_gtab, set_gtab)

    def set_dwi_affine(self):
        self.data_frame['dwi_affine'] =\
            self.data_frame['dwi_file'].apply(_get_affine)

    def get_dwi_affine(self):
        return self.data_frame['dwi_affine']

    dwi_affine = property(get_dwi_affine, set_dwi_affine)

    def __getitem__(self, k):
        return self.data_frame.__getitem__(k)

    def set_brain_mask(self, median_radius=4, numpass=4, autocrop=False,
                       vol_idx=None, dilate=None):
        if ('brain_mask_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['brain_mask_file'] =\
                self.data_frame.apply(_brain_mask,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_brain_mask(self):
        self.set_brain_mask()
        return self.data_frame['brain_mask_file']

    brain_mask = property(get_brain_mask, set_brain_mask)

    def set_dti(self):
        if ('dti_params_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['dti_params_file'] =\
                self.data_frame.apply(_dti,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti(self):
        self.set_dti()
        return self.data_frame['dti_params_file']

    dti = property(get_dti, set_dti)

    def set_dti_fa(self):
        if ('dti_fa_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['dti_fa_file'] =\
                self.data_frame.apply(_dti_fa,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti_fa(self):
        self.set_dti_fa()
        return self.data_frame['dti_fa_file']

    dti_fa = property(get_dti_fa, set_dti_fa)

    def set_dti_md(self):
        if ('dti_md_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['dti_md_file'] =\
                self.data_frame.apply(_dti_md,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_dti_md(self):
        self.set_dti_md()
        return self.data_frame['dti_md_file']

    dti_md = property(get_dti_md, set_dti_md)

    def set_mapping(self):
        if 'mapping' not in self.data_frame.columns or self.force_recompute:
            self.data_frame['mapping'] =\
                self.data_frame.apply(_mapping,
                                      axis=1,
                                      force_recompute=self.force_recompute)

    def get_mapping(self):
        self.set_mapping()
        return self.data_frame['mapping']

    mapping = property(get_mapping, set_mapping)

    def set_streamlines(self):
        if ('streamlines_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['streamlines_file'] =\
                self.data_frame.apply(_streamlines, axis=1,
                                      args=[self.wm_labels],
                                      odf_model=self.odf_model,
                                      directions=self.directions,
                                      force_recompute=self.force_recompute)

    def get_streamlines(self):
        self.set_streamlines()
        return self.data_frame['streamlines_file']

    streamlines = property(get_streamlines, set_streamlines)

    def set_bundles(self):
        if ('bundles_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['bundles_file'] =\
                self.data_frame.apply(_bundles, axis=1,
                                      args=[self.wm_labels],
                                      odf_model=self.odf_model,
                                      directions=self.directions,
                                      force_recompute=self.force_recompute)

    def get_bundles(self):
        self.set_bundles()
        return self.data_frame['bundles_file']

    bundles = property(get_bundles, set_bundles)

    def set_tract_profiles(self):
        if ('tract_profiles_file' not in self.data_frame.columns or
                self.force_recompute):
            self.data_frame['tract_profiles_file'] =\
                self.data_frame.apply(_tract_profiles,
                                      args=[self.wm_labels],
                                      force_recompute=self.force_recompute,
                                      axis=1)

    def get_tract_profiles(self):
        self.set_tract_profiles()
        return self.data_frame['tract_profiles_file']

    tract_profiles = property(get_tract_profiles, set_tract_profiles)


def _get_affine(fname):
    return nib.load(fname).get_affine()


def _get_fname(row, suffix):
    split_fdwi = op.split(row['dwi_file'])
    fname = op.join(split_fdwi[0], split_fdwi[1].split('.')[0] +
                    suffix)
    return fname

if __name__ == '__main__':
    fetch_hcp(['992774', '994273'])
    base_dir = op.join(op.expanduser('~'), 'AFQ_data', 'HCP')
    myafq = AFQ(preproc_path=base_dir, sub_prefix='sub')
