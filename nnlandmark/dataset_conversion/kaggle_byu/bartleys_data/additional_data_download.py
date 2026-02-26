import concurrent.futures
import glob
import os
import shutil
import blosc2
import numpy as np
import zarr
from batchgenerators.utilities.file_and_folder_operations import save_json, maybe_mkdir_p, join, load_json
from cryoet_data_portal import Client, Dataset, Run


class CziiCollector():
    """
    This class was used to redownload Brendan's data and convert his annotations to the original image shape.
    You don't need to use this again. Please use RawDatasetDownloader() instead!
    """
    def __init__(
            self,
            tmp_dir: str = "/media/isensee/data/kaggle_byu_additional_data/tmp/",
            out_dir: str = "/media/isensee/data/kaggle_byu_additional_data/data/",
            labels_csv: str = '/home/isensee/git_repos/nnu-netv2/nnunetv2/dataset_conversion/kaggle_byu/labels.csv',
            dataset_author: str = 'Yi-Wei Chang',
            num_threads: int = 3
    ):
        super().__init__()
        self.tmp_dir = tmp_dir
        self.out_dir = out_dir
        self.dataset_author = dataset_author
        self.num_threads = num_threads

        self.labels = np.loadtxt(labels_csv, delimiter=',', skiprows=1, dtype=str) if labels_csv is not None else None

        # Tmp dir
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        os.makedirs(self.tmp_dir)

        # Out dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def process_tomogram(self, x, tomo_id):
        assert self.labels is not None, "Please provide a labels_csv"
        cols = np.where(self.labels[:, 3] == tomo_id)[0]
        if len(cols) > 0:
            annotations_made_in_shape = np.array((128, 512, 512))
            coords = self.labels[cols, :3].astype(float)

            # we need to revert the coordinates of z
            z_coords = coords[:, 0].astype(float)
            # revert z slicing
            indices = list(np.linspace(0, x.shape[0] - 1, 128).astype(int))
            z_coords_orig = np.array([indices[int(z)] for z in z_coords]).reshape((len(coords), -1))

            # now convert xy of the annotations
            xy_coords = coords[:, 1:3]
            orig_anno = xy_coords * ((np.array(x.shape[1:]) - 1) / (annotations_made_in_shape[1:] - 1))
            orig_anno = np.hstack((z_coords_orig, orig_anno))
            orig_anno = [list(i) for i in orig_anno]
        else:
            orig_anno = None

        return x, orig_anno

    def run(self):
        # ========== Query Datasets ==========
        client = Client()

        # Datasets by Author
        ds_all = Dataset.find(client, [Dataset.authors.name == self.dataset_author])
        print("=" * 25)
        print("N_DATASETS:", len(ds_all))
        print("=" * 25)

        # ========= Process Each Dataset ==========
        for ds in ds_all:
            s = "TOTAL: {:<10}     TITLE: {}".format(len(ds.runs), ds.title)
            print(s)

            # Create dataset directory
            ds_dir = os.path.join(self.out_dir, str(ds.id))
            if not os.path.exists(ds_dir):
                os.mkdir(ds_dir)

            # Define a helper function to process a single run
            def process_run(run):
                outpath = os.path.join(ds_dir, str(run.name)) + ".b2nd"
                if os.path.exists(outpath[:-5] + '.json'):
                    print(f'skipping {run.name}')
                    return

                my_temp = join(self.tmp_dir, run.name)
                maybe_mkdir_p(my_temp)
                try:
                    # Download tomo
                    if len(run.tomograms) == 0:
                        print(f'run {run.name} has no tomograms')
                        return
                    tomo = run.tomograms[0]
                    tomo.download_omezarr(dest_path=my_temp)

                    # Load tomo
                    fpath = glob.glob(join(my_temp, "*"))[0]
                    arr = zarr.open(fpath, mode='r')
                    spacing = arr.attrs['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale']
                    assert spacing[0] == tomo.voxel_spacing
                    arr = arr['0']

                    # Preprocess
                    arr, anno = self.process_tomogram(arr, run.name)

                    # Save
                    clevel: int = 8
                    codec = blosc2.Codec.ZSTD
                    cparams = {'codec': codec, 'clevel': clevel, 'nthreads': 8}
                    blosc2.asarray(
                        np.ascontiguousarray(arr),
                        urlpath=outpath,
                        chunks=(64, 64, 64),
                        blocks=(16, 16, 16),
                        cparams=cparams,
                        mode='w'
                        # mmap_mode='w+'
                    )
                    save_json(
                        {'annotation': anno, 'spacing': spacing},
                        outpath[:-5] + '.json',
                        sort_keys=False
                    )
                    del arr
                    shutil.rmtree(my_temp)

                except KeyboardInterrupt as e:
                    shutil.rmtree(my_temp)
                    raise e

                except Exception as e:
                    import IPython
                    IPython.embed()
                    print(e)
                    print("FAILED:", ds, run, outpath)
                    raise e

            # Process runs in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(process_run, run) for run in ds.runs]
                # Optionally, wait for all to finish and catch exceptions
                for future in concurrent.futures.as_completed(futures):
                    future.result()  # This will re-raise any exception

        return




