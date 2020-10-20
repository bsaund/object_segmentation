"""
This is a helper to automatically download pretrained files.
If this causes issues, download the google drive pretrained model manually
"""

file_id = '17eV88dp33_Kxqt3ke_C1DoRICeOICh-4'


def ycb(dest_dir):
    try:
        from google_drive_downloader import GoogleDriveDownloader as gdd
    except Exception as e:
        print("Failed to import gdd.")
        print("Either `pip install googledrivedownloader`")
        print("or manually download checkpoint from google drive")
        raise e

    gdd.download_file_from_google_drive(file_id=file_id,
                                        dest_path=(dest_dir / 'ycbvideo-mobilenetv2dilated-c1_deepsup.zip').as_posix(),
                                        unzip=True)
