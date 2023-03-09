import os
import requests
import h5py
import subprocess

from tqdm import tqdm

def download_file(url, filename):
    '''
    Function to download a file from a URL with a progress bar
    '''
    # get the response object using the requests library
    r = requests.get(url, stream=True)
    download = False
    try:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))

        with open(filename, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            miniters=1,
            desc=filename
        ) as progress_bar:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        download = True
        

    except requests.exceptions.RequestException as e:
        print('Download failed: ', e)
        with open(os.path.join(original_folder, 'failed_downloads.txt'), 'a') as f:
            f.write('Download failed: ' + str(e) + '\n')

            return download
        
def create_directory(directory):
    '''
    Function to create a directory if it does not exist
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)

    # change the current working directory to the new folder
    os.chdir(directory)

def repack_hdf5(filename):
    '''
    Function to repack an HDF5 file and overwrite the original file with the repacked version
    '''
    # Run ptrepack command to repack the file
    subprocess.run(['ptrepack', '--chunkshape=auto', '--propindexes', '--complevel=9', '--complib=blosc', filename, f"{filename}_repack"])

    # Delete the original file
    os.remove(filename)

    # Rename the repacked file to the original file name
    os.rename(f"{filename}_repack", filename)


####################
# Preprocessing code
# TO BE CHECKED
####################
def preprocess_file(filename):
    '''
    Function to preprocess the file
    '''
    if ".h5" not in filename:
        pass
    else:
        print(f"Preprocessing file: {filename}")

        # Open the file in read-write mode
        with h5py.File(filename, 'r+') as f:
            # List all the groups in the file
            print('Groups:', list(f.keys()))

            # Access a group and list its contents
            group = f['radar']
            print('Group contents:', list(group.keys()))

            # Access a dataset and print its shape and contents
            for key1 in group.keys():
                print(key1)
                group_group = group[key1]
                print('Group contents:', list(group_group.keys()))

                for key2 in group_group.keys():
                    if key2 == 'cube':
                        # Delete the 'cube' dataset
                        del group_group[key2]
                        print('Deleted dataset:', key2)

        repack_hdf5(filename)

def main():
    '''
    Main function
    '''
    # create a directory to store the files
    create_directory('DATA')
    global original_folder
    original_folder= os.getcwd()

    subjects = [f'subject_{i:02d}' for i in range(10, 20)]
    captured_data = 'captured_data'
    sets = [f'set{i:03d}' for i in range(0, 4)]
    files = ['timestamp_speech.csv', 'processed_data.h5']

    # loop over each subject
    for subject in subjects:
        print(f"Processing subject: {subject}")
        create_directory(subject)

        create_directory(captured_data)

        # loop over each set
        for set in sets:
            print(f"Processing set: {set}")
            create_directory(set)

            for file in files:
                print(f"Downloading file: {file}")
                d = download_file(f'https://cloud.ilabt.imec.be/index.php/s/eRkdk6NnJZNL5XG/download?path=%2F{subject}%2F{captured_data}%2F{set}&files={file}', file)
                if d:
                    if '.h5' in file:
                        preprocess_file(file)
                    #try:
                    #    os.remove(file)
                    #except OSError:
                    #    pass
            # change the current working directory to the parent directory
            os.chdir('..')

        # change the current working directory to the parent directory
        os.chdir('..')

        # change the current working directory to the parent directory
        os.chdir('..')


if __name__ == '__main__':
    main()