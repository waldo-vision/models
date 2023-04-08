"""
Downloads youtube videos from a csv of URLs
"""
import argparse
import pandas as pd
from pytube import YouTube

# Setup command line arguments
parser = argparse.ArgumentParser(description="Get URL's from API and store them locally")
parser.add_argument("-i", "--input", help="Path to CSV of Youtube URLs to download",
                    required=True, type=str)
parser.add_argument("-o", "--output", help='Folder To Store Downloaded Videos',
                    required=True, type=str)
parser.add_argument("-r", "--max_resolution", help='The maximum resolution downloads allowed',
                    required=False, type=str, default='1080')

args = vars(parser.parse_args())

def download_best_video(row):
    """Downloads the Highest resolution, Highest FPS stream of a given video URL from YouTube"""
    # create YT object
    uuid = row['uuid']
    url = row['url']
    yt_obj = YouTube(url)
    print(f"Downloading UUID: {uuid} | URL:{url}")

    # get list of all mp4 video streams
    all_streams = yt_obj.streams.filter(file_extension='mp4',type='video')

    # get set of all available resolutions
    available_resolutions = []
    for stream in all_streams:
        available_resolutions.append(stream.resolution.split('p')[0])
    available_res= [*set(available_resolutions)]

    # get highest resolution not greater than max res
    best_res = max([int(res) for res in available_res if int(res) <= int(args['max_resolution'])])
    best_res = str(best_res) + 'p'

    # get highest fps stream at best res
    best_stream = yt_obj.streams.filter(file_extension='mp4', type='video', res=str(best_res)) \
                                .order_by('fps') \
                                .last()

    # download video
    best_stream.download(output_path=args['output'],
                        filename=f'{uuid}.mp4',
                        skip_existing=True,
                        max_retries=1)

    return True

def main(path_to_csv):
    """Download URLs from input CSV from YouTube"""
    # read csv into dataframe
    print(f'Reading Input CSV: {path_to_csv}')
    urls_df = pd.read_csv(path_to_csv)

    # download all videos
    print('Beginning Downloads...')
    urls_df.apply(download_best_video, axis=1)

    print('Downloads Complete')

if __name__ == "__main__":
    main(args['input'])
