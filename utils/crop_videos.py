import argparse
from DefaultVideoCropper import DefaultVideoCropper


parser = argparse.ArgumentParser(description="Crop a video and store frames locally", add_help=False)
parser.add_argument("-i", "--input", help='Input Video', required=True, type=str)
parser.add_argument("-w", "--width", help='Cropping width', required=True, type=int)
parser.add_argument("-h", "--height", help='Cropping height', required=True, type=int)
parser.add_argument("-o", "--output", help='Folder To Store Output', required=True, type=str)
parser.add_argument("-x", "--x_position", help='Crop x position', required=False, default=(0), type=int)
parser.add_argument("-y", "--y_position", help='Crop y position', required=False, default=(0), type=int)

args = vars(parser.parse_args())

def extract_vid_uuid(vid_path:str) -> str:
    vid_name = vid_path.split("\\")[-1]
    uuid = vid_name.split(".")[0]
    return uuid

def main() -> None:
    cropper = DefaultVideoCropper(output_dir=args['output'], cropping_dimensions=(args['width'], args['height']), crop_position=(args['x_position'], args['y_position']))
    uuid = extract_vid_uuid(args['input'])
    cropper.process_video(args['input'], uuid)

if __name__=="__main__":
    main()