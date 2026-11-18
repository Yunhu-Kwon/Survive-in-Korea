"""Run the latest VGG pipeline on Linux with CUDA; no account secrets required."""
import argparse,os,subprocess,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--input',required=True,help='Folder containing official extracted competition data')
parser.add_argument('--output',default='./vgg_final_outputs')
args=parser.parse_args();env=os.environ.copy()
env['STATEFARM_INPUT_ROOT']=str(Path(args.input).resolve())
env['STATEFARM_OUTPUT_ROOT']=str(Path(args.output).resolve())
root=Path(__file__).resolve().parent
for step in ['train_full_cv','train_crop_cv','train_face_full','calibrate_final']:
    subprocess.run([sys.executable,str(root/'src'/(step+'.py'))],env=env,check=True)
