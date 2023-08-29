import modules.BodyMotion as BodyMotion
import modules.preprocessing.normalize_video as norm
import modules.preprocessing.temp_segmentation as temp
import warnings 
import sys
import argparse
warnings.filterwarnings("ignore") 

video_name = "C:\\Users\\panto\\Videos\\video_tirocinio\\test.mp4"
#video_name = "C:\\Users\\panto\\Videos\\video_tirocinio\\video\\test1\\clip0.mp4"
#video_name = "C:\\Users\\panto\\Videos\\video_tirocinio\\video\\test1\\output.mp4"
#out_name = "C:\\Users\\panto\\Videos\\video_tirocinio\\video\\test1"
video_name = "C:\\Users\\panto\\Videos\\video_tirocinio\\video\\test1\\tagliati\\clip1.mp4"
#video_name = "C:\\Users\\panto\\Videos\\video_tirocinio\\video\\clip3.mp4"
#video_name = "output\\new_video.mp4"
output_dir = "output"
file_json = "data\\face_encodings2.json"

#pre.extract_songs(video_name, out_name, plot= True)
#pre.normalize_rotation(video_name, output_dir + "/new_video.mp4", 50)



def main():
    parser = argparse.ArgumentParser(description="Programma con comandi obbligatori e facoltativi")
    
    # Comando obbligatorio
    parser.add_argument("command", type=str, help="Comando da eseguire")
    parser.add_argument("video_name", type=str, help="Percorso del file in input")
    parser.add_argument("output", type=str, help="Percorso del file in output")

    #face_encodings_file = None, segmentation_freq = 0.1, face_detection_freq = 0.3, optical_flow_freq=6, max_width=1000
    parser.add_argument("-fef", "--face_encodings_file", type=str, help="Percorso del file di lettura dei volti")
    parser.add_argument("-sF", "--segmentation_freq", type=float, help="Frequenza di segmentazione")
    parser.add_argument("-fdF", "--face_detection_freq", type=float, help="Frequenza di ricerca dei volti")
    parser.add_argument("-ofF", "--optical_flow_freq", type=float, help="Frequenza di calcolo del flusso ottico")
    parser.add_argument("-mw", "--max_width", type=int, help="Ridimensionamento larghezza video")
    
    args = parser.parse_args()
    
    if args.command == "analizza":
        video_name = args.video_name
        output = args.output

        arg1 = args.face_encodings_file
        arg2 = args.segmentation_freq
        arg3 = args.face_detection_freq
        arg4 = args.optical_flow_freq
        arg5 = args.max_width

        if arg2 is None:
            arg2 = 0.1
        if arg3 is None:
            arg3 = 0.3
        if arg4 is None:
            arg4 = 6
        if arg5 is None:
            arg5 = 1000

        BodyMotion.compute_motion_values(   video_name,
                                    output, 
                                    segmentation_freq = arg2, 
                                    face_detection_freq=arg3,
                                    optical_flow_freq=arg4, 
                                    max_width=arg5, 
                                    face_encodings_file=arg1)
        
    elif args.command == "estrai_canzoni":
        video_name = args.video_name
        output_dir = args.output_name
        temp.extract_songs(video_name, output_dir, plot= False)
    
    elif args.command == "norm_video":
        video_name = args.video_name
        output = args.output + "/new_video.mp4"
        parser.add_argument("-p", "--padding", type=int, help="Dimensione padding verticale in pixel")
        args = parser.parse_args()
        arg1 = args.padding

        if arg1 is None:
            arg1 = 0

        norm.normalize_rotation(video_name, output, arg1)

    else:
        print("Comando non riconosciuto")

if __name__ == "__main__":
    main()




