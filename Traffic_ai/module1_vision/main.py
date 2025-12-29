import os,csv,cv2
from detector import Trafficdetector
from utils import estimate_congestion


# Paths

Image_dir="../archive/bdd100k/bdd100k/images/10k/val"
Output_CSV="../outputs/module1_results.csv"

# script_dir=os.path.dirname(os.path.abspath(__file__))
# Image_dir= os.path.join(script_dir,"..","archive","bdd100k","images","10k","val")
# Output_CSV=os.path.join(script_dir,"..","Outputs","Module1_results.csv")

os.makedirs("../Outputs",exist_ok=True)

detector=Trafficdetector()

with open(Output_CSV,"w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["image_name","vehicle_count","Pedestrian_count","congestion"])

    for img_name in os.listdir(Image_dir):
        if not img_name.endswith(".jpg"):
            continue

        img_path=os.path.join(Image_dir,img_name)
        image=cv2.imread(img_path)

        vehicle_count,pedestrain_count=detector.detect(image)
        congestion=estimate_congestion(vehicle_count=vehicle_count)

        writer.writerow([img_name,vehicle_count,pedestrain_count,congestion])

        print(f"{img_name}->Vehicles:{vehicle_count},Pedestrian:{pedestrain_count},Congestion:{congestion}")
print("Result of module 1 saved")






