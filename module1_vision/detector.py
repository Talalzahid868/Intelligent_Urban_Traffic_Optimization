from ultralytics import YOLO

model = YOLO("yolov8n.pt")


class Trafficdetector:
    def __init__(self, model_path=model):
        self.model = model_path
        self.vehicle_classes = [
            2,
            3,
            5,
            7,
        ]  # COCO class ids for car,motorcycle,bus,truck
        self.pedestrian_class = 0

    def detect(self, image):
        results = self.model(image, verbose=False)
        vehicle_count = 0
        pedestrain_count = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.vehicle_classes:
                    vehicle_count += 1
                elif cls_id == self.pedestrian_class:
                    pedestrain_count += 1

        return vehicle_count, pedestrain_count
