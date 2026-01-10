def estimate_congestion(vehicle_count):

    if vehicle_count < 10:
        return "LOW"
    elif vehicle_count < 25:
        return "Medium"
    else:
        return "High"



